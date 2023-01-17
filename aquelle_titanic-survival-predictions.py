# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import shap

from matplotlib import pyplot as plt

from xgboost import XGBClassifier

from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFECV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test['Survived'] = np.NaN



data = train.append(test, ignore_index = True, sort = True)
#impute missing values

data['FareBin'] = pd.qcut(data.Fare, 4)

data.Cabin = data.Cabin.fillna("U")

data.Embarked = data.Embarked.fillna("S")

data.Fare = data.Fare.fillna(data.Age.median())

    

data['Sex'] = data.Sex.map({'male':0, 'female': 1})

data['FareBin'] = data.FareBin.astype('category').cat.codes

data['Embarked'] = data.Embarked.map({'S':0, 'C': 1, 'Q': 2})

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1



guess_ages = np.zeros((2,3))

guess_ages



for i in range(0, 2):

    for j in range(0, 3):

        guess_df = data[(data['Sex'] == i) & \

                              (data['Pclass'] == j+1)]['Age'].dropna()

        age_guess = guess_df.median()

        

        # Convert random age float to nearest .5 age

        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



for i in range(0, 2):

    for j in range(0, 3):

        data.loc[ (data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j+1),\

                'Age'] = guess_ages[i,j]



data['Age'] = data['Age'].astype(int)

    

data['AgeBin'] = pd.cut(data.Age, 5).cat.codes



data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

replacementTitle = []

data['Title'] = np.where(data.Title == "Master", 2, data.Sex)

data['TicketFreq'] = data.Ticket.map(lambda x: data[data.Ticket == x].Ticket.count())

data['FareAdj'] = data.Fare/data.TicketFreq/10

data.FamilySize = data.SibSp + data.Parch + 1

data['FamAge'] = data.FamilySize + data.Age/70



data2 = data[(data.PassengerId<= 891) & (data.Title == 0)]

data3 = data[(data.PassengerId > 891) & (data.Title == 0)]
sns.kdeplot(data2[(data2.Survived == 1) & (data2.Pclass == 1)].FamAge, clip = (0,4), bw = 0.04)

sns.kdeplot(data2[(data2.Survived == 0) & (data2.Pclass == 1)].FamAge, clip = (0,4), bw = 0.04)
sns.kdeplot(10*data2[data2.Pclass == 1].FareAdj, clip = (0,40))

sns.kdeplot(10*data2[data2.Pclass == 2].FareAdj, clip = (0,40))

sns.kdeplot(10*data2[data2.Pclass == 3].FareAdj, clip = (0,40))
def PredictionToHue(x,p):

    if x > p:

        return [0,1,0,0.5]

    else:

        return [1,0,0,0.5]



xgb = XGBClassifier(max_depth = 5, eta = 0.1, gamma = 0.1, colsample_bytree = 1, min_child_weight = 1,

    n_estimators = 500)

xgb.fit(data2[['FareAdj', 'FamAge']], data2.Survived)

#evaluate xgb on a grid to plot the descision tree

x1s = np.linspace(0,5,100)

x2s = np.linspace(1,3,100)

p = []

for i in x1s:    

    g = pd.DataFrame({'FareAdj': np.full(100, i), 'FamAge': x2s})

    values = xgb.predict(g)

    hues = [PredictionToHue(x, 0.5) for x in values]

    p.append(hues)

#overlay a scatterplot of data3

p = [list(i) for i in zip(*p)] #* is splat operator

plt.figure(num = 1, figsize = (5,5))

plt.imshow(p, extent = [0,5,1,3], origin = "lower", aspect = 'auto')

plt.scatter(data2[(data2.FareAdj < 5) & (data2.FamAge < 3)].FareAdj, data2[(data2.FareAdj < 5) & (data2.FamAge < 3)].FamAge, c = data2[(data2.FareAdj < 5) & (data2.FamAge < 3)].Survived)

plt.show()
data2['p'] = [x[1] for x in xgb.predict_proba(data2[['FareAdj', 'FamAge']])]

roc = pd.DataFrame(columns=['TN', 'FN', 'TP', 'FP', 'FPR', 'TPR'])

for i in range(0,101):

    temp = pd.Series({

        'TN': data2[(data2.Survived == 0) & ~(data2.p >= (i-1)/100)].Survived.count(),

        'FP': data2[(data2.Survived == 0) & (data2.p >= (i-1)/100)].Survived.count(),

        'FN': data2[(data2.Survived == 1) & ~(data2.p >= (i-1)/100)].Survived.count(),

        'TP': data2[(data2.Survived == 1) & (data2.p >= (i-1)/100)].Survived.count()})

    temp['FPR'] = temp.FP/(temp.FP + temp.TN)

    temp['TPR'] = temp.TP/(temp.TP + temp.FN)

    roc = roc.append(temp, ignore_index = True)

rocArea = 0

for i in range(0,100):

    rocArea = rocArea + (roc.TPR[i+1] + roc.TPR[i])*(roc.FPR[i] - roc.FPR[i+1])/2

print("Area under ROC curve: " + str(rocArea))

sns.scatterplot(roc.FPR, roc.TPR, np.linspace(0,1,101), palette = 'RdBu')
data3['p'] = [x[1] for x in xgb.predict_proba(data3[['FareAdj', 'FamAge']])]

p = []

for i in x1s:    

    g = pd.DataFrame({'FareAdj': np.full(100, i), 'FamAge': x2s})

    values = [x[1] for x in xgb.predict_proba(g)]

    hues = [PredictionToHue(x, 0.9) for x in values]

    p.append(hues)

#overlay a scatterplot of data3

p = [list(i) for i in zip(*p)] #* is splat operator

plt.figure(num = 1, figsize = (5,5))

plt.imshow(p, extent = [0,5,1,3], origin = "lower", aspect = 'auto')

plt.scatter(data3[(data3.FareAdj < 5) & (data3.FamAge < 3)].FareAdj, data3[(data3.FareAdj < 5) & (data3.FamAge < 3)].FamAge)

plt.show()

data3[data3.p > 0.9]
data['Surname'] = data.Name.map(lambda x: x[:x.index(',')])

data['GroupId'] = data.Pclass.map(str)+ '-' + data.Ticket.map(str) + '-' + data.Fare.map(str)

data.GroupId = np.where(data.Title == 0, 'None', data.GroupId) 

## Mrs Wilkes (Needs) is Mrs Hocking (Needs) sister

data.GroupId.iloc[892] = data.GroupId.iloc[774]

data['GroupFreq'] = data.GroupId.map(lambda x: data.GroupId[data.GroupId == x].count())

data.GroupId = np.where(data.GroupFreq <= 1, 'None', data.GroupId) 

print('We found ' + str(data.GroupId.unique().size - 1) + ' woman-child-groups')



data['GroupSurvival'] = np.nan

data.GroupSurvival = data.GroupId.map(lambda x: data[data.GroupId == x].Survived.mean())

## classify unknown groups

data.GroupSurvival = np.where((data.Pclass == 3) & (data.GroupSurvival.isna()), 0, data.GroupSurvival)

data.GroupSurvival = np.where((data.Pclass != 3) & (data.GroupSurvival.isna()), 1, data.GroupSurvival)

## make predictions

data['Predict'] = 0

data.Predict[data.Sex == 1] = 1

data.Predict[(data.Title == 1) & (data.GroupSurvival == 0)] = 0

data.Predict[(data.Title == 2) & (data.GroupSurvival == 1)] = 1

print('We found ' + str(data[(data.Title == 2) & (data.Predict == 1) & (data.PassengerId > 891)].Predict.count()) + ' boys predicted to live')

print('We found ' + str(data[(data.Title == 1) & (data.Predict == 0) & (data.PassengerId > 891)].Predict.count()) + ' women predicted to die')
data2 = data[(data.PassengerId <= 891) & (data.Title == 1) & (data.FamilySize == 1)]

data3 = data[(data.PassengerId > 891) & (data.Title == 1) & (data.FamilySize == 1)]



xgb = XGBClassifier(max_depth = 5, eta = 0.1, gamma = 0.1, colsample_bytree = 1, min_child_weight = 1,

    n_estimators = 500)

xgb.fit(data2[['FareAdj', 'FamAge']], data2.Survived)

#evaluate xgb on a grid to plot the descision tree

x1s = np.linspace(0,5,100)

x2s = np.linspace(1,3,100)

p = []

for i in x1s:    

    g = pd.DataFrame({'FareAdj': np.full(100, i), 'FamAge': x2s})

    values = [x[1] for x in xgb.predict_proba(g)]

    hues = [PredictionToHue(x, 0.25) for x in values]

    p.append(hues)

#overlay a scatterplot of data3

p = [list(i) for i in zip(*p)] #* is splat operator

plt.figure(num = 1, figsize = (5,5))

plt.imshow(p, extent = [0,5,1,3], origin = "lower", aspect = 'auto')

plt.scatter(data2[(data2.FareAdj < 5) & (data2.FamAge < 3)].FareAdj, data2[(data2.FareAdj < 5) & (data2.FamAge < 3)].FamAge, c = data2[(data2.FareAdj < 5) & (data2.FamAge < 3)].Survived)

plt.xlim(0.5,1.5)

plt.ylim(1,2)

plt.show()
data3['p'] = [x[1] for x in xgb.predict_proba(data3[['FareAdj', 'FamAge']])]

p = []

for i in x1s:    

    g = pd.DataFrame({'FareAdj': np.full(100, i), 'FamAge': x2s})

    values = [x[1] for x in xgb.predict_proba(g)]

    hues = [PredictionToHue(x, 0.15) for x in values]

    p.append(hues)

#overlay a scatterplot of data3

p = [list(i) for i in zip(*p)] #* is splat operator

plt.figure(num = 1, figsize = (5,5))

plt.imshow(p, extent = [0,5,1,3], origin = "lower", aspect = 'auto')

plt.scatter(data3[(data3.FareAdj < 5) & (data3.FamAge < 3)].FareAdj, data3[(data3.FareAdj < 5) & (data3.FamAge < 3)].FamAge)

plt.ylim(1,2)

plt.xlim(0.5,1.2)

plt.show()

data3[data3.p < 0.15]
data.Predict[data.PassengerId.isin(data3.PassengerId[data3.p < 0.15])] = 0

print(data.iloc[897])
data2 = data[(data.PassengerId <= 891) & (data.Title == 1) & (data.FamilySize != 1) & (data.GroupId == 'None')]

data3 = data[(data.PassengerId > 891) & (data.Title == 1) & (data.FamilySize != 1) & (data.GroupId == 'None')]



xgb = XGBClassifier(max_depth = 5, eta = 0.1, gamma = 0.1, colsample_bytree = 1, min_child_weight = 1,

    n_estimators = 500)

xgb.fit(data2[['FareAdj', 'FamAge']], data2.Survived)

#evaluate xgb on a grid to plot the descision tree

x1s = np.linspace(0,5,100)

x2s = np.linspace(1,3,100)

p = []

for i in x1s:    

    g = pd.DataFrame({'FareAdj': np.full(100, i), 'FamAge': x2s})

    values = [x[1] for x in xgb.predict_proba(g)]

    hues = [PredictionToHue(x, 0.2) for x in values]

    p.append(hues)

#overlay a scatterplot of data3

p = [list(i) for i in zip(*p)] #* is splat operator

plt.figure(num = 1, figsize = (5,5))

plt.imshow(p, extent = [0,5,1,3], origin = "lower", aspect = 'auto')

plt.scatter(data2[(data2.FareAdj < 5) & (data2.FamAge < 3)].FareAdj, data2[(data2.FareAdj < 5) & (data2.FamAge < 3)].FamAge, c = data2[(data2.FareAdj < 5) & (data2.FamAge < 3)].Survived)

plt.xlim(0.5,1.5)

plt.ylim(2,3)

plt.show()
data.Predict[data.PassengerId.isin(data3.PassengerId[data3.p < 0.2])] = 0

print(data.iloc[897])
# Any results you write to the current directory are saved as output.

output = pd.DataFrame({'PassengerId' : data.PassengerId[891:], 'Survived' : data.Predict[891:]})



output.to_csv('submission.csv', index=False)