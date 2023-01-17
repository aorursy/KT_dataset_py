# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Downloadlibraries

import pandas as pd

import numpy as np

import math

import xgboost as xgb

from matplotlib import pyplot as plt

from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

import warnings

warnings.filterwarnings('ignore')

%pylab inline
data = pd.read_csv('../input/train.csv')

data.head()
data.isnull().sum()
#work with categorial sex feature

data['Sex'] = np.where(data['Sex']=='male',1,0)

data.head()
for i in range(0,len(data)):

    data.iloc[i,3] = len(data.iloc[i,3])

    ticket = data.iloc[i,8].split()

    try:

        data.iloc[i,8] = int(ticket[-1])

    except:

        data.iloc[i,8] = 0

data.head()
data = data.drop(['PassengerId'],axis=1)

data.head()
data.isnull().sum()
data = data.drop(['Cabin'], axis=1)

data.head()
data_pos = data[data['Survived']==1]

data_neg = data[data['Survived']==0]

plt.figure(figsize=(9, 9))

plt.subplot(221)

plt.pie([sum(data_pos['Sex'])/len(data_pos),(len(data_pos)-sum(data_pos['Sex']))/len(data_pos)],

                                             labels=['Men', 'Women'], explode=[0, 0.3], autopct = '%1.1f%%', shadow=True,

       colors=['white','cyan'])

plt.title('Survived')

plt.subplot(222)

plt.pie([sum(data_neg['Sex'])/len(data_neg),(len(data_neg)-sum(data_neg['Sex']))/len(data_neg)],

                                             labels=['Men', 'Women'], explode=[0, 0.3], autopct = '%1.1f%%', shadow=True,

       colors=['white','cyan'])

plt.title('Died')

plt.show()
data_men = data[data['Sex']==1]

data_women = data[data['Sex']==0]

plt.figure(figsize=(9, 9))

plt.subplot(221)

plt.pie([sum(data_men['Survived'])/len(data_men),(len(data_men)-sum(data_men['Survived']))/len(data_men)],

                                             labels=['Survived', 'Died'], explode=[0, 0.3], autopct = '%1.1f%%', shadow=True,

       colors=['cyan','white'])

plt.title('Men')

plt.subplot(222)

plt.pie([sum(data_women['Survived'])/len(data_women),(len(data_women)-sum(data_women['Survived']))/len(data_women)],

                                             labels=['Survived', 'Died'], explode=[0, 0.3], autopct = '%1.1f%%', shadow=True,

       colors=['cyan','white'])

plt.title('Women')

plt.show()
plt.figure(figsize=(9, 9))

plt.subplot(221)

plt.hist(data_neg['Age'].dropna(), bins=10, color='b', alpha=0.75)

plt.title('Died')

plt.subplot(222)

plt.hist(data_pos['Age'].dropna(), bins=10, color='r', alpha=0.75)

plt.title('Survived')

plt.show()
data.head()
print('Выживших в выборке {:.2f} %, погибших в выборке: {:.2f} %'.format(sum(data.Survived)/len(data)*100,(1-sum(data.Survived)/len(data))*100))
#let's insert age categpries and try 3 variates nan's replacing

for i in range(0,len(data)):

    age = data.iloc[i,4]

    if math.isnan(data.iloc[i,4]):

        data.iloc[i,4] = 'none group'

    elif int(age)<10:

        data.iloc[i,4] = '<10'

    elif 10<=int(age)<20:

        data.iloc[i,4] = '10-20'

    elif 20<=int(age)<30:

        data.iloc[i,4] = '20-30'

    elif 30<=int(age)<40:

        data.iloc[i,4] = '30-40'

    elif 40<=int(age)<50:

        data.iloc[i,4] = '40-50'

    elif 50<=int(age)<60:

        data.iloc[i,4] = '50-60'

    elif 60<=int(age)<70:

        data.iloc[i,4] = '60-70'

    elif 70<=int(age)<80:

        data.iloc[i,4] = '70-80'

    elif 80<=int(age)<90:

        data.iloc[i,4] = '80-90'

    elif int(age)>90:

        data.iloc[i,4] = '>90'

    data.iloc[i,1] = str(data.iloc[i,1])

data.head()
print(data['Embarked'].value_counts())

print(data['Pclass'].value_counts())

print(data.isnull().sum())

print(len(data))
#Embarked nan replacing

data[data.isnull().any(axis=1)]
data.describe()
data.iloc[829,9]=data[(data['Pclass']=='1') & (data['Fare']>60) & (data['Fare']<100) & (data['Age']=='60-70')]['Embarked'].value_counts().index[0]

data.iloc[61,9]=data[(data['Pclass']=='1') & (data['Fare']>60) & (data['Fare']<100) & (data['Age']=='30-40')]['Embarked'].value_counts().index[0]
print(data.isnull().sum())

print(len(data))
encoder=DictVectorizer(sparse=False).fit(data[['Age','Embarked','Pclass']].T.to_dict().values())

data_cat = encoder.transform(data[['Age','Embarked','Pclass']].T.to_dict().values())

scaler = StandardScaler().fit(data['Fare'].as_matrix().reshape(-1,1))

data_real = scaler.transform(data['Fare'].as_matrix().reshape(-1,1))

data_cat = pd.DataFrame(data_cat, columns=[encoder.feature_names_])

data_real = pd.DataFrame(data_real,columns=['Fare'])

data_cat.head()
data = data.drop(['Fare','Age','Embarked','Pclass'], axis=1)

data = pd.concat([data, data_cat, data_real], axis=1, join_axes=[data.index])

data.head()
label = data['Survived']

data = data.drop(['Survived'], axis=1)

data.head()
class_res = []

#LogisticRegression grid_search

params = {

    'C':[0.25,0.5,1,2,5,10,15],

    'class_weight':['balanced', None],

    'max_iter':[70,100,200],

    'penalty':['l2','l1']

}

grid_cv_log = GridSearchCV(LogisticRegression(), params, scoring='accuracy', cv=3)

grid_cv_log.fit(data.as_matrix(),label.as_matrix())

class_res.append(['logReg',grid_cv_log.best_score_,grid_cv_log.best_params_])

print(grid_cv_log.best_score_)

print(list(zip(data.columns,grid_cv_log.best_estimator_.coef_[0])))
params = {

    'n_neighbors':[3,5,7]

}

grid_cv_kN = GridSearchCV(KNeighborsClassifier(), params, scoring = 'accuracy', cv=3)

grid_cv_kN.fit(data.as_matrix(),label.as_matrix())

class_res.append(['KN',grid_cv_kN.best_score_,grid_cv_kN.best_params_])

print(grid_cv_kN.best_score_)
params = {

    'n_estimators':[10,20,50,70],

    'max_depth':[30,50,None],

    'n_jobs':[-1]

}

grid_cv_rndForest = GridSearchCV(RandomForestClassifier(), params, n_jobs=-1, scoring = 'accuracy', cv=3)

grid_cv_rndForest.fit(data.as_matrix(),label.as_matrix())

class_res.append(['RFC',grid_cv_rndForest.best_score_,grid_cv_rndForest.best_params_])

print(grid_cv_rndForest.best_score_)

print(list(zip(data.columns, grid_cv_rndForest.best_estimator_.feature_importances_)))
params = {

    'learning_rate':[1,2,3],

    'n_estimators':[10,30,50,70],

    'base_estimator':[DecisionTreeClassifier()]

}

grid_cv_ada = GridSearchCV(AdaBoostClassifier(), params, n_jobs=-1, scoring = 'accuracy', cv=3)

grid_cv_ada.fit(data.as_matrix(),label.as_matrix())

class_res.append(['Ada',grid_cv_ada.best_score_,grid_cv_ada.best_params_])

print(grid_cv_ada.best_score_)

print(list(zip(data.columns, grid_cv_ada.best_estimator_.feature_importances_)))
params = {

    'max_depth': [10,20,50,None],

    'class_weight': [None, 'balanced']

}

grid_cv_dTree = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=-1, scoring = 'accuracy', cv=3)

grid_cv_dTree.fit(data.as_matrix(),label.as_matrix())

class_res.append(['DTree',grid_cv_dTree.best_score_,grid_cv_dTree.best_params_])

print(grid_cv_dTree.best_score_)
params = {

    'class_weight': [None, 'balanced'],

    'penalty': ['l2','l1']

}

grid_cv_SGD = GridSearchCV(SGDClassifier(), params, n_jobs=-1, scoring = 'accuracy', cv=3)

grid_cv_SGD.fit(data.as_matrix(),label.as_matrix())

class_res.append(['SGD',grid_cv_SGD.best_score_,grid_cv_SGD.best_params_])

print(grid_cv_SGD.best_score_)
for i in range(0,len(class_res)):

    tmp = i

    for j in range(i+1,len(class_res)):

        if class_res[j][1]>class_res[tmp][1]:

            tmp=j

    chg = class_res[tmp]

    class_res[tmp] = class_res[i]

    class_res[i] = chg

class_res[0:4]
clf_1 = VotingClassifier([('DFC', DecisionTreeClassifier(class_weight=None, max_depth=10)),

                          ('RFC', RandomForestClassifier(max_depth=None, n_estimators=50)),

                         ('LogReg', LogisticRegression(penalty='l1', class_weight='balanced', C=10, max_iter=200))], voting='hard')

clf_2 = VotingClassifier([('RFC', RandomForestClassifier(max_depth=None, n_estimators=50)),

                         ('LogReg', LogisticRegression(penalty='l1', class_weight='balanced', C=10, max_iter=200)),

                         ('Ada', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), learning_rate=2,n_estimators=10))])

cross_1 = cross_val_score(clf_1, data.as_matrix(), label.as_matrix(), cv=5, scoring='accuracy')

cross_2 = cross_val_score(clf_2, data.as_matrix(), label.as_matrix(), cv=5, scoring='accuracy')

print('Первый вариант: ', cross_1.mean())

print('Второй вариант: ', cross_2.mean())

clf_3 = xgb.XGBClassifier()

cross_3 = cross_val_score(clf_3, data.as_matrix(), label.as_matrix(), cv=5, scoring='accuracy')

print(cross_3.mean())
test = pd.read_csv('../input/test.csv')

test.head()
test.isnull().sum()
test['Sex'] = np.where(test['Sex']=='male',1,0)

test.head()
for i in range(0,len(test)):    

    test.iloc[i,2] = len(test.iloc[i,2])

    ticket = test.iloc[i,7].split()

    try:

        test.iloc[i,7] = int(ticket[-1])

    except:

        test.iloc[i,7] = 0

test = test.drop(['PassengerId','Cabin'],axis=1)

test.head()
for i in range(0,len(test)):

    age = test.iloc[i,3]

    if math.isnan(test.iloc[i,3]):

        test.iloc[i,3] = 'none group'

    elif int(age)<10:

        test.iloc[i,3] = '<10'

    elif 10<=int(age)<20:

        test.iloc[i,3] = '10-20'

    elif 20<=int(age)<30:

        test.iloc[i,3] = '20-30'

    elif 30<=int(age)<40:

        test.iloc[i,3] = '30-40'

    elif 40<=int(age)<50:

        test.iloc[i,3] = '40-50'

    elif 50<=int(age)<60:

        test.iloc[i,3] = '50-60'

    elif 60<=int(age)<70:

        test.iloc[i,3] = '60-70'

    elif 70<=int(age)<80:

        test.iloc[i,3] = '70-80'

    elif 80<=int(age)<90:

        test.iloc[i,3] = '80-90'

    elif int(age)>90:

        test.iloc[i,3] = '>90'

    test.iloc[i,0] = str(test.iloc[i,0])

test.head()
test[test.isnull().any(axis=1)]
test.iloc[152,7] = test[(test['Pclass']=='3') & (test['Embarked']=='S')]['Fare'].mean()

test.head()
test.isnull().sum()
test_cat = encoder.transform(test[['Age','Embarked','Pclass']].T.to_dict().values())

test_real = scaler.transform(test['Fare'].as_matrix().reshape(-1,1))

test_cat = pd.DataFrame(test_cat, columns=[encoder.feature_names_])

test_real = pd.DataFrame(test_real,columns=['Fare'])

test = test.drop(['Fare','Age','Embarked','Pclass'], axis=1)

test = pd.concat([test, test_cat, test_real], axis=1, join_axes=[test.index])

test.head()
sample = pd.read_csv('../input/gendermodel.csv')

sample.head()
clf_1 = clf_1.fit(data.as_matrix(),label.as_matrix())

clf_2 = clf_2.fit(data.as_matrix(),label.as_matrix())

clf_3 = clf_3.fit(data.as_matrix(),label.as_matrix())

res_1 = clf_1.predict(test.as_matrix())

res_2 = clf_2.predict(test.as_matrix())

res_3 = clf_3.predict(test.as_matrix())
res_1 = pd.DataFrame(list(zip(sample['PassengerId'].as_matrix(),res_1)), columns=['PassengerId','Survived'])

res_1.head()
res_2 = pd.DataFrame(list(zip(sample['PassengerId'].as_matrix(),res_2)), columns=['PassengerId','Survived'])

res_2.head()
res_3 = pd.DataFrame(list(zip(sample['PassengerId'].as_matrix(),res_3)), columns=['PassengerId','Survived'])

res_3.head()
res_1.to_csv('230117_01.csv', header=True, index=None)

res_2.to_csv('230117_02.csv', header=True, index=None)

res_3.to_csv('230117_03.csv', header=True, index=None)
clf_4 = VotingClassifier([('RFC', RandomForestClassifier(max_depth=None, n_estimators=50)),

                         ('XgBoost', xgb.XGBClassifier()),

                         ('Logreg', LogisticRegression(penalty='l1', class_weight='balanced', C=10, max_iter=200))]).fit(data.as_matrix(),label.as_matrix())

res_4 = clf_4.predict(test.as_matrix())

res_4 = pd.DataFrame(list(zip(sample['PassengerId'].as_matrix(),res_4)), columns=['PassengerId','Survived'])

res_4.to_csv('230117_04.csv', header=True, index=None)