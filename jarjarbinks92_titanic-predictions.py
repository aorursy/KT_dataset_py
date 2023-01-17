# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import VotingClassifier 

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import RFECV

from sklearn import metrics



%matplotlib inline

plt.rcParams['figure.figsize'] = [10,7]



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

data = train_df.append(test_df)
train_df
test_df
train_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
train_df.corr()
g = sns.heatmap(train_df.drop('PassengerId',axis=1).corr(), annot=True, cmap='coolwarm')
#Explore Pclass vs Survived

g = sns.catplot(data=train_df, y='Survived', x='Pclass', kind='bar')
#Explore Age vs Survived

g = sns.catplot(data=train_df, x='Survived', y='Age', kind='bar', height=5)
#Explore SibSp vs Survived

g = sns.catplot(data=train_df, x='SibSp', y='Survived', kind='bar', height=5)
#Explore Parch vs Survived

g = sns.catplot(data=train_df, x='Parch', y='Survived', kind='bar', height=5)
#Explore Fare vs Survived

g = sns.catplot(data=train_df, y='Fare', x='Survived', kind='bar', height=5)
data['Title'] = data['Name']

for name_string in data['Name']:

    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

data['Title'].value_counts()

title_changes = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

data.replace({'Title': title_changes}, inplace=True)

titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

for title in titles:

    age_to_impute = data.groupby('Title')['Age'].median()[titles.index(title)]

    data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = age_to_impute

train_df['Age'] = data['Age'][:891]

test_df['Age'] = data['Age'][891:]

data.drop('Title', axis = 1, inplace = True)
data['Family_Size'] = data['Parch'] + data['SibSp'] + 1

train_df['Family_Size'] = data['Family_Size'][:891]

test_df['Family_Size'] = data['Family_Size'][891:]
data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])

data['Fare'].fillna(data['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5

data['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
for grp, grp_df in data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):   

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin==0.0):

                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 

      data.loc[data['Family_Survival']!=0.5].shape[0])
for _, grp_df in data.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0                      

print("Number of passenger with family/group survival information: " 

      +str(data[data['Family_Survival']!=0.5].shape[0]))

train_df['Family_Survival'] = data['Family_Survival'][:891]

test_df['Family_Survival'] = data['Family_Survival'][891:]
data['Fare'].fillna(data['Fare'].median(), inplace = True)

data['FareBin'] = pd.qcut(data['Fare'], 5)

label = LabelEncoder()

data['FareBin_Code'] = label.fit_transform(data['FareBin'])

train_df['FareBin_Code'] = data['FareBin_Code'][:891]

test_df['FareBin_Code'] = data['FareBin_Code'][891:]

train_df.drop(['Fare'], 1, inplace=True)

test_df.drop(['Fare'], 1, inplace=True)
data['AgeBin'] = pd.qcut(data['Age'], 4)

label = LabelEncoder()

data['AgeBin_Code'] = label.fit_transform(data['AgeBin'])

train_df['AgeBin_Code'] = data['AgeBin_Code'][:891]

test_df['AgeBin_Code'] = data['AgeBin_Code'][891:]

train_df.drop(['Age'], 1, inplace=True)

test_df.drop(['Age'], 1, inplace=True)
train_df['Sex'].replace(['male','female'],[0,1],inplace=True)

test_df['Sex'].replace(['male','female'],[0,1],inplace=True)

train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',

               'Embarked'], axis = 1, inplace = True)

test_df.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',

              'Embarked'], axis = 1, inplace = True)
train_df.head(10)
test_df.head(10)
x = train_df.drop('Survived', axis=1)

y = train_df['Survived']



x_test = test_df.copy()
from sklearn.model_selection import KFold

#Stratified k fold cross validation

kfold = KFold(n_splits=10)
std_scaler = StandardScaler()

x = std_scaler.fit_transform(x)

x_test = std_scaler.transform(x_test)
# Store scores of all models

gs_scores = {}
# Adaboost

DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsadaDTC.fit(x,y)



ada_best = gsadaDTC.best_estimator_



#Best score 

gs_scores['Ada'] = gsadaDTC.best_score_
# ExtraTrees 

ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [None, 'auto'],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsExtC.fit(x,y)



ExtC_best = gsExtC.best_estimator_



# Best score

gs_scores['ET'] = gsExtC.best_score_
# RFC Parameters tunning 

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [None, 'auto'],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsRFC.fit(x,y)



RFC_best = gsRFC.best_estimator_



# Best score

gs_scores['RFC'] = gsRFC.best_score_
# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [None, 'auto'] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsGBC.fit(x,y)



GBC_best = gsGBC.best_estimator_



# Best score

gs_scores['GBC'] = gsGBC.best_score_
# SVM



SVMC = SVC(probability=True)

svc_param_grid = {'kernel':['rbf'],

                  'gamma':[0.001,0.01,0.1,1],

                  'C':[0.1,1,10,50,100,200,300,]}

gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=kfold, scoring='accuracy', n_jobs = -1, verbose = 1)



gsSVMC.fit(x,y)



SVMC_best = gsSVMC.best_estimator_



# Best score

gs_scores['SVC'] = gsSVMC.best_score_
# KNN



knn = KNeighborsClassifier()

n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]

algorithm = ['auto']

weights = ['uniform', 'distance']

leaf_size = list(range(1,50,5))

hyperparams = {'algorithm': algorithm,

               'weights': weights,

               'leaf_size': leaf_size,

               'n_neighbors': n_neighbors}



gsKNN=GridSearchCV(estimator = knn, param_grid = hyperparams, verbose=1, cv=10, scoring = "roc_auc", n_jobs=-1)



gsKNN.fit(x, y)



knn_best = gsKNN.best_estimator_



# Best score

gs_scores['KNN'] = gsKNN.best_score_
gs_scores_df = pd.DataFrame(gs_scores, index=[0])

gs_scores_df
g = sns.catplot(data=gs_scores_df,kind='bar')
# def get_reduced_features(models):

#     model_num=1

#     for model in models:

#         print('Model', model_num)

#         rfecv = RFECV(estimator=model, step=1, cv=kfold, scoring='accuracy')

#         rfecv.fit(x,y)

#         print('Optimal no. of features: ', rfecv.n_features_)

#         feat_support = rfecv.support_

#         feat_grid_score = rfecv.grid_scores_

#         print('Selected columns are: ', *x.columns[feat_support])

#         print('Not selected columns are: ', *x.columns[feat_support!=True])

#         print('Cross val score: ', feat_grid_score.mean(), end='\n\n')

#         model_num+=1

    



# get_reduced_features([RFC_best,ExtC_best,ada_best,GBC_best])
# votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),

# ('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best), ('knn',knn_best)], voting='soft', n_jobs=-1)



# votingC = votingC.fit(x,y)



# from sklearn.model_selection import cross_val_score

# print(cross_val_score(votingC, x, y, cv=10).mean())
#Get predictions for test data

# y_pred = votingC.predict(x_test)

knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 

                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 

                           weights='uniform')

knn.fit(x, y)

y_pred = knn.predict(x_test)
temp = pd.DataFrame(pd.read_csv("/kaggle/input/titanic/test.csv")['PassengerId'])

temp['Survived'] = y_pred

temp.to_csv("../working/submission.csv", index = False)

print("Predictions submitted")