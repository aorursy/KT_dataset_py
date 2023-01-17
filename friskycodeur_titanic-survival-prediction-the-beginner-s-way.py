import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold,GridSearchCV,learning_curve,cross_val_score
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')

df=pd.concat([train,test],axis=0,ignore_index=True)

print(f'Train:{train.shape}\nTest:{test.shape}\nDf:{df.shape}')
df.head()
#columns with missing values

df.isnull().sum()[df.isnull().sum()>0]
df.describe().T
# only describing the categorical columns

df.describe(exclude='number')
#Let's draw a heatmap of correlation between the features

plt.figure(figsize=(12,7))

sns.heatmap(df.drop('PassengerId',axis=1).corr(),annot=True,center=0)
g=sns.FacetGrid(train,col='Survived').map(sns.distplot,'Age',hist=False,kde=True,rug=False,kde_kws={'shade':True})
sns.catplot(x='Pclass',y='Survived',data=train,kind='bar')
sns.catplot(x='Sex',y='Survived',hue='Pclass',data=train,kind='bar')
sns.catplot(x="Embarked", y="Survived", data=train, kind="bar")
df['Family_Size']=df['Parch']+df['SibSp']

df.groupby('Family_Size')['Survived'].mean()
sns.catplot(x='Family_Size',y='Survived',data=df,kind='bar')
#converting the sex column into numerical column

df.Sex=df.Sex.map({'male':0,'female':1}).astype('int')
df['Title'] = df['Name']



for name_string in df['Name']:

    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)



mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}



df.replace({'Title': mapping}, inplace=True)
sns.barplot(x='Title',y='Survived',data=df.iloc[:len(train)])
df[df['Fare'].isnull()]
df['Fare'].fillna(df['Fare'].mean(), inplace=True)
df['FareBin'] = pd.qcut(df['Fare'], 5)



label = LabelEncoder()

df['FareBin_Code'] = label.fit_transform(df['FareBin'])

df.drop(['Fare'], 1, inplace=True)
df['FareBin_Code'].value_counts()
embarked=pd.get_dummies(df['Embarked'],drop_first=True)

df=pd.concat([df,embarked],axis=1)
#  def fix_age(cols):

#     Age=cols[0]

#     Pclass=cols[1]

    

#     if pd.isnull(Age):

#         if Pclass==1:

#             return 37 

#         elif Pclass==2:

#             return 29

#         else:

#             return 24

#     else:

#         return Age 
# df['Age']=df[['Age','Pclass']].apply(fix_age,axis=1)
# filling missing values in 'age' column

titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']



for title in titles:

    age_to_impute = df.groupby('Title')['Age'].median()[titles.index(title)]

    df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = age_to_impute
df['Age'].isnull().sum()
df['AgeBin'] = pd.qcut(df['Age'], 4)



label = LabelEncoder()

df['AgeBin_Code'] = label.fit_transform(df['AgeBin'])
df.sample(2)
df['AgeBin_Code'].value_counts()
df.loc[(df['Cabin'].isnull()),'Cabin_status'] = 0

df.loc[(df['Cabin'].notnull()),'Cabin_status']=1

df.Cabin_status.astype('int')
sns.barplot(x='Cabin_status',y='Survived',data=df.iloc[:len(train)])
df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin','Embarked',

                 'FareBin', 'AgeBin', 'Survived', 'Title', 'Age'], axis = 1, inplace = True)
df.sample(2)
X_train = df[:len(train)]

X_test = df[len(train):]



y_train = train['Survived']
scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
kfold = StratifiedKFold(n_splits=8)
RFC = RandomForestClassifier()



rf_param_grid = {"max_depth": [None],

              "max_features": [3,"sqrt", "log2"],

              "min_samples_split": [n for n in range(1, 9)],

              "min_samples_leaf": [5, 7],

              "bootstrap": [False, True],

              "n_estimators" :[200, 500],

              "criterion": ["gini", "entropy"]}



rf_param_grid_best = {"max_depth": [None],

              "max_features": [3],

              "min_samples_split": [4],

              "min_samples_leaf": [5],

              "bootstrap": [False],

              "n_estimators" :[200],

              "criterion": ["gini"]}



gs_rf = GridSearchCV(RFC, param_grid = rf_param_grid_best, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)



gs_rf.fit(X_train, y_train)



rf_best = gs_rf.best_estimator_

RFC.fit(X_train, y_train)
print(f'RandomForest GridSearch best params: {gs_rf.best_params_}\n')

print(f'RandomForest GridSearch best score: {gs_rf.best_score_}')

print(f'RandomForest score:                 {RFC.score(X_train,y_train)}')
XGB = XGBClassifier()



xgb_param_grid = {'learning_rate':[0.05, 0.1], 

                  'reg_lambda':[0.3, 0.5],

                  'gamma': [0.8, 1],

                  'subsample': [0.8, 1],

                  'max_depth': [2, 3],

                  'n_estimators': [200, 300]

              }



xgb_param_grid_best = {'learning_rate':[0.1], 

                  'reg_lambda':[0.3],

                  'gamma': [1],

                  'subsample': [0.8],

                  'max_depth': [2],

                  'n_estimators': [300]

              }



gs_xgb = GridSearchCV(XGB, param_grid = xgb_param_grid_best, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)



gs_xgb.fit(X_train,y_train)

XGB.fit(X_train, y_train)



xgb_best = gs_xgb.best_estimator_





print(f'XGB GridSearch best params: {gs_xgb.best_params_}\n')

print(f'XGB GridSearch best score: {gs_xgb.best_score_}')

print(f'XGB score:                 {XGB.score(X_train,y_train)}')
results=pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':RFC.predict(X_test)})

results.to_csv("submission.csv", header=True,index=False)



print("The submission file is ready, here's a sample of it!")

print(results.sample(2))