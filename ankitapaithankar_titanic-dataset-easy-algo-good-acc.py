import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# read datasets
train_df=pd.read_csv("../input/titanic/train.csv")
train_df.head()
test_df=pd.read_csv("../input/titanic/test.csv")
test_df.head()
gender_sub=pd.read_csv("../input/titanic/gender_submission.csv")
gender_sub.head()
train_df.Survived.value_counts()
train_df.shape
train_df.info()
round(100*(train_df.isnull().sum()/len(train_df.index)),2)
### check missing values
round(100*(test_df.isnull().sum()/len(test_df.index)),2)
gender_sub['PassengerId']=test_df['PassengerId']
gender_sub.head()
gender_sub.info()
train_df.describe()
train_df.describe(include=['O'])
train_df.Pclass.value_counts()
train_df[['Pclass','Survived']].groupby('Pclass', as_index=False).mean()
train_df[['Sex','Survived']].groupby('Sex', as_index=False).mean()
sns.barplot('Sex','Survived',data=train_df)
sns.barplot('Pclass','Survived',data=train_df)
train_df[['Embarked','Survived']].groupby('Embarked', as_index=False).mean()
sns.barplot('Embarked','Survived',data=train_df)
train_df[['Parch','Survived']].groupby('Parch', as_index=False).mean()
sns.barplot('Parch','Survived',data=train_df)
train_df[['SibSp','Survived']].groupby('SibSp',as_index=False).mean()
sns.barplot('SibSp','Survived',data=train_df)
plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
sns.violinplot('Sex','Age',data=train_df,hue='Survived',split=True)
plt.subplot(1,3,2)
sns.violinplot('Embarked','Age',data=train_df,hue='Survived',split=True)
plt.subplot(1,3,3)
sns.violinplot('Pclass','Age',data=train_df,hue='Survived',split=True)
## creating column label

train_df['Title']=train_df['Name'].str.split(', ', expand=True)[1].str.split('.',expand=True)[0]
test_df['Title']=test_df['Name'].str.split(', ', expand=True)[1].str.split('.',expand=True)[0]
train_df.head()
train_df.Title.value_counts()
train_df['Title']=train_df['Title'].replace(['Dr','Rev','Major','Mlle','Col','Don','Dona','Ms','Mme','Capt','Lady',
                                            'the Countess','Sir','Jonkheer'],'Other')
test_df['Title']=test_df['Title'].replace(['Dr','Rev','Major','Mlle','Col','Don','Dona','Ms','Mme','Capt','Lady',
                                            'the Countess','Sir','Jonkheer'],'Other')
train_df.Title.value_counts()
sns.boxplot(train_df['Age'])
train_df['Age']=train_df['Age'].fillna(train_df['Age'].mean()).astype(int)
test_df['Age']=test_df['Age'].fillna(test_df['Age'].mean()).astype(int)
def age_bins(df):
    df.loc[(df['Age']<=10),'Age'] = 0
    df.loc[(df['Age']>10) & (df['Age']<=21),'Age'] = 1
    df.loc[(df['Age']>21) & (df['Age']<=35),'Age'] = 2
    df.loc[(df['Age']>35) & (df['Age']<=50),'Age'] = 3
    df.loc[(df['Age']>50) & (df['Age']<=65),'Age'] = 4
    df.loc[(df['Age']>65),'Age'] = 5
age_bins(train_df)
age_bins(test_df)
## family size column

train_df['Family_size']=train_df['SibSp'] + train_df['Parch'] + 1
test_df['Family_size']=test_df['SibSp'] + test_df['Parch'] + 1
train_df.head()
test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].median()).astype(int)
train_df['fareperperson']=train_df['Fare']/train_df['Family_size']
test_df['fareperperson']=test_df['Fare']/test_df['Family_size']
def fare_bins(df):
    df.loc[(df['fareperperson']<=10),'fareperperson'] = 0
    df.loc[(df['fareperperson']>10) & (df['fareperperson']<=21),'fareperperson'] = 1
    df.loc[(df['fareperperson']>21) & (df['fareperperson']<=35),'fareperperson'] = 2
    df.loc[(df['fareperperson']>35) & (df['fareperperson']<=50),'fareperperson'] = 3
    df.loc[(df['fareperperson']>50) & (df['fareperperson']<=65),'fareperperson'] = 4
    df.loc[(df['fareperperson']>65),'fareperperson'] = 5
fare_bins(train_df)
fare_bins(test_df)
train_df['Cabin']=train_df['Cabin'].fillna('Missing')
train_df['Cabin']=train_df['Cabin'].str[:1]

test_df['Cabin']=test_df['Cabin'].fillna('Missing')
test_df['Cabin']=test_df['Cabin'].str[:1]
train_df['Cabin'].unique()
#drop the column cabin
train_df.drop(['Name','Ticket','PassengerId','Parch','SibSp','Fare'],axis=1,inplace=True)
test_df.drop(['Name','Ticket','PassengerId','Parch','SibSp','Fare'],axis=1,inplace=True)
train_df.Sex.value_counts()
train_df['Sex']=train_df.Sex.map({'male' : 1, 'female' : 0})
test_df['Sex']=test_df.Sex.map({'male' : 1, 'female' : 0})
train_df.head()
plt.hist(train_df['Age'])
plt.show()
train_df['Embarked']=train_df['Embarked'].fillna('S')
test_df['Embarked']=test_df['Embarked'].fillna('S')
le=preprocessing.LabelEncoder()

train_df['Cabin']=le.fit_transform(train_df['Cabin'])
test_df['Cabin']=le.fit_transform(test_df['Cabin'])

train_df=pd.get_dummies(train_df,drop_first=True)
test_df=pd.get_dummies(test_df,drop_first=True)
train_df.head()
round(100*(train_df.isnull().sum()/len(train_df.index)),2)
train_df.head()
round(100*(train_df.isnull().sum()/len(train_df.index)),2)
test_df.head()
round(100*(test_df.isnull().sum()/len(test_df.index)),2)
train_df.head()
X=train_df.drop(['Survived'],axis=1)
y=train_df['Survived']

dt_default=DecisionTreeClassifier(max_depth=5)
dt_default.fit(X,y)
y_pred=dt_default.predict(test_df)
print(y_pred)
gender_sub['Survived']=y_pred
gender_sub.head()
gender_sub.to_csv('gender_sub.csv',index=False)
logreg=LogisticRegression()
logreg.fit(X,y)
y_pred=logreg.predict(test_df)
gender_sub['Survived']=y_pred
gender_sub.head()
gender_sub.to_csv('gender_sub_LR.csv',index=False)
rf=RandomForestClassifier(max_depth=5)
rf.fit(X,y)
y_pred=rf.predict(test_df)
gender_sub['Survived']=y_pred
gender_sub.head()
gender_sub.to_csv('gender_sub_RF.csv',index=False)
### using grid search CV on RF
param_grid={
    'max_depth' : [3,5,6,8,10,15],
    'min_samples_split': range(5, 500, 20),
    'min_samples_leaf': [3,5],
    'n_estimators' : range(10,100,10),
    'max_features' : range(3,7,1)
}
# rf=RandomForestClassifier()

# grid_model=GridSearchCV(estimator=rf,
#                        param_grid=param_grid,
#                        verbose=1,
#                        cv=3,
#                        n_jobs=-1)
# grid_model.fit(X,y)
# printing the optimal accuracy score and hyperparameters
# print('We can get accuracy of',grid_model.best_score_,'using',grid_model.best_params_)
# model with the best hyperparameters
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=10,
                             min_samples_leaf=3, 
                             min_samples_split=5,
                             max_features=3,
                             n_estimators=20)
rfc.fit(X,y)
y_pred=rfc.predict(test_df)
gender_sub['Survived']=y_pred
gender_sub.head()
gender_sub.to_csv('gender_sub_RF_GSCV.csv',index=False)
xgb=xgb.XGBClassifier()
xgb.fit(X,y)
y_pred=xgb.predict(test_df)
gender_sub['Survived']=y_pred
gender_sub.head()
gender_sub.to_csv('gender_sub_xgb.csv',index=False)
adb=AdaBoostClassifier(learning_rate=0.1,n_estimators=5)
adb.fit(X,y)
y_pred=adb.predict(test_df)
gender_sub['Survived']=y_pred
gender_sub.head()
gender_sub.to_csv('gender_sub_adb.csv',index=False)
lgb=lgb.LGBMClassifier()
lgb.fit(X,y)
y_pred=xgb.predict(test_df)
gender_sub['Survived']=y_pred
gender_sub.head()
gender_sub.to_csv('gender_sub_lgbm.csv',index=False)
votingC = VotingClassifier(estimators=[('LGR', logreg),('LGB',lgb),('RFC',rf),('XGB',xgb),('ADB',adb)], voting='soft', 
                           n_jobs=4,weights=[1,1,2,2,1])

votingC = votingC.fit(X, y)
y_pred=votingC.predict(test_df)
gender_sub['Survived']=y_pred
gender_sub.head()
gender_sub.to_csv('gender_sub_votingC.csv',index=False)