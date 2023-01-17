# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
 
import pandas as pd
import numpy as np
import seaborn as sns

from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import pandas_profiling as pp
import math 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#!pip install m
import matplotlib.pyplot as plt
%matplotlib inline
print("done")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings('ignore')


train_Data=pd.read_csv("/kaggle/input/titanic/train.csv")


test_Data=pd.read_csv("/kaggle/input/titanic/test.csv")

gender_Data=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")



gender_Data.info()
train_Data.info()
test_Data.info()
train_Data.shape
test_Data.info()
test_Data.shape
train_Data.isnull().sum()
df=train_Data
df_1=test_Data
df.describe()
df_1.describe()
print((df.isnull().sum().sort_values(ascending=False)/len(df))*100)
print((df_1.isnull().sum().sort_values(ascending=False))/len(df)*100)
pp.ProfileReport(df)
pp.ProfileReport(df_1)
df['Title']=df.Name.str.extract('([A-Za-z]+)\.')
df_1['Title']=df_1.Name.str.extract('([A-Za-z]+)\.')
sns.catplot(x='Pclass',y='Age',hue='Sex',kind='box',data=df)
plt.xlabel('PClass')
plt.ylabel('Age')

sns.catplot(x='Embarked',y='Age',hue='Sex',kind='box',data=df)
sns.pointplot(x='Survived',y='Age',hue='Sex',data=df,dodge=True,linestyles=['-','--'])
sns.pointplot(x='Parch',y='Survived',hue='Sex',data=df,dodge=True)
sns.boxplot(x='Fare',data=df)
sns.boxplot(x='Age',data=df)
sns.boxplot(x='SibSp',data=df)
df.Title.unique()
def Replace_Title(X_titles,y_replcaed):
    x=dict.fromkeys(X_titles,y_replcaed)
    return x
df.Title=df.Title.replace(Replace_Title(['Ms', 'Mlle'],'Miss'))
df.Title=df.Title.replace(Replace_Title(['Dr', 'Major', 'Col', 'Sir', 'Rev', 'Jonkheer', 'Capt', 'Don'],'Mr'))

df.Title=df.Title.replace(Replace_Title(['Mme', 'Countess', 'Lady', 'Dona'],'Mrs'))



df_1.Title=df_1.Title.replace(Replace_Title(['Ms', 'Mlle'],'Miss'))
df_1.Title=df_1.Title.replace(Replace_Title(['Dr', 'Major', 'Col', 'Sir', 'Rev', 'Jonkheer', 'Capt', 'Don'],'Mr'))

df_1.Title=df_1.Title.replace(Replace_Title(['Mme', 'Countess', 'Lady', 'Dona'],'Mrs'))


df.Title.unique()
df_1.Title.unique()
df.groupby('Title').Survived.mean()

sns.barplot(x='Title',y='Survived',data=df)
df['Ticket_letter']=df.Ticket.apply(lambda x: x[:2])

df['Ticket_length']=df.Ticket.apply(lambda x: len(x))


df_1['Ticket_letter']=df_1.Ticket.apply(lambda x: x[:2])

df_1['Ticket_length']=df_1.Ticket.apply(lambda x: len(x))
df.Ticket_letter
df.SibSp.value_counts()
df_1.SibSp.value_counts()
fig, axarr = plt.subplots(1,2,figsize=(12,6))
a = sns.countplot(df['SibSp'], ax=axarr[0]).set_title('Passengers count by SibSp')
axarr[1].set_title('Survival rate by SibSp')
b = sns.barplot(x='SibSp', y='Survived', data=df, ax=axarr[1]).set_ylabel('Survival rate')
fi,ax=plt.subplots(1,2,figsize=(12,6))

sns.countplot(df['Parch'],ax=ax[0]).set_title('Passengers count by parch ')
ax[1].set_title('Survival Rate of Passengers')
sns.barplot(x='Parch',y='Survived',data=df).set_ylabel('Survival Rate')
# Creation of a new Fam_size column
df['Fam_size'] = df['SibSp'] + df['Parch'] + 1
df_1['Fam_size'] = df_1['SibSp'] + df_1['Parch'] + 1
plt.title('Survival rate by family size')
sns.barplot(x='Fam_size',y='Survived',data=df).set_ylabel('Survival_rate')
df['Family_type']=pd.cut(df.Fam_size,[0,1,4,7,11],labels=['Solo','Small','Big','Very Big'])

df_1['Family_type']=pd.cut(df_1.Fam_size,[0,1,4,7,11],labels=['Solo','Small','Big','Very Big'])
plt.title('Survival rate by Family type')
sns.barplot(x='Family_type',y='Survived',data=df)
y=df['Survived']

features=['Pclass','Title','Fare','Embarked','Ticket_letter','Ticket_length','Family_type']

x=df[features]
x.head()
numerical_cat=['Fare']

categorical_cols=['Pclass','Title','Embarked','Ticket_letter','Ticket_length','Family_type']


#Preprocessing for numerical data

numerical_transformer= SimpleImputer(strategy='median')

#Preprocessing for Categorical data

categorical_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])


#Bundle preprocessing for numerical and categorical data

Preprocessor=ColumnTransformer(transformers=[('num',numerical_transformer,numerical_cat),('cat',categorical_transformer,categorical_cols)])


#Bundle preprocessing and modeling code
titanic_pipeline=Pipeline(steps=[('preprocessor',Preprocessor),('model',DecisionTreeClassifier())])

titanic_pipeline_1=Pipeline(steps=[('preprocessor',Preprocessor),('model_R',RandomForestClassifier())])


titanic_pipeline_2=Pipeline(steps=[('preprocessor',Preprocessor),('model_1',LogisticRegression(penalty='l2',C=2.7825594022071245))])

titanic_pipeline_3=Pipeline(steps=[('preprocessor',Preprocessor),('model_2',XGBClassifier())])

titanic_pipeline_4=Pipeline(steps=[('preprocessor',Preprocessor),('model_3',LGBMClassifier())])


#DecisionTRee
parameters={
    "model":[DecisionTreeClassifier()],
    "model__criterion":['gini','entropy'],
    "model__max_depth":[1,2,4,5,10,None],
    "model__min_samples_split":[2,3,5,10],
    "model__min_samples_leaf":[1,5,10,20]
}
grid_Decision_Tree=GridSearchCV(titanic_pipeline,param_grid=parameters,cv=5).fit(x,y)
grid_Decision_Tree.best_params_
titanic_pipeline_Decision_Tree=Pipeline(steps=[('preprocessor',Preprocessor),('model_DT',DecisionTreeClassifier(criterion='entropy',max_depth=4,min_samples_leaf=1,min_samples_split=2))])

#RandomForest

parameters_R={
    
    
     'model_R':[RandomForestClassifier()],
     'model_R__n_estimators':[2,4,5,8,10,15],
     "model_R__criterion": ["gini", "entropy"],
     'model_R__max_features':['auto','log2'],
     'model_R__max_depth':[1,2,3,5,10],
     "model_R__min_samples_split": [2, 3, 5, 10],
     'model_R__min_samples_leaf':[2,3,5,10],


}
grid=GridSearchCV(titanic_pipeline_1,param_grid=parameters_R,cv=5)
grid_Forest=grid.fit(x,y)
grid_Forest.best_score_
grid_Forest.best_params_
titanic_pipeline_Forest=Pipeline(steps=[('preprocessor',Preprocessor),('model_RF',RandomForestClassifier(criterion='gini',max_depth=5,max_features='auto',min_samples_leaf=5,min_samples_split=2,n_estimators=4))])

#XGboost

parameters_XG={
        
                "model_2__max_depth":[3,4,5,6,7,8],
                "model_2__n_estimators":[5,10,20,50,100],
                "model_2__learning_rate":np.linspace(0.02,0.16,8)
    
    
}
grid_XG=GridSearchCV(titanic_pipeline_3,param_grid=parameters_XG,cv=5)
Grid_XG=grid_XG.fit(x,y)
Grid_XG.best_params_
titanic_pipeline_XG=Pipeline(steps=[('preprocessor',Preprocessor),('model_XG',XGBClassifier(max_depth=5,n_estimators=50,learning_rate=0.1))])

#LGBM

parameters_LGBM={
                    'model_3__n_estimators':[5,50,100],
                    'model_3__max_depth':range(3,8),
                    'model_3__num_leaves':[31,61],
                    'model_3__min_data_in_leaf':[20,30,40],
                    'model_3__learning_rate':np.linspace(0.02,0.16,4)
}
grid_LGBM=GridSearchCV(titanic_pipeline_4,param_grid=parameters_LGBM,cv=5)
grid_Lgbm_=grid_LGBM.fit(x,y)
grid_Lgbm_.best_params_
titanic_pipeline_LGBM=Pipeline(steps=[('preprocessor',Preprocessor),('model_3',LGBMClassifier(n_estimators=100 ,max_depth=5 , num_leaves=31 , min_data_in_leaf= 20, learning_rate=0.16))])

titanic_pipeline_Decision_Tree=Pipeline(steps=[('preprocessor',Preprocessor),('model_DT',DecisionTreeClassifier(criterion='entropy',max_depth=4,min_samples_leaf=1,min_samples_split=2))])

titanic_pipeline_Forest=Pipeline(steps=[('preprocessor',Preprocessor),('model_RF',RandomForestClassifier(criterion='gini',max_depth=5,max_features='auto',min_samples_leaf=5,min_samples_split=2,n_estimators=4))])

titanic_pipeline_XG=Pipeline(steps=[('preprocessor',Preprocessor),('model_XG',XGBClassifier(max_depth=5,n_estimators=50,learning_rate=0.1))])


voting_classifier=Pipeline([['model_V',VotingClassifier(estimators=[('titanic_pipeline_Decision',titanic_pipeline_Decision_Tree),('titanic_pipeline_Forest',titanic_pipeline_Forest),('titanic_pipeline_XG',titanic_pipeline_XG),('titanic_pipeline_LGBM',titanic_pipeline_LGBM)])]])
voting_classifier.fit(x,y)

X_test=df_1[features]
X_test.head()
X_test.isnull().sum()
predictions=voting_classifier.predict(X_test)
output=pd.DataFrame({'PassengerId':df_1.PassengerId,'Survived':predictions})

output.to_csv('sub.csv',index=False)
print('Your submission was successfully saved!')