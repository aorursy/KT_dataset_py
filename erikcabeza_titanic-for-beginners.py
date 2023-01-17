import numpy as np 

import pandas as pd 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go



from sklearn.impute import SimpleImputer

from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder



from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.tree import DecisionTreeClassifier





from sklearn.metrics import accuracy_score



from xgboost import XGBClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error



from keras.utils import to_categorical





trainingData=pd.read_csv('../input/titanic/train.csv')

testData=pd.read_csv('../input/titanic/test.csv')

sampleSubmission=pd.read_csv('../input/titanic/gender_submission.csv')
trainingData.info()
#drop unnecesary columns

train=trainingData.drop('Cabin',axis=1)

train=train.drop("Ticket",axis=1)

train=train.drop("Name",axis=1)

train=train.drop("PassengerId",axis=1)
#heatmap to see the correlation between target and numerical variables

corr=train.corr()

sns.heatmap(corr,annot=True)
#correlation between categorical variables and target variable->ANOVA test

from scipy import stats

F, p = stats.f_oneway(train[train.Sex=='male'].Survived,train[train.Sex=='female'].Survived)

print(F)

#for this variable, you could have used get_dummies too and then calculate the correlation with corr()
from scipy import stats

F, p = stats.f_oneway(train[train.Embarked=='C'].Survived,train[train.Embarked=='S'].Survived,train[train.Embarked=='Q'].Survived)

print(F)
#Some attributes should be categorical, like Pclass 

train['Pclass']=train['Pclass'].apply(str)

from scipy import stats

F, p = stats.f_oneway(train[train.Pclass=='3'].Survived,train[train.Pclass=='1'].Survived,train[train.Pclass=='2'].Survived)

print(F)
#add a new attribute to the dataset-> FareInterval

fare=[]

for i in train['Fare']:

    if i<50:

        fare.append('Under 50')

    else:

        fare.append('Over or equal to 50')

        

    
train['FareInterval']=fare
from scipy import stats

F, p = stats.f_oneway(train[train.FareInterval=='Under 50'].Survived,train[train.FareInterval=='Over or equal to 50'].Survived)

print(F)

#AgeInterval



#first let's fix the NaN values that this column has

train['Age']=train['Age'].fillna(train['Age'].median())



age=[]

for i in train['Age']:

    if i<50:

        age.append('Under 50')

    else:

        age.append('Over or equal to 50')

        
train['AgeInterval']=age

from scipy import stats

F, p = stats.f_oneway(train[train.FareInterval=='Under 50'].Survived,train[train.FareInterval=='Over or equal to 50'].Survived)

print(F)

#select the columns with a significant correlation to the target variable and the target variable

train=train[['FareInterval','Survived','Sex','AgeInterval']]

features=train.drop(columns="Survived")

y=train[['Survived']]

categorical_columns=[column_name for column_name in features.columns if features[column_name].dtype=="object"]

numerical_columns=[column_name for column_name in features.columns if features[column_name].dtype in ["int64", "float64"]]



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_columns),

        ('cat', categorical_transformer, categorical_columns)

    ])
#using RandomForest

modelRF = RandomForestClassifier()



#build the pipeline

pipelineRF = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', modelRF)

                             ])
testData=testData[['Sex','Age','Fare']]
testData.info()
fareTest=[]

for i in testData['Fare']:

    if i<50:

        fareTest.append('Under 50')

    else:

        fareTest.append('Over or equal to 50')

        

testData['FareInterval']=fareTest
testData['Age']=testData['Age'].fillna(testData['Age'].median())

ageTest=[]

for i in testData['Age']:

    if i<50:

        ageTest.append('Under 50')

    else:

        ageTest.append('Over or equal to 50')

        

testData['AgeInterval']=ageTest
testData=testData.drop(['Age','Fare'],axis=1)
testData.info()
features.info()
pipelineRF.fit(features,y)

predictionsRF=pipelineRF.predict(testData)

submission = pd.DataFrame({'PassengerId':sampleSubmission['PassengerId'],'Survived':predictionsRF})

submission
filename = 'Submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)