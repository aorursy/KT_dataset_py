#importing the required packages

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import sklearn

import re

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing
#importing the train data and viewing the first few rows

train = pd.read_csv("../input/train.csv")

train.head()
#viewing the details of the train data

train.describe()
#viewing the details of the train data

train.info()
#we notice that the columns "Age" , "Cabin", and "Embarked" have missing values.
#importing the test data and viewing the first few rows

test = pd.read_csv("../input/test.csv")

test.head()
#viewing the details of the test data

test.info()
#we notice that the columns "Age" , "Cabin", and "Fare" have missing values. 

#Also the test data does not have the "Survived" column, so we have to add the "survived" to the test.

#since we have missing values in both the datasets , we will merge them so that 

#we can performe imputations on them and handle the missing values.
#combinig the test and train data

frames = [train,test]

total_data = pd.concat(frames)
total_data.head()
total_data.info()
#some visualizations
#visualizing the total number of passengers survived

colors=sns.color_palette("husl", 10) 

pd.Series(total_data["Survived"]).value_counts().plot(kind = "bar",

                  color=colors,figsize=(8,6),fontsize=10,rot = 0, title = "Total No. of Passengers")

plt.xlabel('Survival', fontsize=10)

plt.ylabel('No. of Passengers', fontsize=10)
#visualizing the number of passengers survived per pclass

sns.countplot(total_data['Pclass'], hue=total_data['Survived'])
#visualizing the number of passengers survived per Embarked locations

sns.countplot(total_data['Embarked'], hue=total_data['Survived'])
#visualizing the number of passengers survived per Gender

sns.countplot(total_data['Sex'], hue=total_data['Survived'])
#Imputing the missing values
#first we will handle the "PassengerId" column. It is not required in our data for imputations , so we can drop it.

total_data = total_data.drop("PassengerId", axis=1)

total_data.head()
#we will handle the "Cabin" column. It has the maximum missing values. IT seems quite statistically insignificant.

#Hence we should drop that column from our dataset.

total_data = total_data.drop("Cabin", axis=1)

total_data.head()
#To check which rows have null Embarked column

total_data[total_data['Embarked'].isnull()]
#visualizing the passengers embarked and fare based on the pclass

sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data= total_data)

plt.axhline(y=80, color='green')
#we see that most people who had pclass 1 and fare 80 would embark at C, hence fill the null values with C.
total_data['Embarked'].fillna('C', inplace=True)
#To check which rows have null Fare column

total_data[total_data['Fare'].isnull()]
#since that passanger is of pclass 3, and embarked from S, we will fill the missing 

#value with the most common value of fare with this combination.
total_data[(total_data.Pclass==3)&(total_data.Embarked=='S')].Fare.value_counts().head()
total_data['Fare'].fillna('8.0500', inplace=True)
#creating a new column "Title" from "Name"



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



total_data['Title'] = total_data['Name'].apply(get_title)



total_data['Title'] = total_data['Title'].replace(['Countess','Capt', 'Col',\

'Don', 'Dr', 'Major', 'Rev','Jonkheer', 'Dona'], 'Rare')



total_data['Title'] = total_data['Title'].replace('Mlle', 'Miss')

total_data['Title'] = total_data['Title'].replace('Ms', 'Miss')

total_data['Title'] = total_data['Title'].replace('Mme', 'Mrs')

total_data['Title'] = total_data['Title'].replace('Lady', 'Mrs')

total_data['Title'] = total_data['Title'].replace('Sir', 'Mr')
#changing the categorical values into numerical values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder



labelEnc=LabelEncoder()



cat_vars=['Embarked','Sex','Ticket','Title']

for col in cat_vars:

    total_data[col]=labelEnc.fit_transform(total_data[col])



total_data.head()
#Also we see that PassengerId ,Parch and SibSp columns can be combined into a single column

#hence combining the PassengerId ,Parch and SibSp into one column as "FamSize"

total_data['FamSize'] = total_data['Parch'] + total_data['SibSp'] +1
#now we have to fill the missing values in "Age" column.

#There are many missing values in age so instead of simply 

#replacing it with mean/median/mode we will try to predict the missing values with a classifier.
from sklearn.ensemble import RandomForestRegressor

#predicting missing values in age using Random Forest

def fill_missing_age(df):

    

    #Feature set

    age_df = df[['Age','Embarked','Fare','Title', 'FamSize',

                 'Ticket','Pclass']]

    # Split sets into train and test

    train_age  = age_df.loc[ (df.Age.notnull()) ]# known Age values

    test_age = age_df.loc[ (df.Age.isnull()) ]# null Ages

    

    # All age values are stored in a target array

    y = train_age.values[:, 0]

    

    # All the other values are stored in the feature array

    X = train_age.values[:, 1::]

    

    # Create and fit a model

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

   

    rtr.fit(X, y)

    

    # Use the fitted model to predict the missing values

    predictedAges = rtr.predict(test_age.values[:, 1::])

    

    # Assign those predictions to the full data set

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df



total_data=fill_missing_age(total_data)
#The "ticket" column also seems quite statistically insignificant , hence remove that as well.

total_data = total_data.drop("Ticket", axis=1)

total_data.head()
#The "Name" column also seems quite statistically insignificant , hence remove that as well.

total_data = total_data.drop("Name", axis=1)

total_data.head()
#creating the train and test features and labels

train_labels = train.Survived

train_features = total_data.ix[:890]

train_features = train_features.drop(['Survived'],axis=1)

test_features = total_data.iloc[891:1309,:]

test_features = test_features.drop(['Survived'],axis=1)
#Feature Scaling



## scale the train data



scaler1 = StandardScaler().fit(train_features)

train_features = scaler1.transform(train_features)



## scale the test data

scaler2 = StandardScaler().fit(test_features)

test_features = scaler2.transform(test_features)



## Normalize the train labels using LabelEncoder

encoder= preprocessing.LabelEncoder()

train_labels = encoder.fit_transform(train_labels)
import warnings

warnings.filterwarnings("ignore")



##using random forest classifier 

from sklearn.ensemble import RandomForestClassifier

clf= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_split=1e-07, min_samples_leaf=12,

            min_samples_split=5, min_weight_fraction_leaf=0.0,

            n_estimators=500, n_jobs=1, oob_score=True, random_state=42,

            verbose=0, warm_start=False)

clffit = clf.fit(train_features,train_labels)

pred= clf.predict(test_features)

print(clffit.score(train_features,train_labels))



#importing the predicted values into excel file in two columns "PassengerId" and "Survived" 

final= pd.DataFrame()

final['PassengerId']= test.PassengerId

final['Survived']= pred



#submitting the predicted values in a csv file

final.to_csv("result_rf4.csv",index=False)
#using XGBoost classifier

from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb

from xgboost import XGBClassifier



gbm = xgb.XGBClassifier(

 learning_rate = 0.02,

 n_estimators= 2000, 

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,               

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1)



gbmfit = gbm.fit(train_features,train_labels)

pred= gbm.predict(test_features)

print(gbmfit.score(train_features,train_labels))



#importing the predicted values into excel file in two columns "PassengerId" and "Survived" 

final= pd.DataFrame()

final['PassengerId']= test.PassengerId

final['Survived']= pred



#submitting the predicted values in a csv file

final.to_csv("result_xgb1.csv",index=False)


