import pandas as pd

import numpy as np

import seaborn as sns

sns.set(style="whitegrid")

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.simplefilter(action='ignore')



training_data = pd.read_csv('../input/train.csv')

testing_data = pd.read_csv('../input/test.csv')
training_data.head()
testing_data.head()
# Mapping 'Sex' Feature 



sex_dict = {'male':0, 'female':1}
training_data['Sex'] = training_data['Sex'].map(sex_dict) 

training_data.head()
testing_data['Sex'] = testing_data['Sex'].map(sex_dict)

testing_data.head()
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

training_data[["Age"]] = imp.fit_transform(training_data[["Age"]]).ravel()

testing_data[["Age"]] = imp.fit_transform(testing_data[["Age"]]).ravel()

testing_data[["Fare"]] = imp.fit_transform(testing_data[["Fare"]]).ravel()

training_data[["Embarked"]] = training_data[["Embarked"]].fillna(method = 'ffill')

testing_data[["Embarked"]] = testing_data[["Embarked"]].fillna(method = 'ffill')
training_data.isnull().sum(axis=0)
testing_data.isnull().sum(axis=0)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

training_data[["Ticket"]] = le.fit_transform(training_data[["Ticket"]]).ravel()

training_data[["Embarked"]] = le.fit_transform(training_data[["Embarked"]]).ravel()

testing_data[["Ticket"]] = le.fit_transform(testing_data[["Ticket"]])

testing_data[["Embarked"]] = le.fit_transform(testing_data[["Embarked"]])
training_data.head()
testing_data.head()
sns.barplot(x="Sex", y="Survived",hue="Sex", data=training_data)

plt.legend(title='Gender', loc='upper left', labels=['Female', 'Male'])
sns.barplot(x="Pclass", y="Survived", data=training_data)
# Where S = 2 , C = 0, Q = 1

sns.barplot(x="Embarked", y="Survived", data=training_data)
sns.barplot(x="SibSp", y="Survived", data=training_data)
sns.barplot(x="Parch", y="Survived", data=training_data)
train_data = training_data.drop(['PassengerId','Name','Cabin','Survived'],axis = 1)

survived = training_data['Survived']

test_data = testing_data.drop(['PassengerId','Name','Cabin'], axis = 1)
train_data.head()
test_data.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data, survived, test_size=0.2, random_state=42)
import lightgbm as lgb

lgbm = lgb.LGBMClassifier(max_depth = 8,            #maximum_depth_of_tree

                         num_leaves=90,             #no_of_leaves

                         lambda_l1 = 0.1,           #l1_regularization_value

                         lambda_l2 = 0.01,          #l2_regularization_value

                         learning_rate = 0.01,      #learning_rate_for_updating_parameter

                         max_bin= 350,              #maximum_binning_for_unique_values

                         n_estimators = 600,        #num_of_trees_to_fit 

                         reg_alpha = 1.6,           #learning_param

                         colsample_bytree = 0.9,    #randomly_select_feature

                         subsample = 0.9,           #select_feature_w/o_randomness

                         n_jobs = 6)                #no_of_threads_used_for_parallel_processing

lgbm.fit(X_train,y_train)

pred = lgbm.predict(test_data)
submission = pd.DataFrame({"PassengerId": testing_data["PassengerId"],"Survived": pred})

submission.to_csv('submission.csv',index= False)
submission.head()