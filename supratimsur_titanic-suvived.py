# import the required libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pickle
# read the csv files



df = pd.read_csv("../input/titanic/train.csv")

df
df.shape
df.info()
df.isna().sum()
import seaborn as sns

#checking the distrubition of missing columns

sns.pairplot(df)
# handling missing data 

# we will drop the cabin colum as it is mostly not present

df_tmp = df.drop(['Cabin','Ticket'],axis = 1)
# split data into training and test 

from sklearn.model_selection import train_test_split

x = df_tmp.drop(['Survived'] , axis = 1)

y = df_tmp['Survived']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
df_tmp['Age']
# for age lets see its distribution first



sns.distplot(df['Age'].dropna())
# we can see its more a right skewed graph so lets take mode median of the age to fill the missing values.

x_train['Age'].fillna(value = x_train['Age'].median(), inplace = True)

x_test['Age'].fillna(value = x_test['Age'].median(), inplace = True)

x_train.isna().sum()

# now lets convert the object type variable to categorical type

# object_cols1 = list(x_train.select_dtypes(include='object').columns)

# x_train[object_cols1] = x_train[object_cols1].astype('category')



# object_cols2 = list(x_test.select_dtypes(include='object').columns)

# x_test[object_cols2] = x_test[object_cols2].astype('category')





x_train['Embarked'].fillna(x_train['Embarked'].mode()[0], inplace=True)

x_test['Embarked'].fillna(x_test['Embarked'].mode()[0], inplace=True)
# as we can see it throws a error for the name column lets then extract the Mr Mrs etc from the name because the full name

# is not required for us to predict and it is of no use. may be there is a probability that if the passenger is male or female

# chances of survival would increase. so lets refrom the name column.





df['Name'].head(20)
name= []

for i in x_train.Name:

    name.append(i)

name
salutation = []



for i in x_train.Name:

    

    a = i.split(',')

    b = a[1].split('.')

    c = b[0]

    salutation.append(c)

    

salutation

    





x_train['salutation'] = salutation

x_train.drop(['Name'], axis = True, inplace = True)



salutation = []



for i in x_test.Name:

    

    a = i.split(',')

    b = a[1].split('.')

    c = b[0]

    salutation.append(c)

    

salutation

    





x_test['salutation'] = salutation

x_test.drop(['Name'], axis = True, inplace = True)
x_test.info()
# outlier detection



x_train.columns

# 'PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked', 'salutation'

plt.figure(figsize=(12,10))

sns.boxplot(data = x_train )
# dbscan clustering

from sklearn.cluster import DBSCAN



db = DBSCAN(min_samples = 2, eps = 3)

cluster = db.fit_predict(x_train[['Age','Fare']])



list(cluster).count(-1)



# isolation forest

from sklearn.ensemble import IsolationForest

clf = IsolationForest( behaviour = 'new', max_samples=100, random_state = 1, contamination= 'auto')

preds = clf.fit_predict(x_train[['Age','Fare']])

preds
# now lets onehotencode the categorical column

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



object_cols = [col for col in list(x_train.select_dtypes(include = 'object'))]

onehot = OneHotEncoder(sparse= False, handle_unknown='ignore')

ct = ColumnTransformer([('onehot', onehot, object_cols)])



transformed_x_train = ct.fit_transform(x_train)

transformed_x_test = ct.transform(x_test)
# lets train a SGD classifier as the training set contai less than 100k samples



from sklearn.linear_model import SGDClassifier



clf = SGDClassifier(random_state=0)



clf.fit(transformed_x_train, y_train)

clf.score(transformed_x_test,y_test)
# now lest check with randomforest regressor though it is not recommended to use randomforest for less than 100k samples



from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(random_state = 1)



model.fit(transformed_x_train, y_train)

model.score(transformed_x_test,y_test)
# as we can see the accuracy score is worst in randomforest we will stick to the SGD estimator

# now we will try to randomizedsearchcv and gridsearch for tuning hyper parameter



from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



# now check the hyperparameters of the estimator



clf.get_params()
grid = {

 'loss': ['hinge','log', 'modified_huber','squared_hinge', 'perceptron',  'squared_loss','huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],

 'penalty': ['none', 'l2', 'l1','elasticnet']

}



grid_model = GridSearchCV(estimator= SGDClassifier(),

                     param_grid= grid,

                         cv=10,

                         verbose=2,

                         n_jobs=-1)



grid_model.fit(transformed_x_train, y_train)

grid_model.score(transformed_x_test, y_test)
# lets try with xgboost to check wether we can improve the score or not



from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(transformed_x_train, y_train)

xgb.score(transformed_x_test, y_test)
# as the score improves lets try with tuning the hyperparmeters



xgb.get_xgb_params()
# now lets check roc curve and confusion matrix 

from sklearn.metrics import roc_curve



predicted_proba = xgb.predict_proba(transformed_x_test)

positive_proba = predicted_proba[:,1]

roc_curve(y_test,positive_proba)

from sklearn.metrics import plot_roc_curve,confusion_matrix,plot_confusion_matrix



plot_roc_curve(xgb,transformed_x_test,y_test)
y_predict = xgb.predict(transformed_x_test)

cm = confusion_matrix(y_test, y_predict)



plot_confusion_matrix(xgb,transformed_x_test,y_test)
# save model

import joblib

joblib.dump(xgb, 'xgb.joblib')
load_model= joblib.load('xgb.joblib') # load saved model
# now load the test dataset

test = pd.read_csv('../input/titanic/test.csv')

test
# preprocessing test data for prediction

test.isna().sum()



# as we can see most of the cabin data are missing we are gonna remove the cabin column

# and also Ticket column is irrelevant

test.drop(['Ticket','Cabin'], axis = 1, inplace = True)
# filling missing value of age column

test.Age.fillna(value= test.Age.median(), inplace = True)

test.info()
# filling missing value of Fare column



sns.distplot(test.Fare.dropna())
# as it is a left skew data we can use median for  the missing value



test.Fare.fillna(value=test.Fare.mode()[0], inplace = True)
test.isna().sum()
# now remove the Name column and add salutation column as we did with training and validation data

salutation = []



for i in test.Name:

    

    a = i.split(',')

    b = a[1].split('.')

    c = b[0]

    salutation.append(c)

    

salutation

    





test['salutation'] = salutation

test.drop(['Name'], axis = True, inplace = True)
# now hotencode the categorical variable



transformed_test = ct.transform(test)
predicted_values = load_model.predict(transformed_test)

len(predicted_values) # matched
# create the dataframe

submit_df = pd.DataFrame()

submit_df['PassengerID'] = test.PassengerId

submit_df['Survived'] = predicted_values

submit_df
#save the dataframe into csv

submit_df.to_csv("Titanic_Survived.csv" , index = False)