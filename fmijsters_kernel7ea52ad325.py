def fill_missing_values(data):

    data['Age_category'] = 0

    data.loc[(train_data['Age'] < 18) & (data['Age'] >= 0),"Age_category"] =1

    data.loc[(train_data['Age'] < 36) & (data['Age'] >= 18),"Age_category"] =2

    data.loc[(train_data['Age'] < 54) & (data['Age'] >= 36),"Age_category"] =3

    data.loc[(train_data['Age'] < 72) & (data['Age'] >= 54),"Age_category"] =4

    data.loc[(train_data['Age'] < 90) & (data['Age'] >= 72),"Age_category"] =5



    data['Initial']=0

    for i in data:

        data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')

    data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)    



    data['FamilySize'] = data['SibSp'] + data['Parch']

    data.loc[(data.Fare.isnull()),'Fare']=data.Fare.mean()





    data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33

    data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36

    data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5

    data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22

    data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46



    data['Embarked'].fillna('S',inplace=True)

    return(data)
def clean_data(data):

    data['Sex'].replace(['male','female'],[0,1],inplace=True)

    data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

    return data
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

train_filename = '/kaggle/input/titanic/train.csv'

test_filename = '/kaggle/input/titanic/test.csv'



train_data = pd.read_csv(train_filename)

train_data = fill_missing_values(train_data)

train_data = clean_data(train_data)





# features = ['Sex', 'FamilySize', 'Fare' ,'Embarked','Age_category']

features = ['Sex', 'Age','Embarked','FamilySize', 'Fare' ,'Age_category']



y = train_data.Survived

X = train_data[features]



X_train, X_valid, y_train, y_test = train_test_split(X, y, random_state=1)





model_expiriment = RandomForestRegressor(random_state=1,n_estimators=9100)

s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print(OH_X_train.columns)

model_expiriment.fit(OH_X_train,y_train)

predictions_expiriment = model_expiriment.predict(OH_X_valid)



score = mean_absolute_error(y_test, predictions_expiriment)

print('experiment',score)

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



from sklearn.ensemble import VotingClassifier

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



import pandas as pd

train_filename = '/kaggle/input/titanic/train.csv'

test_filename = '/kaggle/input/titanic/test.csv'



train_data = pd.read_csv(train_filename)

train_data = fill_missing_values(train_data)

train_data = clean_data(train_data)



test_data = pd.read_csv(test_filename)

test_data = fill_missing_values(test_data)

test_data = clean_data(test_data)

train_data.columns

features = ['Sex', 'FamilySize', 'Fare' ,'Embarked','Age_category']

baseline_features = ['Sex', 'SibSp', 'Parch', 'Fare' ,'Embarked','Age_category']

y = train_data.Survived

X = train_data[features]

X_baseline = train_data[baseline_features]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)



X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(X_baseline, y, random_state=1)



predictable_test_data = test_data[features]



# model_baseline = RandomForestRegressor(random_state=1,n_estimators=9100)

# model_baseline.fit(X_train_baseline,y_train_baseline)

# predictions_baseline = model_baseline.predict(X_test_baseline)



# model_expiriment = RandomForestRegressor(random_state=1,n_estimators=9100)

# model_expiriment.fit(X_train,y_train_baseline)

# predictions_expiriment = model_expiriment.predict(X_test)

# score = mean_absolute_error(y_test, predictions_expiriment)



estimators=[('RFor',RandomForestClassifier(n_estimators=100,random_state=0)),

            ('LR',LogisticRegression(C=0.05,solver='liblinear')),

            ('DT',DecisionTreeClassifier()),

            ('NB',GaussianNB())]



ensemble=VotingClassifier(estimators=estimators,voting='soft')

ensemble.fit(X,y)

print('The accuracy for ensembled model is:',ensemble.score(X_test,y_test))

cross=cross_val_score(ensemble,X,y, cv = 10,scoring = "accuracy")

print('The cross validated score is',cross.mean())





predictions = ensemble.predict(predictable_test_data)

# score_baseline = mean_absolute_error(y_test_baseline, predictions_baseline)

# score = mean_absolute_error(y_test, predictions_expiriment)

# print('baseline',score_baseline)

# print('experiment',score)







# model = RandomForestRegressor(random_state=1,n_estimators=9100)

# model.fit(X,y)

# predictions = model.predict(predictable_test_data)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': np.round(predictions).astype('int')})

output.to_csv('submission.csv', index=False)