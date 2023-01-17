# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
mydata= pd.read_csv('../input/train.csv')

mytest= pd.read_csv('../input/test.csv')



# See details of each columns we have in mydata

mydata.info()
mydata.head(10)


#Getting initials from Name, putting in new column 'Initials'

mydata.loc[mydata['Name'].str.contains('Mr.',regex=False).values,'Initials']='Mr'

mydata.loc[mydata['Name'].str.contains('Mrs.',regex=False).values,'Initials']='Mrs'

mydata.loc[mydata['Name'].str.contains('Miss.',regex=False).values,'Initials']='Miss'

mydata.loc[mydata['Name'].str.contains('Master.',regex=False).values,'Initials']='Master'

mydata.loc[mydata['Name'].str.contains('Dr.',regex=False).values,'Initials']='Mr'

mydata.loc[(mydata['Initials'].isnull()) & (mydata['Sex'].values=='male'),'Initials']='Mr'

mydata.loc[(mydata['Initials'].isnull()) & (mydata['Sex'].values=='female'),'Initials']='Mrs'

mydata.groupby('Initials')['Age'].mean()
#Filling mising values of Age feature with their respective age group

mydata.loc[(mydata['Initials']=='Master') & (mydata['Age'].isnull()),'Age']=4.57

mydata.loc[(mydata['Initials']=='Miss') & (mydata['Age'].isnull()),'Age']=21.77

mydata.loc[(mydata['Initials']=='Mr') & (mydata['Age'].isnull()),'Age']=33.02

mydata.loc[(mydata['Initials']=='Mrs') & (mydata['Age'].isnull()),'Age']=35.59

mydata.loc[mydata['Age'].isnull(),'Age']=32.36   # setting all others to Male mean--> Dr.

mydata.info()

#mydata[mydata['Age'].isnull()]



mydata['Embarked'] = mydata['Embarked'].fillna('median')
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

def col_encode(data, col):

    #Label encoding and OneHotEncoding

    lenco= LabelEncoder()

    col_labels = lenco.fit_transform(data[col])

    col_mappings = {index: label for index, label in enumerate(lenco.classes_)}

    

    data['col_label']=col_labels

    

    ohe = OneHotEncoder()

    feature_arr = ohe.fit_transform(

                              data[['col_label']]).toarray()

    feature_labels = list(lenco.classes_)

    col_features = pd.DataFrame(feature_arr, 

                            columns=feature_labels)



    data_ohe = pd.concat([data, col_features], axis=1)

    return data_ohe
mydata_ohe = col_encode(mydata,'Sex')

mydata_ohe = col_encode(mydata_ohe,'Embarked')

mydata_ohe 
#selecting some main features to feed our Machine Learning algorithm

main_features=['PassengerId','Survived','Pclass','Age','SibSp','Parch','female','male','C','Q','S']

mydata_final=mydata_ohe[main_features]

mydata_final
# Now saperate the features(X) and the output value (y)

X=mydata_final.drop('Survived',axis=1)

y=mydata_final['Survived'].values

y=y.reshape(891,1)



#splitting the observations into two part, train and test

from sklearn.model_selection import train_test_split as tts



Xtrain,Xtest, ytrain,ytest=tts(X,y, random_state=1)

ytrain=ytrain.ravel()

ytest =ytest.ravel()
mydata.groupby('Survived').count()
#applying Gaussian algorithm

from sklearn.naive_bayes import GaussianNB

model_g=GaussianNB()

y_model_g=model_g.fit(Xtrain,ytrain).predict(Xtest)



from sklearn.metrics import accuracy_score

accuracy_score(ytest,y_model_g)
#LogisticRegression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix

model_lr = LogisticRegression()

model_lr.fit(Xtrain, ytrain)



y_model_lr = model_lr.predict(Xtest)



# Summary of the predictions made by the classifier

print(classification_report(ytest, y_model_lr))

print(confusion_matrix(ytest, y_model_lr))

# Accuracy score

print('accuracy is',accuracy_score(y_model_lr,ytest))
#decision tree

from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier()

model_dt.fit(Xtrain, ytrain)



y_model_dt = model_dt.predict(Xtest)



# Summary of the predictions made by the classifier

print(classification_report(ytest, y_model_dt))

print(confusion_matrix(ytest, y_model_dt))

# Accuracy score

print('accuracy is',accuracy_score(y_model_dt,ytest))
# Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(max_depth=4)

model_rf.fit(Xtrain, ytrain)



y_model_rf = model_rf.predict(Xtest)



# Summary of the predictions made by the classifier

print(classification_report(ytest, y_model_rf))

print(confusion_matrix(ytest, y_model_rf))

# Accuracy score

print('accuracy is',accuracy_score(y_model_rf,ytest))
from xgboost import XGBClassifier

# fit model no training data

model_xg = XGBClassifier()

model_xg.fit(Xtrain, ytrain)

# make predictions for test data

y_pred_xg = model_xg.predict(Xtest)

predictions = [round(value) for value in y_pred_xg]



# evaluate predictions

accuracy = accuracy_score(ytest, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Now testing our model into real test data (the data that our model has not seen previously)

# prepairing our test data



mytest.loc[mytest['Name'].str.contains('Mr.',regex=False).values,'Initials']='Mr'

mytest.loc[mytest['Name'].str.contains('Mrs.',regex=False).values,'Initials']='Mrs'

mytest.loc[mytest['Name'].str.contains('Miss.',regex=False).values,'Initials']='Miss'

mytest.loc[mytest['Name'].str.contains('Master.',regex=False).values,'Initials']='Master'

mytest.loc[(mytest['Initials'].isnull()) & (mytest['Sex'].values=='male'),'Initials']='Mr'

mytest.loc[(mytest['Initials'].isnull()) & (mytest['Sex'].values=='female'),'Initials']='Mrs'



mytest.groupby('Initials').mean().Age
#mydata.loc[mydata['Initials'].isnull()]



mytest.loc[(mytest['Initials']=='Master') & (mytest['Age'].isnull()),'Age']=7.4

mytest.loc[(mytest['Initials']=='Miss') & (mytest['Age'].isnull()),'Age']=21.77

mytest.loc[(mytest['Initials']=='Mr') & (mytest['Age'].isnull()),'Age']=32.34

mytest.loc[(mytest['Initials']=='Mrs') & (mytest['Age'].isnull()),'Age']=38.90

mytest.loc[mytest['Age'].isnull(),'Age']=32.34   # setting all others to Male mean--> Dr.



mytest.describe()
mytest['Embarked'] = mytest['Embarked'].fillna('median')

mytest_ohe = col_encode(mytest,'Sex')

mytest_ohe = col_encode(mytest_ohe,'Embarked')

main_features.pop(1)

mytest_final=mytest_ohe[main_features]

mytest_final
y_predict_test=model_rf.predict(mytest_final)

y_predict_test
y_predict_test_pd = pd.DataFrame(y_predict_test)

y_predict_test_pd.columns=['Predicted_survival']



mytest_out = pd.concat([mytest, y_predict_test_pd], axis=1)

sel_col=['PassengerId','Predicted_survival']

mytest_out[sel_col].to_csv('Predicted_submission.csv')
