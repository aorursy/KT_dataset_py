import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# Loading the data in pandas dataframe

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

test = test.merge(gender_submission,on='PassengerId')

train = pd.read_csv("../input/titanic/train.csv")

train.info()

train.head()
train.Cabin = train.Cabin.fillna('R')

train.Cabin=train.Cabin.str.extract('([A-Z])', expand=False)

train.Cabin.unique()

test.Cabin = test.Cabin.fillna('R')

test.Cabin=test.Cabin.str.extract('([A-Z])', expand=False)

test.Cabin.unique()
train['Title'] = train.Name.str.extract('([A-Za-z]+)\.', expand=False)

train.Title.value_counts()

test['Title'] = test.Name.str.extract('([A-Za-z]+)\.', expand=False)

train.Title.isin(test.Title)

test.Title.value_counts()
def data_clean(df):

    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)

    #assign a value for missing titles

    df['Title'] = df['Title'].fillna('NoTitle')

    #Unify titles

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    new_y = df.Survived

    new_X = df.drop(columns = ['Survived','Ticket','Name'])

    new_X=new_X.set_index('PassengerId')

    new_X.info()

    

    return new_X,new_y

X_train,y_train = data_clean(train)

print("X_train - Info")

X_train.info()

X_test,y_test = data_clean(test)

print("X_test - Info")

X_test.info()
def categor_age_fare(df):

    # fill missing values

    df.Age=df.Age.fillna(df.Age.median())

    #create bands for age

    df.loc[ df['Age'] <= 16, 'Age'] = 0

    df.loc[ (df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

    df.loc[ (df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

    df.loc[ (df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

    df.loc[ df['Age'] > 64, 'Age'] = 4

    df['Age'] = df['Age'].astype(int)



    df.Fare=df.Fare.fillna(df.Age.median())

    #create bands for fare

    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0

    df.loc[ (df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

    df.loc[ (df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2

    df.loc[ df['Fare'] > 31, 'Fare'] = 3

    df.Fare = df.Fare.astype(int)

    return df



X_train = categor_age_fare(X_train)

print("X_train - Info")

X_train.info()

X_test = categor_age_fare(X_test)

print("X_test - Info")

X_test.info()

X= pd.concat([X_train,X_test])

train['Agegrouped']=X_train.Age

train['Faregrouped']=X_train.Fare

X_train=X_train.drop(columns = ['Fare'])

X_test=X_test.drop(columns = ['Fare'])
categorical_feature_mask = X_train.dtypes==object

categorical_cols = X_train.columns[categorical_feature_mask].tolist()
train['LastName'] = train.Name.apply(lambda x:x.split(sep=',')[0])

print(str(train.LastName.value_counts().count())+" unique Lastnames are there\n")
from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()

X_train.Sex = le.fit_transform(X_train.Sex)

X_test.Sex = le.fit_transform(X_test.Sex)

X_train.Embarked = X_train.Embarked.fillna(X_train['Embarked'].value_counts().idxmax())

X_train.Embarked = le.fit_transform(X_train.Embarked)

X_test.Embarked = X_test.Embarked.fillna(X_test['Embarked'].value_counts().idxmax())

X_test.Embarked = le.fit_transform(X_test.Embarked)

X_train.Title = X_train.Title.fillna(X_train['Title'].value_counts().idxmax())

X_train.Title = le.fit_transform(X_train.Title)

X_test.Title = X_test.Title.fillna(X_test['Title'].value_counts().idxmax())

X_test.Title = le.fit_transform(X_test.Title)

#X_train.Cabin = X_train.Cabin.fillna(X_train['Cabin'].value_counts().idxmax())

#X_train.Cabin = le.fit_transform(X_train.Cabin)

#X_test.Cabin = X_test.Cabin.fillna(X_test['Cabin'].value_counts().idxmax())

#X_test.Cabin = le.fit_transform(X_test.Cabin)
#Processing feature: Pclass and Embarked

#splitting each feature into new binary features 



def dummy_data(data, columns):

    for column in columns:

        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)

        data = data.drop(column, axis=1)

    return data





dummy_columns = ["Pclass",'Age','Embarked','Cabin','Title']

X_ohe_train=dummy_data(X_train, dummy_columns)

display(X_ohe_train.head())

X_ohe_test = dummy_data(X_test, dummy_columns)

X_ohe_test.insert(X_ohe_train.columns.get_loc("Cabin_T"),"Cabin_T",0)

display(X_ohe_test.head())
from sklearn.preprocessing import MinMaxScaler

min_max=MinMaxScaler(feature_range=(0, 1))

# To scale data 

X_train_minmax = X_ohe_train

X_test_minmax = X_ohe_test

X_train_minmax.iloc[:,1:3]=min_max.fit_transform(X_train_minmax.iloc[:,1:3])

X_test_minmax.iloc[:,1:3]=min_max.fit_transform(X_test_minmax.iloc[:,1:3])
# of people survived grouped by sex

print("percentage of people survived :",round(y_train.sum()/y_train.count(),2)*100)

groupby_sex = train.groupby('Sex').sum().Survived.to_frame()

groupby_sex['Total'] = train.groupby('Sex').count().Survived

groupby_sex.plot.bar()

plt.show()
# of people survived grouped by age

print("percentage of people survived :",round(y_train.sum()/y_train.count(),2)*100)

groupby_age = train.groupby('Agegrouped').sum().Survived.to_frame()

groupby_age['Total'] = train.groupby('Agegrouped').count().Survived

groupby_age.index = groupby_age.index.map({0:'0 - 16',1:'16 - 32',2:'32 - 48',3:'48 - 64',4:'64+'})

groupby_age.plot.bar()

plt.xlabel('Age Range')

plt.ylabel('Count')

plt.show()
# of people survived grouped by age

print("percentage of people survived :",round(y_train.sum()/y_train.count(),2)*100)

groupby_title = train.groupby('Title').sum().Survived.to_frame()

groupby_title['Total'] = train.groupby('Title').count().Survived

groupby_title.plot.bar()

plt.ylabel('Count')

plt.show()
# of people survived grouped by age

print("percentage of people survived :",round(y_train.sum()/y_train.count(),2)*100)

groupby_age = train.groupby('Faregrouped').sum().Survived.to_frame()

groupby_age['Total'] = train.groupby('Faregrouped').count().Survived

groupby_age.index = groupby_age.index.map({0:'0 - 7.91',1:'7.91 - 14.54',2:'14.54 - 31',3:'>31'})

groupby_age.plot.bar()

plt.xlabel('Fare Range')

plt.ylabel('Count')

plt.show()
# of people survived grouped by sex

print("percentage of people survived :",round(y_train.sum()/y_train.count(),2)*100)

groupby_embarked = train.groupby('Embarked').sum().Survived.to_frame()

groupby_embarked['Total'] = train.groupby('Embarked').count().Survived

groupby_embarked.plot.bar()

plt.show()
# of people survived grouped by sex

print("percentage of people survived :",round(y_train.sum()/y_train.count(),2)*100)

groupby_cabin = ctrain.groupby('Cabin').sum().Survived.to_frame()

groupby_cabin['Total'] = ctrain.groupby('Cabin').count().Survived

groupby_cabin

groupby_cabin.plot.bar()

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn import metrics

lr = LinearRegression()

#X_train = MinMaxScaler().fit_transform(X_train)

lr.fit(X_train_minmax,y_train)

y_pred = lr.predict(X_test_minmax)

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",lr.score(X_test_minmax, y_test))
#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train_minmax,y_train)



y_pred=clf.predict(X_test_minmax)
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

from sklearn.metrics import roc_auc_score

# Model Accuracy, how often is the classifier correct?

Accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:",Accuracy)

roc_score = roc_auc_score(y_test, y_pred)

print("ROC_AUC_SCORE : ",roc_score)

CM = confusion_matrix(y_test, y_pred)

print("Confusion Matrix : \n",CM)

report = classification_report(y_test, y_pred)

print("Classification Report :\n",report)
#Dependencies

import keras

from keras.models import Sequential

from keras.layers import Dense

# Neural network

model = Sequential()

model.add(Dense(24, activation='relu',input_shape=(X_train_minmax.shape[1],)))

model.add(Dense(24, activation='softmax'))

model.add(Dense(16, activation='softmax'))

model.add(Dense(2, activation='softmax'))

model.add(Dense(1, activation='sigmoid'))



#compiling the model

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])



from keras.callbacks import EarlyStopping

#set early stopping monitor so the model stops training when it won't improve anymore

early_stopping_monitor = EarlyStopping(patience=3)



#train model

model.fit(X_train_minmax, y_train, validation_split=0.4, epochs=100, callbacks=[early_stopping_monitor])
y_pred = (model.predict(X_test_minmax)>=0.5).astype('int64')

accuracy = model.evaluate(X_test_minmax,y_test,verbose = 0)[1]

print("Accuracy score of the neural network is :",accuracy)
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

from sklearn.metrics import roc_auc_score

# Model Accuracy, how often is the classifier correct?

Accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:",Accuracy)

roc_score = roc_auc_score(y_test, y_pred)

print("ROC_AUC_SCORE : ",roc_score)

CM = confusion_matrix(y_test, y_pred)

print("Confusion Matrix : \n",CM)

report = classification_report(y_test, y_pred,output_dict=True)

print("Classification Report :\n",report)
final_results = pd.DataFrame(y_pred, columns = ['Survived'])

final_results['PassengerId'] = gender_submission.PassengerId

final_results=final_results.set_index('PassengerId')

final_results.to_csv('FinalResults-'+str(round(accuracy,2))+'-'+str(CM[0][1])+','+str(CM[1][0])+'.csv')

print("Output CSV file has been saved "+'FinalResults-'+str(round(accuracy,2))+'-'+str(CM[0][1])+','+str(CM[1][0])+'.csv')