# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from datetime import datetime

started_at = datetime.now().strftime("%H:%M:%S")



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
print(train_data.head())

print(train_data.describe(include="all"))

# Survived values of 0 indicate they did not survive while 1 indicated that they did survive

# The crosstab function just shows counts but it is easier to absorb the info as percentages so we can calculate that as well with the apply

# Let's see how the passenger's Gender, Class of Travel and Embarkation point are distributed among the survivors

print(pd.crosstab(train_data["Sex"],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))

print("-"*50)

print(pd.crosstab(train_data["Pclass"],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))

print("-"*50)

print(pd.crosstab(train_data["Embarked"],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))

import seaborn as sns

import matplotlib.pyplot as plt



sns.countplot(x = 'Sex', hue = 'Survived', data = train_data)
sns.countplot(x = 'Embarked', hue = 'Survived', data = train_data)

# The graph below shows us that people who got on at Southampton had the best chances of survival. Another useful attribute for our model
sns.countplot(x = 'Pclass', hue = 'Survived', data = train_data)

# The graph below shows us that people who got on at Southampton had the best chances of survival. Another useful attribute for our model
# Let's see if any of the data values are null

print(train_data.shape)

print(train_data.isnull().sum())
print(train_data.Embarked.value_counts(dropna=False))

print(train_data.groupby(train_data['Pclass']).Age.median())

print(train_data.groupby(train_data['Embarked']).Age.median())
def fill_nulls(data):

    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)

    data['Age'].fillna(data['Age'].median(),inplace=True)

    data['Fare'].fillna(data['Fare'].mean(),inplace=True) # In the TEST data we have one null Fare



    return data

    

train_data = fill_nulls(train_data)

test_data = fill_nulls(test_data)
print(train_data.isnull().sum())

print(train_data.Embarked.value_counts(dropna=False))
def extract_honorific_from_name(data):

    data['Honorific'] = data['Name'] # initialize this new column

    titles = data["Name"].str.split(",") # we now have an array

    for indx,title in enumerate(titles): # for each element of the array

        data["Honorific"][indx] = title[1].split(".")[0] # Get the Mr, Mrs, Dr, etc by parsing it out

    return data



train_data = extract_honorific_from_name(train_data)

test_data = extract_honorific_from_name(test_data)



#print(train_data.Honorific.value_counts(dropna=False))

#print("=======================")

#print(test_data.Honorific.value_counts(dropna=False))

test_data['Honorific'] = test_data['Honorific'].str.replace('Dona','Mrs')

#print(test_data.Honorific.value_counts(dropna=False))



print(pd.crosstab(train_data['Honorific'],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))

print(train_data['Honorific'].value_counts())

#blank_array = ['Z' for n in range(len(train_data))]

#train_data['foobar'] = blank_array

#print(train_data['foobar'].value_counts())

#train_data.loc[train_data['Honorific'] == 'Mr', train_data['Honorific']]

# Now to encode. We can't pass data twice as the same encoding for train has to be applied for test also

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

train_data['Embarked'] = enc.fit_transform(train_data['Embarked'])

test_data['Embarked'] = enc.transform(test_data['Embarked'])



train_data['Sex'] = enc.fit_transform(train_data['Sex'])

test_data['Sex'] = enc.transform(test_data['Sex'])



train_data['Honorific'] = enc.fit_transform(train_data['Honorific'])

test_data['Honorific'] = enc.transform(test_data['Honorific'])







#print(train_data.isnull().sum())

print(train_data.Embarked.value_counts(dropna=False))

print(train_data.Sex.value_counts(dropna=False))

print(train_data.Honorific.value_counts(dropna=False))


def massage_data(data):

    data["CoPassengers"] = data["SibSp"] + data["Parch"]

    data['Solitary'] = np.where(data['CoPassengers'] > 0, 1, 0)

    data['Minor'] = np.where(data['Age']<=16, 1, 0)

    data['FareCategory'] = np.where(data['Fare'] <= 32, 'X', 'Y')    

    return(data)
train_data = massage_data(train_data)

test_data = massage_data(test_data)



print(pd.crosstab(train_data['CoPassengers'],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))

print(pd.crosstab(train_data['Solitary'],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))

print(pd.crosstab(train_data['Minor'],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))

print(pd.crosstab(train_data['FareCategory'],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))



# A choice of columns for training the model. It's a good idea to play with this list

columns_for_fitting = ['Age','Minor','Sex','Honorific','Embarked','Pclass','Fare','Solitary'] 
X = train_data[columns_for_fitting]

y = train_data['Survived']

X1 = test_data[columns_for_fitting]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, y, test_size=0.2,random_state=21)

print ('Train set:', X_train.shape,  Y_train.shape)

print ('Test set:', X_test.shape,  Y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.svm import SVC





from sklearn.metrics import accuracy_score

from sklearn import preprocessing



svc_model = SVC()

svc_model.fit(X_train,Y_train)

svc_predictions = svc_model.predict(X_test)

print("SVC = {}".format(accuracy_score(Y_test,svc_predictions)))



x_model = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, objective= 'binary:logistic',gamma=0,

 subsample=0.8,colsample_bytree=0.8,scale_pos_weight=1)

x_model.fit(X_train,Y_train,verbose=False)

x_predictions = x_model.predict(X_test)

print("XGB = {}".format(accuracy_score(Y_test,x_predictions)))



est = [20,50,100,150,250,300,500,750,1000,2500]

est=[550]

for n in est:

    rf_model = RandomForestClassifier(n_estimators=n)

    rf_model.fit(X_train,Y_train)

    rf_predictions = rf_model.predict(X_test)

    #print("RandomForest = {}".format(accuracy_score(Y_test,rf_predictions)))

    print ('For RF: n and Accuracy are', n,accuracy_score(Y_test,rf_predictions))



k = 6

while k < 7:

    k = k + 1

    X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))

    knn_model = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)

    X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))

    knn_predictions = knn_model.predict(X_test)

    #print("KNN = {}".format(accuracy_score(Y_test,knn_predictions)))

    print ('K and Accuracy are', k,accuracy_score(Y_test,knn_predictions))
yhat = rf_model.predict(X1)

yhat[0:5]
pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':yhat}).set_index('PassengerId').to_csv('titanic_submission.csv')

print("Okay , We started at " + started_at + " please check for output data now!")