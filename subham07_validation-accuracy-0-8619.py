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
# Load the full dataset

X = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
print(X.head())
Y=X['Survived'].values.tolist()
#Coding category P as 0 and C as 1

for i in range(len(X)):

    if(X['Category'][i]=='P'):

        X['Category'][i]=0;

    else:

        X['Category'][i]=1;



print(X.head())
print(X['Country'].unique())
print(X.groupby(['Survived','Country']).size())
#Coding Country as numbers

#Coding category P as 0 and C as 1

for i in range(len(X)):

    if(X['Country'][i]=='Sweden'):

        X['Country'][i]=0;

    elif(X['Country'][i]=='Estonia'):

        X['Country'][i]=1;

    elif(X['Country'][i]=='Latvia'):

        X['Country'][i]=2;

    elif(X['Country'][i]=='Finland'):

        X['Country'][i]=3;

    elif(X['Country'][i]=='Russia'):

        X['Country'][i]=4;

    else:

        X['Country'][i]=5;



print(X.head())
#Coding Country as numbers

#Coding Sex as 0 for Male and 1 for Female

for i in range(len(X)):

    if(X['Sex'][i]=='M'):

        X['Sex'][i]=0;

    elif(X['Sex'][i]=='F'):

        X['Sex'][i]=1;



print(X.head())
X=X.drop(columns=['PassengerId','Firstname','Lastname','Survived'])

print(X.head())
X_all=X.values.tolist()
print(X_all)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_all, Y, test_size=0.3, random_state=42)

model=RandomForestClassifier(random_state=42)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print(accuracy_score(y_pred,y_test))
#tuning no. of estimators



n_estimators_list=[1,2,4,8,16,32,64,128,256,512]



for n_estimators in n_estimators_list:

    model=RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    model.fit(X_train,y_train)



    y_pred=model.predict(X_test)

    print(accuracy_score(y_pred,y_test))
#tuning no. of estimators



min_samples_split_list=[0.01,0.1,0.5,1.0,2,4,8,16,32]



for min_samples_split in min_samples_split_list:

    model=RandomForestClassifier(n_estimators=128, min_samples_split=min_samples_split, random_state=42)

    model.fit(X_train,y_train)



    y_pred=model.predict(X_test)

    print(accuracy_score(y_pred,y_test))
print(model.feature_importances_)
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score



clf1=RandomForestClassifier(n_estimators=128, max_depth=100, random_state=42)

clf2=MultinomialNB()

clf3=LogisticRegression(random_state=42)



ensemble =VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')

ensemble.fit(X_train,y_train)

# make a prediction for one example

y_pred=ensemble.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy score: ", accuracy_score(y_pred,y_test))
from sklearn.ensemble import StackingClassifier   

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

clf1=RandomForestClassifier(n_estimators=200, max_depth=100, random_state=42)

clf2=MultinomialNB()

clf3=LogisticRegression(random_state=42)



sclf = StackingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)])

sclf.fit(X_train,y_train)



y_pred = sclf.predict(X_test)



score = accuracy_score(y_test, y_pred)

print("Accuracy score: ", accuracy_score(y_pred,y_test))
model2=LogisticRegression()

model2.fit(X_train,y_train)



y_pred=model2.predict(X_test)

print(accuracy_score(y_pred,y_test))
from xgboost import XGBClassifier

import numpy as np

model3=XGBClassifier()



X_train_n=np.asarray(X_train)

X_test_n=np.asarray(X_test)



model3.fit(X_train_n,y_train)

y_pred=model3.predict(X_test_n)

predictions = [round(value) for value in y_pred]

print(accuracy_score(predictions,y_test))
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_all, Y, test_size=0.3, random_state=42)

model=RandomForestClassifier(random_state=42)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("ValidationAccuracy: ", accuracy_score(y_pred,y_test))