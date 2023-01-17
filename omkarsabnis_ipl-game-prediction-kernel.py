# IMPORTING REQUIRED LIBRARIES

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# LOADING THE GIVEN DATA

train = pd.read_csv("../input/ipledited/train.csv")

test = pd.read_csv("../input/ipl-game-prediction/test.csv")

# VISUALIZING THE DATASET

test.head()
# TO CHECK THE DATASET FOR IMBALANCE - SO WE CAN EITHER UPSAMPLE OR DOWNSAMPLE

train['Winner (team 1=1, team 2=0)'].value_counts()
# BASIC ANALYSIS OF COLUMNS

print(train.columns)

print(train.dtypes)

train.describe(include="all")
# DROPPING THE CATEGORICAL VARIABLES AND SPLITTING THE DATA INTO TRAINING AND TESTING SETS

from sklearn.model_selection import train_test_split

p =  train.drop(['Game ID','Team 1','Team 2','City','DayOfWeek','DateOfGame','TimeOfGame','AvgWindSpeed','AvgHumidity','Winner (team 1=1, team 2=0)'],axis=1)

target = train['Winner (team 1=1, team 2=0)']

x_train,x_val,y_train,y_val = train_test_split(p,target,test_size=0.25,random_state=0)

test_target = test['Winner (team 1=1, team 2=0)']

q = test.drop(['Game ID','Team 1','Team 2','CityOfGame','Day','DateOfGame','TimeOfGame','AvgWindSpeed','AvgHumidity','Winner (team 1=1, team 2=0)'],axis=1)
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

l = LogisticRegression()

l.fit(x_train,y_train)

y_pred = l.predict(x_val)

print(classification_report(y_val,y_pred))

print()

print(confusion_matrix(y_val,y_pred))

print()

print(accuracy_score(y_val,y_pred)*100)

y_lol = l.predict(q)

print()

print(classification_report(test_target,y_lol))

print()

print(confusion_matrix(test_target,y_lol))

print()

print(accuracy_score(test_target,y_lol)*100)
# NORMAL CORRELATION CHECKER

sns.barplot(x="Inn 1 Team 2 wickets taken_catches_runout",y="Winner (team 1=1, team 2=0)",color="yellow",data=train)
# DROPPING THE CATEGORICAL VARIABLES AND SPLITTING THE DATA INTO TRAINING AND TESTING SETS

from sklearn.model_selection import train_test_split

p = train.drop(['Game ID','Team 1','Team 2','City','DayOfWeek','DateOfGame','TimeOfGame','AvgWindSpeed','AvgHumidity','Inn 1 Team 2 wickets taken_catches_runout','Inn 1 Team 2 Extras conceded in_wides_No Balls','Inn 2 Team 2 NOP R>25,SR>125','Inn 2 Team 2 Total 6s','Inn 2 Team 1 Extras conceded in_wides_No Balls','Winner (team 1=1, team 2=0)'],axis=1)

target = train['Winner (team 1=1, team 2=0)']

x_train,x_val,y_train,y_val = train_test_split(p,target,test_size=0.28,random_state=0)

test_target = test['Winner (team 1=1, team 2=0)']

q = test.drop(['Game ID','Team 1','Team 2','CityOfGame','Day','DateOfGame','TimeOfGame','AvgWindSpeed','AvgHumidity','Inn 1 Team 2 wickets taken_catches_runout','Inn 1 Team 2 Extras conceded in_wides_No Balls','Inn 2 Team 2 NOP R>25,SR>125','Inn 2 Team 2 Total 6s','Inn 2 Team 1 Extras conceded in_wides_No Balls','Winner (team 1=1, team 2=0)'],axis=1)
# RETESTING THE DATA TO SEE IF WE CAN GET ANY IMPROVEMENT IN PRECISION, RECALL and ACCURACY

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

l = LogisticRegression()

l.fit(x_train,y_train)

y_pred = l.predict(x_val)

print(classification_report(y_val,y_pred))

print()

print(confusion_matrix(y_val,y_pred))

print()

print(accuracy_score(y_val,y_pred)*100)

y_lol = l.predict(q)

print()

print(classification_report(test_target,y_lol))

print()

print(confusion_matrix(test_target,y_lol))

print()

print(accuracy_score(test_target,y_lol)*100)

y_prob = l.predict_proba(q)
# PRINTING THE TEST DATASET AND RESULTS

print("Team 1", "|", "Team 2","|", "Winner","|", "Probability")

for i in range(1,len(test)):

    print(test["Team 1"][i],"|",test['Team 2'][i],"|", y_lol[i],"|", y_prob[i])



output = pd.DataFrame({'Team 1': test['Team 1'], 'Team 2': test['Team 2'], 'Winner': y_lol, 'Probability' : list(y_prob)})

output.to_csv("Prediction.csv",index=False)
import keras 

from keras.models import Sequential 

from keras.layers import Dense,Dropout



# MODEL

model = Sequential()



# layers

model.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

model.compile(optimizer = 'Adagrad', loss = 'binary_crossentropy', metrics = ['accuracy'])



model.fit(x_train,y_train,batch_size=32,epochs=200)

y_pred = model.predict(x_val)

for i in range(len(y_pred)):

    if(y_pred[i]<0.5):

        y_pred[i] = 0

    else:

        y_pred[i] = 1

print(classification_report(y_val,y_pred))

print()

print(confusion_matrix(y_val,y_pred))

print()

print(accuracy_score(y_val,y_pred)*100)

y_lol = model.predict(q)

for i in range(len(y_lol)):

    if(y_lol[i]<0.5):

        y_lol[i] = 0

    else:

        y_lol[i] = 1

print()

print(classification_report(test_target,y_lol))

print()

print(confusion_matrix(test_target,y_lol))

print()

print(accuracy_score(test_target,y_lol)*100)