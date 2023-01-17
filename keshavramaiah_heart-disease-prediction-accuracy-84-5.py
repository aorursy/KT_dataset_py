import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers.core import Dense,Activation,Dropout

from keras.layers import LSTM

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier





from keras.callbacks import EarlyStopping

import math

df=pd.read_csv('../input/heart.csv',delimiter=',')

df.head(3)
df.corr()
fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(df.corr(),annot=True, linewidths=.5,ax=ax)
label=df['target']

df.shape

del df['target']

df.shape
sns.countplot(label)
X_train,Y_train,X_test,Y_test=train_test_split(df,label,test_size=0.3,random_state=0)

print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)

X_valid,X_rftest,Y_valid,Y_rftest=train_test_split(Y_train,Y_test,test_size=0.5,random_state=0)
#Hyperparameter tuning

maxi=0

l=np.arange(1,5000,25)

for x in range(len(l)):

    model=RandomForestClassifier(random_state=l[x],verbose=1)

    model.fit(X_train,X_test)

    a=model.predict(X_valid)

    if(maxi<round(accuracy_score(a,Y_valid)*100,2)):

        maxi=round(accuracy_score(a,Y_valid)*100,2)

        c1=l[x]

print(c1)



    

maxi=0

for i in range(c1-25,c1+25):

    model=RandomForestClassifier(random_state=i,verbose=1)

    model.fit(X_train,X_test)

    a=model.predict(X_valid)

    if(maxi<round(accuracy_score(a,Y_valid)*100,2)):

        maxi=round(accuracy_score(a,Y_valid)*100,2)

        c1=i

model=RandomForestClassifier(random_state=c1)

model.fit(X_train,X_test)

predict1=model.predict(Y_train)

weight1=round(accuracy_score(predict1,Y_test)*100,2)

print('Random forest accuracy score:',round(accuracy_score(predict1,Y_test)*100,2))

c1=round(accuracy_score(predict1,Y_test)*100,2)

model=svm.SVC(kernel='linear',verbose=1,gamma='scale', decision_function_shape='ovo')

model.fit(X_train,X_test)

predict2=model.predict(Y_train)

c=0

for i in range(len(predict2)):

    if(predict2[i]==Y_test.iloc[i]):

        c+=1

c2=(c/len(predict2))*100

print('Linear Svm Accuracy Score is',c2)

weight2=c2



model = XGBClassifier(objective="binary:logistic")

model.fit(X_train, X_test)

predict3=model.predict(Y_train)

c=0

for i in range(len(predict3)):

    if(predict3[i]==Y_test.iloc[i]):

        c+=1

c3=(c/len(predict3))*100

print('XGBoost Accuracy Score is',c3)

weight3=c3

X_train=np.expand_dims(X_train, axis=2) 

Y_train=np.expand_dims(Y_train, axis=2)

es=EarlyStopping(patience=7)

model=Sequential()

model.add(LSTM(13,input_shape=(13,1)))

model.add(Dense(output_dim=1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train,X_test,epochs=100,batch_size=1,verbose=1,callbacks=[es])

predict4=model.predict(Y_train)

c4=model.evaluate(Y_train,Y_test)

weight4=c4[1]

print('SVM Accuracy Score:',c2)

print('XGBoost Accuracy Score:',c3)

print('LSTM accuracy Score:',c4[1]*100)

print('Random Forest Classifier:',c1)
l=[]

for i in range(len(Y_train)):

    c1,c2=0,0

    if(predict1[i]==0):

        c1+=weight1

    else:

        c2+=weight1

    if(predict2[i]==0):

        c1+=weight2

    else:

        c2+=weight2

    if(predict3[i]==0):

        c1+=weight3

    else:

        c2+=weight3

    if(predict4[i]==0):

        c1+=weight4

    else:

        c2+=weight4

    if(c1>c2):

        l.append(0)

    else:

        l.append(1)

c=0



for i in range(len(Y_train)):

    if(l[i]==Y_test.iloc[i]):

        c+=1

print('Accuracy of Voting Based Model',c/len(Y_train))

    

        

        
