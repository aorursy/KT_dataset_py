# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_trainval=pd.read_csv("/kaggle/input/titanic/train.csv")

df_test=pd.read_csv("/kaggle/input/titanic/test.csv")
df_trainval_C=df_trainval.copy()
print("df_trainval shape: ",df_trainval.shape)

print("df_test shape: ",df_test.shape)
df_trainval.info()
df_test.info()

df_trainval_C.columns
df_trainval.head()
print(df_trainval["Survived"].value_counts())

sns.countplot(x="Survived",data=df_trainval)

plt.show()
print("% Of Males Survived:",(((((df_trainval["Sex"]=="male")&(df_trainval["Survived"]==1)).astype(int)).sum())/(df_trainval["Sex"]=="male").astype(int).sum())*100,"%")

print("% Of Females Survived:",(((((df_trainval["Sex"]=="female")&(df_trainval["Survived"]==1)).astype(int)).sum())/(df_trainval["Sex"]=="female").astype(int).sum())*100,"%")

fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(df_trainval['Survived'],hue=df_trainval["Sex"])

plt.title('Survived labels values')

plt.show()
print("% Of Males Survived from class 1:",(((((df_trainval["Sex"]=="male")&(df_trainval["Survived"]==1)&(df_trainval["Pclass"]==1)).astype(int)).sum())/(((df_trainval["Sex"]=="male")&(df_trainval["Pclass"]==1)).astype(int)).sum())*100,"%")

print("% Of Males Survived from class 2:",(((((df_trainval["Sex"]=="male")&(df_trainval["Survived"]==1)&(df_trainval["Pclass"]==2)).astype(int)).sum())/(((df_trainval["Sex"]=="male")&(df_trainval["Pclass"]==2)).astype(int)).sum())*100,"%")

print("% Of Males Survived from class 3:",(((((df_trainval["Sex"]=="male")&(df_trainval["Survived"]==1)&(df_trainval["Pclass"]==3)).astype(int)).sum())/(((df_trainval["Sex"]=="male")&(df_trainval["Pclass"]==3)).astype(int)).sum())*100,"%")

print("% Of Females Survived from class 1:",(((((df_trainval["Sex"]=="female")&(df_trainval["Survived"]==1)&(df_trainval["Pclass"]==1)).astype(int)).sum())/(((df_trainval["Sex"]=="female")&(df_trainval["Pclass"]==1)).astype(int)).sum())*100,"%")

print("% Of Females Survived from class 2:",(((((df_trainval["Sex"]=="female")&(df_trainval["Survived"]==1)&(df_trainval["Pclass"]==2)).astype(int)).sum())/(((df_trainval["Sex"]=="female")&(df_trainval["Pclass"]==2)).astype(int)).sum())*100,"%")

print("% Of Females Survived from class 3:",(((((df_trainval["Sex"]=="female")&(df_trainval["Survived"]==1)&(df_trainval["Pclass"]==3)).astype(int)).sum())/(((df_trainval["Sex"]=="female")&(df_trainval["Pclass"]==3)).astype(int)).sum())*100,"%")

sns.catplot(x='Survived',hue="Pclass",col='Sex',kind="count",data=df_trainval)

plt.show()
df_trainval["Parch"].value_counts()
print("% of people survived with having 0 Parch: ",(((((df_trainval["Parch"]==0)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["Parch"]==0).astype(int).sum()))*100," %")

print("% of people survived with having 1 Parch: ",(((((df_trainval["Parch"]==1)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["Parch"]==1).astype(int).sum()))*100," %")

print("% of people survived with having 2 Parch: ",(((((df_trainval["Parch"]==2)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["Parch"]==2).astype(int).sum()))*100," %")

print("% of people survived with having 3 Parch: ",(((((df_trainval["Parch"]==3)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["Parch"]==3).astype(int).sum()))*100," %")

print("% of people survived with having 4 Parch: ",(((((df_trainval["Parch"]==4)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["Parch"]==4).astype(int).sum()))*100," %")

print("% of people survived with having 5 Parch: ",(((((df_trainval["Parch"]==5)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["Parch"]==5).astype(int).sum()))*100," %")

print("% of people survived with having 6 Parch: ",(((((df_trainval["Parch"]==6)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["Parch"]==6).astype(int).sum()))*100," %")



sns.catplot(x='Survived',hue="Parch",kind="count",data=df_trainval)
df_trainval["SibSp"].value_counts()
print("% of people survived with having 0 Siblings: ",(((((df_trainval["SibSp"]==0)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["SibSp"]==0).astype(int).sum()))*100," %")

print("% of people survived with having 1 Siblings: ",(((((df_trainval["SibSp"]==1)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["SibSp"]==1).astype(int).sum()))*100," %")

print("% of people survived with having 2 Siblings: ",(((((df_trainval["SibSp"]==2)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["SibSp"]==2).astype(int).sum()))*100," %")

print("% of people survived with having 3 Siblings: ",(((((df_trainval["SibSp"]==3)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["SibSp"]==3).astype(int).sum()))*100," %")

print("% of people survived with having 4 Siblings: ",(((((df_trainval["SibSp"]==4)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["SibSp"]==4).astype(int).sum()))*100," %")

print("% of people survived with having 5 Siblings: ",(((((df_trainval["SibSp"]==5)&(df_trainval["Survived"]==1)).astype(int).sum())/(df_trainval["SibSp"]==5).astype(int).sum()))*100," %")

sns.catplot(x='Survived',hue="SibSp",kind="count",data=df_trainval)
fig,ax=plt.subplots(figsize=(20,5))

df_trainval["Age"].hist()

plt.title("Age Distribution")

plt.show()
plt.subplots(figsize=(20,5))

df_trainval["Fare"].hist()

plt.title("Fare Distribution")

plt.show()
print("% Of male Survived with boarding at Southampton :",(((((df_trainval["Sex"]=="male")&(df_trainval["Survived"]==1)&(df_trainval["Embarked"]=="S")).astype(int)).sum())/(((df_trainval["Sex"]=="male")&(df_trainval["Embarked"]=="S")).astype(int)).sum())*100,"%")

print("% Of female Survived with boarding at Southampton :",(((((df_trainval["Sex"]=="female")&(df_trainval["Survived"]==1)&(df_trainval["Embarked"]=="S")).astype(int)).sum())/(((df_trainval["Sex"]=="female")&(df_trainval["Embarked"]=="S")).astype(int)).sum())*100,"%")

print("% Of male Survived with boarding at Cherbourg :",(((((df_trainval["Sex"]=="male")&(df_trainval["Survived"]==1)&(df_trainval["Embarked"]=="C")).astype(int)).sum())/(((df_trainval["Sex"]=="male")&(df_trainval["Embarked"]=="C")).astype(int)).sum())*100,"%")

print("% Of female Survived with boarding at Cherbourg :",(((((df_trainval["Sex"]=="female")&(df_trainval["Survived"]==1)&(df_trainval["Embarked"]=="C")).astype(int)).sum())/(((df_trainval["Sex"]=="female")&(df_trainval["Embarked"]=="C")).astype(int)).sum())*100,"%")

print("% Of female Survived with boarding at Queenstown :",(((((df_trainval["Sex"]=="male")&(df_trainval["Survived"]==1)&(df_trainval["Embarked"]=="Q")).astype(int)).sum())/(((df_trainval["Sex"]=="male")&(df_trainval["Embarked"]=="Q")).astype(int)).sum())*100,"%")

print("% Of female Survived with boarding at Queenstown :",(((((df_trainval["Sex"]=="female")&(df_trainval["Survived"]==1)&(df_trainval["Embarked"]=="Q")).astype(int)).sum())/(((df_trainval["Sex"]=="female")&(df_trainval["Embarked"]=="Q")).astype(int)).sum())*100,"%")

sns.catplot(x='Survived',col="Embarked",hue="Sex",kind="count",data=df_trainval)

plt.show() 
df_trainval["Sex"].value_counts()
df_trainval["Sex_labels"],Sex_uniques=pd.factorize(df_trainval["Sex"])

df_trainval.head()
df_trainval["Pclass"].value_counts()
Age_mean=df_trainval["Age"].mean()

df_trainval["Age"].fillna(Age_mean,inplace=True)
a,age_bins=pd.qcut(df_trainval["Age"],7,retbins=True)

df_trainval["Age_label"],Age_unique=pd.factorize(a)

df_trainval.head()
df_trainval["Age_label"].value_counts()
age_cat=[]

for i in range(7):

    i1=age_bins[i]

    i2=age_bins[i+1]

    age_cat.append(str(i)+"  :  "+i1.astype(str)+"-"+i2.astype(str))

print("age_bins: ",age_cat)

sns.catplot(x='Survived',hue="Age_label",kind="count",data=df_trainval)

plt.show()
df_trainval.corr()
df_trainval["Parch"].value_counts()
df_trainval["(Sex_labels+1)/Parch"]=(df_trainval["Sex_labels"]+1)/df_trainval["Pclass"]

df_trainval["Parch+SibSp+1"]=df_trainval["SibSp"]+df_trainval["Parch"]+1

df_trainval["Pclass/Parch"]=df_trainval["Parch"]/df_trainval["Pclass"]

df_trainval["Parch+SibSp+sex_label"]=df_trainval["SibSp"]+df_trainval["Parch"]+df_trainval["Sex_labels"]

df_trainval.head()
df_trainval["Embarked"].value_counts()
df_trainval["Embarked_label"],Embarked_unique=pd.factorize(df_trainval["Embarked"])

df_trainval.head()
b,Fare_bins=pd.qcut(df_trainval["Fare"],5,retbins=True)

df_trainval["Fare_labels"],Fare_unique=pd.factorize(b)

df_trainval.head()
df_trainval["Fare_labels"].value_counts()
sns.countplot(x="Survived",hue="Fare_labels",data=df_trainval)

plt.show()
df_trainval["Fare_labels*(Sex_labels+1)"]=df_trainval["Fare_labels"]*(df_trainval["Sex_labels"]+1)
X_trainval=df_trainval.drop(columns=["Survived","Sex","Fare","Embarked","Cabin","Name","Ticket","PassengerId","Age"])

Y_trainval=df_trainval["Survived"]

print("X_trainval shape:  ",X_trainval.shape)

print("Y_trainval shape:  ",Y_trainval.shape)
from sklearn.model_selection import train_test_split

X_train1,X_val1,Y_train1,Y_val1=train_test_split(X_trainval,Y_trainval,test_size=0.2,random_state=42)

print("X_train shape: ",X_train1.shape)

print("X_val shape: ",X_val1.shape)

print("Y_train shape: ",Y_train1.shape)

print("Y_val shape: ",Y_val1.shape)
from sklearn.linear_model import LogisticRegression

model1=LogisticRegression()

model1.fit(X_train1,Y_train1)

print("train set Accuracy: ",model1.score(X_train1,Y_train1))

print("test set Accuracy: ",model1.score(X_val1,Y_val1))
from sklearn.metrics import confusion_matrix

Y_val1_predict= model1.predict(X_val1)

print(confusion_matrix(Y_val1,Y_val1_predict))
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

standard_scaled_features=scaler.fit_transform(X_trainval)

X_trainval_2=pd.DataFrame(standard_scaled_features,index=X_trainval.index,columns=X_trainval.columns)

X_trainval_2.head()

X_train2,X_val2,Y_train2,Y_val2=train_test_split(X_trainval_2,Y_trainval,test_size=0.2,random_state=42)

print("X_train shape: ",X_train2.shape)

print("X_val shape: ",X_val2.shape)

print("Y_train shape: ",Y_train2.shape)

print("Y_val shape: ",Y_val2.shape)
model2=LogisticRegression()

model2.fit(X_train2,Y_train2)

print("train set Accuracy: ",model2.score(X_train2,Y_train2))

print("test set Accuracy: ",model1.score(X_val2,Y_val2))
Y_val2_predict= model2.predict(X_val2)

print(confusion_matrix(Y_val2,Y_val2_predict))
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

minmax_scaled_features=scaler.fit_transform(X_trainval)

X_trainval_3=pd.DataFrame(minmax_scaled_features,index=X_trainval.index,columns=X_trainval.columns)

X_trainval_3.head()

X_train3,X_val3,Y_train3,Y_val3=train_test_split(X_trainval_3,Y_trainval,test_size=0.2,random_state=42)

print("X_train shape: ",X_train3.shape)

print("X_val shape: ",X_val3.shape)

print("Y_train shape: ",Y_train3.shape)

print("Y_val shape: ",Y_val3.shape)
model3=LogisticRegression()

model3.fit(X_train3,Y_train3)

print("train set Accuracy: ",model3.score(X_train3,Y_train3))

print("test set Accuracy: ",model3.score(X_val3,Y_val3))
Y_val3_predict= model3.predict(X_val3)

print(confusion_matrix(Y_val3,Y_val3_predict))
#SVM On Normal Data

from sklearn.svm import LinearSVC

model4=LinearSVC()

model4.fit(X_train1,Y_train1)

print("train set Accuracy: ",model4.score(X_train1,Y_train1))

print("test set Accuracy: ",model4.score(X_val1,Y_val1))
Y_val4_predict= model4.predict(X_val1)

print(confusion_matrix(Y_val1,Y_val4_predict))
#SVM On StandardScaled  Data

model5=LinearSVC()

model5.fit(X_train2,Y_train2)

print("train set Accuracy: ",model5.score(X_train2,Y_train2))

print("test set Accuracy: ",model5.score(X_val2,Y_val2))
Y_val5_predict= model5.predict(X_val2)

print(confusion_matrix(Y_val2,Y_val5_predict))
#SVM On MinMaxScaled Data

model6=LinearSVC()

model6.fit(X_train3,Y_train3)

print("train set Accuracy: ",model6.score(X_train3,Y_train3))

print("test set Accuracy: ",model6.score(X_val3,Y_val3))
#Decision Tree On Normal Data

from sklearn.tree import DecisionTreeClassifier

model7=DecisionTreeClassifier(max_depth=9,min_samples_split=8,min_samples_leaf=2)

model7.fit(X_train1,Y_train1)

print("train set Accuracy: ",model7.score(X_train1,Y_train1))

print("test set Accuracy: ",model7.score(X_val1,Y_val1))
Y_val7_predict= model7.predict(X_val1)

print(confusion_matrix(Y_val1,Y_val7_predict))
#Decision Tree On StandardScaler Data

model8=DecisionTreeClassifier(max_depth=9,min_samples_split=8,min_samples_leaf=2)

model8.fit(X_train2,Y_train2)

print("train set Accuracy: ",model8.score(X_train2,Y_train2))

print("test set Accuracy: ",model8.score(X_val2,Y_val2))
Y_val8_predict= model8.predict(X_val2)

print(confusion_matrix(Y_val2,Y_val8_predict))
#Decision Tree On MinMax Scaled Data

model9=DecisionTreeClassifier(max_depth=9,min_samples_split=8,min_samples_leaf=2)

model9.fit(X_train3,Y_train3)

print("train set Accuracy: ",model9.score(X_train3,Y_train3))

print("test set Accuracy: ",model9.score(X_val3,Y_val3))
Y_val9_predict= model9.predict(X_val3)

print(confusion_matrix(Y_val3,Y_val9_predict))
#Random Forest On Normal Data

from sklearn.ensemble import RandomForestClassifier

model10=RandomForestClassifier(n_estimators=100,max_leaf_nodes=50)

model10.fit(X_train1,Y_train1)

print("train set Accuracy: ",model10.score(X_train1,Y_train1))

print("test set Accuracy: ",model10.score(X_val1,Y_val1))
#Random Forest On StandardScaler Data

model11=RandomForestClassifier(n_estimators=100,max_leaf_nodes=50)

model11.fit(X_train2,Y_train2)

print("train set Accuracy: ",model11.score(X_train2,Y_train2))

print("test set Accuracy: ",model11.score(X_val2,Y_val2))
#Random Forest On MinMax Scaled Data

model12=RandomForestClassifier(n_estimators=100,max_leaf_nodes=50)

model12.fit(X_train3,Y_train3)

print("train set Accuracy: ",model12.score(X_train3,Y_train3))

print("test set Accuracy: ",model12.score(X_val3,Y_val3))
import tensorflow as tf

from keras.layers import Dense, Activation,LeakyReLU

from keras.models import Sequential

from keras import backend as K

K.clear_session()

tf.reset_default_graph()



model13 = Sequential()  

model13.add(Dense(8,input_shape=(12,)))  

model13.add(Activation('sigmoid'))

model13.add(Dense(16))

model13.add(Activation('sigmoid'))

model13.add(Dense(32))

model13.add(Activation('sigmoid'))

model13.add(Dense(2))

model13.add(Activation('sigmoid'))

model13.add(Dense(1))

model13.add(Activation('sigmoid'))
model13.summary()
model13.compile(

    loss='mean_squared_error', 

    optimizer='adagrad',

    metrics=['accuracy']

)
model13.fit(

    X_train3, 

    Y_train3,

    batch_size=32, 

    epochs=100,

    validation_split=0.2

)
print("test set Accuracy: ",model13.evaluate(x=X_val3, y=Y_val3, batch_size=2, verbose=1))
df_test.head()
g1=df_test["PassengerId"]

df_test["Sex_labels"],Sex_test_uniques=pd.factorize(df_test["Sex"])

df_test.drop(columns=["Sex"],inplace=True)

df_test["Age"].fillna(Age_mean,inplace=True)

a1=pd.qcut(df_test["Age"],6)

df_test["Age_label"],Age_test_unique=pd.factorize(a1)

df_test.drop(columns=["Age"],inplace=True)

df_test.drop(columns=["Cabin","Name","Ticket","PassengerId"],inplace=True)

df_test["(Sex_labels+1)/Parch"]=(df_test["Sex_labels"]+1)/df_test["Pclass"]

df_test["Parch+SibSp+1"]=df_test["SibSp"]+df_test["Parch"]+1

df_test["Pclass/Parch"]=df_test["Pclass"]/df_test["Parch"]

df_test["Parch+SibSp+sex_label"]=df_test["SibSp"]+df_test["Parch"]+df_test["Sex_labels"]

df_test["Embarked_label"],Embarked_test_unique=pd.factorize(df_test["Embarked"])

df_test.drop(columns=["Embarked"],inplace=True)

a2=pd.qcut(df_test["Fare"],5)

df_test["Fare_labels"],Fare_uniques=pd.factorize(a2)

df_test.drop(columns=["Fare"],inplace=True)

df_test["Fare_labels*(Sex_labels+1)"]=df_test["Fare_labels"]*(df_test["Sex_labels"]+1)

df_test["Parch+SibSp+sex_label"]=df_test["SibSp"]+df_test["Parch"]+df_test["Sex_labels"]











df_test.head()
#Simple Logistic Regression

result_1=model1.predict(df_test)

Result_logistic_Regression=pd.DataFrame({'PassengerId':g1,'Survived':result_1})

Result_logistic_Regression.to_csv("Result_logistic_Regression_1.csv",index=False,header=True)
scaler=StandardScaler()

df_test_2=scaler.fit_transform(df_test)
#Simple Logistic Regression with standardscaler

result_2=model2.predict(df_test_2)

Result_logistic_Regression_2=pd.DataFrame({'PassengerId':g1,'Survived':result_2})

Result_logistic_Regression_2.to_csv("Result_logistic_Regression_2.csv",index=False,header=True)
scaler=MinMaxScaler()

df_test_3=scaler.fit_transform(df_test)
#Simple Logistic Regression with minmaxscaler

result_3=model3.predict(df_test_3)

Result_logistic_Regression_3=pd.DataFrame({'PassengerId':g1,'Survived':result_3})

Result_logistic_Regression_3.to_csv("Result_logistic_Regression_3.csv",index=False,header=True)
#SVM On Normal Data

result_4=model4.predict(df_test)

Result_logistic_Regression_4=pd.DataFrame({'PassengerId':g1,'Survived':result_4})

Result_logistic_Regression_4.to_csv("Result_SVM_1.csv",index=False,header=True)
#SVM On StandardScaled  Data

result_5=model5.predict(df_test_2)

Result_logistic_Regression_5=pd.DataFrame({'PassengerId':g1,'Survived':result_5})

Result_logistic_Regression_5.to_csv("Result_SVM_2.csv",index=False,header=True)
#SVM On MinMaxScaled Data

result_6=model6.predict(df_test_3)

Result_logistic_Regression_6=pd.DataFrame({'PassengerId':g1,'Survived':result_6})

Result_logistic_Regression_6.to_csv("Result_SVM_3.csv",index=False,header=True)
#Decision Tree On Normal Data

result_7=model7.predict(df_test)

Result_logistic_Regression_7=pd.DataFrame({'PassengerId':g1,'Survived':result_7})

Result_logistic_Regression_7.to_csv("Result_Decision_tree_1.csv",index=False,header=True)
#Decision Tree On StandardScaler Data

result_8=model8.predict(df_test_2)

Result_logistic_Regression_8=pd.DataFrame({'PassengerId':g1,'Survived':result_8})

Result_logistic_Regression_8.to_csv("Result_Decision_tree_2.csv",index=False,header=True)
#Decision Tree On MinMax Scaled Data

result_9=model9.predict(df_test_3)

Result_logistic_Regression_9=pd.DataFrame({'PassengerId':g1,'Survived':result_9})

Result_logistic_Regression_9.to_csv("Result_Decision_tree_3.csv",index=False,header=True)
#Random Forest On Normal Data

result_10=model10.predict(df_test)

Result_logistic_Regression_10=pd.DataFrame({'PassengerId':g1,'Survived':result_10})

Result_logistic_Regression_10.to_csv("Result_Random_Forest_1.csv",index=False,header=True)
#Random Forest On StandardScale Data

result_11=model11.predict(df_test_2)

Result_logistic_Regression_11=pd.DataFrame({'PassengerId':g1,'Survived':result_11})

Result_logistic_Regression_11.to_csv("Result_Random_Forest_2.csv",index=False,header=True)
#Random Forest On MinMax Scaled Data

result_12=model12.predict(df_test_3)

Result_logistic_Regression_12=pd.DataFrame({'PassengerId':g1,'Survived':result_12})

Result_logistic_Regression_12.to_csv("Result_Random_Forest_3.csv",index=False,header=True)
#Random Forest On MinMax Scaled Data

result_12=model12.predict(df_test_3)

Result_logistic_Regression_12=pd.DataFrame({'PassengerId':g1,'Survived':result_12})

Result_logistic_Regression_12.to_csv("Result_Random_Forest_3.csv",index=False,header=True)
#Neural Network On MinMax Scaled Data

result_13=(model13.predict(df_test_3)>0.5).astype(int)

result_13=result_13.squeeze()

Result_logistic_Regression_13=pd.DataFrame({'PassengerId':g1,'Survived':result_13})

Result_logistic_Regression_13.to_csv("Neural_Network_3.csv",index=False,header=True)