

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense
df_main = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

df_main.describe(include='all')

print(df_main.columns)
df_main.head()
df_main.tail()
df_main.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.title('1: Survived  0:not Survived')
plt.show()

fig=plt.figure(figsize=(18,6))
plt.subplot2grid((3,4),(0,0))
df_main.Survived[df_main.Sex=='male'].value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.title("men Survived")

plt.subplot2grid((3,4),(0,1))
df_main.Survived[df_main.Sex =='female'].value_counts(normalize=True).plot(kind='bar',alpha=0.5,color='red')
plt.title("Women Survived")
plt.show()


fig=plt.figure(figsize=(18,6))
plt.subplot2grid((4,4),(0,0))
df_main.Survived[(df_main.Sex=='male') & (df_main.Pclass==1)].value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.title("poor men Survived")

plt.subplot2grid((4,4),(0,1))
df_main.Survived[(df_main.Sex =='male') & (df_main.Pclass==3)].value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.title("rich men Survived")


plt.subplot2grid((4,4),(2,0))
df_main.Survived[(df_main.Sex=='female') & (df_main.Pclass==1)].value_counts(normalize=True).plot(kind='bar',alpha=0.8,color='red')
plt.title("poor women Survived")

plt.subplot2grid((4,4),(2,1))
df_main.Survived[(df_main.Sex =='female') & (df_main.Pclass==3)].value_counts(normalize=True).plot(kind='bar',alpha=0.8,color='red')
plt.title("rich women Survived")
plt.show()
y=df_main[['Survived']]
df_main=df_main.drop(['PassengerId', 'Name','Ticket','Cabin','Survived'], axis=1)
x=df_main.values
imp_mean = SimpleImputer( strategy='most_frequent')
imp_mean.fit(x)
x = imp_mean.transform(x)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,1]=le.fit_transform(x[:,1])
le2=LabelEncoder()
x[:,-1]=le2.fit_transform(x[:,-1])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer([('one_hot_encoder',OneHotEncoder(categories='auto'),[1])],remainder='passthrough')
x=ct.fit_transform(x)
x=x[:,:-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=4,metric='minkowski')
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print('KNN')
cm=confusion_matrix(y_test, y_pred)
print(cm)
acc_gaussian = round(accuracy_score(y_pred, y_test) * 100, 2)
print("accuracy score is ",acc_gaussian)
# Decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print('Decision Tree')
cm2=confusion_matrix(y_test,y_pred)
print(cm2)
acc_gaussian = round(accuracy_score(y_pred, y_test) * 100, 2)
print("accuracy score is ",acc_gaussian)
# gaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
cm2=confusion_matrix(y_test,y_pred)
print(cm2)
acc_gaussian = round(accuracy_score(y_pred, y_test) * 100, 2)
print("accuracy score is ",acc_gaussian)
#Logistic Regression
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
acc_lr=round(accuracy_score(y_pred,y_test)*100,2)
print("accuracy score is ",acc_lr)
#support vector machine
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
acc_svc=round(accuracy_score(y_pred,y_test)*100,2)
print("accuarcy score is",acc_svc)
classifier=Sequential()

classifier.add(Dense(6,activation='relu',kernel_initializer="uniform",input_dim=7))

classifier.add(Dense(6,activation='relu',kernel_initializer="uniform"))

classifier.add(Dense(1,activation='sigmoid',kernel_initializer="uniform"))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x_train,y_train,epochs=100)
y_pred=classifier.predict(x_test)

y_pred=(y_pred>0.5)
cm=confusion_matrix(y_test,y_pred)
print(cm)
acc_Neural=round(accuracy_score(y_pred,y_test)*100,2)
print("accuarcy score is",acc_Neural)
