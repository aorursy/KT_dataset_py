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
pd.set_option('display.max_columns',None)
train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.columns
%matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [10.0, 8.0]
from sklearn.model_selection import train_test_split
training,valid=train_test_split(train,random_state=0,test_size=0.2)
training.head()
import warnings
warnings.filterwarnings(action='ignore')
age=training.Age
age.fillna(age.mean(),inplace=True)
age=age.values
sex=training.Sex
sex=pd.get_dummies(sex)
sex=sex.values[:,0]
fare=training.Fare
fare=fare.fillna(fare.mean())
fare=fare.values
embarked=training.Embarked
embarked=embarked.fillna('S')
embarked=pd.get_dummies(embarked)
embarked=embarked.values
sib=training.SibSp.values
parch=training.Parch.values
pclass=training.Pclass.values
survive=training.Survived.values



sex[:5]
no_of_male=list(sex).count(0)
no_of_female=list(sex).count(1)
male_survive=len([survive[i] for i in range(len(survive)) if sex[i]==0 and survive[i]==1])
female_survive=len([survive[i] for i in range(len(survive)) if sex[i]==1 and survive[i]==1])
print(no_of_male,no_of_female,male_survive,female_survive,len(survive),(male_survive)+(female_survive))
male_survive_rate=male_survive/no_of_male
female_survive_rate=female_survive/no_of_female
female_survive_rate
male_survive_rate
plt.figure(figsize=(10,8))
plt.bar([0],male_survive_rate,0.5,label='MALE')
plt.bar([1],female_survive_rate,0.5,label='FEMALE')
plt.xlabel('sex',fontsize=18)
plt.ylabel('survive rate',fontsize=18)
plt.title('sex vs survive rate')
plt.xticks([0,1],labels=['MALE','FEMALE'])
plt.ylim([0,1])
plt.legend(loc='best')
plt.show()
# def plotdata(x,y,title=None,xlabel=None,ylabel=None):
#     alist=[0]
#     survive_list=[0]
#     newls=[]
#     maximum=max(x)
#     for j in range(1,maximum):
#         alist[0]=0
#         survive_list[0]=0
#         for num,i in enumerate(x):
#             if(i<=j):
#                 alist[0]+=1
#                 if(survive[num]==1):
#                     survive_list[0]+=1          
#         newls.append(survive_list[0]/alist[0])
#     plt.figure()    
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.plot([i for i in range(1,81)],newls)    
#     plt.show()
agelist=[0]
survive_agelist=[0]
newls=[]
for j in range(1,81):
    agelist[0]=0
    survive_agelist[0]=0
    for num,i in enumerate(age):
        if(i<=j):
            agelist[0]+=1
            if(survive[num]==1):
                survive_agelist[0]+=1          
    newls.append(survive_agelist[0]/agelist[0])
plt.figure()    
plt.title('age vs survive rate')
plt.xlabel('age')
plt.ylabel('survive rate')
plt.plot([i for i in range(1,81)],newls)    
plt.show()
agelist=[0]
survive_agelist=[0]
newls=[]
for j in range(1,81):
    agelist[0]=0
    survive_agelist[0]=0
    for num,i in enumerate(age):
        if(sex[num]==0):
            continue
        if(i<=j):
            agelist[0]+=1
            if(survive[num]==1):
                survive_agelist[0]+=1          
    newls.append(survive_agelist[0]/agelist[0])
plt.figure()    
plt.title('age vs Female survive rate')
plt.xlabel('age')
plt.ylabel('Female survive rate')
plt.plot([i for i in range(1,81)],newls)    
plt.show()
agelist=[0]
survive_agelist=[0]
newls=[]
for j in range(1,81):
    agelist[0]=0
    survive_agelist[0]=0
    for num,i in enumerate(age):
        if(sex[num]==1):
            continue
        if(i<=j):
            agelist[0]+=1
            if(survive[num]==1):
                survive_agelist[0]+=1          
    newls.append(survive_agelist[0]/agelist[0])
plt.figure()    
plt.title('age vs Male survive rate')
plt.xlabel('age')
plt.ylabel('Male survive rate')
plt.plot([i for i in range(1,81)],newls)    
plt.show()
agelist=[0]
survive_agelist=[0]
newls=[]
for j in range(1,81):
    agelist[0]=0
    survive_agelist[0]=0
    for num,i in enumerate(fare):
        if(i<=j):
            agelist[0]+=1
            if(survive[num]==1):
                survive_agelist[0]+=1          
    if(agelist[0]==0):
        newls.append(0)
        continue
    newls.append(survive_agelist[0]/agelist[0])
    
plt.figure()    
plt.title('fare vs survive rate')
plt.xlabel('fare')
plt.ylabel('survive rate')
plt.plot([i for i in range(1,81)],newls)    
plt.show()
agelist=[0]
survive_agelist=[0]
newls=[]
for j in range(1,81):
    agelist[0]=0
    survive_agelist[0]=0
    for num,i in enumerate(fare):
        if(sex[num]==0):
            continue
        if(i<=j):
            agelist[0]+=1
            if(survive[num]==1):
                survive_agelist[0]+=1
    if(agelist[0]==0):
        newls.append(0)
        continue
    newls.append(survive_agelist[0]/agelist[0])
plt.figure()    
plt.title('fare vs Female survive rate')
plt.xlabel('fare')
plt.ylabel('Female survive rate')
plt.plot([i for i in range(1,81)],newls)    
plt.show()
agelist=[0]
survive_agelist=[0]
newls=[]
for j in range(1,81):
    agelist[0]=0
    survive_agelist[0]=0
    for num,i in enumerate(fare):
        if(sex[num]==1):
            continue
        if(i<=j):
            agelist[0]+=1
            if(survive[num]==1):
                survive_agelist[0]+=1 
    if(agelist[0]==0):
        newls.append(0)
        continue            
    newls.append(survive_agelist[0]/agelist[0])
plt.figure()    
plt.title('fare vs Male survive rate')
plt.xlabel('fare')
plt.ylabel('Male survive rate')
plt.plot([i for i in range(1,81)],newls)    
plt.show()
training.columns
X_train,X_valid,y_train,y_valid=train_test_split(training[['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']],training[['Survived']],test_size=0.2,random_state=0)

X_train.columns
X_train.Embarked.fillna('S',inplace=True)
X_valid.Embarked.fillna('S',inplace=True)
X_train=pd.get_dummies(X_train)
# X_train=X_train.drop(['Age'],axis=1)
X_valid=pd.get_dummies(X_valid)
# X_valid=X_valid.drop(['Age'],axis=1)
X_valid.head()
X_valid.shape
X_train.shape
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_valid=scaler.transform(X_valid)
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# X_train=scaler.fit_transform(X_train)
# X_valid=scaler.transform(X_valid)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train_pca=pca.fit_transform(X_train)
X_valid_pca=pca.transform(X_valid)

plt.figure()
plt.scatter(X_train_pca[:,0],X_train_pca[:,1],c=y_train.values.reshape(-1,),s=50)
plt.show()
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_valid,y_valid)
!pip install keras_utils
!pip install keras

import tensorflow as tf
print(tf.__version__)
import keras_utils
# from keras_utils import reset_tf_session
from keras.layers import Dense,Activation
from keras.models import Sequential
model=Sequential()
model.add(Dense(10,input_shape=(10,)))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )
    
y_t=y_train.values
y_2t=y_t==0
y_2t=y_2t.astype(int)
y_2t[:5]
y_train_oh=np.append(y_train.values,y_2t,axis=1)
print(y_train_oh.shape)
print(y_train_oh[:5])
X_train.shape
y_v=y_valid.values
y_2v=y_v==0
y_2v=y_2v.astype(int)
y_valid_oh=np.append(y_valid.values,y_2v,axis=1)
vx,tx,vy,ty=train_test_split(X_valid,y_valid_oh,test_size=0.5,random_state=0)
model.fit(X_train,y_train_oh,epochs=100,validation_data=(vx,vy),verbose=0)
test=pd.read_csv('/kaggle/input/titanic/test.csv')
X_test=test[['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']]
X_test.Fare.fillna(32,inplace=True)
X_test.Age.fillna(29,inplace=True)
X_test=pd.get_dummies(X_test)
X_test.head()
X_test=scaler.transform(X_test)
model.predict(tx)[:4]
ty[:5]
def changeto(pred):
    arr=[]
    for i in pred[:,1]:
        if(i>0.5):
            arr.append(1)
        else:
            arr.append(0)
    return np.array(arr)        
pred=changeto(model.predict(tx))
pred[:10]
ty[:10,1]
from sklearn.metrics import confusion_matrix
confusion_matrix(ty[:,1],pred)
from sklearn.metrics import roc_auc_score
roc_auc_score(ty[:,1],model.predict(tx)[:,1])
lr.predict_proba()
roc_auc_score(ty[:,0],lr.predict_proba(tx)[:,1])
from sklearn.metrics import accuracy_score
accuracy_score(ty[:,1],pred)
accuracy_score(ty[:,0],lr.predict(tx))
train_pred=changeto(model.predict(X_train))
accuracy_score(y_train_oh[:,1],train_pred)
test_pred=changeto(model.predict(X_test))
result=pd.DataFrame(zip(test.PassengerId.values,test_pred),columns=['PassengerId','Survived'])
    
result.to_csv('submission.csv',index=False)