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
%matplotlib inline 
import numpy as np 
import scipy as sp 
import matplotlib as mpl
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import pandas as pd 

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')
from pandas.plotting import scatter_matrix

import sys
import os
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

import keras
from keras.models import Sequential
from keras.layers import Dense



test = pd.read_csv('../input/titanic/test.csv', header = 0, dtype={'Age': np.float64})
train  = pd.read_csv('../input/titanic/train.csv' , header = 0, dtype={'Age': np.float64})
titanic_data = [train, test]
train.info()
test.info()
train.tail()
#3.1 check missing and impact on data train 
#3.1.1 check sex and survived
print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
#3.1.2 check pclass and survived
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
#3.1.3 embarked and survived
for dataset in titanic_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
train["Survived"].value_counts().plot(kind="bar")
train["Survived"].value_counts()
train["Age"].hist(width=5)
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(train.corr(), annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
titanic_data=pd.concat([train,test])
titanic_data.shape

def survival_bar(variable):
    Died=train[train["Survived"]==0][variable].value_counts()/len(train["Survived"]==0)
    Survived=train[train["Survived"]==1][variable].value_counts()/len(train["Survived"]==1)
    data=pd.DataFrame([Died,Survived])
    data.index=["Not survived","Survived"]
    data.plot(kind="bar",stacked=True,title="%")
    return data.head()
survival_bar("Sex")
#4.1 mapping and encoding
sex_map={"male":1,"female":0}
train["Sex"]=train["Sex"].map(sex_map)
test["Sex"]=test["Sex"].map(sex_map)
survival_bar("Sex")

#train.columns.to_series().groupby(train.dtypes).groups
train.isna().sum()
#function for insert age
def insert_age_na(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(insert_age_na,axis=1)
train['Age'].isna().sum()
train['NameLength'] = train['Name'].apply(len)
train.head()
train.columns
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)
train['Embarked'].isna().sum()
train['FamilyNo'] = train['SibSp'] + train['Parch'] + 1
train['FamilyNo'].hist(bins=20)
def IsTravelAlone(col):
    if col == 1:
        return 1
    else:
        return 0
train['IsAlone'] = train['FamilyNo'].apply(IsTravelAlone)
sns.countplot(data=train,x=train['IsAlone'])
train.head()
#Drop column not importance
cols_drop = ['PassengerId','Name','Ticket','Cabin']
train.drop(cols_drop, axis=1, inplace = True)

train.head()
categorical_feature = []

for i in range(train.shape[1]):
   
    if train[train.columns[i]].dtype == 'object':
        categorical_feature.append(train.columns[i])
        

categorical_feature
train = pd.get_dummies(data=train,columns=categorical_feature,drop_first=True) 
train.head()
fig = plt.figure(figsize=(15,10))
sns.heatmap(train.corr(),annot=True,cmap="YlGnBu",linewidths=0.2)
plt.show()
#train test split dataset
dfX = train.drop('Survived',axis=1)
dfY = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(dfX, dfY, test_size=0.30, 
                                                    random_state=0)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
#Normalization
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test) 
train.corr().loc['Survived']
acc_score = []

def build_train_predict(clf,X_train,y_train,X_test,strAlg,acc_score):
    model = clf
    
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    plot_score(y_test,pred,strAlg,acc_score)
    return clf,pred
def plot_score(y_test,y_pred,strAlg,lstScore):
  
    lstScore.append([strAlg,accuracy_score(y_test, y_pred)])
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True,cmap="YlGnBu")
    plt.title('Model: {0} \nAccuracy:{1:.3f}'.format(strAlg,accuracy_score(y_test, y_pred)))
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()
def cross_val_MinMax(clf,X_train,y_train,fold):
    scores = cross_val_score(clf,X_train,y_train,cv=fold)
    print('Accuracy Min: {} \nAccuracy Max: {}'.format(scores.min(),scores.max()))
#2.4.1 Decision Tree

model_dt,pred_dt = build_train_predict(DecisionTreeClassifier(),
                                       X_train,y_train,X_test,
                                       'DecisionTreeClassifier',acc_score)
#2.6 ให้แบ่งชุดข้อมูลเป็น 5-fold cross validation

cross_val_MinMax(DecisionTreeClassifier(),X_train,y_train,5)
#ให้แสดงผลลัพท์การจำแนกในรูปแบบของ
#2.7.1 Recall ของแต่ละ class
#2.7.2 Precision ของแต่ละ class
#2.7.3 F-Measure ของแต่ละ class
#2.7.4 Average F-Measure ของทั้งชุดข้อมูล


print(classification_report(y_test,pred_dt))
#2.4.2 Naïve Bayes
model_gnb,pred_gnb = build_train_predict(GaussianNB(),
                                       X_train,y_train,X_test,
                                       'GaussianNB',acc_score)



#2.6 ให้แบ่งชุดข้อมูลเป็น 5-fold cross validation
cross_val_MinMax(GaussianNB(),X_train,y_train,5)
#ให้แสดงผลลัพท์การจำแนกในรูปแบบของ
#2.7.1 Recall ของแต่ละ class
#2.7.2 Precision ของแต่ละ class
#2.7.3 F-Measure ของแต่ละ class
#2.7.4 Average F-Measure ของทั้งชุดข้อมูล


print(classification_report(y_test,pred_gnb))
#2.4.3 Neural Network
dims = X_train.shape[1]
h_dims = int((dims+1)/2)
dims,h_dims

model_ann = Sequential() #initialize

model_ann.add(Dense(units=h_dims,kernel_initializer='uniform',activation='relu',input_dim=dims))

model_ann.add(Dense(units=h_dims,kernel_initializer='uniform',activation='relu'))
model_ann.add(Dense(units=h_dims,kernel_initializer='uniform',activation='tanh'))
model_ann.add(Dense(units=h_dims,kernel_initializer='uniform',activation='relu'))
model_ann.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

model_ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model_ann.fit(X_train,y_train,batch_size=32,epochs=500,verbose=0)


pred_ann = model_ann.predict(X_test)
pred_ann = pred_ann > 0.6
plot_score(y_test,pred_ann,'ANN',acc_score)





#ให้แสดงผลลัพท์การจำแนกในรูปแบบของ
#2.7.1 Recall ของแต่ละ class
#2.7.2 Precision ของแต่ละ class
#2.7.3 F-Measure ของแต่ละ class
#2.7.4 Average F-Measure ของทั้งชุดข้อมูล


print(classification_report(y_test,pred_ann))