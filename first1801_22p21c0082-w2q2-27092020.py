# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_validate

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/titanic/train.csv')

data
gendersubmiss = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

gendersubmiss #An example of what a submission file should look like.
data.isnull().sum() 
testdata = pd.read_csv('/kaggle/input/titanic/test.csv')

testdata 
data.describe()
f,ax=plt.subplots(1,2,figsize=(18,8))

data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=data,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
f,ax=plt.subplots(1,3,figsize=(18,8))

data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

sns.countplot('Embarked',hue='Survived',data=data,ax=ax[2])

ax[2].set_title('Embarked:Survived vs Dead')

plt.show()
sns.catplot(x="Embarked", y="Survived", kind="bar", data=data)
sns.catplot(x="Embarked", kind="count", palette="ch:.25", data=data)
pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient()
data.groupby(['Pclass','Embarked'])['Survived'].sum().plot(kind='bar')
(data.groupby(['Pclass','Embarked'])['Survived'].sum()/data.groupby(['Pclass','Embarked'])['Survived'].count()).plot(kind='bar')
data
Train = data.drop(['Name','Cabin','Ticket'],axis=1)

testdata = testdata.drop(['Name','Cabin','Ticket'],axis=1)

Train
Train['Age'] = Train['Age'].fillna(0)

testdata['Age'] = testdata['Age'].fillna(0)

Train.isnull().sum()
Train[Train['Embarked'].isnull()]
Train['Embarked'] =  Train['Embarked'].fillna('C')

Train.isnull().sum()
Train
Train['Age_range'] = 0

Train
Train['Age'].max()
Train.loc[Train['Age']<20,'Age_range']=0

Train.loc[(Train['Age']>=20) & (Train['Age']<40),'Age_range']=1

Train.loc[(Train['Age']>=40) & (Train['Age']<60),'Age_range']=2

Train.loc[(Train['Age']>=60) & (Train['Age']<=80),'Age_range']=3



testdata.loc[testdata['Age']<20,'Age_range']=0

testdata.loc[(testdata['Age']>=20) & (testdata['Age']<40),'Age_range']=1

testdata.loc[(testdata['Age']>=40) & (testdata['Age']<60),'Age_range']=2

testdata.loc[(testdata['Age']>=60) & (testdata['Age']<=80),'Age_range']=3

Train
testdata
Train['Age_range'].value_counts()
testdata
Train= pd.get_dummies(data = Train, dummy_na=True, prefix= ["Pclass","Sex","Embarked"] ,columns=["Pclass","Sex","Embarked"])

testdata= pd.get_dummies(data = testdata, dummy_na=True, prefix= ["Pclass","Sex","Embarked"] ,columns=["Pclass","Sex","Embarked"])

Train
Train=Train[Train.columns.difference(["Age","PassengerId"])]

testdata=testdata[testdata.columns.difference(["Age","PassengerId"])]
Train["Fare"] = Train["Fare"] /Train["Fare"].max()

testdata["Fare"] = testdata["Fare"]/Train["Fare"].max()
X_train = Train.drop('Survived',axis=1)

Y_train = Train["Survived"]

Xtest = testdata
X_train
X_train=X_train.to_numpy()

Y_train =Y_train.to_numpy()

Xtest =Xtest.to_numpy()
X_train.shape
from sklearn.model_selection import cross_val_predict
def cal_precision_recall_f1(conf_m):

    tp = conf_m[0][0]

    fp = conf_m[0][1]

    fn = conf_m[1][0]

    tn = conf_m[1][1]

    pre = tp/(tp+fp)

    rec = tp/(tp+fn)

    f1 = 2*((pre*rec)/(pre+rec))

    print("precision = ",pre,"\nrecal =",rec,"\nf1 = ",f1)

    return f1
from sklearn.metrics import accuracy_score,classification_report
def Kfold_fit(model,X_train,Y_train,Fold=5):

    f1sum = 0

    i = 0

    model = model

    report = []

    kf = KFold(n_splits=Fold)

    for train_idx, val_idx in kf.split(X_train,Y_train):

        i+=1

        X_t = X_train[train_idx]

        Y_t = Y_train[train_idx]

        X_val = X_train[val_idx]

        Y_val = Y_train[val_idx]

        model.fit(X_t,Y_t)

        

        Predict = model.predict(X_val).reshape((X_val.shape[0],1))

        Predict = np.round(Predict)

        Y_val =Y_val.reshape((Y_val.shape[0],1))

        tp =np.sum((np.round(Predict)==Y_val)&(Y_val==1))

        fp =np.sum((np.round(Predict)!=Y_val)&(Y_val==0))

        tn =np.sum((np.round(Predict)==Y_val)&(Y_val==0))

        fn =np.sum((np.round(Predict)!=Y_val)&(Y_val==1))

        

        pre = tp/(tp+fp)

        rec = tp/(tp+fn)

        

        f1 = 2*((pre*rec)/(pre+rec))

        mse= np.mean((Predict-Y_val)**2)

        

        print(classification_report(Y_val, Predict))

        #print("Precision = ",pre,"\nRecall =",rec,"\nF-measure = ",f1,"\nloss = ",mse)

        f1sum += f1 

        

    #print("Average F-measure =",f1sum/i)

    
from sklearn import tree

TreeCLF = tree.DecisionTreeClassifier()

Kfold_fit(TreeCLF,X_train,Y_train,Fold=5)
from sklearn.naive_bayes import GaussianNB

NBCLF = GaussianNB()

Kfold_fit(NBCLF,X_train,Y_train,Fold=5)
import tensorflow as tf

d_in = (X_train.shape[1],)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(20, input_shape=d_in, 

                                activation=tf.keras.layers.PReLU()

))

model.add(tf.keras.layers.Dense(10,

                                activation=tf.keras.layers.PReLU()))

model.add(tf.keras.layers.Dense(5,

                                activation=tf.keras.layers.PReLU()))

model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(loss=tf.keras.losses.MeanSquaredError(),

                        optimizer=tf.keras.optimizers.SGD(learning_rate=0.09))

model.summary()
Kfold_fit(model,X_train,Y_train,Fold=5)