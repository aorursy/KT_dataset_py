# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head()
train.columns
train.head()
train['Embarked'].value_counts()
plt.subplots(figsize=(15,8))
sns.barplot(x='Embarked',
           y='Survived',
           data=train)

train.head()
train['Sex'].value_counts()
train["Sex"].isnull().value_counts()
train.Pclass.value_counts()
plt.subplots(figsize=(15,10))
plt.title("Passenger Class and Sex Distribution-Survived vs Non-Survived")
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train)
labels=['Upper','Middle','Lower']
val=[0,1,2]
plt.xticks(val,labels)

plt.subplots(figsize=(15,10))
sns.barplot(x='Pclass',
           y="Survived",
           data=train,
           linewidth=2)
plt.title("Passenger Class Distribution-Survied vs Non-Survived", fontsize=25)
plt.xlabel("Socio-Economic class", fontsize=15);
plt.ylabel('% of Passenger Survived', fontsize=15);
labels=['Upper','Middle','Lower']
val=[0,1,2]
plt.xticks(val,labels);
fig=plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.Pclass[train.Survived==0],
               color='gray',
               shade=True,
               label='not survived')
ax=sns.kdeplot(train.Pclass[train.Survived==1],
               color='green',
               shade=True,
               label='survived')
plt.xlabel('Passenger Class',fontsize=15)
plt.ylabel('Frequency of Passenger Survived',fontsize=15)
labels=['Upper','Middle',"Lower"]
plt.xticks(sorted(train.Pclass.unique()), labels)
fig=plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.Age[train.Survived==0],
               color='gray',
               shade=True,
               label='not survived')
ax=sns.kdeplot(train.Age[train.Survived==1],
               color='green',
               shade=True,
               label='survived')
plt.xlabel('Passenger Age',fontsize=15)
plt.ylabel('Frequency of Passenger Survived',fontsize=15)
#labels=['Upper','Middle',"Lower"]
#plt.xticks(sorted(train.Pclass.unique()), labels)
pal={1:'seagreen',0:'gray'}
g=sns.FacetGrid(train,size=5,col="Sex",row="Survived", margin_titles=True,hue="Survived",palette=pal)
g=g.map(plt.hist, "Age",edgecolor='white')
g.fig.suptitle("Survived by Sex and Age", size=25)
plt.subplots_adjust(top=0.90)
train['Age'].quantile(.25)
corr=train.corr()**2
corr.Survived.sort_values(ascending=False)
mask=np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
plt.subplots(figsize=(15,12))
sns.heatmap(train.corr(),annot=True,cmap='RdBu_r',square=.9)
train.head()
train=train.drop('PassengerId',axis=1)
train_survived=train.Survived
train=train.drop("Survived",axis=1)
train.info()
avg_not_survived=train[train["Survived"]==0]["Sex"]#.mean()
avg_not_survived
import scipy.stats as stats
stats.ttest_1samp(a=train[train['Survived']==1]['Sex'])
data.Sex.isnull().value_counts()
data.Sex=data["Sex"].map({'male':0,'female':1}).astype(int)
data.head()
data.SibSp.isnull().value_counts()
data.SibSp.value_counts()
data.Ticket.isnull().value_counts()
data.Ticket.str.extract('([A-Za-z])\.',expand=False).value_counts()
data.head()
testing.head()
testing['Embarked'].describe()
testing['Embarked'].value_counts()
median_age=data['Age'].median()
median_age_test=testing['Age'].median()
testing['Age'].fillna(median_age_test,inplace=True)
testing.Age.head()
testing.head()
data['Age'].fillna(median_age,inplace=True )
data['Age'].head()
testing['Age'].fillna(median_age,inplace=True)
testing['Age'].head()
data_inputs=data[['Pclass','Age','Sex']]
data_inputs.head()
testing_inputs=testing[['Pclass','Age','Sex']]
testing_inputs.head()
data.head()
data_inputs['Pclass'].replace("3rd", 3, inplace=True)
data_inputs['Pclass'].replace("2nd", 2, inplace=True)
data_inputs['Pclass'].replace("1st", 1, inplace=True)
data_inputs['Pclass'].head()
testing_inputs['Pclass'].replace("3rd", 3, inplace=True)
testing_inputs['Pclass'].replace("2nd", 2, inplace=True)
testing_inputs['Pclass'].replace("1st", 1, inplace=True)
testing_inputs['Pclass'].head()
data_inputs['Sex'].head()
testing_inputs['Sex'].head()
data_inputs['Sex'].replace("female",0,inplace=True)
data_inputs['Sex'].replace("male",1,inplace=True)
testing_inputs['Sex'].replace("female",0,inplace=True)
testing_inputs['Sex'].replace("male",1,inplace=True)
data_inputs['Sex'][:5]
data_inputs.head()
testing_inputs['Sex'][:5]
input_train, inputs_test, expected_output_train, expected_output_test=train_test_split(data_inputs, data.Survived, test_size=0.1, random_state=42)
import tensorflow as tf
from tensorflow import keras
data_inputs.loc[0].shape

model = keras.Sequential([
    keras.layers.Dense(input_dim=input_train.shape[1],units=128),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense( activation=tf.nn.softmax,units=2),
    #keras.layers.Dense(1,activation='tanh')
])

model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
input_train.values.shape

model.fit(input_train.values, expected_output_train,  epochs=250)
model.evaluate(inputs_test,expected_output_test)
print(" metrics values: {}%".format(ex[1]*100))

ps=model.predict(inputs_test)
expected_output_test
save=[]
for i in range(len(ps)):
    if ps[i][0]<0.5:
        save.append(1)
        print("Survived")
    else:
        save.append(0)
        print("died")
ps=model.predict(testing_inputs)
save=[]
for i in range(len(ps)):
    if ps[i][0]<0.5:
        save.append(1)
        print("Survived")
    else:
        save.append(0)
        print("died")
submission = pd.DataFrame({
        "PassengerId": testing["PassengerId"],
        "Survived": save
    })

submission
submission.to_csv("titanic.csv", index=False)
print(submission.shape)

submission_predictions = rf.predict(testing_inputs)

submission_predictions =model.predict(testing_inputs)
submission_predictions.shape
testing_inputs.shape
submission_predictions.shape
pd.read_csv("titanic.csv")
