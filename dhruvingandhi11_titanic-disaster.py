import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df1 = pd.read_csv('/kaggle/input/titanic/test.csv')
df.head()
df1.head()
dfy = df.Survived



df.drop(['PassengerId','Survived','Name','Ticket','Cabin','Fare'],axis=1,inplace=True)

df1.drop(['PassengerId','Name','Ticket','Cabin','Fare'],axis=1,inplace=True)
dfy
df.head()
df1.head()
print('shape of training data: ',df.shape)

print('shape of testing data: ',df1.shape )
df.info()
df1.info()
df.describe()
df1.describe()
df.columns
column_df = df.columns



for x in column_df:

  print('unique value information: ',x)

  # print(df[x].unique())

  print('number of unique values: ',df[x].unique().shape[0])

  print('number of null (True if value is NaN) : \n',df[x].isnull().value_counts())

  print('\n-----------------------------------------------------------------\n')
columns_df1 = df1.columns



for x in columns_df1:

  print('unique value information: ',x)

  #print(df1[x].unique())

  print('number of unique values: ',df1[x].unique().shape[0])

  print('number of null (True if value is NaN): \n',df1[x].isnull().value_counts())

  print('\n-------------------------------------------------------------\n')
df.Age.isnull().value_counts()
df.Age.fillna(df.Age.mean(),inplace=True)
df.Age.isnull().value_counts()
df1.Age.isnull().value_counts()
df1.Age.fillna(df1.Age.mean(), inplace = True)
df1.Age.isnull().value_counts()
df.Embarked.isnull().value_counts()
df.Embarked.value_counts()
df.Embarked.fillna('S',inplace = True)
df.Embarked.isnull().value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()



le.fit(df.Sex)

Sex_labeled = le.transform(df.Sex)

df['Sex_labeled'] = Sex_labeled

df.drop(['Sex'],axis = 1, inplace = True)



le.fit(df.Embarked)

Embarked_labeled = le.transform(df.Embarked)

df['Embarked_labeled'] = Embarked_labeled

df.drop(['Embarked'],axis =1, inplace = True)



le.fit(df1.Sex)

Sex_labeled= le.transform(df1.Sex)

df1['Sex_labeled'] = Sex_labeled

df1.drop(['Sex'], axis =1, inplace =True)



le.fit(df1.Embarked)

Embarked_labeled = le.transform(df1.Embarked)

df1['Embarked_labeled'] = Embarked_labeled

df1.drop(['Embarked'], axis =1, inplace =True)
df.head()
df1.head()
df.info()
df1.info()
x = np.array(df)

y = np.array(dfy)



x1 = np.array(df1)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



x = scaler.fit_transform(x)

x1 = scaler.fit_transform(x1)
from sklearn.model_selection import train_test_split as tts

x_train, x_test, y_train, y_test = tts(df, dfy, test_size =0.2)
print('Shape of Train and Test Set: ')

print('x_train: ', x_train.shape)

print('x_test: ', x_test.shape)

print('y_train: ', y_train.shape)

print('y_test: ',y_test.shape)
from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.metrics import roc_auc_score
model_rfc = RFC(max_depth=10, n_estimators =100, random_state = 2)

model_rfc.fit(x_train,y_train)
model_rfc.score(x_test,y_test)
y_pred = model_rfc.predict(df1)

#roc_auc_score(y_test,y_pred)
from sklearn.svm import SVC
model_svm = SVC(C=10)

model_svm.fit(df,dfy)
model_svm.score(df,dfy)
y_pred = model_svm.predict(df1)



#roc_auc_score(y_test,y_pred)
# create Model 



model = keras.Sequential()

model.add(layers.Dense(12 , activation="relu", input_shape=(6,)))

model.add(layers.Dense(6, activation="relu")),

model.add(layers.Dense(1,activation="sigmoid"))



model.summary()
# fit and compile model



model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.AUC()])



history= model.fit(df,dfy,batch_size=23,epochs=100,validation_data=(x_test,y_test))



plt.figure()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])
y_pred = model.predict_classes(df1)



#roc_auc_score(y_test,y_pred)
Survived = np.squeeze(y_pred)
PassengerId = np.arange(892,1310)
ans = pd.DataFrame(list(zip(PassengerId,Survived)),columns=['PassengerId','Survived'])

ans.head()
ans.shape
ans.to_csv("final_ans.csv",index=False)