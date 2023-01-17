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
#학습모델

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import VotingClassifier



#전처리 및 하이퍼파라미터

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



#결과

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



#시각화



import matplotlib.pylab as plt

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
train.info()
train.describe(include='all')
train.shape
train = train.drop_duplicates(keep=False) # 중복행 제거
train.isnull().sum() #결측치 확인
train = train.fillna(train.mean()).dropna() # 나이에 대한 결측치를 채워넣고 탑승항구와 객실번호가 없는 것은 삭제
train['Age'] = train['Age'].astype(int) # 소수점 제거

train['Age'] = train['Age'].astype('float64')
train.isnull().sum(), train.info()
X = train.drop(['PassengerId','Survived','Name','Ticket','Cabin'],axis=1)

y = train['Survived']
X
# X_ols = pd.get_dummies(X,columns =['Sex','Embarked','Pclass','SibSp','Parch'],drop_first=True)

X = pd.get_dummies(X,columns =['Sex','Embarked','Pclass']) 
scaler = StandardScaler() # 스케일링

x = scaler.fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
lr = LogisticRegression()

lr.fit(x_train,y_train)

pred = lr.predict(x_test)

accuracy_score(y_test,pred)
print(classification_report(y_test,pred))
lr = LogisticRegression()

dt = DecisionTreeClassifier()

rf = RandomForestClassifier()

nb = GaussianNB()



eclf_h =VotingClassifier(estimators = [('lr',lr),('dt',dt),('rf',rf),('nb',nb)],voting='hard')

eclf_s =VotingClassifier(estimators = [('lr',lr),('dt',dt),('rf',rf),('nb',nb)],voting='soft')

models = [lr,dt,rf,nb,eclf_h,eclf_s]
for model in models:

  model.fit(x_train,y_train)

  predictions = model.predict(x_test)

  score = model.score(x_test,y_test)

  print(score)
X.columns
test.columns
test_id = test['PassengerId']
test = test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

test = pd.get_dummies(test,columns =['Sex','Embarked','Pclass']) 

test
X.shape,test.shape
test = test.dropna()

test
test =scaler.fit_transform(test)

test
eclf_result = eclf_s.predict(test)
import tensorflow as tf

from tensorflow.keras import layers
np_test = np.array(test)
inputs = tf.keras.Input(x[1].shape)

ly1=tf.keras.layers.Dense(12,activation=tf.nn.relu)(inputs)

ly2=tf.keras.layers.Dense(8,activation=tf.nn.relu)(ly1)

ly3=tf.keras.layers.Dense(16,activation=tf.nn.relu)(ly2)

ly4=tf.keras.layers.Dense(128,activation=tf.nn.relu)(ly3)

ly5=tf.keras.layers.Dense(64,activation=tf.nn.relu)(ly4)

ly6=tf.keras.layers.Dense(32,activation=tf.nn.relu)(ly5)

outputs =tf.keras.layers.Dense(1,activation=tf.nn.relu)(ly6)



model = tf.keras.Model(inputs=inputs,outputs=outputs)

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy','BinaryCrossentropy'])

hist = model.fit(x_train,y_train, epochs = 200, batch_size=32, validation_split=0.2)
model.evaluate(x_test,y_test)
hist.history.keys()
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])