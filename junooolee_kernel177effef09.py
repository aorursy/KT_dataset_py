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



#tensorflow

import tensorflow as tf

from tensorflow.keras import layers
cancer = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
cancer
cancer.info()
cancer = cancer.drop(['Unnamed: 32'],axis=1)
cancer.isnull().sum()
X = cancer.drop(['id','diagnosis'],axis=1)

y = cancer['diagnosis']
X.head()
y.head()
#M :1 B : 0

y.loc[y=='M'] = 1

y.loc[y=='B'] = 0

y
y =y.astype(int)
#newline

np.bincount(y)
X.describe()
X = np.array(X)
plt.boxplot(X[:,])

plt.show()
scaler = StandardScaler() # 스케일링

x = scaler.fit_transform(X)
x_train_all,x_test,y_train_all,y_test = train_test_split(x,y,test_size=0.3)

x_train,x_val,y_train,y_val = train_test_split(x_train_all,y_train_all,test_size=0.3) # newline
lr = LogisticRegression()

dt = DecisionTreeClassifier()

rf = RandomForestClassifier()

nb = GaussianNB()



eclf_h =VotingClassifier(estimators = [('lr',lr),('dt',dt),('rf',rf),('nb',nb)],voting='hard')

eclf_s =VotingClassifier(estimators = [('lr',lr),('dt',dt),('rf',rf),('nb',nb)],voting='soft')

models = [lr,dt,rf,nb,eclf_h,eclf_s]
y_train
for model in models:

  model.fit(x_train,y_train)

  predictions = model.predict(x_test)

  score = model.score(x_test,y_test)

  print(classification_report(y_test,predictions),'\n')

# lr,dt,rf,nb,eclf_h,eclf_s
X_np = np.array(X)

X_np
model = tf.keras.Sequential([

    layers.Input(shape=X_np[1].shape),

    layers.Dense(12,activation='sigmoid'),

    layers.Dense(8,activation='relu'),

    layers.Dense(16,activation='sigmoid'),

    layers.Dense(128,activation='relu'),

    layers.Dense(64,activation='sigmoid'),

    layers.Dense(32,activation='relu'),

    layers.Dense(1,activation='sigmoid')    

])



model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy','BinaryCrossentropy'])

hist = model.fit(x_train,y_train, epochs = 300, batch_size=32, validation_split=0.2, validation_data=(x_val,y_val))
model.evaluate(x_test,y_test)
hist.history.keys()
plt.plot(hist.history['loss'],'x--',label='loss')

plt.plot(hist.history['val_loss'],'x--',label='val_loss')

plt.legend()
plt.plot(hist.history['accuracy'],'x--',label='accuracy')

plt.plot(hist.history['val_accuracy'],'x--',label='val_accuracy')

plt.legend()
x_test
#newline

x_data = x_test[[range(1,20)]]
x_data.shape
pred =model.predict(x_data)

pred