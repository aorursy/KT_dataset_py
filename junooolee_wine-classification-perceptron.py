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
from sklearn.metrics import confusion_matrix

#시각화

import matplotlib.pylab as plt
%matplotlib inline

#tensorflow
import tensorflow as tf
from tensorflow.keras import layers
wine = pd.read_csv('../input/wineuci/Wine.csv',header=None)
wine.head()
wine.columns = ['class','alcohol','malicAcid','ash',\
                'ashalcalinity','magnesium','totalPhenols','flavanoids',\
                'nonFlavanoidPhenols','proanthocyanins','colorIntensity','hue','od280_od315',\
                'proline']
wine
wine.info()
wine.isnull().sum()
X = wine.drop(['class'],axis=1)
y = wine['class']
X.head()
y.head()
scaler = StandardScaler() # 스케일링
x = scaler.fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
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
  print(classification_report(y_test,predictions),'\n')
# lr,dt,rf,nb,eclf_h,eclf_s
x.shape,y.shape
x[1].shape
y = pd.get_dummies(y)
y.columns = ['class1','class2','class3']
y
x_train_all,x_test,y_train_all,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train_all,y_train_all,test_size=0.2)
regular = 0.00001 #regularization
model = tf.keras.Sequential()
x[1].shape
model.add(layers.Dense(12, input_shape = x[1].shape,activation='relu',\
          kernel_regularizer = tf.keras.regularizers.l2(regular),\
          activity_regularizer = tf.keras.regularizers.l2(regular)))
model.add(layers.Dense(8,activation = 'relu'))
model.add(layers.Dense(16,activation = 'relu'))
model.add(layers.Dense(128,activation = 'relu'))
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(32,activation = 'relu'))
model.add(layers.Dense(3,activation = 'softmax'))

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

hist = model.fit(x_train,y_train,epochs =100,validation_split=0.2)
model.evaluate(x_test,y_test)
y_pred =model.predict(x_test)    

y_test_class=np.argmax(y_test.values,axis=1)
y_pred_class=np.argmax(y_pred,axis=1)

y_train
print(classification_report(y_test_class,y_pred_class))
confusion_matrix(y_test_class,y_pred_class)
hist.history.keys()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
y_test_class.shape,y_pred_class.shape
weights, biases = model.layers[1].get_weights()
weights
weights.shape, biases.shape
plt.plot(weights,'x')
plt.plot(biases,'o')
plt.title('L2 - 0.00001')
