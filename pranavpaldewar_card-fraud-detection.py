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
import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.head(10)
df.shape
plt.figure(figsize=(10,6))

sns.distplot(df.Amount**0.5)

plt.figure(figsize=(10,6))

sns.distplot(df.Time)
df.Class.value_counts().plot(kind='bar')
plt.figure(figsize=(10,6))

sns.heatmap(df.corr())
df.describe()
from sklearn.model_selection import train_test_split

y=df.Class

x=df.drop('Class',axis=1)

train_x,valid_x,train_y,valid_y=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50,max_depth=5)

rfc.fit(train_x,train_y)

predictions=rfc.predict(valid_x)
par_grid={'n_estimators':[20,50,70],'max_depth':[3,5,8]}
from sklearn.model_selection import GridSearchCV 

grc=GridSearchCV(rfc,cv=3,param_grid=par_grid)

grc.fit(train_x,train_y)
predic=grc.predict(valid_x)
from sklearn.metrics import mean_absolute_error

print("the accuracy of the model is {}".format(mean_absolute_error(predic,valid_y)))

print("the accuracy of the model is {}".format(mean_absolute_error(predictions,valid_y)))
from sklearn.metrics import confusion_matrix

con_1=confusion_matrix(valid_y,predic)

con_1
sns.heatmap(con_1)


import tensorflow as tf

model1=tf.keras.Sequential([tf.keras.layers.Dense(30,input_shape=[30],activation='relu'),

                          tf.keras.layers.Dense(256,activation='relu'),

                          tf.keras.layers.Dense(1,activation='sigmoid')])

model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model1.fit(train_x,train_y,epochs=100,validation_data=(valid_x,valid_y),verbose=2)
pr=model1.predict(valid_x)

pr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ldr=LinearDiscriminantAnalysis()

ldr.fit(train_x,train_y)

predict_1=ldr.predict(valid_x)

con_2=confusion_matrix(predict_1,valid_y)

con_2
from scikitplot.metrics import plot_roc_curve



plot_roc_curve(predic,valid_y)