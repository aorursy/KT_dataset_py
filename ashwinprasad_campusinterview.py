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
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import scipy
from tensorflow.keras.layers import Input,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy
#importing the dataset
df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
#view dataset and removing nan
df.dropna(inplace=True)
df.head()
#does gender bias exist ?
sns.scatterplot(df['sl_no'],df['salary'],hue=df['gender'])
#science vs commerce vs arts
sns.scatterplot(df['sl_no'],df['salary'],hue=df['hsc_s'])
#correlation between 10th grade and salary
plt.scatter(df['ssc_p'],df['salary'])
plt.xlabel('score in 10th grade')
plt.ylabel('salary')
score = np.array(df['ssc_p'])
salary = np.array(df['salary'])
print("correlation coefficient : ",np.corrcoef(score,salary)[0][1])
#correlation between degree pecentage and salary
plt.scatter(df['degree_p'],df['salary'])
plt.xlabel('degree percentage')
plt.ylabel('salary')
score = np.array(df['degree_p'])
salary = np.array(df['salary'])
print("correlation coefficient : ",np.corrcoef(score,salary)[0][1])
sns.scatterplot(df['sl_no'],df['salary'],hue=df['workex'])
sns.barplot(x=df['workex'],y=df['salary'])
#importing the dataset
df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
#dropping unnecessary columns
df.drop(['mba_p','salary','sl_no','specialisation'],inplace=True,axis=1)
#categorical to dummy
df['gender'] = pd.get_dummies(df['gender'],drop_first=True)
df['ssc_b'] = pd.get_dummies(df['ssc_b'],drop_first=True)
df['hsc_b'] = pd.get_dummies(df['hsc_b'],drop_first=True)
df['hsc_s'] = pd.get_dummies(df['hsc_s'],drop_first=True)
df['degree_t'] = pd.get_dummies(df['degree_t'],drop_first=True)
df['workex'] = pd.get_dummies(df['workex'],drop_first=True)
df['status'] = pd.get_dummies(df['status'],drop_first=True)
df.head()
#creating x and y 
x = df.drop(['status'],axis=1)
y = df['status']

#converting to numpy
x = np.array(x)
y = np.array(y)
#creating the model
i = Input(shape=(10))
X = Dense(1,activation='sigmoid')(i)
model = Model(i,X)
#compiling the model
model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.01),metrics=['accuracy'])
#training the model
train = model.fit(x=x,y=y,epochs=300)
#model accuracy 
plt.plot(train.history['accuracy'],label='training accuracy')
plt.legend()
#model loss
plt.plot(train.history['loss'],label='training loss')
plt.legend()