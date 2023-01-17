# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.metrics import log_loss, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier



from xgboost import XGBClassifier



import tensorflow as tf





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/15cse380-nndl-eval/train.csv')

train
test = pd.read_csv('/kaggle/input/15cse380-nndl-eval/test.csv')

test
sub = pd.read_csv('/kaggle/input/15cse380-nndl-eval/submission.csv')

sub
train['previous_year_rating'].fillna(0.0, inplace=True)

test['previous_year_rating'].fillna(0.0, inplace=True)



train['education'].fillna('Not Graduated', inplace=True)

test['education'].fillna('Not Graduated', inplace=True)
le1 = LabelEncoder().fit(train['department'])

le2 = LabelEncoder().fit(train['region'])

le3 = LabelEncoder().fit(train['education'])

le4 = LabelEncoder().fit(train['gender'])

le5 = LabelEncoder().fit(train['recruitment_channel'])



train['department'] = le1.transform(train['department'])

test['department'] = le1.transform(test['department'])



train['region'] = le2.transform(train['region'])

test['region'] = le2.transform(test['region'])



train['education'] = le3.transform(train['education'])

test['education'] = le3.transform(test['education'])



train['gender'] = le4.transform(train['gender'])

test['gender'] = le4.transform(test['gender'])



train['recruitment_channel'] = le5.transform(train['recruitment_channel'])

test['recruitment_channel'] = le5.transform(test['recruitment_channel'])



train
x = train.iloc[:,1:-1]

y = train.iloc[:,-1]
x.shape
# model = RandomForestClassifier(n_estimators=1000)

# model.fit(x,y)
# xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
# model = tf.keras.Sequential()

# model.add(tf.keras.layers.Dense(256, input_shape=xtrain.shape[1:], activation='relu'))

# model.add(tf.keras.layers.Dropout(0.2))

# model.add(tf.keras.layers.Dense(256, activation='relu'))

# model.add(tf.keras.layers.Dropout(0.2))

# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



# model.compile(optimizer='adam', loss='log_loss', metrics=['accuracy'])



# model.summary()
# model.fit(xtrain,ytrain,epochs=20,batch_size=32,validation_data=(xtest,ytest))
model = GradientBoostingClassifier(n_estimators=1000, max_depth=10, learning_rate=0.005)

model.fit(x,y)
y_pred = model.predict(test.iloc[:,1:])

# y_pred[y_pred>0.5] = 1

# y_pred[y_pred<=0.5] = 0

sub['is_promoted'] = y_pred

sub
sub.to_csv('submission.csv',index=False)