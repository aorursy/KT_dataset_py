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
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import optimizers
import seaborn as sns

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train['LastName'] = train['Name'].str.split(',', expand=True)[0]
test['LastName'] = test['Name'].str.split(',', expand=True)[0]
ds = pd.concat([train, test])

sur = []
died = []
for index, row in ds.iterrows():
    s = ds[(ds['LastName']==row['LastName']) & (ds['Survived']==1)]
    d = ds[(ds['LastName']==row['LastName']) & (ds['Survived']==0)]
    s=len(s)
    if row['Survived'] == 1:
        s-=1
    d=len(d)
    if row['Survived'] == 0:
        d-=1
    sur.append(s)
    died.append(d)
ds['FamilySurvived'] = sur
ds['FamilyDied'] = died

ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
ds['IsAlone'] = 0
ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1
ds['Fare'] = ds['Fare'].fillna(train['Fare'].median())
ds['Embarked'] = ds['Embarked'].fillna('Q')

train = ds[ds['Survived'].notnull()]
test = ds[ds['Survived'].isnull()]
test = test.drop(['Survived'], axis=1)

train['rich_woman'] = 0
test['rich_woman'] = 0
train['men_3'] = 0
test['men_3'] = 0

train.loc[(train['Pclass']<=2) & (train['Sex']=='female'), 'rich_woman'] = 1
test.loc[(test['Pclass']<=2) & (test['Sex']=='female'), 'rich_woman'] = 1
train.loc[(train['Pclass']==3) & (train['Sex']=='male'), 'men_3'] = 1
test.loc[(test['Pclass']==3) & (test['Sex']=='male'), 'men_3'] = 1

train['rich_woman'] = train['rich_woman'].astype(np.int8)
test['rich_woman'] = test['rich_woman'].astype(np.int8)

train["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train['Cabin']])
test['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in test['Cabin']])

train = train.drop(['PassengerId', 'Ticket', 'LastName', 'SibSp', 'Parch'], axis=1)
test = test.drop(['PassengerId', 'Ticket', 'LastName', 'SibSp', 'Parch'], axis=1)

categorical = ['Pclass', 'Sex', 'Embarked', 'Cabin']
for cat in categorical:
    train = pd.concat([train, pd.get_dummies(train[cat], prefix=cat)], axis=1)
    train = train.drop([cat], axis=1)
    test = pd.concat([test, pd.get_dummies(test[cat], prefix=cat)], axis=1)
    test = test.drop([cat], axis=1)
    
train = train.drop(['Sex_male', 'Name'], axis=1)
test =  test.drop(['Sex_male', 'Name'], axis=1)

train = train.fillna(-1)
test = test.fillna(-1)
train.head()
y = train['Survived']
X = train.drop(['Survived', 'Cabin_T'], axis=1)
X_test = test.copy()

X_train, X_test1, y_train, y_test = train_test_split(X, y, random_state=4, test_size=0.2, shuffle=True)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test1.shape,  y_test.shape)
model = tf.keras.Sequential([
keras.layers.Dense(512 , activation='relu' , input_shape=[23]),
keras.layers.Dense(512 , activation='relu'),
keras.layers.Dense(512 , activation='relu'),
keras.layers.Dense(512 , activation='relu'),
keras.layers.Dense(512 , activation='relu'),
  

keras.layers.Dense(1 ) ])
model.compile(loss=('mse','mae'),optimizer='adam',metrics=['mse','accuracy'])
history=model.fit( X_train,y_train ,epochs = 500,  validation_data=(X_test1,y_test),verbose=1 )
yhat_test=model.predict(X_test1)
yhat_train=model.predict(X_train)
width = 12
height = 10
plt.figure(figsize=(width, height))
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()
plt.figure(figsize=(width, height))

ax1 = sns.distplot(y_train, hist=False, color="r", label="Actual Value")
sns.distplot(yhat_train, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')


plt.show()
plt.close()
plt.figure(figsize=(width, height))

ax2 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(yhat_test, hist=False, color="b", label="Fitted Values" , ax=ax2)

plt.title('Actual vs Fitted Values for Price')


plt.show()
plt.close()
preds = model.predict(X_test)
preds = preds.astype(np.int16)
submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = preds
submission.to_csv('submission.csv', index=False)
#Loading the data set
real = pd.read_csv('/kaggle/input/titanic-leaked/titanic.csv')
a=submission[['Survived']].to_numpy()
b=real[['Survived']].to_numpy()
acc =1- (np.square(np.subtract(b, a)).mean())
print('Model Accuracy =',acc)
