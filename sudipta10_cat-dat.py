import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas_profiling

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
df_train = pd.read_csv('../input/cat-in-the-dat/train.csv')

df_train.head()
df_train.shape
df_train.isnull().sum()
df_train.dtypes
fig, ax =plt.subplots(2,2,figsize=(16,8))

sns.countplot(df_train['bin_0'], ax=ax[0,0])

sns.countplot(df_train['bin_1'], ax=ax[0,1])

sns.countplot(df_train['bin_2'], ax=ax[1,0])

sns.countplot(df_train['bin_3'], ax=ax[1,1])

fig.show()
plt.figure(figsize=(10,3))

sns.countplot(df_train['bin_4'])
fig, ax =plt.subplots(2,2,figsize=(16,12))

sns.countplot(df_train['nom_0'], ax=ax[0,0])

sns.countplot(df_train['nom_1'], ax=ax[0,1])

sns.countplot(df_train['nom_2'], ax=ax[1,0])

sns.countplot(df_train['nom_3'], ax=ax[1,1])

fig.show()
plt.figure(figsize=(10,3))

sns.countplot(df_train['nom_4'])
fig, ax =plt.subplots(3,2,figsize=(16,12))

sns.countplot(df_train['ord_0'], ax=ax[0,0])

sns.countplot(df_train['ord_1'], ax=ax[0,1])

sns.countplot(df_train['ord_2'], ax=ax[1,0])

sns.countplot(df_train['ord_3'].sort_values(), ax=ax[1,1])

sns.countplot(df_train['ord_4'].sort_values(), ax=ax[2,0])

sns.countplot(df_train['ord_5'].sort_values(), ax=ax[2,1])

fig.show()
plt.figure(figsize=(10,3))

sns.distplot(df_train['day'])
plt.figure(figsize=(10,3))

sns.distplot(df_train['month'])
plt.figure(figsize=(10,3))

sns.countplot(df_train['target'])
pd.value_counts(df_train['target'])
x = df_train.drop(['target'],axis=1)

y = df_train['target']
train = pd.DataFrame()

label=LabelEncoder()

for c in  x.columns:

    if(x[c].dtype=='object'):

        train[c]=label.fit_transform(x[c])

    else:

        train[c]=x[c]
train.head()
x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=0)
lg = LogisticRegression()
lg.fit(x_train,y_train)
pred = lg.predict(x_test)
print('Accuracy : ',accuracy_score(y_test,pred))
x.head()
one=OneHotEncoder()



one.fit(x)

train=one.transform(x)



print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))
x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=0)
lg1 = LogisticRegression()
lg1.fit(x_train, y_train)
pred1 = lg1.predict(x_test)
print('Accuracy : ',accuracy_score(y_test,pred1))
df_test = pd.read_csv('../input/cat-in-the-dat/test.csv')

df_test.head()
test_id = df_test['id']

df_test = df_test.drop(['id','nom_5','nom_6','nom_7','nom_8','nom_9'], axis=1)

x =x.drop(['id','nom_5','nom_6','nom_7','nom_8','nom_9'], axis=1)
hot = OneHotEncoder()

hot.fit(df_test)

test=hot.transform(df_test)



print('test data set has got {} rows and {} columns'.format(test.shape[0],test.shape[1]))
one=OneHotEncoder()



one.fit(x[:200000])

train=one.transform(x[:200000])



print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))
final_lg = LogisticRegression()
final_lg.fit(train,y[:200000])
actual_pred = final_lg.predict(test)
predict = pd.DataFrame(actual_pred)

id_t = pd.DataFrame(test_id)
submission = pd.concat([id_t,predict], axis=1)

submission.columns = ["id", "target"]
submission.head()
submission.to_csv('submission.csv',index=False)