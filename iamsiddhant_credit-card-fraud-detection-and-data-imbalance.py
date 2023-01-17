import pandas as pd

import numpy as np
train=pd.read_csv('../input/creditcardfraud/creditcard.csv')
train.head()
train.info()
train['Class'].unique()
train.describe()
train.isnull().sum()
train.tail()
train.shape
train.Amount[train.Class == 1].describe()
train.Amount[train.Class == 0].describe()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
y=train['Class']

X=train.drop(['Class'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
model=LogisticRegression(solver='lbfgs',max_iter=1000)
model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test,y_test)
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.gridspec as gridspec
v_features = train.ix[:,1:29].columns
plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(train[v_features]):

    ax = plt.subplot(gs[i])

    sns.distplot(train[cn][train.Class == 1], bins=50)

    sns.distplot(train[cn][train.Class == 0], bins=50)

    ax.set_xlabel('')

    ax.set_title('histogram of feature: ' + str(cn))

plt.show();
correlation_matrix = train.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,vmax=0.8,square = True)

plt.show()
df=train
df.head()
df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
df.head()
y_1=df['Class']

X_1=df.drop(['Class'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.20)
model.fit(X_train,y_train)
model.score(X_test,y_test)
sns.countplot(train['Class'])
sns.boxplot(x="Class", y="Amount", hue="Class",data=train, palette="PRGn",showfliers=False)
train.head()
df_fraud=train[train.Class == 1]
df_fraud.head()
df_fraud.info()
df_genuine=train[train.Class == 0]
df_genuine.info()
# Randomly selecting 4000 rows from the genuine dataset



df_new_genuine=df_genuine.iloc[58457:60457]



#df_new_gen=df_gen.sample(4000)
# combining the both dataset the dataset with genuine transaction details and fraud transaction details

train_new = pd.concat([df_new_genuine, df_fraud],ignore_index=True, sort =False)
train_new.head()
train_new.info()
y_2=train_new['Class']

X_2=train_new.drop(['Class'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=0.20)
model.fit(X_train,y_train)
model.score(X_test,y_test)
train.head()
for _ in range(5):

    df_fraud = pd.concat([df_fraud, df_fraud],ignore_index=True, sort =False)
df_fraud_new=df_fraud
df_fraud_new.info()
df_fraud_new.head()
train_new_1 = pd.concat([df_genuine, df_fraud_new],ignore_index=True, sort =False)
train_new_1 .iloc[np.random.permutation(len(train_new_1 ))]
train_new_1.info()
y_3=train_new_1['Class']

X_3=train_new_1.drop(['Class'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size=0.20)
model.fit(X_train,y_train)
model.score(X_test,y_test)