# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import datetime as dt

from keras.models import Sequential

from keras.layers import Dense

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ndata=pd.read_csv("../input/heart.csv")
ndata.head()
ndata.describe()
fig, ax = plt.subplots(figsize=(12,5))



sns.heatmap(ndata.corr(),annot = True,cmap="YlGnBu");

fig, ax = plt.subplots(figsize=(12,5))

sns.barplot(x = 'sex', y = 'target', data = ndata)

plt.xticks(rotation=20);
sns.violinplot(x="target", y="chol", data=ndata);
sns.boxplot(x="sex", y="chol", data=ndata);
sns.jointplot(x=ndata['chol'], y=ndata['trestbps'], kind="kde", color="y");
sns.catplot(x="target", y="age", hue="sex",data=ndata, kind="violin")
sns.jointplot(x=ndata['age'], y=ndata['thalach'], kind="hex", color="r");
plt.figure(figsize=(14,7))

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

N=303

colors = np.random.rand(N)

ax=plt.scatter(ndata.age, ndata['chol'],c=colors, alpha=0.5)
X = ndata.iloc[:, [0,1,2,3,4,5,6,7,8,10,11]].values

y = ndata.iloc[:, 12].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

X = sc.fit_transform(X)
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(X)

df = pd.DataFrame(data = principalComponents, columns = ['1','2','3'])
from sklearn.model_selection import train_test_split

#split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.linear_model import LogisticRegression

log=LogisticRegression()

log.fit(X_train,y_train)

y_pred = log.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

accuracy

from sklearn.metrics import f1_score

f1_score(y_test,y_pred,average='weighted')
import xgboost as xgb

model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

f1_score(y_test,y_pred,average='weighted')
from sklearn.svm import SVC

clf = SVC(random_state = 100, kernel='rbf')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

accuracy

from sklearn.metrics import f1_score

f1_score(y_test,y_pred,average='weighted')
from sklearn.ensemble import GradientBoostingClassifier

model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

f1_score(y_test,y_pred,average='weighted')
model = Sequential()

model.add(Dense(12, input_dim=11, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=150, batch_size=10,validation_data=(X_test, y_test))

scores = model.evaluate(X, y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))