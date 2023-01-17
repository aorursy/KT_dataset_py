import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

from sklearn.metrics import mean_squared_error
train = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

train.head()
train.dtypes
train.isnull().sum()
len(train)
a = []

for i in range(len(train)):

    if (train.iloc[i,10] == 'new'):

        a.append(1)

    else:

        a.append(0)
temp = train.copy()
temp.drop(labels = ['type'], axis = 1, inplace = True)
temp['type'] =a
temp.head()
temp.fillna(temp.mean(), inplace=True)
temp.isnull().sum()
cor = temp.corr()

sns.set(rc={'figure.figsize':(15,15)})

sns.heatmap(cor, vmin=-1, vmax=1, annot = True)
temp1 = temp.drop(['id','type'], axis = 1)

cor = temp1.corr()

sns.set(rc={'figure.figsize':(7,5)})

sns.heatmap(cor, vmin=-1, vmax=1, annot = True)

temp1 = temp1.drop(['rating'],axis = 1)
"""temp3 = temp.copy()

temp3.drop(['id','rating','feature8','feature7','feature5','type'],axis = 1, inplace = True)

clf = RandomForestClassifier(random_state=42,n_estimators=522)

X = temp3.values

y = temp['rating']

y = y.values

clf.fit(X[:4000],y[:4000])

mse = mean_squared_error(y[4000:],clf.predict(X[4000:])) #0.4040219378427788 0.40036563071297987

print(mse)

for i in temp3.columns:

    temp1 = temp3.copy()

    temp1.drop([i],axis = 1 ,inplace =True)

    clf = RandomForestClassifier(random_state=42,n_estimators=522)

    X = temp1.values

    y = temp['rating']

    y = y.values

    clf.fit(X[:4000],y[:4000])

    print(mse - mean_squared_error(y[4000:],clf.predict(X[4000:])),i)"""

# This code was used to get which columns to drop
"""temp3 = temp.copy()

temp3.drop(['id','rating','feature7','feature8','feature10','type'],axis = 1, inplace = True)

clf = RandomForestClassifier(random_state=42,n_estimators=500)

X = temp3.values

y = temp['rating']

y = y.values

clf.fit(X[:4000],y[:4000])

mse = mean_squared_error(y[4000:],clf.predict(X[4000:])) #0.4040219378427788 0.40036563071297987 0.3875685557586837

print(mse)

mi = mse

for i in range(1,1000):

    clf = RandomForestClassifier(random_state=42,n_estimators=i)

    X = temp3.values

    y = temp['rating']

    y = y.values

    clf.fit(X[:4000],y[:4000])

    ms = mean_squared_error(y[4000:],clf.predict(X[4000:]))

    if (ms<mi):

        mi = ms

        print(i,mi)"""



# This code was used to get best size of n_estimator which turned out to be 522
temp3 = temp.copy()

temp3.drop(['id','rating','feature8','feature7','feature10','type'],axis = 1, inplace = True)#,'feature7','feature8','feature10','type'

clf = RandomForestClassifier(random_state=42, n_estimators=500)

X = temp3.values

y = temp['rating']

y = y.values

clf.fit(X[:4000],y[:4000])

mse = mean_squared_error(y[4000:],clf.predict(X[4000:])) #0.4040219378427788 0.40036563071297987

print(mse)
temp3.columns
clf.feature_importances_/np.max(clf.feature_importances_)
test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
a = []

for i in range(len(test)):

    if (test.iloc[i,10] == 'new'):

        a.append(1)

    else:

        a.append(0)

temp2 = test.copy()

i = temp2['id']

temp2.drop(['id','feature8','feature7','feature10','type'],axis = 1, inplace = True) #,'feature7','feature8','feature10','type'

temp2.fillna(temp2.mean(), inplace=True)
temp2.columns
temp2.isnull().sum()
values = clf.predict(temp2)
i = i.values
values
ans = []

for j in range(len(values)):

    ans.append([i[j],values[j]])
ans = np.array(ans)
dataset = pd.DataFrame({'id': ans[:,0], 'rating': ans[:,1]})
dataset.head()
dataset.to_csv('ans.csv', index=False)