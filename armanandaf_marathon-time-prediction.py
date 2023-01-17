import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt #data visualization

import seaborn as sns #data visualization
df = pd.read_csv("../input/MarathonData.csv")

df.head()
df = df.drop(columns=['id','Marathon','CATEGORY'])

df.head()
df = df.drop(columns=['Name'])

df.head()
df.isna().sum()
df['CrossTraining'].unique()
df['CrossTraining'].fillna('nonct',inplace=True)

df.isna().sum()
dfn = df.dropna(how='any')

dfn.isna().sum()
f,axes = plt.subplots(figsize=(15,5))

sns.swarmplot(x = 'Category', y='MarathonTime',data=df, ax=axes)

plt.title('Time distribution on different Category')

plt.xlabel('Category')

plt.ylabel('Marathon Time')

plt.show()
df = df.dropna(how='any')

df = df.drop(columns=['Category'])

df.head()
df.isna().sum()
df.info()
df['Wall21'] = df['Wall21'].astype(float)

df.info()
plt.scatter(x = df['km4week'], y=df['MarathonTime'])

plt.title('km4week Vs Marathon Time')

plt.xlabel('km4week')

plt.ylabel('Marathon Time')

plt.show()
plt.scatter(x = df['sp4week'], y=df['MarathonTime'])

plt.title('sp4week Vs Marathon Time')

plt.xlabel('sp4week')

plt.ylabel('Marathon Time')

plt.show()
df = df.query('sp4week<2000')
plt.scatter(x = df['sp4week'], y=df['MarathonTime'])

plt.title('sp4week Vs Marathon Time')

plt.xlabel('sp4week')

plt.ylabel('Marathon Time')

plt.show()
plt.scatter(x = df['Wall21'], y=df['MarathonTime'])

plt.title('Wall21 Vs Marathon Time')

plt.xlabel('Wall21')

plt.ylabel('Marathon Time')

plt.show()
f,axes = plt.subplots(figsize=(15,5))

sns.boxplot(x = df['CrossTraining'], y=df['MarathonTime'],ax=axes)

plt.title('CrossTraining Vs Marathon Time')

plt.xlabel('CrossTraining')

plt.ylabel('Marathon Time')

plt.show()
df = pd.get_dummies(df)
correlated = df.corr().abs()['MarathonTime'].sort_values(ascending=False)

correlated
correlated = correlated[:5]

correlated
df = df.loc[:,correlated.index]

df.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score
X = df.drop(columns=['MarathonTime'])

y = df['MarathonTime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
clf = LinearRegression()

clf.fit(X_train,y_train)

r2 = clf.score(X_train,y_train)

def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)

a = adjustedR2(r2, X_train.shape[0],4)

a
prediction = clf.predict(X_test)

rmse = np.sqrt(np.mean((prediction-y_test)**2))

rmse
cv1 = cross_val_score(clf,X_train,y_train,cv=5).mean()

cv1
poly = PolynomialFeatures()

X_train2 = poly.fit_transform(X_train)

X_test2 = poly.fit_transform(X_test)

clf.fit(X_train2,y_train)

PolyR2 = clf.score(X_train2,y_train)

PolyR2
b = adjustedR2(r2, X_train2.shape[0],4)

b
prediction2 = clf.predict(X_test2)

rmse2 = np.sqrt(np.mean((prediction2-y_test)**2))

rmse2
cv2 = cross_val_score(clf,X_train2,y_train,cv=5).mean()

cv2
results = pd.DataFrame({'Model': [],

                        'Root Mean Squared Error (RMSE)':[],

                        'R-squared (test)':[],

                        'Adjusted R-squared (test)':[],

                        '5-Fold Cross Validation':[]})

r = results.shape[0]

results.loc[r] = ['Linear Regression', rmse, r2, a, cv1]

results.loc[r+1] = ['Polynomial Regression', rmse2, PolyR2, b, cv2]
results