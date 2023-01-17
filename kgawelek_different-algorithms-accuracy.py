import numpy as np

import pandas as pd

import math 



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.neighbors    import KNeighborsRegressor

from sklearn.ensemble     import AdaBoostRegressor

from sklearn.ensemble     import RandomForestRegressor



from sklearn.model_selection import train_test_split

from sklearn.preprocessing   import StandardScaler

from sklearn.metrics         import r2_score



import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
diamonds = pd.read_csv('../input/diamonds/diamonds.csv')
diamonds.head()
diamonds.drop('Unnamed: 0', axis='columns', inplace=True)
diamonds.head()
diamonds.shape
diamonds.info()
diamonds.isnull().sum()
msno.matrix(diamonds, figsize=(10,4))
diamonds.describe()
ill_values = (diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)
len(diamonds[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)])
diamonds = diamonds.loc[~ill_values]
len(diamonds[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)])
corr = diamonds.corr()

sns.heatmap(data=corr, square=True , annot=True, cbar=True)

sns.pairplot(diamonds)
sns.kdeplot(diamonds['carat'], shade=True , color='r')
plt.hist(diamonds['carat'], bins=25)
sns.jointplot(x='carat' , y='price' , data=diamonds , size=5)
sns.factorplot(x='cut', data=diamonds , kind='count',aspect=1.5)
sns.factorplot(x='cut', y='price', data=diamonds, kind='violin' ,aspect=1.5)
sns.factorplot(x='color', data=diamonds , kind='count',aspect=1.5)

sns.factorplot(x='color', y='price' , data=diamonds , kind='violin', aspect=1.5)
sns.factorplot(x='clarity', data=diamonds , kind='count',aspect=1.5)

sns.factorplot(x='clarity', y='price' , data=diamonds , kind='violin', aspect=1.5)
labels = diamonds.clarity.unique().tolist()

sizes = diamonds.clarity.value_counts().tolist()

explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

plt.pie(sizes, explode=explode, labels=labels,

        autopct='%1.1f%%', shadow=True, startangle=0)

plt.axis('equal')

plt.title("Percentage of Clarity Categories")

plt.plot()

fig=plt.gcf()

fig.set_size_inches(6,6)

plt.show()
sns.boxplot(x='clarity', y='price', data=diamonds)
plt.hist('depth' , data=diamonds , bins=25)
sns.jointplot(x='depth', y='price', data=diamonds, size=5)
sns.kdeplot(diamonds['x'] ,shade=True , color='r' )

sns.kdeplot(diamonds['y'] , shade=True , color='g' )

sns.kdeplot(diamonds['z'] , shade= True , color='b')

plt.xlim(2,10)

diamonds['volume'] = diamonds['x']*diamonds['y']*diamonds['z']

diamonds.head()
plt.figure(figsize=(5,5))

plt.hist( x=diamonds['volume'] , bins=30 ,color='g')

plt.xlabel('Volume in mm^3')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond\'s Volume')

plt.xlim(0,1000)

plt.ylim(0,50000)
sns.jointplot(x='volume', y='price' , data=diamonds, size=5)
diamonds.drop(['x','y','z'], axis=1, inplace= True)

diamonds.head()
diamonds = pd.get_dummies(diamonds, prefix_sep='_', drop_first=True)

diamonds.head()
X = diamonds.drop(['price'], axis=1)

y = diamonds['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=66)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

X_train
scores = []

models = ['Linear Regression', 'Lasso Regression', 'AdaBoost Regression', 

          'Ridge Regression', 'RandomForest Regression', 

          'KNeighbours Regression']
lr = LinearRegression()

lr.fit(X_train , y_train)

y_pred = lr.predict(X_test)

r2 = r2_score(y_test, y_pred)
scores.append(r2)

print('Linear Regression R2: {0:.2f}'.format(r2))
lasso = Lasso(normalize=True)

lasso.fit(X_train , y_train)

y_pred = lasso.predict(X_test)

r2 = r2_score(y_test, y_pred)

 

scores.append(r2)

print('Lasso Regression R2: {0:.2f}'.format(r2))
adaboost = AdaBoostRegressor(n_estimators=1000)

adaboost.fit(X_train , y_train)

y_pred = adaboost.predict(X_test)

r2 = r2_score(y_test, y_pred)

 

scores.append(r2)

print('AdaBoost Regression R2: {0:.2f}'.format(r2))
ridge = Ridge(normalize=True)

ridge.fit(X_train , y_train)

y_pred = ridge.predict(X_test)

r2 = r2_score(y_test, y_pred)

 

scores.append(r2)

print('Ridge Regression R2: {0:.2f}'.format(r2))
randomforest = RandomForestRegressor()

randomforest .fit(X_train , y_train)

y_pred = randomforest.predict(X_test)

r2 = r2_score(y_test, y_pred)

 

scores.append(r2)

print('Random Forest R2: {0:.2f}'.format(r2))
kneighbours = KNeighborsRegressor()

kneighbours.fit(X_train , y_train)

y_pred = kneighbours.predict(X_test)

r2 = r2_score(y_test, y_pred)

 

scores.append(r2)

print('K-Neighbours Regression R2: {0:.2f}'.format(r2))
ranking = pd.DataFrame({'Algorithms' : models , 'R2-Scores' : scores})

ranking = ranking.sort_values(by='R2-Scores' ,ascending=False)

ranking
sns.barplot(x='R2-Scores' , y='Algorithms' , data=ranking)