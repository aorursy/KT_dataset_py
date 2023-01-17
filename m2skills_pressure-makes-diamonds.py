import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
# reading the dataset
diamond = pd.read_csv("../input/diamonds.csv")
diamond.dtypes
diamond.info()
diamond.head(10)
diamond.describe()
diamond = diamond.drop(diamond.loc[diamond.x <= 0].index)
diamond = diamond.drop(diamond.loc[diamond.y <= 0].index)
diamond = diamond.drop(diamond.loc[diamond.z <= 0].index)
diamond["ratio"] = diamond.x / diamond.y
diamond.drop(['Unnamed: 0'],1, inplace = True)
#correlation matrix for 15 variables with largest correlation
corrmat = diamond.corr()
f, ax = plt.subplots(figsize=(12, 9))
k = 8 #number of variables for heatmap
cols = corrmat.nlargest(k, 'price')['price'].index
cm = np.corrcoef(diamond[cols].values.T)

# Generate a mask for the upper triangle
mask = np.zeros_like(cm, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


hm = sns.heatmap(cm, vmax=1, mask=mask, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
print("Mean Diamond Carat = " + str(np.mean(diamond.carat)))
plt.subplots(figsize=(10,7))
sns.distplot(diamond.carat)
plt.show()
sns.countplot(y = diamond.cut)
plt.show()
print("Mean Diamond Depth Value = " + str(np.mean(diamond.depth)))
plt.subplots(figsize=(10,7))
sns.distplot(diamond.depth)
plt.show()
plt.subplots(figsize=(10,7))
sns.countplot(diamond.color)
plt.show()
from collections import Counter
plt.pie(list(dict(Counter(diamond.color)).values()),
        labels = list(dict(Counter(diamond.color)).keys()),
        shadow = True,
        startangle = 0,
        explode = (0.1,0.1,0.1,0.1,0.1,0.1, 0.1));
plt.legend(list(dict(Counter(diamond.color)).keys()),loc = 2, bbox_to_anchor=(1.1, 1))
plt.show()
sns.countplot(diamond.clarity)
plt.show()
plt.pie(list(dict(Counter(diamond.clarity)).values()),
        labels = list(dict(Counter(diamond.clarity)).keys()),
        shadow = True,
        startangle = 0);
plt.legend(list(dict(Counter(diamond.clarity)).keys()),loc = 2, bbox_to_anchor=(1.1, 1))
plt.show()
print("Mean Diamond Table Value = " + str(np.mean(diamond.table)))
plt.subplots(figsize=(10,7))
sns.distplot(diamond.table)
plt.show()
plt.subplots(figsize=(15,7))
sns.distplot(diamond.price)
plt.show()
sns.set()
cols = diamond.columns
sns.pairplot(diamond[cols], size = 3.5)
plt.show();
diamond_cut = {'Fair':0,
               'Good':1,
               'Very Good':2, 
               'Premium':3,
               'Ideal':4}

diamond_color = {'J':0,
                 'I':1, 
                 'H':2,
                 'G':3,
                 'F':4,
                 'E':5,
                 'D':6}

diamond_clarity = {'I1':0,
                   'SI2':1,
                   'SI1':2,
                   'VS2':3,
                   'VS1':4,
                   'VVS2':5,
                   'VVS1':6,
                   'IF':7}
diamond.cut = diamond.cut.map(diamond_cut);
diamond.clarity = diamond.clarity.map(diamond_clarity);
diamond.color = diamond.color.map(diamond_color);
diamond.head()
diamond.describe()
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

X = diamond.drop(['price'],1)
y = diamond['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# min max or standard scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = LinearRegression()
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5,verbose = 1)
print('Linear regression accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))
classifier = Ridge(normalize=True)
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2',verbose = 1)
print('Ridge regression accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))
classifier = Lasso(normalize=True)
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2',verbose = 1)
print('Lasso accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))
classifier = ElasticNet()
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2',verbose = 1)
print('Elastic Net accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))
from sklearn.neighbors import KNeighborsRegressor
classifier = KNeighborsRegressor(n_neighbors=3)
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2',verbose = 1)
print('KNeighbors accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))
from sklearn.neural_network import MLPRegressor
classifier = MLPRegressor(hidden_layer_sizes=(14, ), learning_rate_init = 0.1)
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2',verbose = 1)
print('MLP accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))
from sklearn.ensemble import GradientBoostingRegressor
classifier = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls',verbose = 1)
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2')
print('Gradient Boosting Regression accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))