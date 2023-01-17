import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import random

import math

import seaborn as sns



from matplotlib.mlab import PCA as mlabPCA

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score

%matplotlib inline



import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")
movies = pd.read_csv("../input/movies.csv", encoding='latin-1')

movies.head()
genre_list = (movies['genre'].value_counts()[:3]).index.tolist()

movies = movies[movies.genre.isin(genre_list)]

new_star    = movies['star'].value_counts()     > 1

new_company = movies['company'].value_counts()  > 1

new_country = movies['country'].value_counts()  > 1

new_director= movies['director'].value_counts() > 1

new_writer  = movies['writer'].value_counts()   > 1



ns  = new_star[new_star].index

nco = new_company[new_company].index

nc  = new_country[new_country].index

nd  = new_director[new_director].index

nw  = new_writer[new_writer].index



movies_2 = movies[(movies['star'].isin(ns)) & \

                  (movies['company'].isin(nco)) & \

                  (movies['country'].isin(nc)) & \

                  (movies['director'].isin(nd)) & \

                  (movies['writer'].isin(nw))]

len(movies_2)

movies_2.columns

movies_2.head()
def onehot(X, cat_columns):

    X = pd.get_dummies(X, columns = cat_columns)

    return X
cat_columns = ['writer','director','country','star','company','rating']

X = onehot(movies_2, cat_columns)

X = X.drop(columns=['genre','name','released', 'votes'])

X.head()

scaler = StandardScaler()



X[['budget', 'gross']] = scaler.fit_transform(X[['budget', 'gross']])

X.head()
y= movies_2.genre

y = y.replace({'Comedy':0, 'Action':1,'Drama':2})

y
y = movies_2.genre

y_dummies = y.str.get_dummies()

y_dummies
sklearn_pca = PCA()

X_pca = sklearn_pca.fit_transform(X)



print(

    'The percentage of total variance in the dataset explained by each',

    'component from Sklearn PCA.\n',

    sklearn_pca.explained_variance_ratio_

)



plt.plot(X_pca)

plt.title('PCA')

plt.show()
mlp = MLPClassifier()
mlp.fit(X_pca,y)

mlp.score(X_pca,y)
test_params = {

    'hidden_layer_sizes': [100,200,300,400],

    

}

grid=GridSearchCV(estimator=mlp, param_grid=test_params)

grid.fit(X_pca,y)



print("Best parameters: ", grid.best_params_)

print("Best grid score: ", grid.best_score_)
test_params = {

    'activation':['identity', 'logistic', 'tanh', 'relu'],

    

}

grid=GridSearchCV(estimator=mlp, param_grid=test_params)

grid.fit(X_pca,y)



print("Best parameters: ", grid.best_params_)

print("Best grid score: ", grid.best_score_)
mlp = MLPClassifier(hidden_layer_sizes=(100,2),activation='logistic',alpha=.25)

mlp.fit(X_pca,y)

mlp.score(X_pca,y)
cross_val_score(mlp, X_pca, y, cv=5)
clf = RandomForestClassifier()

test_params = {

    'max_depth': [2,4,6,8,10],

    

}

grid=GridSearchCV(estimator=clf, param_grid=test_params)

grid.fit(X_pca,y)



print("Best parameters: ", grid.best_params_)

print("Best grid score: ", grid.best_score_)
clf = RandomForestClassifier()

test_params = {

    'n_estimators': [300,400,500],

    

}

grid=GridSearchCV(estimator=clf, param_grid=test_params)

grid.fit(X_pca,y)



print("Best parameters: ", grid.best_params_)

print("Best grid score: ", grid.best_score_)
clf = RandomForestClassifier(max_depth=10,n_estimators=500)

clf.fit(X_pca,y)

clf.score(X_pca,y)
cross_val_score(clf, X_pca, y, cv=10)
clf = RandomForestClassifier(max_depth=10,n_estimators=00)

clf.fit(X_pca,y)

clf.score(X_pca,y)
df1=pd.DataFrame({'songs': [1,2,3,1,3],'length': [4,2,5,4,5]})

df1
df2=pd.DataFrame([[1,2,3],[4,3,3],[4,2,3]])

df2
df1.corr()
def get_redundant_pairs(df):

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations")

print(get_top_abs_correlations(X, 50))