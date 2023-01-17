import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn.preprocessing import power_transform

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from scipy.stats import randint as sp_randint





import os

print(os.listdir("../input"))

%matplotlib inline

import matplotlib.pyplot as plt
df = pd.read_csv("../input/data.csv")

print(df.head())

print(df.info())

print(df.shape)
df.head()
df=df.drop(["id","Unnamed: 32"],axis=1)

df.head()
df.shape
#probando si hay valores nulos

pd.isnull(df).sum()
def mapping(df,feature):

    featureMap=dict()

    count=0

    for i in sorted(df[feature].unique(),reverse=True):

        featureMap[i]=count

        count=count+1

    df[feature]=df[feature].map(featureMap)

    return df
df=mapping(df,feature="diagnosis")
df.sample(5)
plt.figure(figsize=(12,8))

sns.heatmap(df.describe()[1:].transpose(),

            annot=True,linecolor="w",

            linewidth=2,cmap=sns.color_palette("Blues"))

plt.title("Data summary")

plt.show()
# Histograma de sus datos

f, axes = plt.subplots(2,4, figsize=(20, 12))

sns.distplot( df["radius_mean"], ax=axes[0,0])

sns.distplot( df["texture_mean"], ax=axes[0,1])

sns.distplot( df["perimeter_mean"], ax=axes[0,2])

sns.distplot( df["area_mean"], ax=axes[1,0])

sns.distplot( df["smoothness_mean"], ax=axes[1,1])

sns.distplot( df["compactness_mean"], ax=axes[1,2])

sns.distplot( df["concavity_mean"], ax=axes[2,0])

sns.distplot( df["concave points_mean"], ax=axes[2,1])

sns.distplot( df["symmetry_mean"], ax=axes[2,2])

corr=df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.color_palette("Blues")

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.set(style="white")

df = df.loc[:,['radius_worst','perimeter_worst','area_worst']]

g = sns.PairGrid(df, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot, lw=3)
X = observables

y = df['diagnosis']
gnb = GaussianNB()

gnb_scores = cross_val_score(gnb, X, y, cv=10, scoring='accuracy')

print(gnb_scores.mean())
knn = KNeighborsClassifier()



k_range = list(range(1, 30))

leaf_size = list(range(1,30))

weight_options = ['uniform', 'distance']

algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options, 'algorithm': algorithm}
rand_knn = RandomizedSearchCV(knn, param_grid, cv=10, scoring="accuracy", n_iter=100, random_state=42)

rand_knn.fit(X,y)
print(rand_knn.best_score_)

print(rand_knn.best_params_)

print(rand_knn.best_estimator_)
dt_clf = DecisionTreeClassifier(random_state=42)



param_grid = {'max_features': ['auto', 'sqrt', 'log2'],

              'min_samples_split': sp_randint(2, 11), 

              'min_samples_leaf': sp_randint(1, 11)}
rand_dt = RandomizedSearchCV(dt_clf, param_grid, cv=10, scoring="accuracy", n_iter=100, random_state=42)

rand_dt.fit(X,y)

print(rand_dt.best_score_)

print(rand_dt.best_params_)

print(rand_dt.best_estimator_)
sv_clf = SVC(random_state=42)



param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 

               'gamma': [0.001, 0.0001], 'kernel': ['rbf']},]
grid_sv = GridSearchCV(sv_clf, param_grid, cv=10, scoring="accuracy")

grid_sv.fit(X,y)
print(grid_sv.best_score_)

print(grid_sv.best_params_)

print(grid_sv.best_estimator_)
# Tratando de evitar el sobreajuste

stump_clf =  DecisionTreeClassifier(random_state=42, max_depth=1)



param_grid = {"base_estimator__max_features": ['auto', 'sqrt', 'log2'],

              "n_estimators": list(range(1,500)),

              "learning_rate": np.linspace(0.01, 1, num=20),}
ada_clf = AdaBoostClassifier(base_estimator = stump_clf)



rand_ada = RandomizedSearchCV(ada_clf, param_grid, scoring = 'accuracy', n_iter=100, random_state=42)

rand_ada.fit(X,y)
print(rand_ada.best_score_)

print(rand_ada.best_params_)

print(rand_ada.best_estimator_)