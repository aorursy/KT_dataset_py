# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pandas as pd

from pandas import DataFrame,Series

from sklearn import tree

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as smf

import statsmodels.api as sm

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from sklearn import neighbors

from sklearn import linear_model

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/movie_metadata.csv')
df.head(3)
df.columns
df[df.director_name=='James Cameron']
df_float=df.dtypes[df.dtypes!=object].index

df_object=df.dtypes[df.dtypes==object].index

df[df_object].head(3)
df['director_name'].unique
df.groupby('director_name')['director_name'].count()
sns.countplot(x='color',data=df)
sns.countplot(x='language',data=df)
df.groupby('language')['language'].count()
df.groupby('country')['country'].count()
df[df_float].head(3)
X_train=df[df_float]

X_train.fillna(0)

y=X_train['imdb_score']

#X_train.drop(['imdb_score'],axis=1,inplace=True)

X_train.head()[:2]
# GETTING Correllation matrix

X_train1=X_train

corr_mat=X_train.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
X_train.drop(['imdb_score'],axis=1,inplace=True)
np.sum(X_train.isnull())
X_train=X_train.fillna(0)
X_train=X_train.fillna(0)

X_Train=X_train.values

X_Train=np.asarray(X_Train)



# Finding normalised array of X_Train

X_std=StandardScaler().fit_transform(X_Train)

from sklearn.decomposition import PCA

pca = PCA().fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,7,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
covariance=pca.get_covariance()

explained_variance=pca.explained_variance_

explained_variance
with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))

    

    plt.bar(range(15), explained_variance, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
from sklearn.decomposition import PCA

sklearn_pca=PCA(n_components=5)

X_Train=sklearn_pca.fit_transform(X_std)



sns.set(style='darkgrid')

f, ax = plt.subplots(figsize=(8, 8))

# ax.set_aspect('equal')

ax = sns.kdeplot(X_Train[:,0], X_Train[:,1], cmap="Greens",

          shade=True, shade_lowest=False)

ax = sns.kdeplot(X_Train[:,1], X_Train[:,2], cmap="Reds",

          shade=True, shade_lowest=False)

ax = sns.kdeplot(X_Train[:,2], X_Train[:,3], cmap="Blues",

          shade=True, shade_lowest=False)

red = sns.color_palette("Reds")[-2]

blue = sns.color_palette("Blues")[-2]

green = sns.color_palette("Greens")[-2]

ax.text(0.5, 0.5, "2nd and 3rd Projection", size=12, color=blue)

ax.text(-4, 0.0, "1st and 3rd Projection", size=12, color=red)

ax.text(2, 0, "1st and 2nd Projection", size=12, color=green)

plt.xlim(-6,5)

plt.ylim(-2,2)
#Splitting the data into training and testing dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_Train,y,test_size=0.2,random_state=6)
#Linear Regrassion

from sklearn.linear_model import LinearRegression

LR=LinearRegression()

LR.fit(X_train, y_train)

LR_score_train = LR.score(X_train, y_train)

print("Training score: ",LR_score_train)

logis_score_test = LR.score(X_test, y_test)

print("Testing score: ",LR_score_train)



# Filling all Null values

X_train=df[df_float]

X_train=X_train.fillna(0)

columns=X_train.columns.tolist()

y=X_train['imdb_score']

#X_train.drop(['imdb_score'],axis=1,inplace=True)

X_train.head()[:2]
X_train.drop(['imdb_score'],axis=1,inplace=True)
# GETTING Correllation matrix

corr_mat=X_train.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
X_Train=X_train.values

X_Train=np.asarray(X_Train)



# Finding normalised array of X_Train

X_std=StandardScaler().fit_transform(X_Train)
from sklearn.decomposition import PCA

pca = PCA().fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,7,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
covariance=pca.get_covariance()

explained_variance=pca.explained_variance_

explained_variance
with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))

    

    plt.bar(range(15), explained_variance, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
from sklearn.decomposition import PCA

sklearn_pca=PCA(n_components=5)

X_Train=sklearn_pca.fit_transform(X_std)



sns.set(style='darkgrid')

f, ax = plt.subplots(figsize=(8, 8))

# ax.set_aspect('equal')

ax = sns.kdeplot(X_Train[:,0], X_Train[:,1], cmap="Greens",

          shade=True, shade_lowest=False)

ax = sns.kdeplot(X_Train[:,1], X_Train[:,2], cmap="Reds",

          shade=True, shade_lowest=False)

ax = sns.kdeplot(X_Train[:,2], X_Train[:,3], cmap="Blues",

          shade=True, shade_lowest=False)

red = sns.color_palette("Reds")[-2]

blue = sns.color_palette("Blues")[-2]

green = sns.color_palette("Greens")[-2]

ax.text(0.5, 0.5, "2nd and 3rd Projection", size=12, color=blue)

ax.text(-4, 0.0, "1st and 3rd Projection", size=12, color=red)

ax.text(2, 0, "1st and 2nd Projection", size=12, color=green)

plt.xlim(-6,5)

plt.ylim(-2,2)
#Splitting the data into training and testing dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train,y,test_size=0.2,random_state=4)
#Linear Regression

from sklearn.linear_model import LinearRegression

logis = LinearRegression()

logis.fit(X_train, y_train)

logis_score_train = logis.score(X_train, y_train)

print("Training score: ",logis_score_train)

logis_score_test = logis.score(X_test, y_test)

print("Testing score: ",logis_score_test)
#decision tree

from sklearn.ensemble import RandomForestRegressor

dt = RandomForestRegressor()

dt.fit(X_train, y_train)

dt_score_train = dt.score(X_train, y_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(X_test, y_test)

print("Testing score: ",dt_score_test)
#predict

dt.predict(X_test)
X_test.head(2)
y_test.head(3)
#Logistic Regression

from sklearn.linear_model import Ridge

rr = Ridge()

rr.fit(X_train, y_train)

rr_score_train = rr.score(X_train, y_train)

print("Training score: ",rr_score_train)

rr_score_test = rr.score(X_test, y_test)

print("Testing score: ",rr_score_test)
svm_reg=svm.SVR()

svm_reg.fit(X_train, y_train)

y1_svm=svm_reg.predict(X_train)

y1_svm=list(y1_svm)

y2_svm=svm_reg.predict(X_test)

y2_svm=list(y2_svm)
svm_score_train = svm_reg.score(X_train, y_train)

print("Training score: ",svm_score_train)

rr_score_test = svm_reg.score(X_test, y_test)

print("Testing score: ",rr_score_test)
#SVM Residual plot

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":svm_reg.predict(X_train), "true":y_train})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")

plt.title("Residual plot in SVM")
n_neighbors=5

knn=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')

knn.fit(X_train,y_train)

y1_knn=knn.predict(X_train)

y1_knn=list(y1_knn)

kmn_score_train = knn.score(X_train, y_train)

print("Training score: ",kmn_score_train)

kmn_score_test = knn.score(X_test, y_test)

print("Testing score: ",kmn_score_test)
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":knn.predict(X_train), "true":y_train})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")

plt.title("Residual plot in Knn")
#so it shows it is very difficult to predict imdb rating based on feature