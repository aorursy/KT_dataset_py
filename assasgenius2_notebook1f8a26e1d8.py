# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

prices.hist()
#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
from sklearn.model_selection import train_test_split

X, X_draft, y_train, y_draft = train_test_split(X_train, y, test_size=0.5, random_state=42)
plt.plot(X.iloc[:, 3], y_train, 'ro')
print (X.iloc[:, 3])
from scipy.stats import pearsonr

print ("Correlation: ", pearsonr(X.iloc[:, 3], X.iloc[:, 2]))

from sklearn.decomposition import PCA

X=digits.data

y_train=digits.target

# définition de la commande

pca = PCA()

# Estimation, calcul des composantes principales

C = pca.fit(X).transform(X)

# Décroissance de la variance expliquée

plt.plot(pca.explained_variance_ratio_)

plt.xlabel('number of components')

plt.ylabel('explained variance');
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cummulative explained variance');
#Diagramme en boite des premières composantes principales

plt.boxplot(C[:,0:20])
#representing features in the first principle map

target_name=[0,1,2,3,4,5,6,7,8,9]

plt.scatter(C[:,0], C[:,1], c=y_train, label=target_name)
#3D plot

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(C[:, 0], C[:, 1], C[:, 2], c=y_train,

cmap=plt.cm.Paired)

ax.set_title("ACP: trois premieres composantes")

ax.set_xlabel("Comp1")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("Comp2")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("Comp3")

ax.w_zaxis.set_ticklabels([])

plt.show()
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor(min_samples_leaf=10, max_depth=3)

clf = clf.fit(C,y_train)

clf.score(C,y_train)
from sklearn.ensemble import RandomForestRegressor



forest = RandomForestRegressor(n_estimators=1000, 

                               criterion='mse', 

                               random_state=1, 

                               n_jobs=-1)

forest.fit(C, y_train)

y_train_pred = forest.predict(C)

y_test_pred = forest.predict(X_draft)



print('MSE train: %.3f, test: %.3f' % (

        mean_squared_error(y_train, y_train_pred),

        mean_squared_error(y_draft, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (

        r2_score(y_train, y_train_pred),

        r2_score(y_draft, y_test_pred)))