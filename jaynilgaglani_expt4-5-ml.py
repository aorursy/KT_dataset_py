# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error, r2_score,accuracy_score

from sklearn.model_selection import KFold, cross_val_score,GridSearchCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



df = pd.read_csv("../input/train_modified.csv")
df.head()
df.info()
df.describe()
df = pd.DataFrame(preprocessing.normalize(df),columns = df.columns)

df.head()
df.describe()
y = df.Item_Outlet_Sales

X = df.drop('Item_Outlet_Sales',axis=1) 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,shuffle=True)
clf = RandomForestRegressor(max_depth=20,

                                min_samples_split=170, n_estimators=230*3, 

                                random_state=1).fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Test RMSE: ", np.sqrt(mean_squared_error(y_test,y_pred)))

print("Test R^2: ", r2_score(y_test,y_pred))
pca = PCA()  

pX_train = pca.fit_transform(X_train)  

pX_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_ 

print(list(explained_variance))
pca = PCA(n_components=2,random_state=1)

X_train_pca = pca.fit_transform(X_train)  

X_test_pca = pca.transform(X_test)
clf2 = RandomForestRegressor(max_depth=20,

                                min_samples_split=170, n_estimators=230*3, 

                                random_state=42).fit(X_train_pca, y_train)

y_pred2 = clf2.predict(X_test_pca)

print("Test RMSE: ", np.sqrt(mean_squared_error(y_test,y_pred2)))

print("Test R^2: ", r2_score(y_test,y_pred2))
ax = sns.regplot(X_test_pca[:,0],X_test_pca[:,1])

ax.set_xlabel("Principal Component 1")

ax.set_ylabel("Principal Component 2")

ax.set_title("PCA")