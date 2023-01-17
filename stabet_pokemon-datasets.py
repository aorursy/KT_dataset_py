# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import scale

from sklearn.decomposition import PCA



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df = pd.read_csv('../input/Pokemon.csv')



# Drop type 2 

#df2 = df.drop(['Type 2'], axis=1)

#df2.rename(columns={"Type 1": "Type"}, inplace=True)



# Drop type 1

#df1 = df.drop(['Type 1'], axis=1)

#df2.rename(columns={"Type 2": "Type"}, inplace=True)



#df = df1.append(df2, ignore_index=True)



df = df[df['Type 2'].isnull()]



X = df[['Total', 'HP', 'Attack', 'Defense', 'Speed', 'Sp. Atk', 'Sp. Def', 'Legendary', 'Generation']]





# Handle categorical data

df['Type 1'] = pd.Categorical(df['Type 1'])

df['Type 1'] = df['Type 1'].cat.codes





y = df['Type 1']

df.head()







# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



df2 = X

# type

df2['t'] = y



sns.pairplot(df2, hue='t')
#### Get relevant features with PCA

pca = PCA(n_components=8)

pca.fit(X)

#X = pca.transform(X)



X = scale(X)

df2.groupby("t").count()
x1 = X[:,0]

x2 = X[:,1]



plt.scatter(x1, x2, c=y)
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score





svc_scores = cross_val_score(XGBClassifier(learning_rate=10), X, y, cv=5, scoring="r2")

                                              

print("XGB: \n", svc_scores)

print(y.shape)

print(X.shape)
xgb = XGBClassifier()

xgb.fit(X[:300], y[:300])

y_pred = xgb.predict(X[300:])

plt.scatter(y_pred, y[300:])
from xgboost import plot_importance



plot_importance(xgb)
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(alpha=1, hidden_layer_sizes=(30, 30, 30, 30, 30, 30), max_iter=1000)



mlp_scores = cross_val_score(mlp, X, y, cv=5)



print("With a NN: \n", mlp_scores)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()



nb_scores = cross_val_score(gnb, X, y, cv=4)



print("With Naive Bayes: \n", nb_scores)
from sklearn.ensemble import RandomForestClassifier



rndf = RandomForestClassifier(max_depth=2, random_state=0)



rndf_scores = cross_val_score(rndf, X, y, cv=4)



print("With Random Forest: \n", rndf_scores)