# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
stats1 = pd.read_csv('/kaggle/input/league-of-legends-ranked-matches/stats1.csv')

stats1.dtypes
# stats1.iloc[10:11, 0:100]

stats1 = stats1.drop(labels= ['id', 'item1','item2', 'item3', 'item4','item5','item6','trinket'], 

            axis=1)
stats1.describe()
# Models based on PVP action... 

# if you 

# df = stats1.sample(20000)[['win', 'kills', 'deaths', 'assists', 'largestmultikill','firstblood']]

df = stats1

# df = stats1.sample(20000)[['win', 'goldearned', 'dmgtoturrets', 'longesttimespentliving', 'visionscore', 'wardsplaced', 'totminionskilled']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    df, df['win'], random_state=0

)
# Ridge model

from sklearn.linear_model import Ridge



ridge_model_a = Ridge(alpha=100).fit(X_train, y_train)

y_preds = ridge_model_a.predict(X_train)

ridge_model_a.score(X_train, y_train)

ridge_model_a.coef_
from sklearn import linear_model

clf = linear_model.Lasso(alpha = .1)

clf.fit(X_train, y_train)
import altair as alt

source = df.sample(5000)

brush = alt.selection(type='interval', resolve='global')



base = alt.Chart(source).mark_point().encode(

    y='win:N',

    color=alt.condition(brush, 'win:N', alt.ColorValue('gray')),

).add_selection(

    brush

).properties(

    width=250,

    height=250

)



base.encode(x='kills:Q') | base.encode(x='deaths:Q')
import seaborn as sns

import matplotlib.pyplot as plt

corr = stats1.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(14, 14))



sns.heatmap(

corr,

cmap='bwr',

square = True,

    mask = mask,

    linewidth = 0.5,

    alpha = 0.7

)

# from pandas.plotting import scatter_matrix

# import matplotlib.pyplot as plt

# %matplotlib inline

# %config InlineBackend.figure_format='retina'



# grr = scatter_matrix(df, c=df['win'], figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=30, alpha=.8)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=12)

# how do you know what to pick for k????
knn.fit(X_train, y_train)
f"KNN Score: {knn.score(X_test, y_test):2f}"
# Classification: SVM

from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=0, tol=1e-5)

clf.fit(X_train, y_train)