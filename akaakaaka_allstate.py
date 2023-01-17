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
#more imports; mine, not there by default

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression



from sklearn.tree import DecisionTreeRegressor



from sklearn.ensemble import RandomForestRegressor



from sklearn.model_selection import validation_curve

from sklearn.model_selection import learning_curve

from sklearn.model_selection import cross_val_score



from sklearn.metrics import mean_absolute_error

from sklearn.metrics import make_scorer

mae_scoring = make_scorer(mean_absolute_error)



from sklearn.decomposition import PCA



from sklearn.neighbors import KNeighborsRegressor



from numbers import Number
df = pd.read_csv("../input/train.csv")
#find columns w nas

print(df.columns[pd.isnull(df).any()].tolist())





src = list(df.columns)

src.remove('id')

src.remove('loss')

src
#dummy out categorical variables so we can use numerical dimensionality reduction + other methods



ind = 1

for var in src:

    if var[:3] == "cat":

        print(var)

        root = "dum" + str(ind) + "_"

        ind += 1

        dums = pd.get_dummies(df[var])

        dums.columns = [root + str(col) for col in dums.columns]

        df = pd.concat([df,dums], axis=1)

        

        

        

        
#normalize

from numbers import Number

for var in df.columns:

    #for numerical variables

    if isinstance(df[var][0],Number):

        col = np.asarray(df[var])

        df[var] = (col -np.mean(col))/np.std(col)

        

df.var()
df.shape()
X = df._get_numeric_data()

X = X.drop('loss',1)



y = df['loss']
X = (X - X.mean())/np.sqrt(X.var())

X.var()
pca = PCA()



pca.fit(X)

exp = pca.explained_variance_ratio_

cumvar = np.cumsum(exp)

print(cumvar)



#save a copy of X, get a transformed copy

Xbase = X.copy()

XT = pca.transform(np.asarray(X))
XT = XT[:,:-3]
reg = KNeighborsRegressor()

tovary = 'n_neighbors'

vary_range = range(3,30,3)





train_scores, valid_scores = validation_curve(reg, X, y, tovary, vary_range, scoring = mae_scoring)



trainavg = np.mean(train_scores, axis=1)

validavg = np.mean(valid_scores, axis=1)



plt.clf()

plt.plot(vary_range, trainavg, label='train scores', marker = '.', c = 'k')

plt.plot(vary_range, validavg, label='valid scores', marker = '.', c = 'g')

plt.show()

reg = RandomForestRegressor(criterion = 'mae', n_estimators = 3)



score = cross_val_score(reg,X,y,scoring = mae_scoring)

score
