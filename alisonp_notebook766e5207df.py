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


from sklearn.feature_selection import SelectKBest

from sklearn.decomposition import PCA

from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

# load data

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = pd.read_csv('../input/Indian_Diabetes.csv', names=names)

#dataframe.head()

dfcompare = dataframe.drop(['class'], axis =1)

print (dfcompare.head())
array = dataframe.values

X = array[:,0:8]

Y = array[:,8]

print ("feature extraction using SelectKBest")

test = SelectKBest(score_func=chi2, k=4)

fit = test.fit(X, Y)

# summarize scores

np.set_printoptions(precision=3)

print(fit.scores_)

features = fit.transform(X)

# summarize selected features

print(features[0:5,:])

print (dfcompare.head())

print("")

print ("***********************************")

print("")

print ("feature extraction: Extra Trees Classfier")

model = ExtraTreesClassifier()

model.fit(X, Y)

print(model.feature_importances_)

print (dfcompare.head())



print("")

print ("***********************************")

print("")

print ("feature extraction: PCA")

pca = PCA(n_components=3)

fit = pca.fit(X)

# summarize components

print("Explained Variance: :",fit.explained_variance_ratio_)

print(fit.components_)

print (dfcompare.head())



print("")

print ("***********************************")

print("")

print ("feature extraction: RFE")

model = LogisticRegression()

rfe = RFE(model)

fit = rfe.fit(X, Y)

print("Num Features: ",fit.n_features_)

print("Selected Features: ",fit.support_)

print("Feature Ranking: ",fit.ranking_)

print (dfcompare.head())