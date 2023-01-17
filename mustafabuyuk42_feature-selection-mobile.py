import pandas as pd

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

data = pd.read_csv("../input/train.csv")

X = data.iloc[:,0:20]  #independent columns

y = data.iloc[:,-1]    #target column i.e price range

#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
data.head()
X = data.iloc[:,0:20]  #independent columns

y = data.iloc[:,-1]    #target column i.e price range

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

dt_clf2 = DecisionTreeClassifier(min_samples_split=10)

Naive_bayes2 = GaussianNB()

Naive_bayes2.fit(X, y)

score2 = Naive_bayes2.score(X,y)

print("score for naive bayes:",score2)

dt_clf2.fit(X, y)

score_dt2 = dt_clf2.score(X, y)

print("score for decision tree:",score_dt2)


data_ =data.loc[:,["ram","px_height","battery_power","px_width","price_range"]]
data_.head()
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(min_samples_split=10)



Naive_bayes = GaussianNB()

X_feature = data_.iloc[:,0:4] # ONLY 4 FEATURE SUBSETS ARE SELECTED

Y_label = data_.iloc[:,-1]

Naive_bayes.fit(X_feature, Y_label)

score = Naive_bayes.score(X_feature,Y_label)

print("score for naive bayes:",score)

dt_clf.fit(X_feature, Y_label)

score_dt = dt_clf.score(X_feature, Y_label)

print("score for decision tree:",score_dt)
X_feature.head()
import pandas as pd

import numpy as np

data = pd.read_csv("../input/train.csv")

X = data.iloc[:,0:20]  #independent columns

y = data.iloc[:,-1]    #target column i.e price range

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
import pandas as pd

import numpy as np

import seaborn as sns

data = pd.read_csv("../input/train.csv")

X = data.iloc[:,0:20]  #independent columns

y = data.iloc[:,-1]    #target column i.e price range

#get correlations of each features in dataset

corrmat = data.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")