import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

starbucks = pd.read_csv('/kaggle/input/starbucks-customer-retention-malaysia-survey/Starbucks satisfactory survey encode cleaned.csv')

starbucks.columns 
starbucks.head()
print("Starbucks data set dimensions : {}".format(starbucks.shape))
# review the target class size

# 0 yes will continue buying starbucks, 1 no will not continue buying starbucks

starbucks.groupby('loyal').size()
#delete Id column since we won't need this

starbucks = starbucks.drop('Id', axis=1)
#starbucks.groupby('loyal').hist(figsize=(20, 20))
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



X = starbucks.iloc[:,0:21]  #independent columns

y = starbucks.iloc[:,-1]    #target column i.e price range



#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
feature_names = ['priceRate', 'membershipCard', 'spendPurchase', 'productRate', 'status', 'visitNo', 'timeSpend', 'ambianceRate','location','method']

X = starbucks[feature_names]

y = starbucks.loyal
#import libraries

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
#initialize

models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('SVC', SVC()))

models.append(('LR', LogisticRegression()))

models.append(('DT', DecisionTreeClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))

models.append(('GB', GradientBoostingClassifier()))
#evaluation method - train test split

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = starbucks.loyal, random_state=0)
names = []

scores = []

for name, model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    scores.append(accuracy_score(y_test, y_pred))

    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})

print(tr_split)