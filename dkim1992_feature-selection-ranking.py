# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/voice.csv')
data.sample(5)
sns.heatmap(data.corr())
data.info()
data.isnull().any()
data.label.value_counts()
# convert class label into binary number, 1: female, 0: male

data.label = np.where(data.label.values == 'female', 1, 0)

data.label.value_counts()
X = data.drop('label', axis = 1)

y = data.label
X.shape, y.shape
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



clf = RandomForestClassifier(n_estimators = 50, max_depth = 4)



scores = []

num_features = len(X.columns)

for i in range(num_features):

    col = X.columns[i]

    score = np.mean(cross_val_score(clf, X[col].values.reshape(-1,1), y, cv=10))

    scores.append((int(score*100), col))



print(sorted(scores, reverse = True))



def print_best_worst (scores):

    scores = sorted(scores, reverse = True)

    

    print("The 5 best features selected by this method are :")

    for i in range(5):

        print(scores[i][1])

    

    print ("The 5 worst features selected by this method are :")

    for i in range(5):

        print(scores[len(scores)-1-i][1])
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

# {'logisticregression__C': [1, 10, 100, 1000]

param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100]}

pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l1'))     

grid = GridSearchCV(pipe, param_grid, cv = 10)

grid.fit(X, y)

print(grid.best_params_)
X_scaled = StandardScaler().fit_transform(X)

clf = LogisticRegression(penalty = 'l1', C = 0.1)

clf.fit(X_scaled,y)
zero_feat = []

nonzero_feat = []

# type(clf.coef_)

for i in range(num_features):

    coef = clf.coef_[0,i]

    if coef == 0:

        zero_feat.append(X.columns[i])

    else:

        nonzero_feat.append((coef, X.columns[i]))

        

print ('Features that have coeffcient of 0 are: ', zero_feat)

print ('Features that have non-zero coefficients are:')

print (sorted(nonzero_feat, reverse = True))

        
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100]}

pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l2'))     

grid = GridSearchCV(pipe, param_grid, cv = 10)

grid.fit(X, y)

print(grid.best_params_)
X_scaled = StandardScaler().fit_transform(X)

clf = LogisticRegression(penalty = 'l2', C = 1)

clf.fit(X_scaled,y)
abs_feat = []

for i in range(num_features):

    coef = clf.coef_[0,i]

    abs_feat.append((abs(coef), X.columns[i]))

        

print (sorted(abs_feat, reverse = True))
print_best_worst(abs_feat)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2, mutual_info_classif



test = SelectKBest(score_func=chi2, k=2)

test.fit(X, y)
scores = []

for i in range(num_features):

    score = test.scores_[i]

    scores.append((score, X.columns[i]))

        

print (sorted(scores, reverse = True))
print_best_worst(scores)
test = SelectKBest(score_func = mutual_info_classif, k=2)

test.fit(X, y)
scores = []

for i in range(num_features):

    score = test.scores_[i]

    scores.append((score, X.columns[i]))

        

print (sorted(scores, reverse = True))
print_best_worst(scores)
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression





rfe = RFE(LogisticRegression(), n_features_to_select=1)

rfe.fit(X,y)
scores = []

for i in range(num_features):

    scores.append((rfe.ranking_[i],X.columns[i]))

    

print_best_worst(scores)
from sklearn.ensemble import RandomForestClassifier



rfe = RFE(RandomForestClassifier(), n_features_to_select = 1)

rfe.fit(X,y)
scores = []

for i in range(num_features):

    scores.append((rfe.ranking_[i],X.columns[i]))

    

print_best_worst(scores)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()

clf.fit(X,y)

scores = []

for i in range(num_features):

    scores.append((clf.feature_importances_[i],X.columns[i]))

        

print_best_worst(scores)
from sklearn.model_selection import cross_val_score



scores = []

clf = RandomForestClassifier()

score_normal = np.mean(cross_val_score(clf, X, y, cv = 10))



# X_shuffled = X.copy()

# np.random.shuffle(X_shuffled[X.columns[i]])



# X_shuffled.meanfreq

for i in range(num_features):

    X_shuffled = X.copy()

    scores_shuffle = []

    for j in range(3):

        np.random.seed(j*3)

        np.random.shuffle(X_shuffled[X.columns[i]])

        score = np.mean(cross_val_score(clf, X_shuffled, y, cv = 10))

        scores_shuffle.append(score)

        

    scores.append((score_normal - np.mean(scores_shuffle), X.columns[i]))

    
scores,score_normal
print_best_worst(scores)
from sklearn.linear_model import RandomizedLogisticRegression



clf = RandomizedLogisticRegression()

clf.fit(X,y)

zero_feat = []

nonzero_feat = []

# type(clf.coef_)

for i in range(num_features):

    coef = clf.scores_[i]

    if coef == 0:

        zero_feat.append(X.columns[i])

    else:

        nonzero_feat.append((coef, X.columns[i]))

        

print ('Features that have coeffcient of 0 are: ', zero_feat)

print ('Features that have non-zero coefficients are:')

print (sorted(nonzero_feat, reverse = True))