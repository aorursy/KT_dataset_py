# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Load dataset and draw shape.

dataset = pd.read_csv('../input/iris/Iris.csv')

dataset.shape
dataset.head()
dataset.drop('Id', axis = 1, inplace = True)

dataset.describe()
dataset.isnull().values.any()
speciesMap={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}

dataset['Species']=dataset['Species'].map(speciesMap)

dataset.shape
dataset.head
## Correlation

import seaborn as sns

import matplotlib.pyplot as plt

#get correlations of each features in dataset

corrmat = dataset.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
dataset.hist(figsize=(10,10))
from sklearn.model_selection import train_test_split

#(Removed SepalWidth as it doesn't much help in predicting the outcome)

feature_columns = ['SepalLengthCm','PetalLengthCm','PetalWidthCm']

predicted_class = ['Species']

X = dataset[feature_columns].values

y = dataset[predicted_class].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =0 )
# Standardize train and test sets.

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()



X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
X_train


from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression,LogisticRegression, RidgeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
# Create objects of required models.

models = []

models.append(("LR",LinearRegression()))

models.append(("LR",LogisticRegression()))

models.append(("GNB",GaussianNB()))

models.append(("KNN",KNeighborsClassifier()))

models.append(("DecisionTree",DecisionTreeClassifier()))

models.append(("LDA",  LinearDiscriminantAnalysis()))

models.append(("QDA",  QuadraticDiscriminantAnalysis()))

models.append(("AdaBoost", AdaBoostClassifier()))

models.append(("SVM Linear",SVC(kernel="linear")))

models.append(("SVM RBF",SVC(kernel="rbf")))

models.append(("Random Forest",  RandomForestClassifier()))

models.append(("Bagging",BaggingClassifier()))

models.append(("GradientBoosting",GradientBoostingClassifier()))

models.append(("LinearSVC",LinearSVC()))

models.append(("Ridge",RidgeClassifier()))
results = {}

#for name,model in models:

    #kfold = KFold(n_splits=10, random_state=0)

    #cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")

    #results.append(tuple([name,cv_result.mean(),cv_result.std()]))

#results.sort(key=lambda x: x[1], reverse = True)    

#for i in range(len(results)):

    #print('{:20s} {:2.2f} (+/-) {:2.2f} '.format(results[i][0] , results[i][1] * 100, results[i][2] * 100))

for name,model in models:

    scores = cross_val_score(model, X_train, y_train.ravel(), cv=5)

    results[name] = scores
 

for name, scores in results.items():

    print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100*scores.mean(), 100*scores.std() * 2))


import xgboost

clf=xgboost.XGBClassifier()
clf=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.5, gamma=0.0,

              learning_rate=0.1, max_delta_step=0, max_depth=6,

              min_child_weight=7, missing=None, n_estimators=100, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)



clf.fit(X_train, y_train)
yPredict=clf.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cf = confusion_matrix(y_test, yPredict)

print(cf)

print("Accuracy = {0:.2f}".format(accuracy_score(y_test, yPredict)))
# Creating a pickle file for the classifier

import pickle

filename = 'first-Iris.pkl'

pickle.dump(clf, open(filename, 'wb'))