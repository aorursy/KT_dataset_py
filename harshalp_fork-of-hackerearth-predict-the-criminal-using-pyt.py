# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/criminal_train.csv')

test = pd.read_csv('../input/criminal_test.csv')
#checking for missing value

pd.isnull(train)
train = train.dropna()

print(train.shape)

print(list(train.columns))
#correlation

train.corr()

# Kendall Tau correlation

train.corr('kendall')



# Spearman Rank correlation

train.corr('spearman')
train.head()
test.head()
train.isnull().sum()
# check out summary statistics of numeric columns

train.describe()
train.dtypes
sns.countplot(x='Criminal', data=train);
#EDA

sns.countplot(x='IFATHER', data=train);
pd.Categorical(train)
#feature 'criminal' split (faceted) over the feature 'IFATHER'.

sns.factorplot(x='Criminal', col='IFATHER', kind='count', data=train);

sns.factorplot(x='Criminal', col='NRCH17_2', kind='count', data=train);
train.groupby('Criminal').PDEN10.describe()
train.groupby('Criminal').IRHH65_2.describe()
# separating our independent and dependent variable

X = train.drop(['Criminal'], axis=1)

y = train["Criminal"]

print (X.head(1))

print (y.head(1))
#split into train and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = .33, random_state = 1)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
final_test = sc.transform(test)## Necessary modules for creating models. 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix

from sklearn.metrics import confusion_matrix
#logistc regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

logreg_accy = round(accuracy_score(y_pred,y_test), 3)

print (logreg_accy)
print (classification_report(y_test, y_pred, labels=logreg.classes_))

print (confusion_matrix(y_pred, y_test))
accuracy_score(y_pred, y_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
#knn classifier

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(weights="uniform", )

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

knn_accy = round(accuracy_score(y_test, y_pred), 3)

print (knn_accy)
#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_test)

gaussian_accy = round(accuracy_score(y_pred, y_test), 3)

print(gaussian_accy)

#support vector machine



from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

svc_accy = round(accuracy_score(y_pred, y_test), 3)

print(svc_accy)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier



dectree = DecisionTreeClassifier( max_depth=5, 

                                class_weight = 'balanced',

                                min_weight_fraction_leaf = 0.01)

dectree.fit(x_train, y_train)

y_pred = dectree.predict(x_test)

dectree_accy = round(accuracy_score(y_pred, y_test), 3)

print(dectree_accy)
#random Forest

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(n_estimators=100)

#randomforest = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

random_accy = round(accuracy_score(y_pred, y_test), 3)

print (random_accy)

# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gradient = GradientBoostingClassifier()

gradient.fit(x_train, y_train)

y_pred = gradient.predict(x_test)

gradient_accy = round(accuracy_score(y_pred, y_test), 3)

print(gradient_accy)
test_prediction = logreg.predict(final_test)

test.shape

test.head()

test.to_csv( 'sample_submission.csv' , index = False )