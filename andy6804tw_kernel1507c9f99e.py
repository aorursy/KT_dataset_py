# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



iris = pd.read_csv("../input/Iris.csv") #load the dataset



iris.head()



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder



print(iris['Species'].unique())

# generate y feature encoder

ency = LabelEncoder()

ency.fit(['Iris-setosa', 'Iris-versicolor','Iris-virginica'])

irisData=iris

irisData['Type']=ency.transform(irisData['Species'])

irisData.head()
#missing data

miss_sum = irisData.isnull().sum().sort_values(ascending=False)

miss_sum
from pandas.plotting import scatter_matrix

scatter_matrix( irisData[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']],figsize=(10, 10),color='b')
corr = irisData[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].corr()

print(corr)
import seaborn as sns

plt.figure(figsize=(8,8))

sns.heatmap(corr, square=True, annot=True, cmap="RdBu_r") #center=0, cmap="YlGnBu"
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,5), sharey=True)

axes[0].boxplot(irisData['SepalLengthCm'],showmeans=True)

axes[0].set_title('SepalLengthCm')



axes[1].boxplot(irisData['SepalWidthCm'],showmeans=True)

axes[1].set_title('SepalWidthCm')



axes[2].boxplot(irisData['PetalLengthCm'],showmeans=True)

axes[2].set_title('PetalLengthCm')



axes[3].boxplot(irisData['PetalWidthCm'],showmeans=True)

axes[3].set_title('PetalWidthCm')

#IQR = Q3-Q1

IQR = np.percentile(irisData['SepalWidthCm'],75) - np.percentile(irisData['SepalWidthCm'],25)
#outlier = Q3 + 1.5*IQR 

irisData[irisData['SepalWidthCm'] > np.percentile(irisData['SepalWidthCm'],75)+1.5*IQR]
#outlier = Q1 - 1.5*IQR

irisData[irisData['SepalWidthCm'] < np.percentile(irisData['SepalWidthCm'],25)-1.5*IQR]
from sklearn.model_selection import train_test_split



X = irisData[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y = irisData['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics



# create KNN model

knnModel = KNeighborsClassifier(n_neighbors=3)

knnModel.fit(X_train_std, y_train)



print(metrics.classification_report(y_test, knnModel.predict(X_test_std)))

print(metrics.confusion_matrix(y_test, knnModel.predict(X_test_std)))

from sklearn.ensemble import RandomForestClassifier



rfcModel = RandomForestClassifier(n_estimators=500, criterion='gini', max_features='auto', oob_score=True)

rfcModel.fit(X_train, y_train) #不標準化



print("oob_score(accuary):",rfcModel.oob_score_)

print(metrics.classification_report(y_test, rfcModel.predict(X_test)))
from sklearn.naive_bayes import GaussianNB

gnbModel = GaussianNB()

gnbModel.fit(X_train_std, y_train)



print(metrics.classification_report(y_test, gnbModel.predict(X_test_std)))

print(metrics.confusion_matrix(y_test, gnbModel.predict(X_test_std)))
from sklearn.svm import SVC



svcModel = SVC(C=1.0, kernel="rbf", probability=True)

svcModel.fit(X_train_std, y_train)



print(metrics.classification_report(y_test, svcModel.predict(X_test_std)))

print(metrics.confusion_matrix(y_test, svcModel.predict(X_test_std)))
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

lrModel = LogisticRegression()

lrModel.fit(X_train_std,y_train)

print(metrics.classification_report(y_test, lrModel.predict(X_test_std)))

print(metrics.confusion_matrix(y_test, lrModel.predict(X_test_std)))
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

dtModel = DecisionTreeClassifier()

dtModel.fit(X_train_std,y_train)

print(metrics.classification_report(y_test, dtModel.predict(X_test_std)))

print(metrics.confusion_matrix(y_test, dtModel.predict(X_test_std)))
#from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import StackingClassifier

import xgboost as xgb



clf1 = KNeighborsClassifier(n_neighbors=3, weights='uniform')

clf2 = RandomForestClassifier(n_estimators=500, criterion='gini', max_features='auto', oob_score=True)

clf3 = GaussianNB()

clf4 = SVC(C=1.0, kernel="rbf", probability=True)

meta_clf = xgb.XGBClassifier(n_estimators= 2000, max_depth= 4)

stacking_clf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=meta_clf)



clf1.fit(X_train_std, y_train)

clf2.fit(X_train, y_train)

clf3.fit(X_train_std, y_train)

clf4.fit(X_train_std, y_train)

stacking_clf.fit(X_train_std, y_train)



print('KNN Score:',clf1.score(X_test_std, y_test))

print('RF Score:',clf2.score(X_test, y_test))

print('GNB Score:',clf3.score(X_test_std, y_test))

print('SVC Score:',clf4.score(X_test_std, y_test))

print('Stacking Score:',stacking_clf.score(X_test_std, y_test))
import xgboost as xgb



xgbModel = xgb.XGBClassifier(n_estimators= 2000, max_depth= 4).fit(X_train, y_train)



print(metrics.classification_report(y_test, xgbModel.predict(X_test)))

print("Score:", xgbModel.score(X_test, y_test))
print(xgbModel.feature_importances_)
from xgboost import plot_importance

plot_importance(xgbModel, )

plt.show()
pred = xgbModel.predict(X_test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']])

pred