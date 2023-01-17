# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv") 

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train.head()
test.head()
train.info()
train.SalePrice.describe()

plt.hist(train.SalePrice, color='green')

plt.show()
target = np.log(train.SalePrice)

plt.hist(target, color='green')

plt.show()
num_features = train.select_dtypes(include=[np.number])

num_features.dtypes
num_features.head()
corr = num_features.corr()

print (corr['SalePrice'].sort_values(ascending=False)[:7])
train.OverallQual.unique()

qualityPivot = train.pivot_table(index='OverallQual',

                  values='SalePrice', aggfunc=np.median)

qualityPivot
qualityPivot.plot(kind='bar', color='green')

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
plt.scatter(x=train['GrLivArea'], y=target, color='green')

plt.ylabel('Sale Price')

plt.xlabel('Above ground living area square feet')

plt.show()
plt.scatter(x=train['GarageArea'], y=target, color='green')

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
train = train[train['GarageArea'] < 900]

test = test[test['GarageArea'] < 900]

plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice), color='green')

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

testdata = test.select_dtypes(include=[np.number]).interpolate().dropna()
# testdata = test.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0)

X = data.iloc[:, :-1].values

y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(11)
from sklearn import preprocessing
# lab_enc = preprocessing.LabelEncoder()

# y_train_encoded = lab_enc.fit_transform(y_train)

# y_test_encoded = lab_enc.fit_transform(y_test)

knn.fit(X_train, y_train)
y_preds = knn.predict(X_test)
print('Accuracy is:', accuracy_score(y_test, y_preds))

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy is: ', logreg.score(X_test, y_test))
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy is:",accuracy_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

rf_clf=RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train,y_train)
y_pred=rf_clf.predict(X_test)
print("Accuracy is:",accuracy_score(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Accuracy is:",metrics.accuracy_score(y_test, y_pred))
from sklearn.svm import SVC

svclassifier = SVC()
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy is:",metrics.accuracy_score(y_test, y_pred))
import xgboost as xgb

xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed=123 )
xg_cl.fit(X_train, y_train)
y_pred = xg_cl.predict(X_test)
print("Accuracy is:",metrics.accuracy_score(y_test, y_pred))
for i,j in zip(y_test, y_preds):

    print('currentPrice: ', i, ' predictedPrice: ',j)