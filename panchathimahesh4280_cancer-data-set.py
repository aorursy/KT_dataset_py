# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

cancer = pd.read_csv("../input/cancer.csv")
cancer.head()
cancer.shape
types = cancer.dtypes

types
cancer.isnull().sum()
cancer["radius_mean"].fillna(cancer["radius_mean"].mean(),inplace=True)
cancer.info()
cancer.drop(["id"],axis=1,inplace=True)
cancer.head()
cancer['diagnosis'].replace(['M','B'],[0,1],inplace=True)
cancer.head()
#dividig the data set independent variable and depended variable

#my target variable is diagnosis



X = cancer.iloc[1:31]

y = cancer.iloc[0]
X.head()
y.head()
X.describe()
## it used to histograms 

from matplotlib import pyplot

cancer.hist(figsize=(12,10))

pyplot.show()
correlations = cancer.corr()

fig = pyplot.figure(figsize=(12,10))

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

pyplot.show()
import pandas  

from pandas.plotting import scatter_matrix



dataCorr = cancer.corr()

pandas.plotting.scatter_matrix(dataCorr,figsize=(12,10))

pyplot.show()
from sklearn.preprocessing import MinMaxScaler

from numpy import set_printoptions

array = cancer.values

X = array[:,1:31]

y = array[:,0]



scaler = MinMaxScaler(feature_range=(0,1))

rescaledX = scaler.fit_transform(X)

print(rescaledX[0:5,:])
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler().fit(X)

scaled_X = scalar.transform(X)

print(scaled_X)


import warnings

warnings.filterwarnings('ignore')



from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

# feature extraction

model = LogisticRegression()

rfe = RFE(model, 3)

fit = rfe.fit(X, y)

print(fit.n_features_)

print(fit.support_)  # Selected Features:

print(fit.ranking_)  # Feature Ranking
# feature extraction model

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(X, y)

print(model.feature_importances_)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



test_size = 0.33

seed = 7

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

model = LogisticRegression()

model.fit(X_train, y_train)

result = model.score(X_test, y_test)

print(result*100.0)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()

model.fit(X_train,y_train)

pred=model.predict(X_test)

pred
y_pred=model.predict(X_test)

y_pred
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,pred)*100,"%")
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

model.fit(X_train,y_train)
pred=model.predict(X_test)

pred
y_pred=model.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,pred)*100,"%")
from sklearn.metrics import confusion_matrix



print(confusion_matrix(y_test, y_pred)) 
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt  

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_pred, y_test)

roc_auc = auc(fpr, tpr)



#plt.figure()

plt.plot(fpr, tpr, color='darkorange', 

         label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
pipe_lr = make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))

pipe_lr.fit(X_train,y_train)
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', random_state=1))

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, 

                                                          X=X_train, y=y_train, 

                                                train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')

plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')

plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()

plt.xlabel('Number of training samples')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')

plt.ylim([0.8, 1.0])

plt.show()
from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(estimator=pipe_lr,X=X_train,y=y_train,param_name='logisticregression__C',param_range=param_range,cv=10)

train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,color='blue', marker='o', markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15,color='blue')

plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')

plt.fill_between(param_range,test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()

plt.xscale('log')

plt.legend(loc='lower right')

plt.xlabel('Parameter C')

plt.ylabel('Accuracy')

plt.ylim([0.8, 1.03])

plt.show()
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']}, {'svc__C': param_range, 'svc__gamma': param_range,'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

gs = gs.fit(X_train, y_train)

print(gs.best_score_)

print(gs.best_params_)
clf = gs.best_estimator_

clf.fit(X_train, y_train)

print('Test accuracy: %.3f' % clf.score(X_test, y_test))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
#SAVING MODEL

from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.externals.joblib import dump

from sklearn.externals.joblib import load
data = pd.read_csv("../input/cancer.csv")

#print(data.shape)



#array = data.values



X = array[:,1:31]

Y = array[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

# Fit the model on 33%

model = LogisticRegression()

model.fit(X_train, y_train)



filename = 'finalized_model.sav'

dump(model, filename)



loaded_model = load(filename)

result = loaded_model.score(X_test, y_test)

print(result)