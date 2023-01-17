import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import math

import numpy as np

import pandas as pd

import random



# importing sklearn libraries

from sklearn import neural_network, linear_model, preprocessing, svm, tree

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.naive_bayes import GaussianNB



# importing keras libraries

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor



import warnings



# supressing the warning on the usage of Linear Regression model

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

%matplotlib inline

pd.set_option("display.max_rows", 1000)    #設定最大能顯示1000rows

pd.set_option("display.max_columns", 1000) #設定最大能顯示1000columns

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 指定默認字形：解決plot不能顯示中文問題

mpl.rcParams['axes.unicode_minus'] = False
forest_fires = pd.read_csv("/kaggle/input/forest-fires-data-set/forestfires.csv")

forest_fires.head(10)
forest_fires.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)

forest_fires.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
forest_fires.describe()
forest_fires['area'].values[forest_fires['area'].values > 0] = 1

forest_fires = forest_fires.rename(columns={'area': 'label'})

forest_fires
forest_fires.corr()
forest_fires.corr()['label'].sort_values(ascending=False)
# Logistic Regression Classification

import pandas as pd

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



scaler = StandardScaler()

scaler.fit(forest_fires.drop('label',axis=1))

scaled_features = scaler.transform(forest_fires.drop('label',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=forest_fires.columns[:-1])

df_feat.head()



X = df_feat

y = forest_fires['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)



logmodel = LogisticRegression(solver='liblinear')

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



from sklearn import metrics

logmodel.score(X_train,y_train)

print("Accuracy:",metrics.accuracy_score(y_test, predictions))

print("Precision:",metrics.precision_score(y_test, predictions))

print("Recall:",metrics.recall_score(y_test, predictions))



#使用混淆矩陣

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
classes={0:'safe',1:'On Fire'}

x_new=[[1, 4, 9 ,1 ,91.5, 130.1, 807.1, 7.5, 21.3, 35, 2.2, 0]]

y_predict=logmodel.predict(x_new)

print(classes[y_predict[0]])
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(forest_fires.drop('label',axis=1))

scaled_features = scaler.transform(forest_fires.drop('label',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=forest_fires.columns[:-1])

df_feat.head()
from sklearn.model_selection import train_test_split

X = df_feat

y = forest_fires['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)
from sklearn.neighbors import KNeighborsClassifier

#從k值=1開始測試

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)
error_rate = []



for i in range(1,60):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))





plt.figure(figsize=(10,6))

plt.plot(range(1,60),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

print('WITH K=7')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
knn = KNeighborsClassifier(n_neighbors=17)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

print('WITH K=17')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
knn.score(X_test, y_test)
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, pred))

print("Precision:",metrics.precision_score(y_test, pred))

print("Recall:",metrics.recall_score(y_test, pred))
classes={0:'safe',1:'On Fire'}

x_new=[[1, 4, 9 ,1 ,91.5, 130.1, 807.1, 7.5, 21.3, 35, 2.2, 0]]

y_predict=knn.predict(x_new)

print(classes[y_predict[0]])
# Support Vector Machine

from sklearn import metrics

from sklearn.svm import SVC

# fit a SVM model to the data



X = forest_fires.drop('label', axis=1)

y = forest_fires['label']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=101)



svc = SVC()

svc.fit(X_train, y_train)

# make predictions

prediction = svc.predict(X_test)

# summarize the fit of the model

print(metrics.classification_report(y_test, prediction))

print(metrics.confusion_matrix(y_test, prediction))



print("Accuracy:",metrics.accuracy_score(y_test, prediction))

print("Precision:",metrics.precision_score(y_test, prediction))

print("Recall:",metrics.recall_score(y_test, prediction))



classes={0:'safe',1:'On Fire'}

x_new=[[1, 4, 9 ,1 ,91.5, 130.1, 807.1, 7.5, 21.3, 35, 2.2, 0]]

y_predict=svc.predict(x_new)

print(classes[y_predict[0]])
# Decision Tree Classifier

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier



X = forest_fires.drop('label', axis=1)

y = forest_fires['label']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=101)



d_tree = DecisionTreeClassifier()

d_tree.fit(X_train, y_train)



# make predictions

predicted = d_tree.predict(X_test)

# summarize the fit of the model

print(metrics.classification_report(y_test, predicted))

print(metrics.confusion_matrix(y_test, predicted))



print("Accuracy:",metrics.accuracy_score(y_test, predicted))

print("Precision:",metrics.precision_score(y_test, predicted))

print("Recall:",metrics.recall_score(y_test, predicted))



classes={0:'safe',1:'On Fire'}

x_new=[[1, 4, 9 ,1 ,91.5, 130.1, 807.1, 7.5, 21.3, 35, 2.2, 0]]

y_predict=d_tree.predict(x_new)

print(classes[y_predict[0]])
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

X = forest_fires.drop('label', axis=1)

y = forest_fires['label']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=101)



# fit a k-nearest neighbor model to the data

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)



print(knn)

# make predictions

predicted = knn.predict(X_test)

# summarize the fit of the model

print(metrics.classification_report(y_test, predicted))

print(metrics.confusion_matrix(y_test, predicted))



print("Accuracy:",metrics.accuracy_score(y_test, predicted))

print("Precision:",metrics.precision_score(y_test, predicted))

print("Recall:",metrics.recall_score(y_test, predicted))



classes={0:'safe',1:'On Fire'}

x_new=[[1, 4, 9 ,1 ,91.5, 130.1, 807.1, 7.5, 21.3, 35, 2.2, 0]]

y_predict=knn.predict(x_new)

print(classes[y_predict[0]])
# Gaussian Naive Bayes

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB



X = forest_fires.drop('label', axis=1)

y = forest_fires['label']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=101)



# fit a Naive Bayes model to the data

G_NB = GaussianNB()

G_NB.fit(X_train,y_train)

print(G_NB)

# make predictions



predict = G_NB.predict(X_test)

# summarize the fit of the model

print(metrics.classification_report(y_test, predict))

print(metrics.confusion_matrix(y_test, predict))



print("Accuracy:",metrics.accuracy_score(y_test, predict))

print("Precision:",metrics.precision_score(y_test, predict))

print("Recall:",metrics.recall_score(y_test, predict))



classes={0:'safe',1:'On Fire'}

x_new=[[1, 4, 9 ,1 ,91.5, 130.1, 807.1, 7.5, 21.3, 35, 2.2, 0]]

y_predict=G_NB.predict(x_new)

print(classes[y_predict[0]])
# Compare Algorithms

import pandas

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



# load dataset

X = forest_fires.drop('label', axis=1)

y = forest_fires['label']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=101)



# prepare configuration for cross validation test harness

seed = 7

# prepare models

models = []

models.append(('LR', LogisticRegression(max_iter=5000)))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('MLP', MLPClassifier()))

models.append(('GradientBoost',GradientBoostingClassifier()))

models.append(('AdaBoost',AdaBoostClassifier()))

models.append(('Bagging',BaggingClassifier()))

models.append(('RandomForest',RandomForestClassifier()))

models.append(('ExtraTrees',ExtraTreesClassifier()))



# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10)

    cv_results = model_selection.cross_val_score(model, X, y,   cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)