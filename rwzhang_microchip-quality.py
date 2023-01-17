import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import os
data = np.loadtxt('../input/ex2data2.txt', delimiter = ',')
x = data[:,:2]
y = data[:,2].astype(int)
color = ['blue', 'red']
plt.figure(figsize = (8, 8))
plt.xlabel('test 1 score')
plt.ylabel('test 2 score')
plt.scatter(x[y==0,0], x[y==0,1], c=color[0], marker='+', s=30)
plt.scatter(x[y==1,0], x[y==1,1], c=color[1])
plt.legend(('rejected', 'accepted'))
plt.show()
np.random.seed(2018)
train = np.random.choice([True, False], len(data), replace=True, p=[0.5, 0.5])
x_train = data[train,:2]
y_train = data[train,2].astype(int)
x_test = data[~train,:2]
y_test = data[~train,2].astype(int)
poly2=preprocessing.PolynomialFeatures(2)
poly3=preprocessing.PolynomialFeatures(3)
poly4=preprocessing.PolynomialFeatures(4)
x2=poly2.fit_transform(x)
x3=poly3.fit_transform(x)
x4=poly4.fit_transform(x)
x_train_2=x2[train,:]
x_test_2=x2[~train,:]
x_train_3=x3[train,:]
x_test_3=x3[~train,:]
x_train_4=x4[train,:]
x_test_4=x4[~train,:]
regr1=linear_model.LogisticRegression(C=100)
regr2=linear_model.LogisticRegression(C=100)
regr3=linear_model.LogisticRegression(C=100)
regr4=linear_model.LogisticRegression(C=100)
regr1.fit(x_train ,y_train)
regr2.fit(x_train_2,y_train)
regr3.fit(x_train_3,y_train)
regr4.fit(x_train_4,y_train)
print('LR-model score:',regr1.score(x_test,y_test))
print('2-degree-polynomialFeatures-LR-model score:',regr2.score(x_test_2,y_test))
print('3-degree-polynomialFeatures-LR-model score:',regr3.score(x_test_3,y_test))
print('4-degree-polynomialFeatures-LR-model score:',regr4.score(x_test_4,y_test))
from sklearn.metrics import recall_score
prediction_2 = regr2.predict(x_test_2)
print('recall-score of 2-degree-polynomialFeatures-LR-model:',recall_score(y_test, prediction_2))
from sklearn.metrics import precision_score
print('precision-score of 2-degree-polynomialFeatures-LR-model:',precision_score(y_test, prediction_2))
gnb1=GaussianNB()
gnb2=GaussianNB()
gnb3=GaussianNB()
gnb4=GaussianNB()
gnb1.fit(x_train ,y_train)
gnb2.fit(x_train_2 ,y_train)
gnb3.fit(x_train_3 ,y_train)
gnb4.fit(x_train_4 ,y_train)
print('GNB-model score:',gnb1.score(x_test,y_test))
print('2-degree-polynomialFeatures-GNB-model score:',gnb2.score(x_test_2,y_test))
print('3-degree-polynomialFeatures-GNB-model score:',gnb3.score(x_test_3,y_test))
print('4-degree-polynomialFeatures-GNB-model score:',gnb4.score(x_test_4,y_test))
from sklearn.metrics import recall_score
gnb3_prediction = gnb3.predict(x_test_3)
print('recall-score of 3-degree-polynomialFeatures-GNB-model:',recall_score(y_test, gnb3_prediction))
from sklearn.metrics import precision_score
print('precision-score of 3-degree-polynomialFeatures-GNB-model:',precision_score(y_test,gnb3_prediction))