# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# ignore warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/heart.csv")

data.info() # to see type of the features





data.head() # to see features and target variable (1: Male  | 0: Famale)

data.describe()

# KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'sex'], data.loc[:,'sex']

knn.fit(x,y)

prediction = knn.predict(x)

print('Prediction: {}'.format(prediction))
# train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.5,random_state = 1)

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'sex'], data.loc[:,'sex']

knn.fit(x_train,y_train)

y_prediction = knn.predict(x_test)

#print('Prediction: {}'.format(prediction))

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy
# Model complexity

neig = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('-value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_prediction)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
data.columns




x = np.array(data.loc[:,'trestbps']).reshape(-1,1)

y = np.array(data.loc[:,'chol']).reshape(-1,1)

# Scatter

plt.figure(figsize=[10,10])

plt.scatter(x=x,y=y)

plt.xlabel('trestbps')

plt.ylabel('chol')

plt.show()
# LinearRegression

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

# Predict space

predict_space = np.linspace(min(x), max(x)).reshape(-1,1)

# Fit

reg.fit(x,y)

# Predict

predicted = reg.predict(predict_space)

# R^2 

print('R^2 score: ',reg.score(x, y))

# Plot regression line and scatter

plt.plot(predict_space, predicted, color='black', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('trestbps')

plt.ylabel('chol')

plt.show()
# CV

from sklearn.model_selection import cross_val_score

reg = LinearRegression()

k = 5

cv_result = cross_val_score(reg,x,y,cv=k) # uses R^2 as score 

print('CV Scores: ',cv_result)

print('CV scores average: ',np.sum(cv_result)/k)
# Ridge

from sklearn.linear_model import Ridge

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2, test_size = 0.3)

ridge = Ridge(alpha = 0.1, normalize = True)

ridge.fit(x_train,y_train)

ridge_predict = ridge.predict(x_test)

print('Ridge score: ',ridge.score(x_test,y_test))
# Lasso

from sklearn.linear_model import Lasso

x = np.array(data.loc[:,['trestbps','chol']])

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 3, test_size = 0.3)

lasso = Lasso(alpha = 0.1, normalize = True)

lasso.fit(x_train,y_train)

ridge_predict = lasso.predict(x_test)

print('Lasso score: ',lasso.score(x_test,y_test))

print('Lasso coefficients: ',lasso.coef_)



# Confusion matrix with random forest

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

x,y = data.loc[:,data.columns != 'sex'], data.loc[:,'sex']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

rf = RandomForestClassifier(random_state = 4)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print('Confusion matrix: \n',cm)

print('Classification report: \n',classification_report(y_test,y_pred))

# visualize with seaborn library

sns.heatmap(cm,annot=True,fmt="d") 

plt.show()
# ROC Curve with logistic regression

from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

# abnormal = 1 and normal = 0



x,y = data.loc[:,(data.columns != 'sex') & (data.columns != 'sex')], data.loc[:,'sex']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

logreg = LogisticRegression()

logreg.fit(x_train,y_train)

y_pred_prob = logreg.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.show()
# grid search cross validation with 1 hyperparameter

from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors': np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV

knn_cv.fit(x,y)# Fit



# Print hyperparameter

print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 

print("Best score: {}".format(knn_cv.best_score_))
# grid search cross validation with 2 hyperparameter

# 1. hyperparameter is C:logistic regression regularization parameter

# 2. penalty l1 or l2

# Hyperparameter grid

param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 12)

logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg,param_grid,cv=3)

logreg_cv.fit(x_train,y_train)



# Print the optimal parameters and best score

print("Tuned hyperparameters : {}".format(logreg_cv.best_params_))

print("Best Accuracy: {}".format(logreg_cv.best_score_))