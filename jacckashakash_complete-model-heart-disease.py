import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import rcParams

from matplotlib.cm import rainbow

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import plotly.graph_objs as go

import plotly.tools as tls

import os

import gc



import re



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
dataset=pd.read_csv('../input/heart-disease-uci/heart.csv')

dataset.head()
dataset.info()

dataset.describe()

#print(dataset['sex'].head())
print('~> Have not heart disease (target = 0):\n   {}%'.format(100 - round(dataset['target'].mean()*100, 2)))

print('\n~> Have heart disease (target= 1):\n   {}%'.format(round(dataset['target'].mean()*100, 2)))
rcParams['figure.figsize'] = 20, 14

plt.matshow(dataset.corr())

plt.yticks(np.arange(dataset.shape[1]), dataset.columns)

plt.xticks(np.arange(dataset.shape[1]), dataset.columns)

plt.colorbar()
plt.figure(figsize=(6,6))

dataset.groupby("target").count().plot.bar()
dataset.hist()
import seaborn as sns

sns.pairplot(dataset, palette='rainbow')
import seaborn as sns

plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)

sns.violinplot(x = 'target', y = 'chol', data = dataset[0:])

plt.show()
sns.lmplot(x='chol',y='target',data=dataset)
plt.figure(figsize=(5, 5))

unique_variations = dataset['chol'].value_counts()

print('Number of Unique chol:', unique_variations.shape[0])

# the top 10 variations that occured most

print(unique_variations.head(10))

s = sum(unique_variations.values);

h = unique_variations.values/s;

plt.plot(h, label="Histrogram of Variations")

plt.xlabel('Index of a chol')

plt.ylabel('Number of Occurances')

plt.legend()

plt.grid()

plt.show()

c = np.cumsum(h)

print(c)

plt.figure(figsize=(5, 5))

plt.plot(c,label='Cumulative distribution of Variations')

plt.grid()

plt.legend()

plt.show()
rcParams['figure.figsize'] = 8,6

plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green'])

plt.xticks([0, 1])

plt.xlabel('Target Classes')

plt.ylabel('Count')

plt.title('Count of each Target Class')
nan_rows = dataset[dataset.isnull().any(1)]

print (nan_rows)
categorical_feature_mask = dataset.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = dataset.columns[categorical_feature_mask].tolist()

print(categorical_cols)

print("number of categorical features ",len(categorical_cols))
standardScaler = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
y = dataset['target']

X = dataset.drop(['target'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
# This function plots the confusion matrices given y_i, y_i_hat.

def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)

    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    

    A =(((C.T)/(C.sum(axis=1))).T)

    

    

    B =(C/C.sum(axis=0))

  

    plt.figure(figsize=(20,4))

    

    labels = [1,2]

    # representing A in heatmap format

    cmap=sns.light_palette("blue")

    plt.subplot(1, 3, 1)

    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Confusion matrix")

    

    plt.subplot(1, 3, 2)

    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Precision matrix")

    

    plt.subplot(1, 3, 3)

    # representing B in heatmap format

    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Recall matrix")

    

    plt.show()
knn_scores = []

for k in range(1,21):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    knn_classifier.fit(X_train, y_train)

    knn_scores.append(knn_classifier.score(X_test, y_test))
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics.classification import accuracy_score, log_loss

plt.figure(figsize=(8,5))

plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')

for i in range(1,21):

    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))

plt.xticks([i for i in range(1, 21)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')



#############we get k best value that is 8

#now again train our model

clf=KNeighborsClassifier(n_neighbors=8)

clf.fit(X_train,y_train)

clf = CalibratedClassifierCV(clf, method="sigmoid")

clf.fit(X_train, y_train)

predict_y=clf.predict_proba(X_test)

print("The log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



#########Plot confusion atrix

print(len(predict_y))

#print(len(y_test))

plot_confusion_matrix(y_test, clf.predict(X_test))



svc_scores = []

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for i in range(len(kernels)):

    svc_classifier = SVC(kernel = kernels[i])

    svc_classifier.fit(X_train, y_train)

    svc_scores.append(svc_classifier.score(X_test, y_test))


colors = rainbow(np.linspace(0, 1, len(kernels)))

plt.bar(kernels, svc_scores, color = colors)

for i in range(len(kernels)):

    plt.text(i, svc_scores[i], svc_scores[i])

plt.xlabel('Kernels')

plt.ylabel('Scores')

plt.title('Support Vector Classifier scores for different kernels')

##################### with best kernel 





clf=SVC(kernel ='linear')

#clf.fit(X_train,y_train)

clf = CalibratedClassifierCV(clf, method="sigmoid")

clf.fit(X_train, y_train)

predict_y=clf.predict_proba(X_test)

print("The log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



#########Plot confusion atrix

print(len(predict_y))

#print(len(y_test))

plot_confusion_matrix(y_test, clf.predict(X_test))
dt_scores = []

for i in range(1, len(X.columns) + 1):

    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)

    dt_classifier.fit(X_train, y_train)

    dt_scores.append(dt_classifier.score(X_test, y_test))
plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')

for i in range(1, len(X.columns) + 1):

    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))

plt.xticks([i for i in range(1, len(X.columns) + 1)])

plt.xlabel('Max features')

plt.ylabel('Scores')

plt.title('Decision Tree Classifier scores for different number of maximum features')



clf=DecisionTreeClassifier(max_features = 10, random_state = 0)

clf.fit(X_train,y_train)

clf = CalibratedClassifierCV(clf, method="sigmoid")

clf.fit(X_train, y_train)

predict_y=clf.predict_proba(X_test)

print("The log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



#########Plot confusion atrix

print(len(predict_y))

#print(len(y_test))

plot_confusion_matrix(y_test, clf.predict(X_test))
rf_scores = []

estimators = [10, 100, 200, 500, 1000]

for i in estimators:

    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)

    rf_classifier.fit(X_train, y_train)

    rf_scores.append(rf_classifier.score(X_test, y_test))
colors = rainbow(np.linspace(0, 1, len(estimators)))

plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)

for i in range(len(estimators)):

    plt.text(i, rf_scores[i], rf_scores[i])

plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])

plt.xlabel('Number of estimators')

plt.ylabel('Scores')

plt.title('Random Forest Classifier scores for different number of estimators')



clf=RandomForestClassifier(n_estimators = 500, random_state = 0)

clf.fit(X_train,y_train)

clf = CalibratedClassifierCV(clf, method="sigmoid")

clf.fit(X_train, y_train)

predict_y=clf.predict_proba(X_test)

print("The log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



#########Plot confusion atrix

print(len(predict_y))

#print(len(y_test))

plot_confusion_matrix(y_test, clf.predict(X_test))

import xgboost as xgb

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 4



d_train = xgb.DMatrix(X_train, label=y_train)

d_test = xgb.DMatrix(X_test, label=y_test)



watchlist = [(d_train, 'train'), (d_test, 'valid')]



bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=20, verbose_eval=10)



xgdmat = xgb.DMatrix(X_train,y_train)

predict_y = bst.predict(d_test)

print("The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



print(len(predict_y))

#print(len(y_test))

plot_confusion_matrix(y_test, clf.predict(X_test))


alpha = [10 ** x for x in range(-5, 2)] # hyperparam for SGD classifier.







log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(X_train, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(X_train, y_train)

    predict_y = sig_clf.predict_proba(X_test)

    log_error_array.append(log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, log_error_array,c='g')

for i, txt in enumerate(np.round(log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(X_train, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(X_train, y_train)



predict_y = sig_clf.predict_proba(X_train)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(X_test)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

predicted_y =np.argmax(predict_y,axis=1)

print("Total number of data points :", len(predicted_y))

plot_confusion_matrix(y_test, predicted_y)
clf=RandomForestClassifier(n_estimators = 500, random_state = 0)

clf.fit(X_train,y_train)

features = X_train.columns

importances = clf.feature_importances_

indices = (np.argsort(importances))[-25:]

plt.figure(figsize=(10,12))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='r', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
error_rate=np.array([0.3828,0.4158,0.3972,0.5409,0.3803,0.4135])

plt.figure(figsize=(16,5))

print(error_rate)



#plt.scatter(error_rate,range(1,7))

#seed = 7

# prepare models

models = ['LR','XGBOOST','RF','DT','SVM','KNN']

plt.xlabel(models)

plt.plot(error_rate)

lowest_loss=np.argmin(error_rate)

print("lowest logg loss : ",min(error_rate))

print(models[lowest_loss])