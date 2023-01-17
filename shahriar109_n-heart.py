# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score
dataset = pd.read_csv('../input/heartdataset/Data_Train.csv')

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 13].values

Z = dataset.drop(['Column14'], axis = 1)
dataset
#from sklearn.preprocessing import Imputer

from sklearn.impute import SimpleImputer

#imputer = SimpleImputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = SimpleImputer(missing_values =np.nan, strategy='mean')

imputer = imputer.fit(x[:, 0:13])

x[:, 0:13] = imputer.transform(x[:, 0:13])
x[0]
import matplotlib.pyplot as plt

from matplotlib import rcParams

from matplotlib.cm import rainbow

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 20, 14

plt.matshow(dataset.corr())

plt.yticks(np.arange(dataset.shape[1]), dataset.columns)

plt.xticks(np.arange(dataset.shape[1]), dataset.columns)

plt.colorbar()
rcParams['figure.figsize'] = 8,6

plt.bar(dataset['Column14'].unique(), dataset['Column14'].value_counts(), color = ['red', 'green'])

plt.xticks([0, 1])

plt.xlabel('Target Classes')

plt.ylabel('Count')

plt.title('Count of each Target Class')
from sklearn.preprocessing import StandardScaler

x_std = StandardScaler().fit_transform(x)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_std,y,test_size=0.3,random_state=1)
from sklearn.ensemble import RandomForestClassifier



rf_scores = []

estimators = [10, 100, 200, 500, 1000]

for i in estimators:

    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    rf_scores.append(rf_classifier.score(X_test, y_test))
colors = rainbow(np.linspace(0, 1, len(estimators)))

plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)

for i in range(len(estimators)):

    plt.text(i, rf_scores[i], rf_scores[i])

plt.xticks(ticks = [i for i in range(len(estimators))], labels = [5,10,20,30,50])

plt.xlabel('Number of estimators')

plt.ylabel('Scores')

plt.title('Random Forest Classifier scores for different number of estimators')
print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[3]*100, [ 30]))
# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)
from sklearn.svm import SVC

svc_scores = []

precision = []

recall = []

f1 = []

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for i in range(len(kernels)):

    svc_classifier = SVC(kernel = kernels[i])

    svc_classifier.fit(X_train, y_train)

    svc_scores.append(svc_classifier.score(X_test, y_test))

    precision.append(precision_score(y_test, y_pred))

    recall.append(recall_score(y_test, y_pred))

    f1.append(f1_score(y_test, y_pred))
colors = rainbow(np.linspace(0, 1, len(kernels)))

plt.bar(kernels, svc_scores, color = colors)

for i in range(len(kernels)):

    plt.text(i, svc_scores[i], svc_scores[i])

plt.xlabel('Kernels')

plt.ylabel('Scores')

plt.title('Support Vector Classifier scores for different kernels')
print("The score for Support Vector Classifier is {}% with {} kernel.".format(svc_scores[2]*100, 'rbf'))
print("For Linear Kernel:")

print("Precision: {}%".format(precision[0]*100))

print("Recall: {}%".format(recall[0]*100))

print("F1 Score:  {}%".format(f1[0]*100))



print("\nFor Poly Kernel:")

print("Precision: {}%".format(precision[1]*100))

print("Recall: {}%".format(recall[1]*100))

print("F1 Score:  {}%".format(f1[1]*100))



print("\nFor RBF Kernel:")

print("Precision: {}%".format(precision[2]*100))

print("Recall: {}%".format(recall[2]*100))

print("F1 Score:  {}%".format(f1[2]*100))



print("\nFor Sigmoid Kernel:")

print("Precision: {}%".format(precision[3]*100))

print("Recall: {}%".format(recall[3]*100))

print("F1 Score:  {}%".format(f1[3]*100))





# precision tp / (tp + fp)

#precision = precision_score(y_test, y_pred)

#print('Precision: %f' % precision)



# recall: tp / (tp + fn)

#recall = recall_score(y_test, y_pred)

#print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

#f1 = f1_score(y_test, y_pred)

#print('F1 score: %f' % f1)
from sklearn.neighbors import KNeighborsClassifier

knn_scores = []

for k in range(1,30):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    knn_classifier.fit(X_train, y_train)

    y_pred = knn_classifier.predict(X_test)

    knn_scores.append(knn_classifier.score(X_test, y_test))
plt.plot([k for k in range(1, 30)], knn_scores, color = 'red')

for i in range(1,30):

    plt.text(i, knn_scores[i-1], (i))

plt.xticks([i for i in range(1, 30)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')
print("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[12]*100, 13))
# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)
from sklearn.metrics import accuracy_score

from sklearn import metrics
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)
from sklearn.tree import DecisionTreeClassifier

dt_scores = []

for i in range(1, 12):

    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = i)

    dt_classifier.fit(X_train, y_train)

    y_pred =  dt_classifier.predict(X_test)

    dt_scores.append(dt_classifier.score(X_test, y_test))
plt.plot([i for i in range(1, 12)], dt_scores, color = 'green')

for i in range(1, 12):

    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))

plt.xticks([i for i in range(1,12)])

plt.xlabel('Max features')

plt.ylabel('Scores')

plt.title('Decision Tree Classifier scores for different number of maximum features')
print("The score for Decision Tree Classifier is {}% with {} maximum features.".format(dt_scores[7]*100, 8))
#print("The score for Decision Tree Classifier is {}% with only {} features.".format(dt_scores[1]*100, 2))
# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)
from keras.models import Sequential

from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 13))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

training = classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 150)

#testing = classifier.fit(X_test, y_test, batch_size = 10, nb_epoch = 200)
from sklearn.metrics import accuracy_score 

y_pred = classifier.predict(X_test)

print("Accuracy: ",metrics.accuracy_score(y_test, y_pred.round())*100,"%")
# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred.round())

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred.round())

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred.round())

print('F1 score: %f' % f1)
from sklearn.neural_network import MLPClassifier

mlprc_regular = MLPClassifier()

mlprc_regular.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)
from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression()

lgr.fit(X_train, y_train)

y_pred = lgr.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)