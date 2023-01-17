#pip install scikit-learn==0.22.1
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input/heartdata"))



# Any results you write to the current directory are saved as output.
from sklearn import metrics 

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

dataset = pd.read_csv('../input/heartdata/heart.csv')

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 13].values

                
dataset
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(x[:, 0:13])

x[:, 0:13] = imputer.transform(x[:, 0:13])
x[0]
import matplotlib.pyplot as plt

from matplotlib import rcParams

from matplotlib.cm import rainbow

%matplotlib inline

rcParams['figure.figsize'] = 20,14

plt.matshow(dataset.corr())

plt.yticks(np.arange(dataset.shape[1]), dataset.columns)

plt.xticks(np.arange(dataset.shape[1]), dataset.columns)

plt.colorbar()
rcParams['figure.figsize'] = 8,6

plt.bar(dataset['Column14'].unique(), dataset['Column14'].value_counts(), color = ['red' , 'green'])

plt.xticks([0, 1])

plt.xlabel('Target Classes')

plt.ylabel('Count')

plt.title('Count of each Target Class')

x_c,y_c= dataset['Column14'].value_counts()

print("Count of heart disease: ", x_c)

x_c,y_c= dataset['Column14'].value_counts()

print("Count of without heart disease: ", y_c)
from sklearn.preprocessing import StandardScaler

x_std = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA

pca = PCA().fit(x_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,14)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variances')
from sklearn.decomposition import PCA

sklearn_pca = PCA(n_components=11)

pca_heart = sklearn_pca.fit_transform(x_std)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(pca_heart,y,test_size=0.3,random_state=1)
from sklearn import svm

clf=svm.SVC(kernel='linear') 

clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy: %.2f%%" % (accuracy*100.0))
confusion_matrix(y_test,y_pred)
#precision tp / (tp + fp)

precision=precision_score(y_test, y_pred)

print("Precision: %.2f%%" % (precision * 100.0))



# recall: tp / (tp + fn)

recall=recall_score(y_test, y_pred)

print("recall: %.2f%%" % (recall * 100.0))





# f1: 2 tp / (2 tp + fp + fn)

f1=f1_score(y_test, y_pred)

print("F1: %.2f%%" % (f1 * 100.0))
from sklearn import svm

clf=svm.SVC(kernel='rbf')

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print("Accuracy: %.2f%%" %(accuracy*100.0))
confusion_matrix(y_test,y_pred)
#precision tp / (tp + fp)

precision=precision_score(y_test, y_pred)

print("Precision: %.2f%%" % (precision * 100.0))



# recall: tp / (tp + fn)

recall=recall_score(y_test, y_pred)

print("recall: %.2f%%" % (recall * 100.0))





# f1: 2 tp / (2 tp + fp + fn)

f1=f1_score(y_test, y_pred)

print("F1: %.2f%%" % (f1 * 100.0))
from sklearn import svm

clf=svm.SVC(kernel='poly') 

clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy: %.2f%%" % (accuracy*100.0))
confusion_matrix(y_test,y_pred)

#precision tp / (tp + fp)

precision=precision_score(y_test, y_pred)

print("Precision: %.2f%%" % (precision * 100.0))



# recall: tp / (tp + fn)

recall=recall_score(y_test, y_pred)

print("recall: %.2f%%" % (recall * 100.0))





# f1: 2 tp / (2 tp + fp + fn)

f1=f1_score(y_test, y_pred)

print("F1: %.2f%%" % (f1 * 100.0))
from sklearn import svm

clf=svm.SVC(kernel='sigmoid') 

clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy: %.2f%%" % (accuracy*100.0))
confusion_matrix(y_test,y_pred)
#precision tp / (tp + fp)

precision=precision_score(y_test, y_pred)

print("Precision: %.2f%%" % (precision * 100.0))



# recall: tp / (tp + fn)

recall=recall_score(y_test, y_pred)

print("recall: %.2f%%" % (recall * 100.0))





# f1: 2 tp / (2 tp + fp + fn)

f1=f1_score(y_test, y_pred)

print("F1: %.2f%%" % (f1 * 100.0))
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=700,random_state=0)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy: %.2f%%" % (accuracy*100.0))
confusion_matrix(y_test,y_pred)
#precision tp / (tp + fp)

precision=precision_score(y_test, y_pred)

print("Precision: %.2f%%" % (precision * 100.0))



# recall: tp / (tp + fn)

recall=recall_score(y_test, y_pred)

print("recall: %.2f%%" % (recall * 100.0))





# f1: 2 tp / (2 tp + fp + fn)

f1=f1_score(y_test, y_pred)

print("F1: %.2f%%" % (f1 * 100.0))
from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(max_features = 11, random_state = 11)

dt_classifier.fit(X_train, y_train)

y_pred =  dt_classifier.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy: %.2f%%" % (accuracy*100.0))
confusion_matrix(y_test,y_pred)
#precision tp / (tp + fp)

precision=precision_score(y_test, y_pred)

print("Precision: %.2f%%" % (precision * 100.0))



# recall: tp / (tp + fn)

recall=recall_score(y_test, y_pred)

print("recall: %.2f%%" % (recall * 100.0))





# f1: 2 tp / (2 tp + fp + fn)

f1=f1_score(y_test, y_pred)

print("F1: %.2f%%" % (f1 * 100.0))
#Import knearest neighbors Classifier model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=16)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy: %.2f%%" % (accuracy*100.0))
confusion_matrix(y_test,y_pred)
#precision tp / (tp + fp)

precision=precision_score(y_test, y_pred)

print("Precision: %.2f%%" % (precision * 100.0))



# recall: tp / (tp + fn)

recall=recall_score(y_test, y_pred)

print("recall: %.2f%%" % (recall * 100.0))





# f1: 2 tp / (2 tp + fp + fn)

f1=f1_score(y_test, y_pred)

print("F1: %.2f%%" % (f1 * 100.0))
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy: %.2f%%" % (accuracy*100.0))
confusion_matrix(y_test,y_pred)
#precision tp / (tp + fp)

precision=precision_score(y_test, y_pred)

print("Precision: %.2f%%" % (precision * 100.0))



# recall: tp / (tp + fn)

recall=recall_score(y_test, y_pred)

print("recall: %.2f%%" % (recall * 100.0))





# f1: 2 tp / (2 tp + fp + fn)

f1=f1_score(y_test, y_pred)

print("F1: %.2f%%" % (f1 * 100.0))
from keras.models import Sequential

from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

training = classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 150)
from sklearn.metrics import accuracy_score 

y_pred = classifier.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred.round()))

accuracy = accuracy_score(y_test,y_pred.round())

print("Accuracy: %.2f%%" % (accuracy*100.0))
confusion_matrix(y_test,y_pred.round())
#precision tp / (tp + fp)

precision=precision_score(y_test,y_pred.round())

print("Precision: %.2f%%" % (precision * 100.0))



# recall: tp / (tp + fn)

recall=recall_score(y_test, y_pred.round())

print("recall: %.2f%%" % (recall * 100.0))





# f1: 2 tp / (2 tp + fp + fn)

f1=f1_score(y_test, y_pred.round())

print("F1: %.2f%%" % (f1 * 100.0))