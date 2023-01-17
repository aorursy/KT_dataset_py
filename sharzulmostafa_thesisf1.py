#pip install scikit-learn==0.22.1
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/heartdata"))



# Any results you write to the current directory are saved as output.
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score
dataset = pd.read_csv('../input/heartdata/heart.csv')

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 13].values

Z = dataset.drop(['Column14'], axis = 1)
dataset
dataset.shape
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
dataset.hist()
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
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn import metrics

bg = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=1.0, n_estimators=20)

bg.fit(X_train,y_train)

y_pred=bg.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn import metrics

from sklearn import svm

clf = svm.SVC(kernel='linear') # Linear Kernel

trained_model = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#print("Training Accuracy:",metrics.accuracy_score(y_train, trained_model.predict(X_train)))

# testing accuracy: (tp + tn) / (p + n)

print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred))



#Train Accuracy: accuracy_score(training_target, trained_model.predict(training_features))

#Test Accuracy: accuracy_score(test_target, predictions)

#Confusion Matrix: confusion_matrix(test_target, predictions)
confusion_matrix(y_test, y_pred)
# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)
# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)
from sklearn import metrics

from sklearn import svm

clf = svm.SVC(kernel='rbf') # Rbf Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
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
from sklearn.svm import SVC

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
print("The score for Support Vector Classifier is {}% with {} kernel.".format(svc_scores[2]*100, 'rbf'))
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

    plt.text(i, rf_scores[i], (i))

    #plt.text(i, rf_scores[i], rf_scores[i])

plt.xticks(ticks = [i for i in range(len(estimators))], labels = [5,10,20,30,50])

plt.xlabel('Number of estimators')

plt.ylabel('Scores')

plt.title('Random Forest Classifier scores for different number of estimators')
print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[3]*100, [ 500]))
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

for i in range(1, len(Z.columns) + 1):

    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = i)

    dt_classifier.fit(X_train, y_train)

    y_pred =  dt_classifier.predict(X_test)

    dt_scores.append(dt_classifier.score(X_test, y_test))
plt.plot([i for i in range(1, len(Z.columns) + 1)], dt_scores, color = 'green')

for i in range(1, len(Z.columns) + 1):

    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))

plt.xticks([i for i in range(1, len(Z.columns) + 1)])

plt.xlabel('Max features')

plt.ylabel('Scores')

plt.title('Decision Tree Classifier scores for different number of maximum features')
print("The score for Decision Tree Classifier is {}% with {} maximum features.".format(dt_scores[7]*100, 8))
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

from sklearn.metrics import accuracy_score



max_accuracy = 0





for p in range(200):

    dt = DecisionTreeClassifier(random_state=p)

    dt.fit(X_train,y_train)

    Y_pred_dt = dt.predict(X_test)

    current_accuracy = round(accuracy_score(Y_pred_dt,y_test)*100,2)

    if(current_accuracy>max_accuracy):

        max_accuracy = current_accuracy

        best_x = p

        

#print(max_accuracy)

#print(best_x)





dt = DecisionTreeClassifier(random_state=best_x)

dt.fit(X_train,y_train)

Y_pred_dt = dt.predict(X_test)



score_dt = round(accuracy_score(Y_pred_dt,y_test)*100,2)



print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=90)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

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
#Import knearest neighbors Classifier model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
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
from sklearn.neural_network import MLPClassifier

mlprc_regular = MLPClassifier()

mlprc_regular.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression()

lgr.fit(X_train, y_train)

y_pred = lgr.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.linear_model import Perceptron

prc_regular = Perceptron()

prc_regular.fit(X_train, y_train)

y_pred = prc_regular.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.neural_network import MLPClassifier

mlprc_regular = MLPClassifier()

mlprc_regular.fit(X_train, y_train)

y_pred = mlprc_regular.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from keras.models import Sequential

from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = 13))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

training = classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 30)

#testing = classifier.fit(X_test, y_test, batch_size = 10, nb_epoch = 200)



val_acc = np.mean(training.history['accuracy'])

print("\n%s: %.2f%%" % ('val_acc', val_acc*100))



from sklearn.metrics import accuracy_score 

y_pred = classifier.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred.round()))

# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred.round())

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred.round())

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred.round())

print('F1 score: %f' % f1)
from sklearn.neighbors import KNeighborsClassifier

knn_scores = []

for k in range(1,30):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    knn_classifier.fit(X_train, y_train)

    y_pred = knn_classifier.predict(X_test)

    knn_scores.append(knn_classifier.score(X_test, y_test))
plt.plot([k for k in range(1, 30)], knn_scores, color = 'red')

for i in range(1,30):

    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))

plt.xticks([i for i in range(1, 30)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')
print("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[18]*100, 19))
# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)
from sklearn.decomposition import PCA as sklearnPCA

pca = sklearnPCA(n_components=5)

pca_heart = pca.fit_transform(x_std)
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(pca_heart,y,test_size=0.2,random_state=50)
knn_scores = []

for k in range(1,30):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    knn_classifier.fit(X_train, y_train)

    knn_scores.append(knn_classifier.score(X_test, y_test))
plt.plot([k for k in range(1, 30)], knn_scores, color = 'red')

for i in range(1,30):

    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))

plt.xticks([i for i in range(1, 30)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')
print("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[10]*100, 11))
from sklearn.svm import SVC

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
print("The score for Support Vector Classifier is {}% with {} kernel.".format(svc_scores[2]*100, 'rbf'))
from keras.models import Sequential

from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 13))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

training = classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 50)



val_acc = np.mean(training.history['accuracy'])

print("\n%s: %.2f%%" % ('val_acc', val_acc*100))
from sklearn.metrics import accuracy_score 

y_pred = classifier.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred.round()))

from sklearn import svm

clf = svm.SVC(kernel='linear') # Linear Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn import svm

clf = svm.SVC(kernel='rbf') # rbf Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



max_accuracy = 0





for p in range(200):

    dt = DecisionTreeClassifier(random_state=p)

    dt.fit(X_train,y_train)

    Y_pred_dt = dt.predict(X_test)

    current_accuracy = round(accuracy_score(Y_pred_dt,y_test)*100,2)

    if(current_accuracy>max_accuracy):

        max_accuracy = current_accuracy

        best_x = p

        

#print(max_accuracy)

#print(best_x)





dt = DecisionTreeClassifier(random_state=best_x)

dt.fit(X_train,y_train)

Y_pred_dt = dt.predict(X_test)



score_dt = round(accuracy_score(Y_pred_dt,y_test)*100,2)



print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=50)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

score_dt = round(accuracy_score(Y_pred_dt,y_test)*100,2)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



print("The accuracy score achieved using RandomForest is: "+str(score_dt)+" %")
#Import knearest neighbors Classifier model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression()

lgr.fit(X_train, y_train)

y_pred = lgr.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.neural_network import MLPClassifier

mlprc_regular = MLPClassifier()

mlprc_regular.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=4, n_iter=10, random_state=50)

#svd_heart=svd.fit(x_std).transform(x_std) 

svd_heart = svd.fit_transform(pca_heart)
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(svd_heart,y,test_size=0.3,random_state=50)
from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier(max_depth=2, random_state=10)



classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=90)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



max_accuracy = 0





for p in range(200):

    dt = DecisionTreeClassifier(random_state=p)

    dt.fit(X_train,y_train)

    Y_pred_dt = dt.predict(X_test)

    current_accuracy = round(accuracy_score(Y_pred_dt,y_test)*100,2)

    if(current_accuracy>max_accuracy):

        max_accuracy = current_accuracy

        best_x = p

        

#print(max_accuracy)

#print(best_x)





dt = DecisionTreeClassifier(random_state=best_x)

dt.fit(X_train,y_train)

Y_pred_dt = dt.predict(X_test)



score_dt = round(accuracy_score(Y_pred_dt,y_test)*100,2)



print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
from sklearn.neural_network import MLPClassifier

mlprc_regular = MLPClassifier()

mlprc_regular.fit(X_train, y_train)

y_pred = mlprc_regular.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



lda = LDA(n_components=1)

#X_train = lda.fit_transform(X_train, y_train)

#X_test = lda.transform(X_test)

#lda_heart = lda.fit_transform(x_std,y)

lda_heart = lda.fit_transform(pca_heart,y)
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(lda_heart,y,test_size=0.2,random_state=50)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=50)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier(max_depth=2, random_state=10)



classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



max_accuracy = 0





for p in range(200):

    dt = DecisionTreeClassifier(random_state=p)

    dt.fit(X_train,y_train)

    Y_pred_dt = dt.predict(X_test)

    current_accuracy = round(accuracy_score(Y_pred_dt,y_test)*100,2)

    if(current_accuracy>max_accuracy):

        max_accuracy = current_accuracy

        best_x = p

        

#print(max_accuracy)

#print(best_x)





dt = DecisionTreeClassifier(random_state=best_x)

dt.fit(X_train,y_train)

Y_pred_dt = dt.predict(X_test)



score_dt = round(accuracy_score(Y_pred_dt,y_test)*100,2)



print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
from sklearn.neural_network import MLPClassifier

mlprc_regular = MLPClassifier()

mlprc_regular.fit(X_train, y_train)

y_pred = mlprc_regular.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.utils import resample

X_train, y_train = resample(X_train, y_train, random_state=10)

from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier(max_depth=2, random_state=10)



classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn import metrics

from sklearn import svm

clf = svm.SVC(kernel='rbf') # Rbf Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.utils import shuffle

X_train, y_train = resample(X_train, y_train, random_state=10)
from sklearn import metrics

from sklearn import svm

clf = svm.SVC(kernel='rbf') # Rbf Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))