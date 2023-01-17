import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import itertools

warnings.filterwarnings("ignore")

%matplotlib inline

from PIL import Image

data = pd.read_csv(r"../input/predicting-a-pulsar-star/pulsar_stars.csv")

data.head(10)
data.info()
data.describe()
data.shape
data.isnull().sum()
dt = data['target_class'].value_counts()

print(dt)
sns.pairplot(data, hue='target_class',palette='cubehelix',kind = 'reg')
plt.figure(figsize=(16,10))



plt.subplot(2,2,1)

sns.violinplot(data=data,y=" Mean of the integrated profile",x="target_class")
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

data_x = data.drop(["target_class"],axis=1)



scaler = MinMaxScaler(feature_range=(0, 1))

X = scaler.fit_transform(data_x)



X_train,X_test,Y_train,Y_test = train_test_split(X,data["target_class"].values,random_state = 42,test_size= 0.15)
type(Y_train)


print("X_train shape: {}, X_test: {}".format(X_train.shape,X_test.shape))

print("Y_train shape: {}, Y_test: {}".format(Y_train.shape,Y_test.shape))

from sklearn.linear_model import LogisticRegression 

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm  

from sklearn.neighbors import KNeighborsClassifier  

from sklearn import metrics 
logr = LogisticRegression()

logr.fit(X_train,Y_train)

y_predict_lr = logr.predict(X_test)

acc_log = metrics.accuracy_score(y_predict_lr,Y_test)

print('The accuracy of the Logistic Regression is', acc_log)
len(y_predict_lr)
Y_test = Y_test.reshape((-1,1))
Y_test.shape
predict = logr.predict(X_test)
def predtest(val):

    zero = []

    one = []

    for i in range(2685):

        counter = np.argmax(val[i])

        if counter == 0:

            zero.append(counter)

        else:

            one.append(counter)

            

    #plt

    print(len(zero),len(one))

    
predtest(y_predict_lr)
dt = DecisionTreeClassifier()

dt.fit(X_train,Y_train)

y_predict_dt = dt.predict(X_test)

acc_dt = metrics.accuracy_score(y_predict_dt,Y_test)

print('The accuracy of the Decision Tree is', acc_dt)
sv = svm.SVC() #select the algorithm

sv.fit(X_train,Y_train) # we train the algorithm with the training data and the training output

y_predict_svm = sv.predict(X_test) #now we pass the testing data to the trained algorithm

acc_svm = metrics.accuracy_score(y_predict_svm,Y_test)

print('The accuracy of the SVM is:', acc_svm)
knc = KNeighborsClassifier(n_neighbors=5) 

knc.fit(X_train,Y_train)

y_predict_knn = knc.predict(X_test)

acc_knn = metrics.accuracy_score(y_predict_knn,Y_test)

print('The accuracy of the KNN is', acc_knn)
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB() 

gnb.fit(X_train, Y_train) 

gnb_pred = gnb.predict(X_test)

acc_gnb = metrics.accuracy_score(gnb_pred,Y_test)

print("The accuracy of the Gaussian Naive Bayes is:", acc_gnb)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

rf = RandomForestClassifier(n_estimators = 40)

rf.fit(X_train,Y_train)

rf_pred = rf.predict(X_test)

acc_rfc = metrics.accuracy_score(rf_pred,Y_test)

print("The accuracy of theRandomForestClassifier is:", acc_rfc)



cm = confusion_matrix(Y_test,rf.predict(X_test))

sns.heatmap(cm,annot=True,fmt="d")
a_index = list(range(1,11))

a = pd.Series()

x = [1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    kcs = KNeighborsClassifier(n_neighbors=i) 

    kcs.fit(X_train,Y_train)

    y_pred = kcs.predict(X_test)

    a=a.append(pd.Series(metrics.accuracy_score(y_pred,Y_test)))

plt.plot(a_index, a)

plt.xticks(x)

# accuracy for various values of n for K-Nearest nerighbours
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Decision Tree', 'Support Vector Machines',

              'K-Nearest Neighbours','Gaussian Naive Bayes','Random Forest Classifier'],

    'Score': [acc_log, acc_dt, acc_svm, acc_knn,acc_gnb,acc_rfc]})

models.sort_values(by='Score', ascending=False)
from sklearn.metrics import roc_curve

fpr_lr, tpr_lr, thresholds = roc_curve(Y_test, y_predict_lr)

plt.plot([0, 1], [0, 1], 'k--',color="grey")

plt.plot(fpr_lr, tpr_lr,color="red")

plt.title('Logistic Regression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')
X_train.shape
X_train, X_test, Y_train, Y_test = X_train.T, X_test.T, Y_train.T, Y_test.T
Y_train = Y_train.reshape(-1,1)

X_train = X_train.T
Y_train.shape
print(Y_train.shape)



print(X_train.shape)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library

from keras.layers import Dense # build our layers library

def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))

    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 50)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.var()
print(accuracies)
print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))
def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'tanh', input_dim = X_train.shape[1]))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'tanh'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 50)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 2)
print(accuracies)
mean = accuracies.mean()

variance = accuracies.var()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))
plt.bar(y_predict_lr,height=5)
sns.lineplot(data=accuracies)