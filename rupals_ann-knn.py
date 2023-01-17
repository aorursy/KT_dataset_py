#from google.colab import files

#uploaded = files.upload() 

#Please select file location
#import statements

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, KFold, StratifiedShuffleSplit

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

import random

from sklearn.svm import SVC

import sklearn.metrics as sk

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import recall_score,auc, roc_auc_score, roc_curve,confusion_matrix,classification_report
#change the dataset location

df = pd.read_csv('/kaggle/input/bank-marketing/bank-additional-full.csv', sep = ';')

df.shape
#viewing data

df.head()
#data info

df.info()

#No null values in the data
#Removing non-relevant variables

df1=df.drop(columns=['day_of_week','month','contact','poutcome'],axis=1)

df1
#Replacing all the binary variables to 0 and 1

df1.y.replace(('yes', 'no'), (1, 0), inplace=True)

df1.default.replace(('yes', 'no'), (1, 0), inplace=True)

df1.housing.replace(('yes', 'no'), (1, 0), inplace=True)

df1.loan.replace(('yes', 'no'), (1, 0), inplace=True)

df1
#creating Dummies for categorical variables

df2 = pd.get_dummies(df1)

df2.head()
#Removing extra dummy variables & checking descriptive stats

df3=df2.drop(columns=['job_unknown','marital_divorced','education_unknown'],axis=1)

df3.describe().T
#Correlation plot

plt.figure(figsize=(14,8))

df3.corr()['y'].sort_values(ascending = False).plot(kind='bar')
#Creating binary classification target variable

df_target=df3[['y']].values

df_features=df3.drop(columns=['y'],axis=1).values

x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.3, random_state = 0)
sc = StandardScaler()

x1_train = sc.fit_transform(x1_train)

x1_test = sc.transform(x1_test)
# Making the Confusion Matrix

def confusionmat(y,y_hat):

  from sklearn.metrics import confusion_matrix,accuracy_score

  cm = confusion_matrix(y, y_hat)

  accu=accuracy_score(y,y_hat)

  print(cm,"\n")

  print("The accuracy is",accu)
#Accuracy and Loss Curves

def learningcurve(history):

  # list all data in history

  print(history.history.keys())

  # summarize history for accuracy

  plt.plot(history.history['accuracy'])

  plt.plot(history.history['val_accuracy'])

  plt.title('model accuracy')

  plt.ylabel('accuracy')

  plt.xlabel('epoch')

  plt.legend(['train', 'test'], loc='upper left')

  plt.show()

  # summarize history for loss

  plt.plot(history.history['loss'])

  plt.plot(history.history['val_loss'])

  plt.title('model loss')

  plt.ylabel('loss')

  plt.xlabel('epoch')

  plt.legend(['train', 'test'], loc='upper left')

  plt.show()
  # Applying k-Fold Cross Validation

  from sklearn.model_selection import cross_val_score

  def kfold(x1,y1):

    return cross_val_score(estimator = classifier, X = x1, y = y1, cv = 10)
from sklearn.model_selection import learning_curve

def knn_learningcurve(c, df_features, df_target):

  train_sizes, train_scores, test_scores = learning_curve(c, df_features, df_target,cv=10,n_jobs=-1)

  train_scores_mean = np.mean(train_scores, axis=1)

  train_scores_std = np.std(train_scores, axis=1)

  test_scores_mean = np.mean(test_scores, axis=1)

  test_scores_std = np.std(test_scores, axis=1)



  plt.figure()

  plt.title("KNNClassifier")

  plt.legend(loc="best")

  plt.xlabel("Training examples")

  plt.ylabel("Score")



  plt.grid()



  plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                      train_scores_mean + train_scores_std, alpha=0.1,

                      color="r")

  plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                      test_scores_mean + test_scores_std, alpha=0.1, color="g")

  plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

              label="Training score")

  plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

              label="Cross-validation score")



  plt.legend(loc="best")

  # sizes the window for readability and displays the plot

  # shows error from 0 to 1.1

  plt.ylim(-.1,1.1)

  plt.show
#Roc Curve

def roc_auc(yTest,y_pred):

    sns.set()

    fpr, tpr, thresholds = roc_curve(yTest, y_pred)

    roc_auc = auc(fpr,tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)

    plt.legend(loc='lower right')

    plt.plot([0,1],[0,1],'r--')

    plt.xlim([-0.1,1.0])

    plt.ylim([-0.1,1.01])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(16,activation="relu"))



# Adding the second hidden layer

classifier.add(Dense(32,activation="relu"))



# Adding the output layer

classifier.add(Dense(1,activation="sigmoid"))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



epoch=[100,250,500]

for e in epoch:

  # Fitting the ANN to the Training set

  history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=e,validation_split=0.3)

  # Predicting the Test set results

  y_pred = classifier.predict_classes(x1_test)

  pre_score = sk.average_precision_score(y1_test, y_pred)

  classifier.summary()

  test_results = classifier.evaluate(x1_test, y1_test)

  print("For epoch = {0}, the model test accuracy is {1}.".format(e,test_results[1]))

  print("The model test average precision score is {}.".format(pre_score))

  confusionmat(y1_test,y_pred)

  learningcurve(history)

# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(16,activation="relu"))



# Adding the output layer

classifier.add(Dense(1,activation="sigmoid"))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=100,validation_split=0.3)

# Predicting the Test set results

y_pred = classifier.predict_classes(x1_test)

pre_score = sk.average_precision_score(y1_test, y_pred)

classifier.summary()

test_results = classifier.evaluate(x1_test, y1_test)

print("For epoch = {0}, the model test accuracy is {1}.".format(100,test_results[1]))

print("The model test average precision score is {}.".format(pre_score))

confusionmat(y1_test,y_pred)

learningcurve(history)



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(16,activation="relu"))



# Adding the second hidden layer

classifier.add(Dense(32,activation="relu"))



# Adding the output layer

classifier.add(Dense(1,activation="sigmoid"))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=100,validation_split=0.3)

# Predicting the Test set results

y_pred = classifier.predict_classes(x1_test)

pre_score = sk.average_precision_score(y1_test, y_pred)

classifier.summary()

test_results = classifier.evaluate(x1_test, y1_test)

print("For epoch = {0}, the model test accuracy is {1}.".format(100,test_results[1]))

print("The model test average precision score is {}.".format(pre_score))

confusionmat(y1_test,y_pred)

learningcurve(history)



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(16,activation="relu"))



# Adding the second hidden layer

classifier.add(Dense(32,activation="relu"))



# Adding the third hidden layer

classifier.add(Dense(32,activation="relu"))



# Adding the output layer

classifier.add(Dense(1,activation="sigmoid"))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=100,validation_split=0.3)

# Predicting the Test set results

y_pred = classifier.predict_classes(x1_test)

pre_score = sk.average_precision_score(y1_test, y_pred)

classifier.summary()

test_results = classifier.evaluate(x1_test, y1_test)

print("For epoch = {0}, the model test accuracy is {1}.".format(100,test_results[1]))

print("The model test average precision score is {}.".format(pre_score))

confusionmat(y1_test,y_pred)

learningcurve(history)



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(16,activation="relu"))



# Adding the second hidden layer

classifier.add(Dense(16,activation="relu"))



# Adding the output layer

classifier.add(Dense(1,activation="sigmoid"))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=100,validation_split=0.3)

# Predicting the Test set results

y_pred = classifier.predict_classes(x1_test)

pre_score = sk.average_precision_score(y1_test, y_pred)

classifier.summary()

test_results = classifier.evaluate(x1_test, y1_test)

print("For epoch = {0}, the model test accuracy is {1}.".format(100,test_results[1]))

print("The model test average precision score is {}.".format(pre_score))

confusionmat(y1_test,y_pred)

learningcurve(history)



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(32,activation="relu"))



# Adding the second hidden layer

classifier.add(Dense(16,activation="relu"))



# Adding the output layer

classifier.add(Dense(1,activation="sigmoid"))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=100,validation_split=0.3)

# Predicting the Test set results

y_pred = classifier.predict_classes(x1_test)

pre_score = sk.average_precision_score(y1_test, y_pred)

classifier.summary()

test_results = classifier.evaluate(x1_test, y1_test)

print("For epoch = {0}, the model test accuracy is {1}.".format(100,test_results[1]))

print("The model test average precision score is {}.".format(pre_score))

confusionmat(y1_test,y_pred)

learningcurve(history)



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(32,activation="relu"))



# Adding the second hidden layer

classifier.add(Dense(32,activation="relu"))



# Adding the output layer

classifier.add(Dense(1,activation="sigmoid"))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=100,validation_split=0.3)

# Predicting the Test set results

y_pred = classifier.predict_classes(x1_test)

pre_score = sk.average_precision_score(y1_test, y_pred)

classifier.summary()

test_results = classifier.evaluate(x1_test, y1_test)

print("For epoch = {0}, the model test accuracy is {1}.".format(100,test_results[1]))

print("The model test average precision score is {}.".format(pre_score))

confusionmat(y1_test,y_pred)

learningcurve(history)



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(32,activation="tanh"))



# Adding the second hidden layer

classifier.add(Dense(16,activation="tanh"))



# Adding the output layer

classifier.add(Dense(1,activation="sigmoid"))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=100,validation_split=0.3)

# Predicting the Test set results

y_pred = classifier.predict_classes(x1_test)

pre_score = sk.average_precision_score(y1_test, y_pred)

classifier.summary()

test_results = classifier.evaluate(x1_test, y1_test)

print("For epoch = {0}, the model test accuracy is {1}.".format(100,test_results[1]))

print("The model test average precision score is {}.".format(pre_score))

confusionmat(y1_test,y_pred)

learningcurve(history)



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(32,activation="sigmoid"))



# Adding the second hidden layer

classifier.add(Dense(16,activation="sigmoid"))



# Adding the output layer

classifier.add(Dense(1,activation="sigmoid"))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=100,validation_split=0.3)

# Predicting the Test set results

y_pred = classifier.predict_classes(x1_test)

pre_score = sk.average_precision_score(y1_test, y_pred)

classifier.summary()

test_results = classifier.evaluate(x1_test, y1_test)

print("For epoch = {0}, the model test accuracy is {1}.".format(100,test_results[1]))

print("The model test average precision score is {}.".format(pre_score))

confusionmat(y1_test,y_pred)

learningcurve(history)



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(32,activation="softmax"))



# Adding the second hidden layer

classifier.add(Dense(16,activation="softmax"))



# Adding the output layer

classifier.add(Dense(1,activation="sigmoid"))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=100,validation_split=0.3)

# Predicting the Test set results

y_pred = classifier.predict_classes(x1_test)

pre_score = sk.average_precision_score(y1_test, y_pred)

classifier.summary()

test_results = classifier.evaluate(x1_test, y1_test)

print("For epoch = {0}, the model test accuracy is {1}.".format(100,test_results[1]))

print("The model test average precision score is {}.".format(pre_score))

confusionmat(y1_test,y_pred)

learningcurve(history)



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(32,activation="softmax"))



# Adding the second hidden layer

classifier.add(Dense(16,activation="softmax"))



# Adding the output layer

classifier.add(Dense(1,activation="sigmoid"))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=100,validation_split=0.3)

# Predicting the Test set results

y_pred = classifier.predict_classes(x1_test)

pre_score = sk.average_precision_score(y1_test, y_pred)

classifier.summary()

test_results = classifier.evaluate(x1_test, y1_test)

print("For epoch = {0}, the model test accuracy is {1}.".format(e,test_results[1]))

print("The model test average precision score is {}.".format(pre_score))

confusionmat(y1_test,y_pred)

learningcurve(history)
from sklearn.neighbors import KNeighborsClassifier

k_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

accu_KNN=[]

for i in k_list:

  # Fitting K-NN to the Training set

  classifier = KNeighborsClassifier(n_neighbors = i)

  history=classifier.fit(x1_train, y1_train)



  # Predicting the Test set results

  y_pred = classifier.predict(x1_test)



  # 10 fold cross validation

  accuracies = kfold(x1_train, y1_train)



  accu_KNN+=[accuracies.mean()]

  std=accuracies.std()

  report=sk.classification_report(y1_test,y_pred)

  confusionmat(y1_test, y_pred)



  print("KNN with n= ",i)

  print("The Classification report\n",report,end='\n')

  print("The Accuracy score with only 1 Training and Testing Data Set\n",sk.accuracy_score(y1_test,y_pred)*100,end='\n')

  #after using cross validation with 10 folds

  print("The mean of the accuracy scores with using 10 fold-cross validation\n",accuracies.mean()*100,end='\n')

  print("The Standard Deviation of the accuracy scores with using 10 fold-cross validation\n",std*100,end='\n')



  knn_learningcurve(classifier, df_features, df_target)
plt.plot(k_list,accu_KNN)

plt.xlabel("k number")

plt.ylabel("Accuracy")

plt.show()
# Fitting K-NN to the Training set

classifier = KNeighborsClassifier(n_neighbors = 9,metric="manhattan")

history=classifier.fit(x1_train, y1_train)

# Predicting the Test set results

y_pred = classifier.predict(x1_test)



# 10 fold cross validation

accuracies = kfold(x1_train, y1_train)

accu_KNN=accuracies.mean()

std=accuracies.std()

report=sk.classification_report(y1_test,y_pred)

confusionmat(y1_test, y_pred)



print("KNN with n= ",9)

print("The Classification report\n",report,end='\n')

print("The Accuracy score with only 1 Training and Testing Data Set\n",sk.accuracy_score(y1_test,y_pred)*100,end='\n')



knn_learningcurve(classifier, df_features, df_target)
# Fitting K-NN to the Training set

classifier = KNeighborsClassifier(n_neighbors = 9,metric="chebyshev")

history=classifier.fit(x1_train, y1_train)

# Predicting the Test set results

y_pred = classifier.predict(x1_test)



# 10 fold cross validation

accuracies = kfold(x1_train, y1_train)

accu_KNN=accuracies.mean()

std=accuracies.std()

report=sk.classification_report(y1_test,y_pred)

confusionmat(y1_test, y_pred)



print("KNN with n= ",9)

print("The Classification report\n",report,end='\n')

print("The Accuracy score with only 1 Training and Testing Data Set\n",sk.accuracy_score(y1_test,y_pred)*100,end='\n')

#after using cross validation with 10 folds

print("The mean of the accuracy scores with using 10 fold-cross validation\n",accuracies.mean()*100,end='\n')

print("The Standard Deviation of the accuracy scores with using 10 fold-cross validation\n",std*100,end='\n')



knn_learningcurve(classifier, df_features, df_target)
# Fitting K-NN to the Training set

classifier = KNeighborsClassifier(n_neighbors = 9,metric="euclidean")

history=classifier.fit(x1_train, y1_train)

# Predicting the Test set results

y_pred = classifier.predict(x1_test)



# 10 fold cross validation

accuracies = kfold(x1_train, y1_train)

accu_KNN=accuracies.mean()

std=accuracies.std()

report=sk.classification_report(y1_test,y_pred)

confusionmat(y1_test, y_pred)



print("KNN with n= ",9)

print("The Classification report\n",report,end='\n')

print("The Accuracy score with only 1 Training and Testing Data Set\n",sk.accuracy_score(y1_test,y_pred)*100,end='\n')

#after using cross validation with 10 folds

print("The mean of the accuracy scores with using 10 fold-cross validation\n",accuracies.mean()*100,end='\n')

print("The Standard Deviation of the accuracy scores with using 10 fold-cross validation\n",std*100,end='\n')



knn_learningcurve(classifier, df_features, df_target)
# Fitting K-NN to the Training set

classifier = KNeighborsClassifier(n_neighbors = 9,metric="manhattan")

history=classifier.fit(x1_train, y1_train)

# Predicting the Test set results

y_pred = classifier.predict(x1_test)



report=sk.classification_report(y1_test,y_pred)

confusionmat(y1_test, y_pred)



print("KNN with n= ",9)

print("The Classification report\n",report,end='\n')

print("The Accuracy score with only 1 Training and Testing Data Set\n",sk.accuracy_score(y1_test,y_pred)*100,end='\n')

knn_learningcurve(classifier, df_features, df_target)
roc_auc(y1_test,y_pred)
distance_list=['euclidean','manhattan','chebyshev']

Accuracy_test=[88.5137,88.5800,88.4031]

 

d1=pd.DataFrame(list(zip(distance_list,Accuracy_test)),columns=['distance_list','Accuracy'])

print(d1)



# this is for plotting purpose

index = np.arange(len(distance_list))

plt.bar(distance_list, Accuracy_test,  width=0.3)

plt.xlabel('Distance metric')

plt.ylabel('Accuracy')

plt.title('Comparison')

plt.figure(figsize=(2,2))

plt.show()
# knn and nn

distance_list=['Neual Network','KNN']

Accuracy_test=[88.9265,88.5800]

 

d1=pd.DataFrame(list(zip(distance_list,Accuracy_test)),columns=['Algorithm','Accuracy'])

print(d1)



# this is for plotting purpose

index = np.arange(len(distance_list))

plt.bar(distance_list, Accuracy_test,  width=0.3)

plt.xlabel('Algorithms')

plt.ylabel('Accuracy')

plt.title('Comparison')

plt.figure(figsize=(1,1))

plt.show()
# comparison between svm  knn nn decision trees ensemble



Algorithms=['SVM-rbf','Decision Trees','Boosting','Neural Networks','KNN-Manhattan']

Accuracy_test=[88.78,85.29,89.52,88.93,88.58]

 

d1=pd.DataFrame(list(zip(Algorithms,Accuracy_test)),columns=['Algorithms','Accuracy'])

print(d1)



# this is for plotting purpose

plt.bar(Algorithms, Accuracy_test, width=0.3)

plt.xlabel('Algorithms')

plt.tick_params(axis='x', rotation=60)

plt.ylabel('Accuracy')

plt.title('Comparison')

plt.figure(figsize=(2,2))

plt.show()