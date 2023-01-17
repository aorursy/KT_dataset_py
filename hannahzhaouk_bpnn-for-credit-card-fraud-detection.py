# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from imblearn.over_sampling import ADASYN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

%matplotlib inline

import tensorflow as tf
from tensorflow import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
credit_card=pd.read_csv("../input/creditcard.csv")
credit_card.head()
#Take a look of our dataset and variables to get brief understanding of the data we'll be working with
#Check the size of our dataset
credit_card.shape
credit_card["Class"].value_counts()
#Check if there is missing value
null_data=pd.isnull(credit_card).sum()
print(null_data)
f, (fraud, normal) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 50

fraud.hist(credit_card.Time[credit_card.Class == 1], bins = bins)
fraud.set_title('Fraud')

normal.hist(credit_card.Time[credit_card.Class == 0], bins = bins)
normal.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Number of Transactions')
plt.show()
f, (fraud, normal) = plt.subplots(2, 1, sharex=False, figsize=(12,7))

bins = 30

fraud.hist(credit_card.Amount[credit_card.Class == 1], bins = bins)
fraud.set_title('Fraud')

normal.hist(credit_card.Amount[credit_card.Class == 0], bins = bins)
normal.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()
f, (fraud, normal) = plt.subplots(2, 1, sharex=False, figsize=(12,8))

fraud.scatter(credit_card.Time[credit_card.Class == 1], credit_card.Amount[credit_card.Class == 1])
fraud.set_title('Fraud')

normal.scatter(credit_card.Time[credit_card.Class == 0], credit_card.Amount[credit_card.Class == 0])
normal.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
v_variable=credit_card.iloc[:,1:29].columns
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(credit_card[v_variable]):
    ax = plt.subplot(gs[i])
    sns.distplot(credit_card[cn][credit_card.Class == 1], bins=50)
    sns.distplot(credit_card[cn][credit_card.Class == 0], bins=50)
    ax.set_xlabel('')
    plt.legend(credit_card["Class"])
    ax.set_title('histogram of feature: ' + str(cn))
    
plt.show()
#select the values of "V1" to "Amount"
creditcard_v=credit_card.iloc[:,1:30].values
print(creditcard_v)
creditcard_v.shape
#Set 30 variables as "X" and "Class" as y
X=creditcard_v
y=credit_card["Class"].values

sss=StratifiedShuffleSplit(n_splits=5, test_size=0.3,random_state=0)
sss.get_n_splits(X,y)
#So this is the cross-validator that we are using
print(sss)
#Split train/test sets of X and y
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train,X_test=X[train_index], X[test_index]
    y_train,y_test=y[train_index], y[test_index]
#Let's see the number of sets in each class of training and testing datasets
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())
pca=PCA(n_components=2)
data_2d=pd.DataFrame(pca.fit_transform(X_train))
data_2d
data_2d=pd.concat([data_2d, pd.DataFrame(y_train)], axis=1)
data_2d.columns=["x","y","fraud"]
data_2d
#visualise the 2D training data
sns.set_style("darkgrid")
sns.lmplot(x="x", y="y", data=data_2d, fit_reg=False, hue="fraud",height=5, aspect=2)
plt.title("Scatter Plot of imbalanced Training Data")
ada= ADASYN()
x_resample,y_resample=ada.fit_sample(X_train,y_train)
#concat oversampled "x" and "y" into one DataFrame
data_oversampled=pd.concat([pd.DataFrame(x_resample),pd.DataFrame(y_resample)],axis=1)
#replace column labels using the labels of original datasets
data_oversampled.columns=credit_card.columns[1:31]
#while the label of column 30 is "Class",we can rename it to "fraud"
data_oversampled.rename(columns={"Class":"fraud"},inplace=True)
data_oversampled["fraud"].value_counts()
#reduce dimensionality to 2 dimensions
oversampled_train2d=pd.DataFrame(pca.fit_transform(data_oversampled.iloc[:,0:29]))
oversampled_train2d=pd.concat([oversampled_train2d,data_oversampled["fraud"]],axis=1)
oversampled_train2d.columns= ["x","y","fraud"]
#visualise data
sns.set_style("darkgrid")
sns.lmplot(x="x", y="y", data=oversampled_train2d, fit_reg=False, hue="fraud",height=5, aspect=2)
plt.title("Scatter Plot of balanced Training Data")
y_resample
#use one-hot encoding to reformat
Y_resample=keras.utils.to_categorical(y_resample,num_classes=None)
print(Y_resample)
ANN=keras.Sequential()

ANN.add(keras.layers.Dense(4, input_shape=(29,),activation="relu"))
ANN.add(keras.layers.Dense(2, activation="softmax"))
ANN.compile(keras.optimizers.Adam(lr=0.04), "categorical_crossentropy",metrics=['accuracy'])
ANN.summary()
ANN.fit(x_resample,Y_resample, epochs=10, verbose=0)
#Evaluate the model based on accuracy
Y_test=keras.utils.to_categorical(y_test,num_classes=None)
control_accuracy=ANN.evaluate(X_test,Y_test)[1]
print("Accuracy: {}".format(control_accuracy))
print("\n")

#Use sklearn to calculate precision and recall of the model
#Create a classification report
prediction_control=ANN.predict(X_test, verbose=1)
labels=["Normal","Fraud"]
y_pred=np.argmax(prediction_control,axis=1)
precision=precision_score(y_test,y_pred,labels=labels)
recall=recall_score(y_test,y_pred,labels=labels)
print("Fraud Precision:{}".format(precision))
print("Fraud Recall:{}".format(recall))

print("\n")
print(classification_report(y_test,y_pred, target_names=labels,digits=8))
ANN_exp=keras.Sequential()
ANN_exp.add(keras.layers.Dense(4, input_shape=(29,),activation="relu"))
ANN_exp.add(keras.layers.Dense(4, activation="relu"))
ANN_exp.add(keras.layers.Dense(2, activation="softmax"))
    
ANN_exp.compile(keras.optimizers.Adam(lr=0.04), "categorical_crossentropy",metrics=["accuracy"])
    
ANN_exp.fit(x_resample,Y_resample, epochs=10, verbose=0)
#Evaluate the model based on accuracy
Y_test=keras.utils.to_categorical(y_test,num_classes=None)
exp_accuracy=ANN_exp.evaluate(X_test,Y_test)[1]
print("Accuracy: {}".format(exp_accuracy))
print("\n")

#Use sklearn to calculate precision and recall of the model
#Create a classification report
prediction_exp=ANN_exp.predict(X_test, verbose=1)
labels=["Normal","Fraud"]
y_pred_exp=np.argmax(prediction_exp,axis=1)
precision=precision_score(y_test,y_pred_exp,labels=labels)
recall=recall_score(y_test,y_pred_exp,labels=labels)
print("Fraud Precision:{}".format(precision))
print("Fraud Recall:{}".format(recall))

print("\n")
print(classification_report(y_test,y_pred_exp, target_names=labels,digits=8))
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cnf_matrix_control = confusion_matrix(y_test,y_pred)

class_names = ["Normal","Fraud"]
plt.figure(figsize=(6,6))
plot_confusion_matrix(cnf_matrix_control
                      , classes=class_names
                      , title='Confusion matrix of control arm')
plt.show()

cnf_matrix_exp = confusion_matrix(y_test,y_pred_exp)

class_names = ["Normal","Fraud"]
plt.figure(figsize=(6,6))
plot_confusion_matrix(cnf_matrix_exp
                      , classes=class_names
                      , title='Confusion matrix of experimental arm')
plt.show()
results_control_evaluation= []
for i in range(0,30):
    model=keras.Sequential()
    model.add(keras.layers.Dense(4, input_shape=(29,),activation="relu"))
    model.add(keras.layers.Dense(2, activation="softmax"))
    model.compile(keras.optimizers.Adam(lr=0.04), "categorical_crossentropy",metrics=["accuracy"])
    
    model.fit(x_resample,Y_resample, epochs=10, verbose=0)
    
    Y_test=keras.utils.to_categorical(y_test,num_classes=None)
    accuracy=model.evaluate(X_test,Y_test)[1]
    
    prediction_control=model.predict(X_test, verbose=0)
    labels=["Normal","Fraud"]
    y_pred=np.argmax(prediction_control,axis=1)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    evaluation=pd.DataFrame([accuracy,precision,recall])
    results_control_evaluation.append(evaluation)
result_control=pd.concat(results_control_evaluation,axis=1)
print(result_control)
results_experimental_evaluation= []
for i in range(0,30):
    model_exp=keras.Sequential()
    model_exp.add(keras.layers.Dense(4, input_shape=(29,),activation="relu"))
    model_exp.add(keras.layers.Dense(4, activation="relu"))
    model_exp.add(keras.layers.Dense(2, activation="softmax"))
    
    model_exp.compile(keras.optimizers.Adam(lr=0.04), "categorical_crossentropy",metrics=["accuracy"])
    
    model_exp.fit(x_resample,Y_resample, epochs=10, verbose=0)
    
    Y_test=keras.utils.to_categorical(y_test,num_classes=None)
    accuracy=model_exp.evaluate(X_test,Y_test)[1]

    prediction_experimental=model_exp.predict(X_test, verbose=0)
    labels=["Normal","Fraud"]
    y_exp_pred=np.argmax(prediction_control,axis=1)
    precision=precision_score(y_test,y_exp_pred)
    recall=recall_score(y_test,y_exp_pred)
    evaluation=pd.DataFrame([accuracy,precision,recall])
    results_experimental_evaluation.append(evaluation)
result_experimental=pd.concat(results_experimental_evaluation,axis=1)
print(result_experimental)
from scipy import stats

alpha = 0.05;

s, p = stats.normaltest(result_control.iloc[0,:])
if p < alpha:
  print('Control data is not normal')
else:
  print('Control data is normal')
print("p-value:{}".format(p))
print("\n")
s, p = stats.normaltest(result_experimental.iloc[0,:])
if p < alpha:
  print('Experimental data is not normal')
else:
  print('Experimental data is normal')

print ("p-value:{}".format(p))
median_control_acc=result_control.iloc[0,:].median()
median_control_preci=result_control.iloc[1,:].median()
median_control_recall=result_control.iloc[2,:].median()

median_exp_acc=result_experimental.iloc[0,:].median()
median_exp_preci=result_experimental.iloc[1,:].median()
median_exp_recall=result_experimental.iloc[2,:].median()
print("Median control accuracy:{}".format(median_control_acc))
print("Median experim accuracy:{}".format(median_exp_acc))
print("\n")

#Significance test of accuracy
s, p = stats.wilcoxon(result_control.iloc[0,:], result_experimental.iloc[0,:])
print("p-value:{}".format(p))
if p < 0.05:
  print('null hypothesis rejected, significant difference between the data-sets')
else:
  print('null hypothesis accepted, no significant difference between the data-sets')
print("Median control precision:{}".format(median_control_preci))
print("Median experim precision:{}".format(median_exp_preci))
print("\n")

#Significance test of Precision
s, p = stats.wilcoxon(result_control.iloc[1,:], result_experimental.iloc[1,:])
print("p-value:{}".format(p))
if p < 0.05:
  print('null hypothesis rejected, significant difference between the data-sets')
else:
  print('null hypothesis accepted, no significant difference between the data-sets')
print("Median control recall:{}".format(median_control_recall))
print("Median experim recall:{}".format(median_exp_recall))
print("\n")

#Significance test of Precision
s, p = stats.wilcoxon(result_control.iloc[2,:], result_experimental.iloc[2,:])
print("p-value:{}".format(p))
if p < 0.05:
  print('null hypothesis rejected, significant difference between the data-sets')
else:
  print('null hypothesis accepted, no significant difference between the data-sets')
result_accuracy=pd.concat([result_control.iloc[0,:],result_experimental.iloc[0,:]],axis=1)
result_precision=pd.concat([result_control.iloc[1,:],result_experimental.iloc[1,:]],axis=1)
result_recall=pd.concat([result_control.iloc[2,:],result_experimental.iloc[2,:]],axis=1)

result_accuracy.columns=["control","experimental"]
result_precision.columns=["control","experimental"]
result_recall.columns=["control","experimental"]
result_accuracy.boxplot()
plt.title("Boxplot of Accuracy")
result_precision.boxplot()
plt.title("Boxplot of Precision")
result_recall.boxplot()
plt.title("Boxplot of Recall")