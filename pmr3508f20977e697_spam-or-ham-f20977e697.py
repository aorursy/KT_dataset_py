#Importing stuff that I will need in this notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_extraction, model_selection, naive_bayes, metrics
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,fbeta_score
from scipy import stats
train = pd.read_csv( "../input/spambase/train_data.csv")
testTOsubmit = pd.read_csv( "../input/spambase/test_features.csv")
print ("train size: ", train.shape)
print ("test size: ", testTOsubmit.shape)
train.head()
train.info()
# Plot a graph with the proportions of spam or ham in our data

count_Class=pd.value_counts(train["ham"], sort= True)
count_Class.plot(kind= 'bar', color= ["green", "red"])
plt.title('Is it a ham?')
plt.show()
train.describe()
# Drop the columns that is not about frquency

train2 = train.drop(columns = ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'Id'])
# Separate the ham and spam for better analisys

HAMdata = train2[train2['ham']==True]
SPAMdata = train2[train2['ham']==False]
# Drop the column that classify if it is ham or not

HAMdata.drop(columns = ['ham'])
SPAMdata.drop(columns = ['ham'])
# Discovering some information about our data that can be useful

HAMline = HAMdata.shape[0]
SPAMline = SPAMdata.shape[0]
ALLlines = SPAMline + HAMline
SPAMperc = SPAMline * 100 / ALLlines
HAMperc = HAMline * 100 / ALLlines

print('     Useful Information')
print()
print ("HAM data size: ", HAMdata.shape)
print ("SPAM data size: ", SPAMdata.shape)
print("HAM percentage: ", HAMperc)
print("SPAM percentage: ", SPAMperc)
# Set the frequency of words and chars in the non-spam email


sums1 = HAMdata.select_dtypes(pd.np.number).sum().rename('total')
print('Frequency of words and chars in the non-spam email')
print()
print(sums1)
# Set the frequency of words and chars in the spam email

sums2 = SPAMdata.select_dtypes(pd.np.number).sum().rename('total')
print('Frequency of words and chars in the spam email')
print()
print(sums2)
# Plot the graph of more frequent words in non-spam messages

sums1.plot.bar(legend = False)
plt.title('More frequent words in non-spam messages')
plt.xlabel('Words or Chars')
plt.ylabel('Numbers')
plt.show()
# Plot the graph of more frequent words in spam messages

sums2.plot.bar(legend = False)
plt.title('More frequent words in spam messages')
plt.xlabel('Words or Chars')
plt.ylabel('Numbers')
plt.show()
train.drop(columns = ['word_freq_you'])
# Set the frequency of words and chars in each non-spam email

sums1FRACTION = sums1 / HAMline

# Set the frequency of words and chars in each spam email

sums2FRACTION = sums2 / SPAMline
# Set the difference between frequencies of words and chars

difSums = sums2FRACTION - sums1FRACTION
print(difSums)
# Plot the graph of the difference between frequencies of words and chars

difSums.plot.bar(legend = False)
plt.title('Difference between frequencies of words and chars')
plt.xlabel('Words or Chars')
plt.ylabel('Numbers')
plt.show()
# Drop the columns that is about frquency and the Id column

columns = ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'ham']
train3 = pd.DataFrame(train, columns=columns)
# Separate the ham and spam for better analisys

HAMcarac = train3[train3['ham']==True]
SPAMcarac = train3[train3['ham']==False]
# Drop the column that classify if it is ham or not

HAMcarac.drop(columns = ['ham'])
SPAMcarac.drop(columns = ['ham'])
# Set the rate of the types of length in the non-spam email

sums3 = HAMcarac.select_dtypes(pd.np.number).sum().rename('total')
sums3FRACTION = sums3 / HAMline
print('Rate of the types of length in the non-spam email')
print()
print(sums3FRACTION)
# Set the rate of the types of length in the spam email

sums4 = SPAMcarac.select_dtypes(pd.np.number).sum().rename('total')
sums4FRACTION = sums4 / SPAMline
print('Rate of the types of length in the spam email')
print()
print(sums4FRACTION)
#Spliting the data in target(output) and features(input)

outputDATA = train['ham']
inputDATA = train.drop(columns=['ham'])
#Spliting the data in train output, train inputs, test output and test inputs

data_train, data_test, target_train, target_test = train_test_split(
    inputDATA,
    outputDATA,
    random_state = 0) 
#Creating the object pertaining to the Naive Bayes classifier for normal probability distribution.

gnb = GaussianNB()
gnb.fit(data_train, target_train)

#Predicting with Naive Bayes classifier

predictions = gnb.predict(data_train)
print(predictions)
print ('Accuracy Score: ' + str(accuracy_score(target_train, predictions)))
print()
print (classification_report(target_train, predictions))
print (confusion_matrix(target_train, predictions))
print('     Confusion Matrix')
m_confusion_test = metrics.confusion_matrix(target_train, predictions)
pd.DataFrame(m_confusion_test, columns = ['Predicted Ham', 'Predicted Spam'],
            index = ['Actual Ham', 'Actual Spam'])
# Setting some information that will be useful

TP = m_confusion_test[0][0]
TN = m_confusion_test[1][1]
FP = m_confusion_test[1][0]
FN = m_confusion_test[0][1]
Trues = TP + TN
Falses = FP + FN
All = TP + TN + FP + FN
recall = TP / (TP + FN)
precision = TP / (TP + FP)
specificity = TN / (TN + FP)
F3_score = 10 * precision * recall / (9 * precision + recall)
print ('F3 Score: ' + str(F3_score))
# Calculate the Fpr and Tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(target_train, predictions)
roc_auc = metrics.auc(fpr, tpr)

# Ploting the (ROC Curve)

plt.title('Receiver Operating Characteristic (The ROC cruve)')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
CVlist = [5, 7, 10, 13, 15, 17, 20, 25, 30]
for i in CVlist:
    scores = cross_val_score(gnb, data_train, target_train, cv=i)
    strI = str(i)
    mean = str(np.mean(scores))
    print('Scores with CV equal to ' + strI)
    print(scores)
    print()
    print('Mean of scores: ' + mean)
    print()
    print()
#Spliting the data in target(output) and features(input)

outputDATA = train['ham']
inputDATA = train.drop(columns=['ham'])
#Spliting the data in train output, train inputs, test output and test inputs

dataVtrain, dataVtest, targetVtrain, targetVtest = train_test_split(
    inputDATA,
    outputDATA,
    random_state = 0) 
# Setting the K numbers and our datas of test and train

neighbors = [3,5,7,9,13,15,20,25]
Xtrain = dataVtrain
Ytrain = targetVtrain
Xtest = dataVtest
Ytest = targetVtest
# Here, we define a function that test (with CV equal to 3) various K-nn and yours misclassification error

print('With CV = 3')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=3, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 3)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 5) various K-nn and yours misclassification error

print('With CV = 5')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 5)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 7) various K-nn and yours misclassification error

print('With CV = 7')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=7, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 7)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 9) various K-nn and yours misclassification error

print('With CV = 9')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=9, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 9)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 13) various K-nn and yours misclassification error

print('With CV = 13')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=13, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 13)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 15) various K-nn and yours misclassification error

print('With CV = 15')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=15, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 15)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 20) various K-nn and yours misclassification error

print('With CV = 20')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=20, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 20)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 25) various K-nn and yours misclassification error

print('With CV = 25')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=25, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 25)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Set the K-nn with K=13 and CV=20

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(Xtrain,Ytrain)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=20)
print(scores)
# Set the prediction

YtrainPred = knn.predict(Xtrain)
print ('Accuracy Score: ' + str(accuracy_score(Ytrain, YtrainPred)))
print()
print (classification_report(Ytrain, YtrainPred))
print (confusion_matrix(Ytrain, YtrainPred))
print('     Confusion Matrix')
n_confusion_test = metrics.confusion_matrix(Ytrain, YtrainPred)
pd.DataFrame(n_confusion_test, columns = ['Predicted Ham', 'Predicted Spam'],
            index = ['Actual Ham', 'Actual Spam'])
# Setting some information that will be useful

TP = n_confusion_test[0][0]
TN = n_confusion_test[1][1]
FP = n_confusion_test[1][0]
FN = n_confusion_test[0][1]
Trues = TP + TN
Falses = FP + FN
All = TP + TN + FP + FN
recall = TP / (TP + FN)
precision = TP / (TP + FP)
specificity = TN / (TN + FP)
F3_score = 10 * precision * recall / (9 * precision + recall)
print ('F3 Score: ' + str(F3_score))
# Calculate the Fpr and Tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Ytrain, YtrainPred)
roc_auc = metrics.auc(fpr, tpr)

# Ploting the (ROC Curve)

plt.title('Receiver Operating Characteristic (The ROC cruve)')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# Open the file that contains a sample how to send ours predictions

sample = pd.read_csv("../input/pmrdatasettarefa/sample_submission_1.csv")
print ("submission size: ", sample.shape)
sample.head()
# Predicting and sending the predictions

predictionsTOsubmit = gnb.predict(testTOsubmit)
str(predictionsTOsubmit)
ids = testTOsubmit['Id']
submission = pd.DataFrame({'Id':ids,'ham':predictionsTOsubmit[:]})
submission.to_csv("predictions.csv", index = False)
submission.shape
submission.head()