# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import itertools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data_path = '../input/creditcard.csv'
data = pd.read_csv(data_path)
featureInfo = {'Min':data.min(), 'Mean':data.mean(), 'Std':data.std(), 'Max':data.max()}
print(pd.DataFrame(data=featureInfo))
print('===============================================================')
data.info()
print('===============================================================')
numEachClass = data['Class'].value_counts()
plt.title("Number of samples of each class")
plt.xlabel("Class")
plt.ylabel("Number of samples")
numEachClass.plot.bar()
numPositives = numEachClass.tolist()[1]
numNegatives = numEachClass.tolist()[0]
totalNumSamples = len(data)
print("Total: " + repr(totalNumSamples))
print('Positive samples: ' + repr(numPositives) + "\taccount for: " + repr(numPositives/totalNumSamples*100) + "%")
print("Negative samples: " + repr(numNegatives) + "\taccount for: " + repr(numNegatives/totalNumSamples*100) + "%")
plt.figure(figsize=(15,5))
plt.subplot(121)
AmntNotFraud = data[data.Class == 0].Amount
plt.title('Not Fraud, Mean= %0.2f' %AmntNotFraud.mean())
plt.xlabel('Amount')
AmntNotFraud.plot.hist()
plt.subplot(122)
AmntFraud = data[data.Class == 1].Amount
plt.title('Fraud, Mean= %0.2f' %AmntFraud.mean())
plt.xlabel('Amount')
AmntFraud.plot.hist()
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.title('Not Fraud, Amount <= 4000')
plt.xlabel('Amount')
data[(data.Class == 0) & (data.Amount <= 4000)].Amount.plot.hist()
plt.subplot(122)
plt.title('Fraud, Amount <= 4000')
plt.xlabel('Amount')
data[(data.Class == 1) & (data.Amount <= 4000)].Amount.plot.hist()
import seaborn as sns

plt.figure(figsize=(20,10))
plt.title('Correlation values between features')
sns.heatmap(data.corr(), fmt='.1f', annot=True)
timeCol = data.loc[:,['Time','Class']]
timeCol.Time /= 3600
timeCol.Time = timeCol.Time.astype(int)
timeCol.Time = timeCol.Time%24

timeNegative = timeCol.loc[timeCol.Class == 0,:].groupby('Time')['Class'].count()
timePositive = timeCol.loc[timeCol.Class == 1,:].groupby('Time')['Class'].count()
#Plotting
fig = plt.figure(figsize=(20,8))
sbplt1 = plt.subplot(121)
sbplt1.set_ylabel('No. of transactions')
sbplt1.set_title('Not fraud')
timeNegative.plot.bar()
sbplt2 = plt.subplot(122)
sbplt2.set_ylabel('No. of transactions')
sbplt2.set_title('Fraud')
timePositive.plot.bar()

data.iloc[:,1:30] = StandardScaler().fit_transform(data.iloc[:,1:30])
data_incl_time = data.copy()
data = data.drop(['Time'],axis=1)
#data.describe()
data_incl_time.Time /= 3600
data_incl_time.Time = data_incl_time.Time.astype(int)
data_incl_time.Time = data_incl_time.Time %24
#data_incl_time.describe()
# Option 1: Dataset includes 'Time' feature
X = data_incl_time.loc[:,data_incl_time.columns != 'Class']
y = data_incl_time.loc[:,data_incl_time.columns == 'Class']
# Option 2: Dataset excludes 'Time' feature
""" 
X = data.loc[:,data.columns != 'Class']
y = data.loc[:,data.columns == 'Class']
"""
# Prepare train set, test set
    # Without Stratification
trainValidX, testX, trainValidy, testy = train_test_split(X, y, test_size = 0.2, random_state = 48)

    # With Stratification
sTrainValidX, sTestX, sTrainValidy, sTesty = train_test_split(X, y, test_size = 0.2, random_state = 48, stratify=y)

    #Simple Stratified Holdout CV
sTrainX, sValidX, sTrainy, sValidy = train_test_split(sTrainValidX, sTrainValidy, test_size = 0.25, random_state = 48, stratify=sTrainValidy)
print('Train set: ~80%\tTest set: ~20%')
dsbTrain = trainValidy['Class'].value_counts()
dsbTest = testy['Class'].value_counts()
prcTrain = dsbTrain[1]/numPositives
prcTest = dsbTest[1]/numPositives
print('Without stratification:\n\t+ Train set: Negative= ' + repr(dsbTrain[0]) + '\tPositive= ' + repr(dsbTrain[1]) +' ~%0.3f'%prcTrain)
print('\t+ Test set: Negative= ' + repr(dsbTest[0]) + '\tPositive= ' + repr(dsbTest[1]) + ' ~%0.3f'%prcTest)
print('==============================')
dsbTrain = sTrainValidy['Class'].value_counts()
dsbTest = sTesty['Class'].value_counts()
prcTrain = dsbTrain[1]/numPositives
prcTest = dsbTest[1]/numPositives
print('With stratification:\n\t+ Train set: Negative= ' + repr(dsbTrain[0]) + '\tPositive= ' + repr(dsbTrain[1]) +' ~%0.3f'%prcTrain)
print('\t+ Test set: Negative= ' + repr(dsbTest[0]) + '\tPositive= ' + repr(dsbTest[1]) + ' ~%0.3f'%prcTest)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, mean_absolute_error, roc_curve, auc, average_precision_score

def evaluate(param, weight, trainX, trainy, testX, testy):
    print("======================================================")
    print("\nRegularization Parameter = ", param, "\n")
    logRegr = LogisticRegression(C = param, penalty = 'l1', class_weight = {0:1/(1+weight), 1:weight/(1+weight)})
    logRegr.fit(trainX, trainy.values.ravel())
    
    prediction = logRegr.predict(testX.values)

    print("MAE:", mean_absolute_error(testy, prediction))
    print("Accuracy score: ", accuracy_score(testy, prediction, normalize = False), "/", len(validy))
    print("F1 score: ", f1_score(testy, prediction))
    print("Precision score: ", precision_score(testy, prediction))
    print("Recall score: ", recall_score(testy, prediction))
    print("AUPRC: ", average_precision_score(testy.values.ravel(), logRegr.predict_proba(testX.values)))
    
    return prediction
print("Evaluating on test set. Regularizing param = ", 0.3)
logRegr = LogisticRegression(C = 0.3, penalty = 'l1')
logRegr.fit(trainValidX, trainValidy.values.ravel())
prediction = logRegr.predict(testX.values)
print("Recall score: ", recall_score(testy, prediction))
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def draw_cnf_matrix(testy, prediction):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(testy, prediction)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[0,1], title='Confusion matrix, without normalization')

draw_cnf_matrix(testy, prediction)
prediction_proba = logRegr.predict_proba(testX.values)[:,1]

def draw_roc_curve(testy, prediction_proba):
    fpr, tpr, thresholds = roc_curve(testy.values.ravel(), prediction_proba, pos_label = 1)

    #plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

draw_roc_curve(testy, prediction_proba)
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

def draw_prc(testy, prediction_proba):
    precision, recall, threshold = precision_recall_curve(testy.values.ravel(), prediction_proba)
    ap = average_precision_score(testy.values.ravel(), prediction_proba)
    threshold = np.append(threshold, 1)
    plt.step(recall, precision, color='r', where='post', label = 'AUPRC = %0.2f' %ap)
    plt.title('Precision Recall Curve')
    plt.legend(loc='lower left')
    
draw_prc(testy, prediction_proba)
from sklearn.model_selection import StratifiedKFold
def select_skfold(X, y):
    
    print('Stratified k-fold cross validation:')
    reg_params = [0.03, 0.3, 3.0, 30, 300]
    print('Regularizing parameter range:\n', reg_params)
    
    folds_num = 4
    folds = StratifiedKFold(n_splits = folds_num)
    print('Num of folds = ', folds_num)
    
    best_recall = 0;
    best_param = 0;
    
    print('Looping...')
    for param in reg_params:
        print('Param = ', param)
        logRegr = LogisticRegression(C = param, penalty = 'l1')
        avg_recall = 0
        
        for train, test in folds.split(X, y):
            trainX, testX = X.iloc[train,:], X.iloc[test,:]
            trainy, testy = y.iloc[train,:], y.iloc[test,:]
            logRegr.fit(trainX, trainy.values.ravel())
            kfold_predict = logRegr.predict(testX.values)
            rec = recall_score(testy.values, kfold_predict)
            avg_recall += rec
        
        avg_recall /= folds_num
        print('Average recall score = ', avg_recall)
        
        if rec > best_recall:
            best_recall = rec
            best_param = param
    
    return best_param

def draw_prthreshold(test, p, weight = 1):
    ap = average_precision_score(test.values.ravel(), p)
    precision, recall, t = precision_recall_curve(test.values.ravel(), p)
    t = np.append(t, 1)
    plt.title('N/P class weight: 1:'+ repr(weight) + ' ; AUPRC = {0:0.6f}'.format(ap))
    plt.plot(t, recall, color='r')
    plt.plot(t, precision, color='b')

    plt.xlabel('Threshold')
    plt.legend(('recall','precision'))
from itertools import cycle

best_param = select_skfold(trainValidX, trainValidy)
weight_range = [1,2,5,10,20,50,100,200,300,400,500]
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue','black', 'purple'])
plt.figure(1, figsize=(20,8))
j = 1
for weight, color in zip(weight_range, colors):
    print('===============================================')
    print('Positive class weight = ', weight)
    logRegr = LogisticRegression(C = best_param, penalty = 'l1', class_weight = {0:1/(1+weight), 1:weight/(1+weight)})
    logRegr.fit(trainValidX, trainValidy.values.ravel())
    prediction = logRegr.predict(testX.values)
    
    print("F1 score: ", f1_score(testy, prediction), "\tPrecision score: ", precision_score(testy, prediction), "\tRecall score: ", recall_score(testy, prediction))
    proba = logRegr.predict_proba(testX.values)[:,1]
    ap = average_precision_score(testy.values.ravel(), proba)
    print("Accuracy score: ", accuracy_score(testy, prediction, normalize = True), '\tAUPRC: ', ap)
    
    plt.figure(2, figsize=(28,15))
    plt.subplot(3,4,j)
    j += 1
    draw_prthreshold(testy, proba)
    
    plt.figure(1)
    plt.plot(recall, precision, color=color, label='PosClassWeight= %s'%weight)
    plt.legend(loc="lower left")
from sklearn.ensemble import RandomForestClassifier

weight_range = [1,2,5,10,20,50,100,200,300,400,500]
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue','black', 'purple'])
plt.figure(0, figsize=(20,8))
j = 1
for weight, color in zip(weight_range, colors):
    print('===============================================')
    print('Positive class weight = ', weight)
    ranFor = RandomForestClassifier(class_weight = {0:1/(1+weight), 1:weight/(1+weight)})
    ranFor.fit(trainValidX, trainValidy.values.ravel())
    prediction = ranFor.predict(testX.values)
    
    print("F1 score: ", f1_score(testy, prediction), "\tPrecision score: ", precision_score(testy, prediction), "\tRecall score: ", recall_score(testy, prediction))
    proba = ranFor.predict_proba(testX.values)[:,1]
    precision, recall, threshold = precision_recall_curve(testy.values.ravel(), proba)
    threshold = np.append(threshold, 1)
    ap = average_precision_score(testy.values.ravel(), proba)
    print("Accuracy score: ", accuracy_score(testy, prediction, normalize = True), '\tAUPRC: ', ap)
    
    plt.figure(1, figsize=(28,15))
    plt.subplot(3,4,j)
    j += 1
    plt.title('N/P class weight: 1:'+ repr(weight) + ' ; AUPRC = {0:0.6f}'.format(ap))
    plt.plot(threshold, recall, color='r')
    plt.plot(threshold, precision, color='b')

    plt.xlabel('Threshold')
    plt.legend(('recall','precision'))
    
    plt.figure(0)
    plt.plot(recall, precision, color=color, label='PosClassWeight= %s'%weight)
    plt.legend(loc="lower left")
from sklearn.cluster import KMeans

kmclstr = KMeans(n_clusters = 10, random_state = 48)
kmclstr.fit(trainValidX)
centroids_pos = kmclstr.cluster_centers_
centroids_pred = kmclstr.predict(testX)
distances = [np.linalg.norm(a - b) for a,b in zip(testX.as_matrix(), centroids_pos[centroids_pred])]
print("Done")
from sklearn.metrics import roc_auc_score

prediction = np.array(distances)
threshold = 98
threshold_step = 0.05
threshold_end = 99.9
while threshold <= threshold_end:
    print("=======================================")
    print(threshold, "-th percentile")
    proba = prediction_proba.copy()
    #print(proba)
    print(proba.max())
    proba[distances >= np.percentile(distances, threshold)] = 1
    proba[distances < np.percentile(distances, threshold)] = 0
    #print(proba)
    print("F1 score: ", f1_score(testy, proba), "\tPrecision score: ", precision_score(testy, proba), "\tRecall score: ", recall_score(testy, proba))
    print("Accuracy score: ", accuracy_score(testy, proba, normalize = True), "\tAUPRC score: ", average_precision_score(testy, prediction_proba))
    print(roc_auc_score(testy, prediction_proba))
    threshold += threshold_step
from sklearn.ensemble import IsolationForest

isoFor = IsolationForest(contamination = 0.01, random_state = 48)
isoFor.fit(sTrainValidX)
isoFor_predict = isoFor.predict(sTestX)
isoFor_decision = isoFor.decision_function(sTestX)
isoFor_decision = MinMaxScaler().fit_transform(isoFor_decision.reshape(-1,1))
isoFor_decision = 1 - isoFor_decision

isoFor_predict[isoFor_predict == 1] = 0
isoFor_predict[isoFor_predict == -1] = 1
print("F1 score: ", f1_score(sTesty, isoFor_predict), "\tPrecision score: ", precision_score(sTesty, isoFor_predict), "\tRecall score: ", recall_score(sTesty, isoFor_predict))
print("Accuracy score: ", accuracy_score(sTesty, isoFor_predict, normalize = True))#, "\tAUPRC score: ", average_precision_score(testy, proba))
print("AUROC score", roc_auc_score(sTesty, isoFor_decision))

cnf_matrix = confusion_matrix(sTesty, isoFor_predict)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1],
                      title='Confusion matrix, without normalization')
