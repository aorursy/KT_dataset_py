import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
#import seaborn as seabornInstance 
#from sklearn.model_selection import train_test_split 
from tpot import TPOTClassifier
import sklearn.metrics as metrics
%matplotlib inline
#flightdata = pd.read_csv("https://introtomlsampledata.blob.core.windows.net/data/flightdelays/flightdelays.csv")
flightdata = pd.read_csv(r"../input/flightdelays/flightdelays.csv")
print(flightdata.shape)
flightdata.columns
flightdata.dtypes
flightdata.head(10)
#flightdata.describe()
carrierlist = list(flightdata.Carrier.unique())
carrierlist.sort()
carrierdict = {carrierlist[i]: list(range(len(carrierlist)))[i] for i in range(len(carrierlist))} 
flightdata["Carrier"] = flightdata["Carrier"].replace(carrierdict) 
flightdata.Carrier.unique()
train = flightdata[flightdata["Month"] < 10]
test = flightdata[flightdata["Month"] >= 10]
print(train.shape, test.shape)
train = train.drop(
    ["Month", "Year", "Year_R", "Timezone", "Timezone_R"], axis=1)
test = test.drop(["Month", "Year", "Year_R", "Timezone", "Timezone_R"], axis=1)
print(train.shape, test.shape)
trainX = train.drop(["ArrDel15"],axis = 1)
trainy = train["ArrDel15"]
print(trainX.shape,trainy.shape)
testX = test.drop(["ArrDel15"],axis = 1)
testy = test["ArrDel15"]
print(testX.shape,testy.shape)
pipeline_optimizer = TPOTClassifier(generations=100, population_size=20, cv=5,
                                    random_state=42, verbosity=2, max_time_mins = 60)

pipeline_optimizer.fit(trainX, trainy)
predicted_classes = pipeline_optimizer.predict(testX)
accuracy = metrics.accuracy_score(testy,predicted_classes)
accuracy
confusion = metrics.confusion_matrix(testy, pipeline_optimizer.predict(testX))
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
# use float to perform true division, not integer division
print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(testy, pipeline_optimizer.predict(testX)))
classification_error = (FP + FN) / float(TP + TN + FP + FN)

print(classification_error)
print(1 - metrics.accuracy_score(testy, pipeline_optimizer.predict(testX)))
sensitivity = TP / float(FN + TP)

print(sensitivity)
print(metrics.recall_score(testy, pipeline_optimizer.predict(testX)))
specificity = TN / (TN + FP)

print(specificity)
false_positive_rate = FP / float(TN + FP)

print(false_positive_rate)
print(1 - specificity)
precision = TP / float(TP + FP)

print(precision)
print(metrics.precision_score(testy, pipeline_optimizer.predict(testX)))
# calculate the fpr and tpr for all thresholds of the classification
probs = pipeline_optimizer.predict_proba(testX)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(testy, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(testy, preds))