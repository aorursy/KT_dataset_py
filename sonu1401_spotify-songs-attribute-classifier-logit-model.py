import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df=pd.read_csv("../input/data.csv")
df.head()
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='target',data=df,palette='rainbow')
sns.countplot(x='target',data=df,palette='rainbow',hue='mode')
sns.countplot(x='target',data=df,hue='time_signature',palette='rainbow')
df.info()
df1=df.drop(['song_title','artist'],axis=1)
df1.info()
y=df1['target']
x=df1.drop(['target'],axis=1)
y.shape
plt.figure(figsize=(15,5))
sns.heatmap(df1.corr(),annot=True)
# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=0)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
predictions[1:10]
#plt.scatter(y_test,predictions)
sns.distplot(predictions-y_test,hist=True,vertical=False,color='r')
logmodel.coef_
logmodel.intercept_
#Calculate Accuracy 
from sklearn import metrics
metrics.accuracy_score(y_test,predictions)
1-metrics.accuracy_score(y_test,predictions)
y_test.value_counts()
# calculate the percentage of ones
y_test.mean()
# calculate the percentage of zeros
1-y_test.mean()
# calculate null accuracy (for binary classification problems coded as 0/1)
max(y_test.mean(), 1 - y_test.mean())
#print first 20 actual value and its corresponding predictions value ¶

print('20 Actual Value:',y_test.values[0:20])
print('20 Predicted Value:',predictions[0:20])
from sklearn import metrics
confusion_matrix=metrics.confusion_matrix(y_test,predictions)
list1 = ["Actual 0", "Actual 1"]
list2 = ["Predicted 0", "Predicted 1"]
pd.DataFrame(confusion_matrix, list1,list2)
# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, predictions)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print('TN:',TN)
print('TP:',TP)
print('FN:',FN)
print('FP:',FP)
Accuracy_Score=(TP+TN)/(TP+TN+FP+FN)
Accuracy_Score
metrics.accuracy_score(y_test,predictions)
Misclassification_Rate=(FP+FN)/(TP+TN+FP+FN)
Misclassification_Rate
1-metrics.accuracy_score(y_test,predictions)
Sensitivity=TP/(TP+FN)
Sensitivity
metrics.recall_score(y_test, predictions)
Specificity=TN/(TN+FP)
Specificity
FPR=FP/(FP+TN)
FPR
FNR=FN/(FN+TP)
FNR
Precision=TP/(TP+FP)
Precision
metrics.precision_score(y_test,predictions)
Prevalence = (TP+FN)/(TP+FN+FP+TN)
Prevalence
FDR= FP/(TP+FP)
FDR
FOR = FN/(TN+FN)
FOR
NPV = TN/(TN+FN)
NPV
PLR = Sensitivity/FPR
PLR
NLR=FNR/Specificity
NLR
DOR=PLR/NLR
DOR
F1_Score=2/((1/Sensitivity)+(1/Precision))
F1_Score
#print first 20 actual value and its corresponding predictions value ¶

print('10 Actual Value:',y_test.values[0:10])
print('10 Predicted Value:',predictions[0:10])
# print the first 10 predicted probabilities of class membership
logmodel.predict_proba(X_test)[:10]

#predict_proba gives you the probabilities for the target (0 and 1 in your case) in array form. 
#The number of probabilities for each row is equal to the number of categories in target variable 4
#(2 in your case).

# i.e. prob of 0 is 0.55004114 and prob of 1 is 0.44995886 hence in logmodel.predict(X_test)[0:10] will give 
# output as 0 as class variable(0) has dominancy in terms of probablity value
# print the first 10 predicted probabilities for class 0
print(logmodel.predict_proba(X_test)[0:10, 0])

print("$$$$$$$$$$$$$$$$$$$$$$#####################$$$$$$$$$$$$$$$$")

# print the first 10 predicted probabilities for class 1
print(logmodel.predict_proba(X_test)[0:10, 1])
# store the predicted probabilities for class 1
y_pred_prob_1 = logmodel.predict_proba(X_test)[:, 1]

# store the predicted probabilities for class 0
y_pred_prob_0 = logmodel.predict_proba(X_test)[:, 0]

y_pred_prob=logmodel.predict_proba(X_test)
plt.figure(figsize=(15,5))
plt.hist(y_pred_prob_1,bins=8,color='g')
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities of music like')
plt.xlabel('Predicted probability whether i would like music')
plt.ylabel('Frequency')


plt.figure(figsize=(15,5))
plt.hist(y_pred_prob_0,bins=8,color='r')
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities of donot likemusic')
plt.xlabel('Predicted probability whether i would not like music')
plt.ylabel('Frequency')
# predict like or dislike of music if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize
y_pred_class = binarize([y_pred_prob_1], 0.3)[0]
# print the first 10 predicted probabilities
print(y_pred_prob[0:10])

print("##############################################")


# print the first 10 predicted classes with the lower threshold
print(y_pred_class[0:10])
# previous confusion matrix (default threshold of 0.5)
print(confusion)

# new confusion matrix (threshold of 0.3)
print(metrics.confusion_matrix(y_test, y_pred_class))
# sensitivity has increased (used to be  0.5830618892508144)
print("Before:",179/ float(179 + 128))

# sensitivity has increased (used to be  0.5830618892508144)
print("Now at 0.3 threshold:",307/ float(307 + 0))

# specificity has decreased (used to be 0.46488294314381273)
print("Specificity Before:",139 / float(139 + 160))

# specificity has decreased (used to be 0.46488294314381273)
print("Specificity at 0.3 threshold:",0 / float(0 + 299))
# IMPORTANT: first argument is true values, second argument is predicted probabilities
FPR, TPR, thresholds = metrics.roc_curve(y_test, predictions)
plt.plot(FPR, TPR)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, predictions))
# calculate cross-validated AUC
from sklearn.cross_validation import cross_val_score
cross_val_score(logmodel, x, y, cv=10, scoring='roc_auc').mean()
plt.figure(figsize=(15,5))
sns.heatmap(df1.corr(),annot=True)

from pandas.plotting import scatter_matrix
scatter_matrix(df1, alpha=0.2, figsize=(20,20), diagonal='kde')
df1.corr(method='pearson', min_periods=1)
df.info()
df2=df
y1=df2['target']
x1=df2.drop(['song_title','artist','target','loudness','energy','danceability','instrumentalness'],axis=1)
plt.figure(figsize=(15,5))
sns.heatmap(x1.corr(),annot=True)
# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.30,random_state=0)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X1_train,y1_train)

predictions1 = logmodel.predict(X1_test)
predictions1[1:10]
#Calculate Accuracy 
from sklearn import metrics
metrics.accuracy_score(y1_test,predictions1)
from sklearn import metrics
confusion_matrix=metrics.confusion_matrix(y1_test,predictions1)
list1 = ["Actual 0", "Actual 1"]
list2 = ["Predicted 0", "Predicted 1"]
pd.DataFrame(confusion_matrix, list1,list2)