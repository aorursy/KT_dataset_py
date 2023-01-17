# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# Reading the dataset
data = pd.read_csv("../input/Financial Distress.csv")

# Checking the dataset
data.head()
data.tail()
data.describe()

print("Number of companies:",data.Company.unique().shape)
data = data[data.columns.drop(list(data.filter(regex='x80')))] # Since it is a categorical feature with 37 features.

# Creating target vector and feature matrix
Y = data.iloc[:,2].values
for y in range(0,len(Y)): # Coverting target variable from continuous to binary form
       if Y[y] > -0.5:
              Y[y] = 0
       else:
              Y[y] = 1
X = data.iloc[:,3:].values
# Counting number of observations for Healthy and and Bankrupt Companies:
num_zeros = 0
for num in Y:
       if num == 0:
              num_zeros = num_zeros+1
num_ones = len(Y) - num_zeros 

print("Number of observations for BANKRUPT companies(1's):",num_ones)
print("Number of observations for HEALTHY companies(0's):",num_zeros)


# Splitting the data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 0.30, random_state = 0)
X_train_wo_sampling = X_train
y_train_wo_sampling = y_train

# Creating more samples units for the bankrupt companies(undersampled data)
y_train = (np.matrix(y_train)).T
y_train = pd.DataFrame(y_train)
y_train.columns = ["Financial_Distress"]
X_train = pd.DataFrame(X_train)
frame = [X_train,y_train]
train_data = pd.concat(frame,axis = 1)
bankrupt_companies = train_data[train_data.Financial_Distress == 1]

feat_mat = bankrupt_companies.iloc[:,:-1].values
response = bankrupt_companies.iloc[:,-1].values
col_mean = np.zeros(shape=(82,1)) 
col_std = np.zeros(shape=(82,1)) 
Dim_1 = np.shape(feat_mat)
for i in range(0,Dim_1[1]): # Logic to calculate mean and standard deviation for each column
       col_mean[i,0] = np.mean(feat_mat[:,i])
       col_std[i,0] = np.std(feat_mat[:,i])
col_mean_and_std = np.hstack((col_mean,col_std))

added_data = np.zeros(shape=(1200,Dim_1[1])) 
for i in range (0,len(col_mean_and_std)):
       mean_ = col_mean_and_std[i,0]
       std_ = col_mean_and_std[i,1]
       added_data[:,i] = np.random.normal(mean_,std_,1200)
added_y = np.ones(shape=(1200,1)) # Creating labels for the added data

X_resampled = np.vstack((X_train,added_data)) # Combining the original data + added data
y_train = np.array(y_train)
y_resampled = np.vstack((y_train,added_y))



# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_resampled = sc.fit_transform(X_resampled)
X_test = sc.transform(X_test)
# Fitting XGBClassifier to the training data: Model_1
from xgboost import XGBClassifier
classifier_1 = XGBClassifier()
classifier_1.fit(X_resampled,y_resampled)

# Fitting SVM to the training data: Model 2
from sklearn.svm import SVC
classifier_2 = SVC(kernel = 'linear', C = 1, probability = True, random_state = random.seed(123)) # poly, sigmoid
classifier_2.fit(X_resampled, y_resampled)

# Creating and Fitting Random Forest Classifier to the training data: Model 3
from sklearn.ensemble import RandomForestClassifier
classifier_3 = RandomForestClassifier(n_estimators = 5, criterion = 'entropy')
classifier_3.fit(X_resampled, y_resampled)

# Fitting classifier to the training data: Model 4 
from sklearn.linear_model import LogisticRegression
classifier_4 = LogisticRegression(penalty = 'l1', random_state = 0)
classifier_4.fit(X_resampled, y_resampled)

# Fitting Balanced Bagging Classifier to the training data: Model 5
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier
classifier_5 = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'),
                                       n_estimators = 5, bootstrap = True)
classifier_5.fit(X_resampled,y_resampled)

# Fitting Decision Tree to the training data: Model 6
from sklearn.tree import DecisionTreeClassifier
classifier_6 = DecisionTreeClassifier()
classifier_6.fit(X_resampled,y_resampled)

# Fitting Naive Bayes to the training data: Model 7
from sklearn.naive_bayes import GaussianNB
classifier_7 = GaussianNB()
classifier_7.fit(X_resampled,y_resampled)

# Predicting the results
y_pred_1 = classifier_1.predict(X_test)
y_pred_2 = classifier_2.predict(X_test)
y_pred_3 = classifier_3.predict(X_test)
y_pred_4 = classifier_4.predict(X_test)
y_pred_5 = classifier_5.predict(X_test)
y_pred_6 = classifier_6.predict(X_test)
y_pred_7 = classifier_7.predict(X_test)

# Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(y_test,y_pred_1)
accuracy_1 = (cm_1[0,0]+cm_1[1,1])/len(y_test)

cm_2 = confusion_matrix(y_test,y_pred_2)
accuracy_2 = (cm_2[0,0]+cm_2[1,1])/len(y_test)

cm_3 = confusion_matrix(y_test,y_pred_3)
accuracy_3 = (cm_3[0,0]+cm_3[1,1])/len(y_test)

cm_4 = confusion_matrix(y_test,y_pred_4)
accuracy_4 = (cm_4[0,0]+cm_4[1,1])/len(y_test)

cm_5 = confusion_matrix(y_test,y_pred_5)
accuracy_5 = (cm_5[0,0]+cm_5[1,1])/len(y_test)

cm_6 = confusion_matrix(y_test,y_pred_6)
accuracy_6 = (cm_6[0,0]+cm_6[1,1])/len(y_test)

cm_7 = confusion_matrix(y_test,y_pred_7)
accuracy_7 = (cm_7[0,0]+cm_7[1,1])/len(y_test)

from sklearn.metrics import precision_recall_fscore_support
precision_1, recall_1, f_score_1, support = precision_recall_fscore_support(y_test, y_pred_1, average = None)
print("\nFor Model 1 - XGBoost:")
print("Precision:",precision_1)
print("Recall:",recall_1)
print("F-Score:",f_score_1)
print("Accuracy_XGBoost:",accuracy_1*100,'%')

precision_2, recall_2, f_score_2, support = precision_recall_fscore_support(y_test, y_pred_2, average = None)
print("\nFor Model 2 - SVC:")
print("Precision:",precision_2)
print("Recall:",recall_2)
print("F-Score:",f_score_2)
print("Accuracy_SVC:",accuracy_2*100,'%') 

precision_3, recall_3, f_score_3, support = precision_recall_fscore_support(y_test, y_pred_3, average = None)
print("\nFor Model 3 - Random Forest:")
print("Precision:",precision_3)
print("Recall:",recall_3)
print("F-Score:",f_score_3)
print("Accuracy_RF:",accuracy_3*100,'%') 

precision_4, recall_4, f_score_4, support = precision_recall_fscore_support(y_test, y_pred_4, average = None)
print("\nFor Model 4 - Logistic:")
print("Precision:",precision_4)
print("Recall:",recall_4)
print("F-Score:",f_score_4)
print("Accuracy_Logistic:",accuracy_4*100,'%') 

precision_5, recall_5, f_score_5, support = precision_recall_fscore_support(y_test, y_pred_5, average = None)
print("\nFor Model 5 - BalancedBaggingClassifier:")
print("Precision:",precision_5)
print("Recall:",recall_5)
print("F-Score:",f_score_5)
print("Accuracy_BalancedBagging:",accuracy_5*100,'%')

precision_6, recall_6, f_score_6, support = precision_recall_fscore_support(y_test, y_pred_6, average = None)
print("\nFor Model 6 - Decision Tree Classifier:")
print("Precision:",precision_6)
print("Recall:",recall_6)
print("F-Score:",f_score_6)
print("Accuracy_DecisionTree:",accuracy_6*100,'%') 

precision_7, recall_7, f_score_7, support = precision_recall_fscore_support(y_test, y_pred_7, average = None)
print("\nFor Model 7 - Naive Bayes Classifier:")
print("Precision:",precision_7)
print("Recall:",recall_7)
print("F-Score:",f_score_7)
print("Accuracy_NaiveBayes:",accuracy_7*100,'%')
# Comparing the Accuracies of various models
ACCURACY = np.vstack((accuracy_1,accuracy_2,accuracy_3,accuracy_4,accuracy_5,accuracy_6,accuracy_7))
number = np.array([1,2,3,4,5,6,7])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(number, ACCURACY, color = 'r', marker = 'o', linewidths = 3)
for j in range(0,len(ACCURACY)):
       ax.annotate('%0.3f' % (ACCURACY[j]),(number[j], ACCURACY[j]))

plt.xlabel('MODELS')    
plt.ylabel('ACCURACY')
# Comparing the F-Values of various models
F_SCORE = np.vstack((f_score_1[1],f_score_2[1],f_score_3[1],f_score_4[1],f_score_5[1],f_score_6[1],f_score_7[1]))
number = np.array([1,2,3,4,5,6,7])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(number, F_SCORE, marker = 'x', linewidths = 3)
for i in range(0,len(F_SCORE)):
       ax.annotate('%0.3f' % (F_SCORE[i]),(number[i], F_SCORE[i]))

plt.xlabel('MODELS')    
plt.ylabel('F-SCORE')

  

# Plot ROC curves
from sklearn.metrics import roc_curve, auc
from scipy import interp

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

probas_1 = classifier_1.predict_proba(X_test)
probas_2 = classifier_2.predict_proba(X_test)
probas_3 = classifier_3.predict_proba(X_test)
probas_4 = classifier_4.predict_proba(X_test)
probas_5 = classifier_5.predict_proba(X_test)
probas_6 = classifier_6.predict_proba(X_test)
probas_7 = classifier_7.predict_proba(X_test)

probas = np.vstack((probas_1,probas_2,probas_3,probas_4,probas_5,probas_6,probas_7))

# Compute ROC curve and area the curve
pointer = [0,1102,2204,3306,4408,5510,6612,7714] 
for a in range(0,7):
       index_1 = pointer[a]
       index_2 = pointer[a+1]
       fpr, tpr, thresholds = roc_curve(y_test, probas[index_1:index_2, 1])
       tprs.append(interp(mean_fpr, fpr, tpr))
       tprs[-1][0] = 0.0
       roc_auc = auc(fpr, tpr)
       aucs.append(roc_auc)
       plt.plot(fpr, tpr, lw=2, alpha=0.8,
                label='Model %d (AUC = %0.2f)' % (a+1, roc_auc))
       
       a += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
# Testing various oversampling techniques
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Random Oversampler
ros = RandomOverSampler(random_state=0)
X_resampled_ros, y_resampled_ros = ros.fit_sample(X_train, y_train)
print(sorted(Counter(y_resampled_ros).items()))

# Synthetic Minority Oversampling Technique(SMOTE) 
X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(X_train, y_train)
print(sorted(Counter(y_resampled_smote).items()))

# Adaptive Synthetic (ADASYN) sampling method
X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(X_train, y_train)
print(sorted(Counter(y_resampled_adasyn).items()))

# Feature scaling
X_resampled_ros = sc.fit_transform(X_resampled_ros)
X_resampled_smote = sc.fit_transform(X_resampled_smote)
X_resampled_adasyn = sc.fit_transform(X_resampled_adasyn)
X_train_wo_sampling = sc.fit_transform(X_train_wo_sampling)
# Fitting Balanced Bagging Classifier to the training data: Model 5
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier
classifier_ros = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'),
                                       n_estimators = 5, bootstrap = True)
classifier_ros.fit(X_resampled_ros,y_resampled_ros)

classifier_smote = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'),
                                       n_estimators = 5, bootstrap = True)
classifier_smote.fit(X_resampled_smote,y_resampled_smote)

classifier_adasyn = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'),
                                       n_estimators = 5, bootstrap = True)
classifier_adasyn.fit(X_resampled_adasyn,y_resampled_adasyn)

classifier_wo_sampling = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'),
                                       n_estimators = 5, bootstrap = True)
classifier_wo_sampling.fit(X_train_wo_sampling,y_train_wo_sampling)
# Predicting the results
y_pred_ros = classifier_ros.predict(X_test)
y_pred_smote = classifier_smote.predict(X_test)
y_pred_adasyn = classifier_adasyn.predict(X_test)
y_pred_wo_sampling = classifier_wo_sampling.predict(X_test)
# Creating the confusion matrix
cm_ros = confusion_matrix(y_test,y_pred_ros)
accuracy_ros = (cm_ros[0,0]+cm_ros[1,1])/len(y_test)

cm_smote = confusion_matrix(y_test,y_pred_smote)
accuracy_smote = (cm_smote[0,0]+cm_smote[1,1])/len(y_test)

cm_adasyn = confusion_matrix(y_test,y_pred_adasyn)
accuracy_adasyn = (cm_adasyn[0,0]+cm_adasyn[1,1])/len(y_test)

cm_wo_sampling = confusion_matrix(y_test,y_pred_wo_sampling)
accuracy_wo_sampling = (cm_wo_sampling[0,0]+cm_wo_sampling[1,1])/len(y_test)


from sklearn.metrics import precision_recall_fscore_support
precision_ros, recall_ros, f_score_ros, support = precision_recall_fscore_support(y_test, y_pred_ros, average = None)
print("\nFor RandomOversampling:")
print("Precision:",precision_ros)
print("Recall:",recall_ros)
print("F-Score:",f_score_ros)
print("Accuracy_RandomOversampling:",accuracy_ros*100,'%')

from sklearn.metrics import precision_recall_fscore_support
precision_smote, recall_smote, f_score_smote, support = precision_recall_fscore_support(y_test, y_pred_smote, average = None)
print("\nFor SMOTE:")
print("Precision:",precision_smote)
print("Recall:",recall_smote)
print("F-Score:",f_score_smote)
print("Accuracy_smote:",accuracy_smote*100,'%')

from sklearn.metrics import precision_recall_fscore_support
precision_adasyn, recall_adasyn, f_score_adasyn, support = precision_recall_fscore_support(y_test, y_pred_adasyn, average = None)
print("\nFor ADASYN:")
print("Precision:",precision_adasyn)
print("Recall:",recall_adasyn)
print("F-Score:",f_score_adasyn)
print("Accuracy_adasyn:",accuracy_adasyn*100,'%')

from sklearn.metrics import precision_recall_fscore_support
precision_wo_sampling, recall_wo_sampling, f_score_wo_sampling, support = precision_recall_fscore_support(y_test, y_pred_adasyn, average = None)
print("\nWithout Sampling:")
print("Precision:",precision_wo_sampling)
print("Recall:",recall_wo_sampling)
print("F-Score:",f_score_wo_sampling)
print("Accuracy_wo_sampling:",accuracy_wo_sampling*100,'%')
# Comparing the F-Values of various models
F_SCORE = np.vstack((f_score_ros[1],f_score_smote[1],f_score_adasyn[1],f_score_wo_sampling[1],f_score_5[1],)) #,f_score_2[1]
number = np.array([1,2,3,4,5])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(number, F_SCORE, marker = 'x', linewidths = 3)
for i in range(0,len(F_SCORE)):
       ax.annotate('%0.3f' % (F_SCORE[i]),(number[i], F_SCORE[i]))

plt.xlabel('Sampling Techniques')    
plt.ylabel('F-SCORE')