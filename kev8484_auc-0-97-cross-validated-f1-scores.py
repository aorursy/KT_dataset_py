%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
data = pd.read_csv("../input/creditcard.csv")

print("No. of Rows: \t\t", data.shape[0])

print("No. of Columns: \t", data.shape[1])

data.head()
norm_count = data[data['Class'] == 0].shape[0] # Normal transactions

fraud_count = data[data['Class'] == 1].shape[0] # Fraudulent transcations

total_count = data.shape[0]

print("No. of normal transactions: \t\t", norm_count)

print("No. of fraudulent transactions: \t", fraud_count)

print("% normal transactions: \t\t", norm_count/total_count * 100)

print("% fraudulent transcations: \t", fraud_count/total_count * 100)

pd.value_counts(data['Class'], sort = True).sort_index().plot(kind='bar')

plt.title("Class Histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
from sklearn.preprocessing import StandardScaler



data['Amount_scl'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

data = data.drop(['Time','Amount'],axis=1)

data.head()
X = data.ix[:, data.columns != 'Class'] # features

y = data['Class'] # labels

print("X.shape: ", X.shape)

print("y.shape: ", y.shape)
from sklearn.model_selection import train_test_split



# 70% training data, 30% testing data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
# Get the indices of the fraudulent and normal classes:

fraud_idx = np.array(y_train[y_train == 1].index)

num_fraud = len(fraud_idx)

normal_idx = y_train[y_train == 0].index



# From the normal indices, sample a random subset (subset size = # of frauds):

normal_idx_sample = np.random.choice(normal_idx, num_fraud, replace=False)

normal_idx_sample = np.array(normal_idx_sample)



# Group together our normal and fraud indices:

# (we'll have a balanced class distribution, 50% normal, 50% fraud)

undersample_idx = np.concatenate([fraud_idx,normal_idx_sample])



# Grab the records at the indices:

undersample_data = data.iloc[undersample_idx,:]



# Split into features and labels:

X_undersample = undersample_data.ix[:, undersample_data.columns != 'Class']

y_undersample = undersample_data['Class']



norm_count = undersample_data[undersample_data['Class'] == 0].shape[0]

fraud_count = undersample_data[undersample_data['Class'] == 1].shape[0]



print("---Undersampled Data Set---")

print("No. of normal transactions: \t", norm_count)

print("No. of fraudulent transactions: \t\t", fraud_count)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import auc,roc_auc_score,roc_curve,recall_score,f1_score
def get_best_hypers_lr(X, y):

    """ Search parameter space for the optimal values.

    

             -Perform Logistic Regression using a range of C parameter values and two different

             penalty terms (L1 & L2)

             -Compute mean recall,accuracy and f1-scores using kfold cross validation 

             for each run

             -Output the C parameter and penalty term with the best f1 score

    """

    c_range = [0.01, 0.1, 1.0, 10.0, 100.0]

    f1_max = 0

    best_c = 0

    penalty = ''

    

    for c_param in c_range:

        print('='*25)

        print('C parameter: ', c_param)

        print('='*25)

        print('')

    

        print('-'*25)

        print('L1-penalty')

        print('-'*25)

        print('')

        

        lr_l1 = LogisticRegression(C=c_param, penalty='l1')

        acc_score = cross_val_score(lr_l1, X, y, cv=5)

        recall_score = cross_val_score(lr_l1, X, y, cv=5, scoring='recall')

        f1_score = cross_val_score(lr_l1, X,y,cv=5, scoring='f1')

        l1_f1=np.mean(f1_score)

        

        print("Mean Accuracy: %0.3f (+/- %0.3f)" % (np.mean(acc_score), np.std(acc_score)) )

        print("Mean Recall: %0.3f (+/- %0.3f)" % (np.mean(recall_score), np.std(recall_score)) )

        print("Mean F1: %0.3f (+/- %0.3f)" % (np.mean(f1_score), np.std(f1_score)) )

        print('')

        

        print('-'*25)

        print('L2-penalty')

        print('-'*25)

        print('')

        

        lr_l2 = LogisticRegression(C=c_param, penalty='l2')

        score = cross_val_score(lr_l2, X, y, cv=5)

        recall_score = cross_val_score(lr_l2, X, y, cv=5, scoring='recall')

        f1_score = cross_val_score(lr_l2, X, y, cv=5, scoring='f1')

        l2_f1 = np.mean(f1_score)

        

        print("Mean Accuracy: %0.3f (+/- %0.3f)" % (np.mean(acc_score), np.std(acc_score)) )

        print("Mean Recall: %0.3f (+/- %0.3f)" % (np.mean(recall_score), np.std(recall_score)) )

        print("Mean F1: %0.3f (+/- %0.3f)" % (np.mean(f1_score), np.std(f1_score)) )

        print('')

        

        # compare l1_f1 & l2_f1:

        if l2_f1 > l1_f1:

            # compare to max:

            if l2_f1 > f1_max:

                f1_max = l2_f1

                best_c = c_param

                penalty='l2'

        else:

            # compare to max:

            if l1_f1 > f1_max:

                f1_max = l1_f1

                best_c = c_param

                penalty='l1'

            



    print('*'*25)

    print('Optimal C parameter = ', best_c)

    print('Optimal penalty = ', penalty)

    print('Optimal F1 = ', f1_max)

    print('*'*25)

    

    return best_c, penalty
best_c_lr, penalty_lr = get_best_hypers_lr(X_undersample,y_undersample)
# Use best hyperparameters:

lr = LogisticRegression(C=best_c_lr, penalty=penalty_lr)

# Train on full undersample data set:

lr.fit(X_undersample, y_undersample)

# Test on unseen test data set:

y_pred_score = lr.decision_function(X_test.values)

# Compute ROC metrics:

fpr, tpr, thresholds = roc_curve(y_test.values, y_pred_score)

# Get AUC:

roc_auc = auc(fpr,tpr)



# Plot ROC:

plt.title('ROC Curve - Linear Regression')

plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)

plt.plot([0,1],[0,1],'r--')

plt.legend(loc='lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from sklearn.svm import LinearSVC
def get_best_hypers_svc(X, y):

    """ Search parameter space for the optimal values.

    

             -Perform Support Vector Classifier using a range of C parameter values and two different

             penalty terms (L1 & L2)

             -Compute mean recall, accuracy, and f1-scores using kfold cross validation

             for each run

             -Output the C parameter and penalty term with the best f1-score

    """

    c_range = [0.01, 0.1, 1.0, 10.0, 100.0]

    f1_max = 0

    best_c = 0

    penalty = ''

    

    for c_param in c_range:

        print('='*25)

        print('C parameter: ', c_param)

        print('='*25)

        print('')

    

        print('-'*25)

        print('L1-penalty')

        print('-'*25)

        print('')

        

        svc_l1 = LinearSVC(C=c_param, penalty='l1', dual=False)

        acc_score = cross_val_score(svc_l1, X, y, cv=5)

        recall_score = cross_val_score(svc_l1, X, y, cv=5, scoring='recall')

        f1_score = cross_val_score(svc_l1, X, y, cv=5, scoring='f1')

        l1_f1 = np.mean(f1_score)

        

        print("Mean Accuracy: %0.3f (+/- %0.3f)" % (np.mean(acc_score), np.std(acc_score)) )

        print("Mean Recall: %0.3f (+/- %0.3f)" % (np.mean(recall_score), np.std(recall_score)) )

        print("Mean F1: %0.3f (+/- %0.3f)" % (np.mean(f1_score), np.std(f1_score)) )

        print('')

        

        print('-'*25)

        print('L2-penalty')

        print('-'*25)

        print('')

        

        svc_l2 = LinearSVC(C=c_param, penalty='l2')

        score = cross_val_score(svc_l2, X, y, cv=5)

        recall_score = cross_val_score(svc_l2, X, y, cv=5, scoring='recall')

        f1_score = cross_val_score(svc_l2, X, y, cv=5, scoring='f1')

        l2_f1 = np.mean(f1_score)

        

        print("Mean Accuracy: %0.3f (+/- %0.3f)" % (np.mean(acc_score), np.std(acc_score)) )

        print("Mean Recall: %0.3f (+/- %0.3f)" % (np.mean(recall_score), np.std(recall_score)) )

        print("Mean F1: %0.3f (+/- %0.3f)" % (np.mean(f1_score), np.std(f1_score)) )

        print('')

        

        # compare l1_recall & l2_recall:

        if l2_f1 > l1_f1:

            # compare to max:

            if l2_f1 > f1_max:

                f1_max = l2_f1

                best_c = c_param

                penalty='l2'

        else:

            # compare to max:

            if l1_f1 > f1_max:

                f1_max = l1_f1

                best_c = c_param

                penalty='l1'

            



    print('*'*25)

    print('Optimal C parameter = ', best_c)

    print('Optimal penalty = ', penalty)

    print('Optimal F1 = ', f1_max)

    print('*'*25)

    

    return best_c, penalty
best_c_svc, penalty_svc = get_best_hypers_svc(X_undersample, y_undersample)
# Use best hyperparameters:

dual_svc = (penalty_svc == 'l2') # 'dual' option must be set to false if penalty is 'l1'

svc = LinearSVC(C=best_c_svc, penalty=penalty_svc, dual=dual_svc)

# Train on full undersample data set:

svc.fit(X_undersample, y_undersample)

# Test on unseen test data set:

y_pred_score = svc.decision_function(X_test.values)

# Compute ROC metrics:

fpr, tpr, thresholds = roc_curve(y_test.values, y_pred_score)

# Get AUC:

roc_auc = auc(fpr,tpr)



# Plot ROC:

plt.title('ROC Curve - SVC')

plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)

plt.plot([0,1],[0,1],'r--')

plt.legend(loc='lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=0)

acc_score = cross_val_score(dt, X_undersample, y_undersample, cv=5)

recall_score = cross_val_score(dt, X_undersample, y_undersample, cv=5, scoring='recall')

f1_score = cross_val_score(dt, X_undersample, y_undersample, cv=5, scoring='f1')

print("Accuracy Score: %0.3f (+/- %0.3f)" % (np.mean(acc_score), np.std(acc_score)) )

print("Recall Score: %0.3f (+/- %0.3f)" % (np.mean(recall_score), np.std(recall_score)) )

print("Mean F1: %0.3f (+/- %0.3f)" % (np.mean(f1_score), np.std(f1_score)) )
# Train on full undersample data set:

dt.fit(X_train, y_train)

# Test on unseen test data set:

y_pred_score = dt.predict_proba(X_test.values)[:,1]

# Compute ROC metrics:

fpr, tpr, thresholds = roc_curve(y_test.values,y_pred_score)

# Get AUC:

roc_auc = auc(fpr, tpr)

                         

                                            

# Plot ROC:

plt.title('ROC Curve - DecisionTree')

plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)

plt.plot([0,1],[0,1],'r--')

plt.legend(loc='lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)

acc_score = cross_val_score(rfc, X_undersample, y_undersample, cv=5)

recall_score = cross_val_score(rfc, X_undersample, y_undersample, cv=5, scoring='recall')

f1_score = cross_val_score(rfc, X_undersample, y_undersample, cv=5, scoring='f1')

print("Accuracy Score: %0.3f (+/- %0.3f)" % (np.mean(acc_score), np.std(acc_score)) )

print("Recall Score: %0.3f (+/- %0.3f)" % (np.mean(recall_score), np.std(recall_score)) )

print("Mean F1: %0.3f (+/- %0.3f)" % (np.mean(f1_score), np.std(f1_score)) )
# Train on full undersample data set:

rfc.fit(X_train, y_train)

# Test on unseen test data set:

y_pred_score = rfc.predict_proba(X_test.values)[:,1]

# Compute ROC metrics:

fpr, tpr, thresholds = roc_curve(y_test.values,y_pred_score)

# Get AUC:

roc_auc = auc(fpr, tpr)

                         

                                            

# Plot ROC:

plt.title('ROC Curve - RandomForest')

plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)

plt.plot([0,1],[0,1],'r--')

plt.legend(loc='lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()