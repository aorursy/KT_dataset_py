import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

%matplotlib inline

random.seed(123) #making the results reproducible

filename = '../input/creditcard.csv'
original_data = pd.read_csv(filename)
data = original_data.copy() # create a copy for use leave the original in case we need it later.
data.head()
amounts = data.loc[:, ['Amount']]
data['Amount'] = StandardScaler().fit_transform(amounts)

data.head()
data = data.drop(columns=['Time'])
data.head()
class_dist = pd.value_counts(data['Class'], sort = True)
class_dist
print('%.2f%% of transactions are fraudulent.' % (class_dist[1]/sum(class_dist)*100) )
class_1 = data[data.Class == 1]
class_0 = data[data.Class == 0]
print(class_0.shape)
print(class_1.shape)
class_0_split = [round(x * class_0.shape[0]) for x in [0.7, 0.15, 0.15]]
class_1_split = [round(x * class_1.shape[0]) for x in [0.7, 0.15, 0.15]]

print('Non-fraudulent data split %s' % class_0_split)
print('Fraudulent data split %s' % class_1_split)

print('\nCheck for rounding issues as we want to avoid increasing the total with rounding up: ')
print(sum(class_0_split) / class_0.shape[0])
print(sum(class_1_split) / class_1.shape[0])
class_0_train = class_0.sample(n = 344, random_state = 123,  replace = False) #make our random state reproducible
class_1_train = class_1.sample(n = 344, random_state = 234,  replace = False)
class_0_cv = class_0.drop(class_0_train.index).sample(n = 42647, random_state = 345,  replace = False)
class_0_test = class_0.drop(class_0_train.index).drop(class_0_cv.index).sample(n = 42647, random_state = 456,  replace = False)

class_1_cv = class_1.drop(class_1_train.index).sample(n = 74, random_state = 567, replace = False)
class_1_test = class_1.drop(class_1_train.index).drop(class_1_cv.index)
train_set = pd.concat([class_0_train, class_1_train])
cv_set = pd.concat([class_0_cv, class_1_cv])
test_set = pd.concat([class_0_test, class_1_test])
#train_set.to_csv('training.csv')
#cv_set.to_csv('cv.csv')
#test_set.to_csv('test.csv')
from sklearn.linear_model import LogisticRegression

X_train = train_set.iloc[:, 0:29]
y_train = train_set.iloc[:, 29]

lr = LogisticRegression(penalty= 'l1', C=1) # default regularisation value for now.

lr.fit(X_train, y_train)
lr.score(X_train, y_train)
X_cv = cv_set.iloc[:, 0:29]
y_cv = cv_set.iloc[:, 29]
print(lr.score(X_cv, y_cv))
y_cv_predict = lr.predict(X_cv)

cv_set['Predicted'] = y_cv_predict

cv_set.head()
class_list = list(cv_set['Class'])
pred_list = list(cv_set['Predicted'])

def calc_prec_recall(class_list, pred_list):
    
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0



    for i in range(cv_set.shape[0]):
        if class_list[i] == 1:
            if pred_list[i] == 1:
                true_pos += 1
            else:
                false_neg += 1
            
        else:
            if pred_list[i] == 0:
                true_neg += 1
            else:
                false_pos += 1
            
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return precision, recall                                            
                      
cv_prec, cv_recall = calc_prec_recall(class_list, pred_list)                   
print('Precision = %.2f%%' % (100 * cv_prec))
print('Recall = %.2f%%' % (100 * cv_recall))
def f1_score(precision, recall):
    return (2 * precision * recall / (precision + recall))
                                     
fscore = f1_score(cv_prec, cv_recall)
print(fscore)
C_set = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]     # A set of C values to try

def logistic_test(X_train, y_train, X_cv, y_cv, C_in):
    
    lr = LogisticRegression(penalty= 'l1', C=C_in)

    lr.fit(X_train, y_train)
    
    train_score = lr.score(X_train, y_train)
    cv_score = lr.score(X_cv, y_cv)
    
    class_list = y_cv.tolist()
    pred_list = lr.predict(X_cv).tolist()
    
    cv_prec, cv_recall = calc_prec_recall(class_list, pred_list) 

    fscore = f1_score(cv_prec, cv_recall)
    
    return train_score, cv_score, fscore

training_scores = []
cv_scores = []
cv_f1 = []
best_f1_score = 0

for C in C_set:
    print('\n-----------------------------------------------------------------------')
    print('Fitting logistic regression with regularisation parameter %f' % C)
    print('-----------------------------------------------------------------------')
    a, b, c = logistic_test(X_train, y_train, X_cv, y_cv, C)    
    print('Training score = %f' % a)
    print('Cross-validation score = %f' % b)
    print('F1 score = %f' %c)
    training_scores.append(a)
    cv_scores.append(b)
    cv_f1.append(c)
    if c > best_f1_score:
        best_f1_score =c
    
plt.plot(C_set, training_scores, '-r', label='Training')
plt.plot(C_set, cv_scores, '-b', label='Cross Validation')
plt.xscale('log')
plt.xlabel('C (inverse regularisation parameter)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(C_set, cv_f1, '-r', label='Cross Validation F1 Score')
plt.xscale('log')
plt.xlabel('C (inverse regularisation parameter)')
plt.ylabel('F1 Score')
plt.legend()
plt.show()
lr = LogisticRegression(penalty= 'l1', C=0.1)
lr.fit(X_train, y_train)
X_test = test_set.iloc[:, 0:29]
y_test = test_set.iloc[:, 29]

class_list_test = list(test_set['Class'])
pred_list_test = lr.predict(X_test)

test_prec, test_recall = calc_prec_recall(class_list_test, pred_list_test) # pretty sure this is wrong - class list needs to be from test data
test_f1 = f1_score(test_prec, test_recall)

print('Accuracy on unseen test data: %.2f%%' % (100 * lr.score(X_test, y_test)))
print('Recall on unseen test data: %.2f%%' % (100*test_recall))
print('Precision on unseen test data: %.2f%%' % (100*test_prec))
print('F1 Score on unseen test data: %.2f%%' % (100 * test_f1))
from sklearn.svm import SVC, LinearSVC #import both the Linear and SVM classifier

svm_lin = LinearSVC(C=1.0) # C is effectively a regularisation parameter, we will experiement with changing this later

svm_lin.fit(X_train, y_train)

print("Training set linear classifier score: %.2f%%" % (100 * svm_lin.score(X_train, y_train)))
print("Cross-validation set linear classifier score: %.2f%%" % (100 * svm_lin.score(X_cv, y_cv)))
svm_lin_pred_list = svm_lin.predict(X_cv).tolist()
cv_prec, cv_recall = calc_prec_recall(class_list, svm_lin_pred_list)
fscore = f1_score(cv_prec, cv_recall)
print("The cross-validation precision is: %.2f%%" % (100*cv_prec))
print("The cross-validation recall is: %.2f%%" % (100*cv_recall))
print("The cross-validation F1 score is: %.2f%%" % (100*fscore))
svm_rbf = SVC(C=1.0, kernel='rbf', gamma='auto') # Using the default C for now and allow the algorithm to choose gamma
svm_rbf.fit(X_train, y_train)
print("Training set rbf classifier score: %.2f%%" % (100 * svm_rbf.score(X_train, y_train)))
print("Cross-validation set rbf classifier score: %.2f%%" % (100 * svm_rbf.score(X_cv, y_cv)))
svm_rbf_pred_list = svm_rbf.predict(X_cv).tolist()
cv_prec, cv_recall = calc_prec_recall(class_list, svm_rbf_pred_list)
fscore = f1_score(cv_prec, cv_recall)
print("The cross-validation precision is: %.2f%%" % (100*cv_prec))
print("The cross-validation recall is: %.2f%%" % (100*cv_recall))
print("The cross-validation F1 score is: %.2f%%" % (100*fscore))
#C_set = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]     # A set of C values to try
#gamma_set = [0.001, 0.003, 0.01, 0.03]     # A set of C values to try

C_set = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]     # A set of C values to try
gamma_set = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]     # A set of C values to try

def rbf_test(X_train, y_train, X_cv, y_cv, C_in, gamma_in):
    
    svm_rbf = SVC(C= C_in, kernel='rbf', gamma=gamma_in)

    svm_rbf.fit(X_train, y_train)
    
    train_score = svm_rbf.score(X_train, y_train)
    cv_score = svm_rbf.score(X_cv, y_cv)
    
    class_list = y_cv.tolist()
    pred_list = svm_rbf.predict(X_cv).tolist()
    
    cv_prec, cv_recall = calc_prec_recall(class_list, pred_list) 

    fscore = f1_score(cv_prec, cv_recall)
    
    return train_score, cv_score, fscore


Index = ['gamma = %s' % g for g in gamma_set]
Columns = ['C = %s' % C for C in C_set]

rbf_training_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_f1_df = pd.DataFrame(index=Index, columns=Columns)

rbf_best_f1_score = 0

for C in C_set:
    for gamma in gamma_set:
        print('\n-----------------------------------------------------------------------')
        print('Fitting Gaussian SVM with regularisation parameter %f and gamma, %f' % (C, gamma))
        print('-----------------------------------------------------------------------')
        a, b, c = rbf_test(X_train, y_train, X_cv, y_cv, C, gamma)    
        #print('Training score = %f' % a)
        #print('Cross-validation score = %f' % b)
        #print('F1 score = %f' %c)
        
        rbf_training_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = a
        rbf_cv_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = b
        rbf_cv_f1_df.iloc[gamma_set.index(gamma), C_set.index(C)] = c
        
        
        training_scores.append(a)
        cv_scores.append(b)
        cv_f1.append(c)
        if c > best_f1_score:
            best_f1_score =c
# For some reason when I looked at my data fram the first time, the plots didn't work as expected
# I looked at the data type for each column using the dtype attribute and they were all objects
# I'm not sure of the reason for this, but it could be that I initialised the arrays without any data
# We can convert all the values to float to fix this.


rbf_training_scores_df = rbf_training_scores_df.astype(float)
rbf_cv_scores_df = rbf_cv_scores_df.astype(float)
rbf_cv_f1_df = rbf_cv_f1_df.astype(float)

plt.figure(figsize=(12,6))
sns.heatmap(rbf_training_scores_df, annot=True, linewidths = 0.5, vmin=0.7, vmax=1)
plt.yticks(rotation=0)
plt.title('Training Data Accuracy', pad=10)
plt.show()
plt.figure(figsize=(12,6))
sns.heatmap(rbf_cv_scores_df, annot=True, linewidths = 0.5, vmin=0.7, vmax=1)
plt.title('Cross-Validation Data Accuracy', pad=10)
plt.yticks(rotation=0)
plt.show()
plt.figure(figsize=(12,6))
sns.heatmap(rbf_cv_f1_df, annot=True, linewidths = 0.5, vmin=0, vmax=0.7)
plt.yticks(rotation=0)
plt.title('F1 Score on Cross-Validation Data', pad=10)
plt.show()
C_set = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]     # A set of C values to try
gamma_set = [0.0001, 0.0003, 0.001, 0.003]     # A set of C values to try

Index = ['gamma = %s' % g for g in gamma_set]
Columns = ['C = %s' % C for C in C_set]

rbf_training_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_f1_df = pd.DataFrame(index=Index, columns=Columns)

rbf_best_f1_score = 0

for C in C_set:
    for gamma in gamma_set:
        #print('\n-----------------------------------------------------------------------')
        #print('Fitting Gaussian SVM with regularisation parameter %f and gamma, %f' % (C, gamma))
        #print('-----------------------------------------------------------------------')
        a, b, c = rbf_test(X_train, y_train, X_cv, y_cv, C, gamma)    
        #print('Training score = %f' % a)
        #print('Cross-validation score = %f' % b)
        #print('F1 score = %f' %c)
        
        rbf_training_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = a
        rbf_cv_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = b
        rbf_cv_f1_df.iloc[gamma_set.index(gamma), C_set.index(C)] = c
        
        
        training_scores.append(a)
        cv_scores.append(b)
        cv_f1.append(c)
        if c > best_f1_score:
            best_f1_score =c
rbf_training_scores_df = rbf_training_scores_df.astype(float)
rbf_cv_scores_df = rbf_cv_scores_df.astype(float)
rbf_cv_f1_df = rbf_cv_f1_df.astype(float)

plt.figure(figsize=(12,6))
sns.heatmap(rbf_training_scores_df, annot=True, linewidths = 0.5, vmin=0.7, vmax=1)
plt.yticks(rotation=0)
plt.title('Training Data Accuracy', pad=10)
plt.show()
plt.figure(figsize=(12,6))
sns.heatmap(rbf_cv_f1_df, annot=True, linewidths = 0.5, vmin=0, vmax=0.8)
plt.yticks(rotation=0)
plt.title('F1 Score on Cross-Validation Data', pad=10)
plt.show()
C_set = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]     # A set of C values to try
gamma_set = [3e-06, 0.00001, 0.00003, 0.0001, 0.0003]     # A set of C values to try

Index = ['gamma = %s' % g for g in gamma_set]
Columns = ['C = %s' % C for C in C_set]

rbf_training_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_f1_df = pd.DataFrame(index=Index, columns=Columns)

rbf_best_f1_score = 0

for C in C_set:
    for gamma in gamma_set:
        print('\n-----------------------------------------------------------------------')
        print('Fitting Gaussian SVM with regularisation parameter %f and gamma, %f' % (C, gamma))
        print('-----------------------------------------------------------------------')
        a, b, c = rbf_test(X_train, y_train, X_cv, y_cv, C, gamma)    
        #print('Training score = %f' % a)
        #print('Cross-validation score = %f' % b)
        #print('F1 score = %f' %c)
        
        rbf_training_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = a
        rbf_cv_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = b
        rbf_cv_f1_df.iloc[gamma_set.index(gamma), C_set.index(C)] = c
        
        
        training_scores.append(a)
        cv_scores.append(b)
        cv_f1.append(c)
        if c > best_f1_score:
            best_f1_score =c
rbf_training_scores_df = rbf_training_scores_df.astype(float)
rbf_cv_scores_df = rbf_cv_scores_df.astype(float)
rbf_cv_f1_df = rbf_cv_f1_df.astype(float)

plt.figure(figsize=(12,6))
sns.heatmap(rbf_training_scores_df, annot=True, linewidths = 0.5, vmin=0.7, vmax=1)
plt.yticks(rotation=0)
plt.title('Training Data Accuracy', pad=10)
plt.show()
plt.figure(figsize=(12,6))
sns.heatmap(rbf_cv_f1_df, annot=True, linewidths = 0.5, vmin=0, vmax=0.9)
plt.yticks(rotation=0)
plt.title('F1 Score on Cross-Validation Data', pad=10)
plt.show()
best_svm_rbf1 = SVC(C= 1.6, kernel='rbf', gamma=3e-05)
best_svm_rbf1.fit(X_train, y_train)
best_pred_list1 = best_svm_rbf1.predict(X_test).tolist()
best_precision1, best_recall1 = calc_prec_recall(y_test.tolist(), best_pred_list1)

best_svm_rbf2 = SVC(C= 0.4, kernel='rbf', gamma=1e-04)
best_svm_rbf2.fit(X_train, y_train)
best_pred_list2 = best_svm_rbf2.predict(X_test).tolist()
best_precision2, best_recall2 = calc_prec_recall(y_test.tolist(), best_pred_list2)

best_fscore1 = f1_score(best_precision1, best_recall1)
best_fscore2 = f1_score(best_precision2, best_recall2)
print("Model 1")
print("F1 Score for 1st model on test data: %.2f%%" % (100 * best_fscore1))
print("Precision: %.2f%%" % (100* best_precision1))
print("Recall: %.2f%%" % (100* best_recall1))

print("\nModel 2")
print("F1 Score for 2nd model on test data: %.2f%%" % (100 * best_fscore2))
print("Precision: %.2f%%" % (100* best_precision2))
print("Recall: %.2f%%" % (100* best_recall2))