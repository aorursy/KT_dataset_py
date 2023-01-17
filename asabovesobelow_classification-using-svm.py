import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

main_file_path = "../input/creditcard.csv"
data = pd.read_csv(main_file_path)
print(data.columns)
#class = 1: fraudulent, class = 0: genuine
fraud_count = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)
genui_indices = data[data.Class == 0].index
#Random sample
random_sample = np.random.choice(genui_indices,fraud_count,replace = False)
genui_indices = np.array(random_sample)
#indices of chosen sample
indices = np.concatenate([fraud_indices,genui_indices])
filtered_data = data.iloc[indices,:]
#split input
X = filtered_data.loc[:,filtered_data.columns != 'Class']
y = filtered_data.loc[:,filtered_data.columns == 'Class']
y = np.array(y).T[0]

#undersampling, ratio of positive:negative = 1:1
print("Percentage of normal transactions: ", len(filtered_data[filtered_data.Class == 0])/len(filtered_data))
print("Percentage of fraud transactions: ", len(filtered_data[filtered_data.Class == 1])/len(filtered_data))
print("Total number of transactions in resampled data: ", len(filtered_data))
from sklearn.model_selection import train_test_split,KFold,cross_validate
from sklearn import svm
from sklearn.metrics import recall_score,accuracy_score

#train, test splitting
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.1,random_state=0)
#train, validation splitting
C = [100,1,0.01]
kernels = ['linear', 'rbf', 'sigmoid']

#cross-validation
for i in kernels:
    for j in C:
        SVM = svm.SVC(C = j, kernel = i, gamma = 'scale')
        kfold = KFold(n_splits = 3, shuffle = True, random_state = 0)
        results = cross_validate(estimator=SVM,X=train_X,y=train_y,scoring=['accuracy','recall','precision'],cv=kfold)
        average_recall = np.mean(results['test_recall'])
        average_accuracy = np.mean(results['test_accuracy'])
        average_precision = np.mean(results['test_precision'])
        standard_deviation_recall = np.std(results['test_recall'])
        print('Paremeter C: ',j,'; Kernel: ',i)
        print('Average recall: ',average_recall,'; average accuracy: ',average_accuracy,'; average precision: ',average_precision)
        print('Standard deviation of recall: ',standard_deviation_recall)
"""
Paremeter C:  0.01 ; Kernel:  linear
Average recall:  0.8101962373514097 ; average accuracy:  0.8983050847457626 ; average precision:  0.9833017814669192
Standard deviation of recall:  0.047515628390786754
Paremeter C:  1 ; Kernel:  linear
Average recall:  0.8082375478927203 ; average accuracy:  0.8983050847457626 ; average precision:  0.9856146469049696
Standard deviation of recall:  0.04131096242673804
Paremeter C:  100 ; Kernel:  linear
Average recall:  0.8190834070144414 ; average accuracy:  0.9016949152542373 ; average precision:  0.9811009252185724
Standard deviation of recall:  0.04904421340349213
"""
#-> Hard-margin SVM is more efficient; kernel: linear, C = 100
from sklearn.metrics import classification_report
C = 100
kernel = 'linear'
SVM = svm.SVC(C, kernel, gamma = 'scale')
SVM.fit(train_X,train_y)
y_pred = SVM.predict(test_X)
print(classification_report(test_y,y_pred, target_names = ['Fraud', 'Genuine']))
print('Accuracy score: ',accuracy_score(test_y,y_pred))