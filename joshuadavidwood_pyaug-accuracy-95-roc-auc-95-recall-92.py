import pandas as pd

import sklearn

import math

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from pyaug import PyAugLinear, PyAugNormal, PyAugLogistic, PyAugLaplace

from collections import Counter

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, accuracy_score, recall_score
df = pd.read_csv('../input/creditcard.csv')



print('DataFrame Shape:', df.shape, '\n')

print('Number of NaN/Null Values:')

print(df.isna().sum(), '\n')

print('DataFrame Head:')

print(df.head())
df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

plt.show()
correlation_matrix = df.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix, vmax=0.8, square = True)

plt.show()
df['Time'].hist(bins=100)

plt.xlim([0, 175000])

plt.xlabel('Time (s)')

plt.ylabel('Frequency')

plt.show()
df['Amount'].hist(bins=500)

plt.xlim([0, 1500])

plt.xlabel('Amount')

plt.ylabel('Frequency')

plt.show()
scaler = StandardScaler()



X = df.drop('Class', axis = 1) # Remove target feature.

y = df['Class']



print('X DataFrame Shape:', X.shape)

print('y DataFrame Shape:', y.shape)



X_scaled = StandardScaler().fit_transform(X).reshape(-1, 1) # Reshape the NumPy array.

X_scaled = pd.DataFrame(X, columns = X.columns.values)



print(X.describe())
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 0)



lr = LogisticRegression(C=1e5, solver='liblinear') # Initiate LogisticRegression model.



lr.fit(X_train, y_train) # Fit the LogisticRegression model.

y_pred = lr.predict(X_test) # Predict y_pred values using X_test features.





accuracy_baseline = float('%.3f' % accuracy_score(y_test, y_pred))

ROCAUC_baseline =  float('%.3f' % roc_auc_score(y_test, y_pred))

recall_baseline = float('%.3f' % recall_score(y_test, y_pred))



print('Accuracy:', accuracy_baseline)

print('ROC AUC Score:', ROCAUC_baseline)

print('Recall Score:', recall_baseline)
df = pd.read_csv('../input/creditcard.csv')

df_features = df.drop('Class', axis = 1) # Remove target feature.

df_labels = df['Class']



def undersampling():

    n_fraud = Counter(df_labels)[1] # Count the number of fraudulent transactions.



    df_class0 = df.loc[df['Class'] == 0]

    df_class0 = df_class0.sample(Counter(df_labels)[1]) # Take a sample of normal transactions with the same length as the fraudulent transactions.



    df_class1 = df.loc[df['Class'] == 1]





    df_undersample = pd.concat([df_class0, df_class1]) # Concatenate the DataFrames.



    df_undersample_features = df_undersample.drop('Class', axis = 1) # Remove target feature.

    df_undersample_labels = df_undersample['Class']



    X_scaled = StandardScaler().fit_transform(df_undersample_features).reshape(-1, 1) # Reshape the NumPy array. 

    X_scaled = pd.DataFrame(df_undersample_features, columns = df_undersample_features.columns.values)



    X_train, X_test, y_train, y_test = train_test_split(df_undersample_features, df_undersample_labels, test_size = 0.3, random_state = 0)



    lr = LogisticRegression(C = 1e5, solver='liblinear') # Initiate LogisticRegression model.



    lr.fit(X_train, y_train) # Fit the LogisticRegression model.

    y_pred = lr.predict(X_test) # Predict y_pred values using X_test features.





    #Assess Machine Learning Model.

    results_matrix = confusion_matrix(y_test, y_pred)

    accuracyscore = float('%.3f' % accuracy_score(y_test, y_pred))

    ROCAUCscore =  float('%.3f' % roc_auc_score(y_test, y_pred))

    recallscore = float('%.3f' % recall_score(y_test, y_pred))

    

    metrics = [accuracyscore, ROCAUCscore, recallscore]

    return metrics





#Iterating and Evaluating Average Metrics.

iteration_results_undersampling = []

for _ in range(10):

    iteration_results_undersampling.append(undersampling())

 

undersampling_df = pd.DataFrame(iteration_results_undersampling, columns = ['Accuracy', 'ROCAUC', 'Recall'])



average_accuracy_undersampling = undersampling_df['Accuracy'].mean()

average_ROCAUC_undersampling = undersampling_df['ROCAUC'].mean()

average_recall_undersampling = undersampling_df['Recall'].mean()



print(undersampling_df.head())

print('Average Accuracy', float('%.3f' % average_accuracy_undersampling))

print('Average ROCAUC', float('%.3f' % average_ROCAUC_undersampling))

print('Average Recall', float('%.3f' % average_recall_undersampling))
df = pd.read_csv('../input/creditcard.csv')

df_features = df.drop('Class', axis = 1)

df_labels = df['Class']





def PyAugOversampling():

    #Scale DataFrame.

    df_scaled = StandardScaler().fit_transform(df_features).reshape(-1, 30) # Reshape the NumPy array. 

    df_scaled = pd.DataFrame(df_scaled, columns = df_features.columns.values)



    #Concatenate Scaled DataFrame with Unscaled Labels.

    df_scaled = pd.concat([df_scaled, df_labels], axis = 1)



    df_class0 = df_scaled.loc[df_scaled['Class'] == 0] #Extract Class = 0 (Non-Fraud).

    df_class0 = df_class0.sample(2000) #Take Random Sample of 2000.

    df_class1 = df_scaled.loc[df_scaled['Class'] == 1] #Extract Class = 1 (Fraud).



    #Define Features and Targets for Class 0 and Class 1.

    X0 = df_class0.drop('Class', axis = 1) #Remove target feature.

    y0 = df_class0['Class']



    X1 = df_class1.drop('Class', axis = 1) #Remove target feature.

    y1 = df_class1['Class']



    X_train0, X_test0, y_train0, y_test0 = train_test_split(X0, y0, test_size = 0.5, random_state = 0)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.5, random_state = 0)





    Xtrain1_ytrain1_concat = pd.concat([X_train1, y_train1], axis = 1) #Concatenate X_train1 and y_train1 to augment 1000 news rows.

    aug_train1 = PyAugLinear(Xtrain1_ytrain1_concat, 1000, 3, ['Class']) #Generate 1000 synthetic results using PyAug.



    aug_train1_features = aug_train1.drop('Class', axis = 1) #Remove target feature.

    aug_train1_class = aug_train1['Class']



    X_train = pd.concat([X_train0, aug_train1_features], axis = 0) #Concatenate X_train0 and aug_train1_features.

    y_train = pd.concat([y_train0, aug_train1_class], axis = 0) #Concatenate y_train0 and aug_train1_class.



    X_test = pd.concat([X_test0.sample(len(X_test1)), X_test1], axis = 0) #Make Sure Sample size correct

    y_test = pd.concat([y_test0.sample(len(y_test1)), y_test1], axis = 0) #Make Sure Sample size correct





    #Machine Learning.

    lr = LogisticRegression(C = 1e5, solver = 'liblinear') # Initiate LogisticRegression model.

    lr.fit(X_train, y_train) # Fit the LogisticRegression model.

    y_pred = lr.predict(X_test) # Predict y_pred values using X_test features.



    

    #Assess Machine Learning Model.

    results_matrix = confusion_matrix(y_test, y_pred)

    accuracyscore = float('%.3f' % accuracy_score(y_test, y_pred))

    ROCAUCscore =  float('%.3f' % roc_auc_score(y_test, y_pred))

    recallscore = float('%.3f' % recall_score(y_test, y_pred))

    

    metrics = [accuracyscore, ROCAUCscore, recallscore]

    return metrics





#Iterating and Evaluating Average Metrics.

iteration_results_oversampling = []

for _ in range(10):

    iteration_results_oversampling.append(PyAugOversampling())

 

PyAug_df = pd.DataFrame(iteration_results_oversampling, columns = ['Accuracy', 'ROCAUC', 'Recall'])



average_accuracy_oversampling = PyAug_df['Accuracy'].mean()

average_ROCAUC_oversampling = PyAug_df['ROCAUC'].mean()

average_recall_oversampling = PyAug_df['Recall'].mean()



print(PyAug_df.head())

print('Average Accuracy', float('%.3f' % average_accuracy_oversampling))

print('Average ROCAUC', float('%.3f' % average_ROCAUC_oversampling))

print('Average Recall', float('%.3f' % average_recall_oversampling))
x_values = ['Baseline', 'Undersampling', 'Oversampling (PyAug)']

accuracy_data = [accuracy_baseline, average_accuracy_undersampling, average_accuracy_oversampling]

ROCAUC_data = [ROCAUC_baseline, average_ROCAUC_undersampling, average_ROCAUC_oversampling]

recall_data = [recall_baseline, average_recall_undersampling, average_recall_oversampling]



plt.bar(x_values, accuracy_data, color = ['r', 'g', 'b'])

plt.ylabel('Accuracy Metric')

plt.ylim((0.5, 1.05))

plt.show()



plt.bar(x_values, ROCAUC_data, color = ['r', 'g', 'b'])

plt.ylabel('ROC AUC Metric')

plt.ylim((0.5, 1.05))

plt.show()



plt.bar(x_values, recall_data, color = ['r', 'g', 'b'])

plt.ylabel('Recall Metric')

plt.ylim((0.5, 1.05))

plt.show()





#Assessing Accuracy Improvement

print('Accuracy Improvement:', ((average_accuracy_oversampling - average_accuracy_undersampling) * 100), '%')

print('ROC AUC Improvement:', ((average_ROCAUC_oversampling - average_ROCAUC_undersampling)* 100), '%')

print('Recall Improvement:', ((average_recall_oversampling - average_recall_undersampling) * 100), '%')