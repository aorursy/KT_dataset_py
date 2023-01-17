from IPython.display import Image

Image("../input/svmimages/images/svm35.png") 
Image("../input/svmimages/images/svm3.png")
Image("../input/svmimages/images/svm5.png")
Image("../input/svmimages/images/svm4.png")
Image("../input/svmimages/images/svm7.png")
Image("../input/svmimages/images/svm8.png")
Image("../input/svmimages/images/svm9.png")
Image("../input/svmimages/images/svm10.png")
#### for the point X4:



Image("../input/svmimages/images/svm11.png")
Image("../input/svmimages/images/svm12.png")
Image("../input/svmimages/images/svm13.png")
Image("../input/svmimages/images/svm14.png")
Image("../input/svmimages/images/SVM15.PNG")
Image("../input/svmimages/images/svm16.png")
Image("../input/svmimages/images/svm17.png")



Image("../input/svmimages/images/svm18.png")
Image("../input/svmimages/images/svm20.png")
Image("../input/svmimages/images/svm21.png")
Image("../input/svmimages/images/svm22.png")
Image("../input/svmimages/images/svm23.png")
Image("../input/svmimages/images/svm24.png")
Image("../input/svmimages/images/svm25.png")
Image("../input/svmimages/images/svm26.png")
Image("../input/svmimages/images/svm27.png")
Image("../input/svmimages/images/svm28.png")
Image("../input/svmimages/images/svm36.png")
Image("../input/svmimages/images/svm30.png")
Image("../input/svmimages/images/svm31.png")
### Image 1

Image("../input/svmimages/images/svm32.png")
#Image 2

Image("../input/svmimages/images/svm33.png")
Image("../input/svmimages/images/svm34.png")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for data visualization

import seaborn as sns # for statistical data visualization

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = '../input/pulsar-stars/pulsar_stars.csv'



df = pd.read_csv(data)
df.shape
# let's preview the dataset



df.head()
# Now, I will view the column names to check for leading and trailing spaces.



# view the column names of the dataframe



col_names = df.columns



col_names
# remove leading spaces from column names



df.columns = df.columns.str.strip()
# view column names again



df.columns

# rename column names because column name is very long



df.columns = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness', 

              'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']
# view the renamed column names



df.columns
# check distribution of target_class column



df['target_class'].value_counts()
# view the percentage distribution of target_class column



df['target_class'].value_counts()/np.float(len(df))
# view summary of dataset



df.info()
# check for missing values in variables



df.isnull().sum()
# Outliers in numerical variables



# view summary statistics in numerical variables



round(df.describe(),2)


plt.figure(figsize=(24,20))





plt.subplot(4, 2, 1)

fig = df.boxplot(column='IP Mean')

fig.set_title('')

fig.set_ylabel('IP Mean')





plt.subplot(4, 2, 2)

fig = df.boxplot(column='IP Sd')

fig.set_title('')

fig.set_ylabel('IP Sd')





plt.subplot(4, 2, 3)

fig = df.boxplot(column='IP Kurtosis')

fig.set_title('')

fig.set_ylabel('IP Kurtosis')





plt.subplot(4, 2, 4)

fig = df.boxplot(column='IP Skewness')

fig.set_title('')

fig.set_ylabel('IP Skewness')





plt.subplot(4, 2, 5)

fig = df.boxplot(column='DM-SNR Mean')

fig.set_title('')

fig.set_ylabel('DM-SNR Mean')





plt.subplot(4, 2, 6)

fig = df.boxplot(column='DM-SNR Sd')

fig.set_title('')

fig.set_ylabel('DM-SNR Sd')





plt.subplot(4, 2, 7)

fig = df.boxplot(column='DM-SNR Kurtosis')

fig.set_title('')

fig.set_ylabel('DM-SNR Kurtosis')





plt.subplot(4, 2, 8)

fig = df.boxplot(column='DM-SNR Skewness')

fig.set_title('')

fig.set_ylabel('DM-SNR Skewness')
# plot histogram to check distribution





plt.figure(figsize=(24,20))





plt.subplot(4, 2, 1)

fig = df['IP Mean'].hist(bins=20)

fig.set_xlabel('IP Mean')

fig.set_ylabel('Number of pulsar stars')





plt.subplot(4, 2, 2)

fig = df['IP Sd'].hist(bins=20)

fig.set_xlabel('IP Sd')

fig.set_ylabel('Number of pulsar stars')





plt.subplot(4, 2, 3)

fig = df['IP Kurtosis'].hist(bins=20)

fig.set_xlabel('IP Kurtosis')

fig.set_ylabel('Number of pulsar stars')







plt.subplot(4, 2, 4)

fig = df['IP Skewness'].hist(bins=20)

fig.set_xlabel('IP Skewness')

fig.set_ylabel('Number of pulsar stars')





plt.subplot(4, 2, 5)

fig = df['DM-SNR Mean'].hist(bins=20)

fig.set_xlabel('DM-SNR Mean')

fig.set_ylabel('Number of pulsar stars')







plt.subplot(4, 2, 6)

fig = df['DM-SNR Sd'].hist(bins=20)

fig.set_xlabel('DM-SNR Sd')

fig.set_ylabel('Number of pulsar stars')





plt.subplot(4, 2, 7)

fig = df['DM-SNR Kurtosis'].hist(bins=20)

fig.set_xlabel('DM-SNR Kurtosis')

fig.set_ylabel('Number of pulsar stars')





plt.subplot(4, 2, 8)

fig = df['DM-SNR Skewness'].hist(bins=20)

fig.set_xlabel('DM-SNR Skewness')

fig.set_ylabel('Number of pulsar stars')
# Declare feature vector and target variable



X = df.drop(['target_class'], axis=1)



y = df['target_class']
# split X and y into training and testing sets



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# check the shape of X_train and X_test



X_train.shape, X_test.shape
# Feature Scaling 





cols = X_train.columns
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)



X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])

X_test = pd.DataFrame(X_test, columns=[cols])
X_train.describe()
# import SVC classifier

from sklearn.svm import SVC





# import metrics to compute accuracy

from sklearn.metrics import accuracy_score





# instantiate classifier with default hyperparameters

svc=SVC() 





# fit classifier to training set

svc.fit(X_train,y_train)



# make predictions on test set

y_pred=svc.predict(X_test)





# compute and print accuracy score

print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# instantiate classifier with rbf kernel and C=100

svc=SVC(C=100.0) 





# fit classifier to training set

svc.fit(X_train,y_train)





# make predictions on test set

y_pred=svc.predict(X_test)





# compute and print accuracy score

print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# instantiate classifier with rbf kernel and C=1000

svc=SVC(C=1000.0) 





# fit classifier to training set

svc.fit(X_train,y_train)





# make predictions on test set

y_pred=svc.predict(X_test)



# compute and print accuracy score

print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# instantiate classifier with linear kernel and C=1.0

linear_svc=SVC(kernel='linear', C=1.0) 





# fit classifier to training set

linear_svc.fit(X_train,y_train)





# make predictions on test set

y_pred_test=linear_svc.predict(X_test)





# compute and print accuracy score

print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
# Run SVM with linear kernel and C=100.0



# instantiate classifier with linear kernel and C=100.0

linear_svc100=SVC(kernel='linear', C=100.0) 





# fit classifier to training set

linear_svc100.fit(X_train, y_train)





# make predictions on test set

y_pred=linear_svc100.predict(X_test)



# compute and print accuracy score

print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Run SVM with linear kernel and C=1000.0



# instantiate classifier with linear kernel and C=1000.0

linear_svc1000=SVC(kernel='linear', C=1000.0) 





# fit classifier to training set

linear_svc1000.fit(X_train, y_train)





# make predictions on test set

y_pred=linear_svc1000.predict(X_test)



# compute and print accuracy score

print('Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
y_pred_train = linear_svc.predict(X_train)



y_pred_train
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
# print the scores on training and test set



print('Training set score: {:.4f}'.format(linear_svc.score(X_train, y_train)))



print('Test set score: {:.4f}'.format(linear_svc.score(X_test, y_test)))
# check class distribution in test set



y_test.value_counts()
# check null accuracy score



null_accuracy = (3306/(3306+274))



print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
# instantiate classifier with polynomial kernel and C=1.0

poly_svc=SVC(kernel='poly', C=1.0) 





# fit classifier to training set

poly_svc.fit(X_train,y_train)





# make predictions on test set

y_pred=poly_svc.predict(X_test)





# compute and print accuracy score

print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# instantiate classifier with polynomial kernel and C=100.0

poly_svc100=SVC(kernel='poly', C=100.0) 





# fit classifier to training set

poly_svc100.fit(X_train, y_train)





# make predictions on test set

y_pred=poly_svc100.predict(X_test)





# compute and print accuracy score

print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# instantiate classifier with sigmoid kernel and C=1.0

sigmoid_svc=SVC(kernel='sigmoid', C=1.0) 





# fit classifier to training set

sigmoid_svc.fit(X_train,y_train)





# make predictions on test set

y_pred=sigmoid_svc.predict(X_test)





# compute and print accuracy score

print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# instantiate classifier with sigmoid kernel and C=100.0

sigmoid_svc100=SVC(kernel='sigmoid', C=100.0) 





# fit classifier to training set

sigmoid_svc100.fit(X_train,y_train)





# make predictions on test set

y_pred=sigmoid_svc100.predict(X_test)





# compute and print accuracy score

print('Model accuracy score with sigmoid kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Print the Confusion Matrix and slice it into four pieces



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred_test)



print('Confusion matrix\n\n', cm)



print('\nTrue Positives(TP) = ', cm[0,0])



print('\nTrue Negatives(TN) = ', cm[1,1])



print('\nFalse Positives(FP) = ', cm[0,1])



print('\nFalse Negatives(FN) = ', cm[1,0])
# visualize confusion matrix with seaborn heatmap



cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 

                                 index=['Predict Positive:1', 'Predict Negative:0'])



sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred_test))
# Classification accuracy

TP = cm[0,0]

TN = cm[1,1]

FP = cm[0,1]

FN = cm[1,0]
# print classification accuracy



classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)



print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
 # print classification error



classification_error = (FP + FN) / float(TP + TN + FP + FN)



print('Classification error : {0:0.4f}'.format(classification_error))
 # print precision score



precision = TP / float(TP + FP)





print('Precision : {0:0.4f}'.format(precision))
# Recall 



recall = TP / float(TP + FN)



print('Recall or Sensitivity : {0:0.4f}'.format(recall))
# plot ROC Curve



from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)



plt.figure(figsize=(6,4))



plt.plot(fpr, tpr, linewidth=2)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12



plt.title('ROC curve for Predicting a Pulsar Star classifier')



plt.xlabel('False Positive Rate (1 - Specificity)')



plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
# compute ROC AUC



from sklearn.metrics import roc_auc_score



ROC_AUC = roc_auc_score(y_test, y_pred_test)



print('ROC AUC : {:.4f}'.format(ROC_AUC))
# calculate cross-validated ROC AUC 



from sklearn.model_selection import cross_val_score



Cross_validated_ROC_AUC = cross_val_score(linear_svc, X_train, y_train, cv=10, scoring='roc_auc').mean()



print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))