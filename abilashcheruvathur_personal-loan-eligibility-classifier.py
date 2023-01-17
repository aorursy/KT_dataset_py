#Importing required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#reading source file

data=pd.read_csv("/kaggle/input/Bank_Personal_Loan_Modelling.csv")
#Checking the head of the data

data.head(10)
#Checking the data types of the columns

data.dtypes



#All datda are of type integer except CCAvg which is of Float
#Checking the information

data.info()
#checking the shape of the data-set

data.shape

# We can infer there are 5000 data with 14 columns each
#To check if there are any null values present

nulllvalues=data.isnull().sum()

print(nulllvalues)

#There are no null values present in the data-set
#To check if there are any NaN values present

NaNvalues=data.isna().sum()

print(NaNvalues)

#There are no NaN values present in the data-set
#To check the data

data.describe().T
#Droping ID column as it is not used for analysis

data.drop(['ID'],axis=1,inplace=True)
#New data head

data.head(10)
#Distribution of continous data



plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,3,1)

plt.title('Age')

sns.distplot(data['Age'],color='red')



#Subplot 2

plt.subplot(1,3,2)

plt.title('Experience')

sns.distplot(data['Experience'],color='blue')



#Subplot 3

plt.subplot(1,3,3)

plt.title('Income')

sns.distplot(data['Income'],color='green')







plt.figure(figsize=(30,6))



#Subplot 1- Boxplot

plt.subplot(1,3,1)

plt.title('Age')

sns.boxplot(data['Age'],orient='vertical',color='red')



#Subplot 2

plt.subplot(1,3,2)

plt.title('Experience')

sns.boxplot(data['Experience'],orient='vertical',color='blue')



#Subplot 3

plt.subplot(1,3,3)

plt.title('Income')

sns.boxplot(data['Income'],orient='vertical',color='green')

#Histogram for CCAvg,Motage- Continous data

plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,2,1)

plt.title('CCAvg')

sns.distplot(data['CCAvg'],color='red')



#Subplot 2

plt.subplot(1,2,2)

plt.title('Mortgage')

sns.distplot(data['Mortgage'],color='blue')



plt.figure(figsize=(30,6))



#Subplot 1- Boxplot

plt.subplot(1,2,1)

plt.title('CCAvg')

sns.boxplot(data['CCAvg'],orient='vertical',color='red')



#Subplot 2

plt.subplot(1,2,2)

plt.title('Mortgage')

sns.boxplot(data['Mortgage'],orient='vertical',color='blue')

# Distribution of Categorical data



plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,3,1)

plt.title('Family')

sns.countplot(data['Family'],color='cyan')



#Subplot 2

plt.subplot(1,3,2)

plt.title('Education')

sns.countplot(data['Education'],color='violet')



#Subplot 3

plt.subplot(1,3,3)

plt.title('Securities Account')

sns.countplot(data['Securities Account'],color='green')



plt.figure(figsize=(30,6))



#Subplot 4

plt.subplot(1,3,1)

plt.title('CD Account')

sns.countplot(data['CD Account'],color='red')



#Subplot 5

plt.subplot(1,3,2)

plt.title('Online')

sns.countplot(data['Online'],color='blue')



#Subplot 6

plt.subplot(1,3,3)

plt.title('CreditCard')

sns.countplot(data['CreditCard'],color='orange')
#To find the correlation between the continous variables

correlation=data.corr()

correlation.style.background_gradient(cmap='coolwarm')
sns.heatmap(correlation)
columns=['Age','Experience','Income','CCAvg','Mortgage','Personal Loan']
sns.pairplot(data[columns])
sns.countplot(data['Personal Loan'])
#Calculate baseline proportion - ratio of Yes to No to identify data imbalance

Value_counts = data['Personal Loan'].value_counts(normalize=True)

print(Value_counts)
#Converting all categorical variable to categorical data type

data['Personal Loan']=data['Personal Loan'].astype('category')

data['Family']=data['Family'].astype('category')

data['Education']=data['Education'].astype('category')

data['Securities Account']=data['Securities Account'].astype('category')

data['CD Account']=data['CD Account'].astype('category')
data.dtypes
#Dropping the non targeted columns

data.drop(["ZIP Code","Age","Experience","Online","CreditCard"],axis=1,inplace = True)

data.head()
#Normalzing the continous data Income,CCAvg,Mortgage

#Removing other cols except continous data for calculating z score(Normalizing)

cols = list(data.columns)

cols.remove('Education')

cols.remove('Personal Loan')

cols.remove('CD Account')

cols.remove('Family')

cols.remove('Securities Account')

cols
for col in cols:

    col_zscore = col + '_zscore'

    data[col_zscore] = (data[col] - data[col].mean())/data[col].std(ddof=0) 

data.head()

#Calculated Z scre for Income CCAVg and mortgage
#importing necessary libraries

from sklearn.model_selection import train_test_split
# Deternmining the indepedent and dependent variales (X and Y)

X=data.drop(["Personal Loan","Income","CCAvg","Mortgage"],axis=1)
Y=data['Personal Loan']
# Checking the shape of X and Y

X.shape
Y.shape
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.3,random_state=1)

Xtrain.head()
#importing necessary libraries

from sklearn.linear_model import LogisticRegression
model_log_regression=LogisticRegression(solver="liblinear")
model_log_regression.fit(Xtrain,Ytrain)

coef_df = pd.DataFrame(model_log_regression.coef_)

coef_df['intercept'] = model_log_regression.intercept_

print(coef_df)
#Checking the score for logistic regression

logistic_regression_Trainscore=model_log_regression.score(Xtrain,Ytrain)

print("The score for Logistic regression-Training Data is {0:.2f}%".format(logistic_regression_Trainscore*100))

logistic_regression_Testscore=model_log_regression.score(Xtest,Ytest)

print("The score for Logistic regression-Test Data is {0:.2f}%".format(logistic_regression_Testscore*100))
#Predicting the Y values

Ypred=model_log_regression.predict(Xtest)



#Misclassification error

LR_MSE=1-logistic_regression_Testscore

print("Misclassification error of Logistical Regression model is {0:.1f}%".format(LR_MSE*100))
#Confusion Matrix

from sklearn import metrics

cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)
accuracy_score=metrics.accuracy_score(Ytest,Ypred)

percision_score=metrics.precision_score(Ytest,Ypred)

recall_score=metrics.recall_score(Ytest,Ypred)

print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))

print("The Percission of this model is {0:.2f}%".format(percision_score*100))

print("The Recall score of this model is {0:.2f}%".format(recall_score*100))
#AUC ROC curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



logit_roc_auc = roc_auc_score(Ytest, model_log_regression.predict(Xtest))

fpr, tpr, thresholds = roc_curve(Ytest, model_log_regression.predict_proba(Xtest)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
auc_score = metrics.roc_auc_score(Ytest, model_log_regression.predict_proba(Xtest)[:,1])

print("The AUC score is {0:.2f}".format(auc_score))
#Importing necessary Libraries and spliting data

from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore



#Spliting the data for K-Nearest Neighbors

XTrain,XTest,YTrain,YTest=train_test_split(X,Y,test_size=0.3,random_state=1)
#Finding the optimal value of k nearest neigbours, not considering 1 as just checking 1 nearest neighbor is not recomended.

neighbours=np.arange(2,30,2)

ac_scores=[]

for k in neighbours:

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(XTrain,YTrain)

    score=knn.score(XTest,YTest)

    ac_scores.append(score)

Max_score=max(ac_scores)#finding the max score

neighbours[ac_scores.index(max(ac_scores))]

model_knn=KNeighborsClassifier(n_neighbors=6)
#Fitting the model

model_knn.fit(XTrain,YTrain)

YPred=model_knn.predict(XTest)
#Checking the score for KNN

KNN_score=model_knn.score(XTest,YTest)

print("The score for KNN is {0:.2f}%".format(KNN_score*100))
MSE=1-KNN_score

print("The misclassification error is {0:.2f}%".format(MSE*100))



#Calculating the model score for accuracy

from sklearn import metrics

KNN_train_Score=model_knn.score(XTrain, YTrain)

print("KNN model Accuracy for training data is {0:.2f}%".format(KNN_train_Score*100))



KNN_test_Score=model_knn.score(XTest, YTest)

print("KNN model Accuracy for training data is {0:.2f}%".format(KNN_test_Score*100))


#Confusion Matrix

cm1=metrics.confusion_matrix(YTest, YPred, labels=[1, 0])



df_cm1 = pd.DataFrame(cm1, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm1, annot=True)
accuracy_score=metrics.accuracy_score(YTest,YPred)

percision_score=metrics.precision_score(YTest,YPred)

recall_score=metrics.recall_score(YTest,YPred)

print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))

print("The Percission of this model is {0:.2f}%".format(percision_score*100))

print("The Recall score of this model is {0:.2f}%".format(recall_score*100))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 10)
#Import necessary Linraries

from sklearn.naive_bayes import GaussianNB
model_NB = GaussianNB()
#Fitting the model

model_NB.fit(X_train,Y_train)
#Predicting the Y value

Y_pred=model_NB.predict(X_test)
#Checking the score for Naive Bayes

NB_score_train=model_NB.score(X_train,Y_train)

print("Naive Bayes model Accuracy for training data is {0:.2f}%".format(NB_score_train*100))

NB_score_test=model_NB.score(X_test,Y_test)

print("Naive Bayes model Accuracy for test data is {0:.2f}%".format(NB_score_test*100))

NB_MSE=1-NB_score_test

print("The misclassification error of Naive Bayes model is {0:.2f}%".format(NB_MSE*100))
#Confusion Matrix

cm2=metrics.confusion_matrix(Y_test, Y_pred, labels=[1, 0])



df_cm2 = pd.DataFrame(cm2, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm2, annot=True)
accuracy_score=metrics.accuracy_score(Y_test,Y_pred)

percision_score=metrics.precision_score(Y_test,Y_pred)

recall_score=metrics.recall_score(Y_test,Y_pred)

print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))

print("The Percission of this model is {0:.2f}%".format(percision_score*100))

print("The Recall score of this model is {0:.2f}%".format(recall_score*100))