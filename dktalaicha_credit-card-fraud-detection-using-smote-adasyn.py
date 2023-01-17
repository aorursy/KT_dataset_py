# Import Libraries

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# import cufflinks as cf

import plotly

import datetime

import math

import matplotlib

import sklearn

from IPython.display import HTML

from IPython.display import YouTubeVideo



import pickle

import os



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



# Print versions of libraries

print(f"Numpy version : Numpy {np.__version__}")

print(f"Pandas version : Pandas {pd.__version__}")

print(f"Matplotlib version : Matplotlib {matplotlib.__version__}")

print(f"Seaborn version : Seaborn {sns.__version__}")

print(f"SkLearn version : SkLearn {sklearn.__version__}")

# print(f"Cufflinks version : cufflinks {cf.__version__}")

print(f"Plotly version : plotly {plotly.__version__}")



# Magic Functions for In-Notebook Display

%matplotlib inline



# Setting seabon style

sns.set(style='darkgrid', palette='colorblind')
df = pd.read_csv('../input/creditcardfraud/creditcard.csv', encoding='latin_1')
# Converting all column names to lower case

df.columns = df.columns.str.lower()
df.head()
df.tail()
# Customising default values to view all columns

pd.options.display.max_rows = 100

pd.options.display.max_columns = 100



# pd.set_option('display.max_rows',1000)
df.head(10)
df.info()
print(df['class'].value_counts())

print('\n')

print(df['class'].value_counts(normalize=True))
df["class"].value_counts().plot(kind = 'pie',explode=[0, 0.1],figsize=(6, 6),autopct='%1.1f%%',shadow=True)

plt.title("Fraudulent and Non-Fraudulent Distribution",fontsize=20)

plt.legend(["Fraud", "Genuine"])

plt.show()
df[['time','amount']].describe()
# Dealing with missing data

df.isnull().sum().max()
plt.figure(figsize=(8,6))

plt.title('Distribution of Transaction Amount', fontsize=14)

sns.distplot(df['amount'], bins=100)

plt.show()
fig, axs = plt.subplots(ncols=2,figsize=(16,4))

sns.distplot(df[df['class'] == 1]['amount'], bins=100, ax=axs[0])

axs[0].set_title("Distribution of Fraud Transactions")



sns.distplot(df[df['class'] == 0]['amount'], bins=100, ax=axs[0])

axs[1].set_title("Distribution of Genuine Transactions")



plt.show()
print("Fraud Transaction distribution : \n",df[(df['class'] == 1)]['amount'].value_counts().head())

print("\n")

print("Maximum amount of fraud transaction - ",df[(df['class'] == 1)]['amount'].max())

print("Minimum amount of fraud transaction - ",df[(df['class'] == 1)]['amount'].min())
print("Genuine Transaction distribution : \n",df[(df['class'] == 0)]['amount'].value_counts().head())

print("\n")

print("Maximum amount of Genuine transaction - ",df[(df['class'] == 0)]['amount'].max())

print("Minimum amount of Genuine transaction - ",df[(df['class'] == 0)]['amount'].min())
plt.figure(figsize=(8,6))

sns.boxplot(x='class', y='amount',data = df)

plt.title('Amount Distribution for Fraud and Genuine transactions')

plt.show()
plt.figure(figsize=(8,6))

plt.title('Distribution of Transaction Time', fontsize=14)

sns.distplot(df['time'], bins=100)

plt.show()
fig, axs = plt.subplots(ncols=2, figsize=(16,4))



sns.distplot(df[(df['class'] == 1)]['time'], bins=100, color='red', ax=axs[0])

axs[0].set_title("Distribution of Fraud Transactions")



sns.distplot(df[(df['class'] == 0)]['time'], bins=100, color='green', ax=axs[1])

axs[1].set_title("Distribution of Genuine Transactions")



plt.show()
plt.figure(figsize=(12,8))

ax = sns.boxplot(x='class', y='time',data = df)



# Change the appearance of that box

ax.artists[0].set_facecolor('#90EE90')

ax.artists[1].set_facecolor('#FA8072')



plt.title('Time Distribution for Fraud and Genuine transactions')

plt.show()
fig, axs = plt.subplots(nrows=2,sharex=True,figsize=(16,6))



sns.scatterplot(x='time',y='amount', data=df[df['class']==1], ax=axs[0])

axs[0].set_title("Distribution of Fraud Transactions")



sns.scatterplot(x='time',y='amount', data=df[df['class']==0], ax=axs[1])

axs[1].set_title("Distribution of Genue Transactions")



plt.show()
# Finging unique values for each column

df[['time','amount','class']].nunique()
fig = px.scatter(df, x="time", y="amount", color="class", 

                 marginal_y="violin",marginal_x="box", trendline="ols", template="simple_white")

fig.show()
df[['time','amount','class']].corr()['class'].sort_values(ascending=False).head(10)
plt.title('Pearson Correlation Matrix')

sns.heatmap(df[['time', 'amount','class']].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="winter",

            linecolor='w',annot=True);
df.shape
df['class'].value_counts(normalize=True)
# Converting time from second to hour

df['time'] = df['time'].apply(lambda sec : (sec/3600))
# Calculating hour of the day

df['hour'] = df['time']%24   # 2 days of data

df['hour'] = df['hour'].apply(lambda x : math.floor(x))
# Calculating First and Second day

df['day'] = df['time']/24   # 2 days of data

df['day'] = df['day'].apply(lambda x : 1 if(x==0) else math.ceil(x))
df[['time','hour','day','amount','class']]
# calculating fraud transaction daywise

dayFrdTran = df[(df['class'] == 1)]['day'].value_counts()



# calculating genuine transaction daywise

dayGenuTran = df[(df['class'] == 0)]['day'].value_counts()



# calculating total transaction daywise

dayTran = df['day'].value_counts()



print("No of transaction Day wise:")

print(dayTran)



print("\n")



print("No of fraud transaction Day wise:")

print(dayFrdTran)



print("\n")



print("No of genuine transactions Day wise:")

print(dayGenuTran)



print("\n")



print("Percentage of fraud transactions Day wise:")

print((dayFrdTran/dayTran)*100)
fig, axs = plt.subplots(ncols=3, figsize=(16,4))



sns.countplot(df['day'], ax=axs[0])

axs[0].set_title("Distribution of Total Transactions")



sns.countplot(df[(df['class'] == 1)]['day'], ax=axs[1])

axs[1].set_title("Distribution of Fraud Transactions")



sns.countplot(df[(df['class'] == 0)]['day'], ax=axs[2])

axs[2].set_title("Distribution of Genuine Transactions")



plt.show()
# Time plots 

fig , axs = plt.subplots(nrows = 1 , ncols = 2 , figsize = (15,8))



sns.distplot(df[df['class']==0]['time'].values , color = 'green' , ax = axs[0])

axs[0].set_title('Genuine Transactions')



sns.distplot(df[df['class']==1]['time'].values , color = 'red' ,ax = axs[1])

axs[1].set_title('Fraud Transactions')



fig.suptitle('Comparison between Transaction Frequencies vs Time for Fraud and Genuine Transactions')

plt.show()
# Let's see if we find any particular pattern between time ( in hours ) and Fraud vs Genuine Transactions



plt.figure(figsize=(12,10))



sns.distplot(df[df['class'] == 0]["hour"], color='green') # Genuine - green

sns.distplot(df[df['class'] == 1]["hour"], color='red') # Fraudulent - Red



plt.title('Fraud vs Genuine Transactions by Hours', fontsize=15)

plt.xlim([0,25])

plt.show()
plt.figure(figsize=(8,6))

df[['time','hour','day','amount','class']].groupby('hour').count()['class'].plot()

plt.show()
df.hist(figsize = (25,25))

plt.show()
df.reset_index(inplace = True , drop = True)
# Scale amount by log

df['amount_log'] = np.log(df.amount + 0.01)
from sklearn.preprocessing import StandardScaler # importing a class from a module of a library



ss = StandardScaler() # object of the class StandardScaler ()

df['amount_scaled'] = ss.fit_transform(df['amount'].values.reshape(-1,1))
from sklearn.preprocessing import MinMaxScaler



mm = MinMaxScaler() # object of the class StandardScaler ()

df['amount_minmax'] = mm.fit_transform(df['amount'].values.reshape(-1,1))
#Feature engineering to a better visualization of the values



# Let's explore the Amount by Class and see the distribuition of Amount transactions

fig , axs = plt.subplots(nrows = 1 , ncols = 4 , figsize = (16,4))



sns.boxplot(x ="class",y="amount",data=df, ax = axs[0])

axs[0].set_title("Class vs Amount")



sns.boxplot(x ="class",y="amount_log",data=df, ax = axs[1])

axs[1].set_title("Class vs Log Amount")



sns.boxplot(x ="class",y="amount_scaled",data=df, ax = axs[2])

axs[2].set_title("Class vs Scaled Amount")



sns.boxplot(x ="class",y="amount_minmax",data=df, ax = axs[3])

axs[3].set_title("Class vs Min Max Amount")



# fig.suptitle('Amount by Class', fontsize=20)

plt.show()
df[['time','hour','day','amount','amount_log','amount_scaled','amount_minmax','class']]
CreditCardFraudDataCleaned = df



# Saving the Python objects as serialized files can be done using pickle library

# Here let us save the Final Data set after all the transformations as a file

with open('CreditCardFraudDataCleaned.pkl', 'wb') as fileWriteStream:

    pickle.dump(CreditCardFraudDataCleaned, fileWriteStream)

    # Don't forget to close the filestream!

    fileWriteStream.close()

    

print('pickle file is saved at Location:',os.getcwd())
# Reading a Pickle file

with open('CreditCardFraudDataCleaned.pkl', 'rb') as fileReadStream:

    CreditCardFraudDataFromPickle = pickle.load(fileReadStream)

    # Don't forget to close the filestream!

    fileReadStream.close()

    

# Checking the data read from pickle file. It is exactly same as the DiamondPricesData

df = CreditCardFraudDataFromPickle

df.head()
df.shape
df.head()
df.columns
# Separate Target Variable and Predictor Variables

# Here I am keeping the log amount and dropping the amount and scaled amount columns.

X = df.drop(['time','class','hour','day','amount','amount_minmax','amount_scaled'],axis=1)

y = df['class']
X
# Load the library for splitting the data

from sklearn.model_selection import train_test_split
# Split the data into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=101)
# Quick sanity check with the shapes of Training and testing datasets

print("X_train - ",X_train.shape)

print("y_train - ",y_train.shape)

print("X_test - ",X_test.shape)

print("y_test - ",y_test.shape)
from sklearn.linear_model import LogisticRegression # Importing Classifier Step
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0) 





logreg = LogisticRegression()

logreg.fit(X_train, y_train) 
y_pred = logreg.predict(X_test)
from sklearn import metrics
# https://en.wikipedia.org/wiki/Precision_and_recall

print(metrics.classification_report(y_test, y_pred))
print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

# print('Confusion Matrix : \n', cnf_matrix)

print("\n")
# Predicted values counts for fraud and genuine of test dataset

pd.Series(y_pred).value_counts()
# Actual values counts for fraud and genuine of test dataset

pd.Series(y_test).value_counts()
103/147
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)

cnf_matrix
# Heatmap for Confusion Matrix

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, annot_kws={"size": 25}, cmap="winter" ,fmt='g')



plt.title('Confusion matrix', y=1.1, fontsize = 22)

plt.ylabel('Actual',fontsize = 18)

plt.xlabel('Predicted',fontsize = 18)



# ax.xaxis.set_ticklabels(['Genuine', 'Fraud']); 

# ax.yaxis.set_ticklabels(['Genuine', 'Fraud']);



plt.show()
92/147
metrics.roc_auc_score(y_test , y_pred) 
y_pred_proba = logreg.predict_proba(X_test)

y_pred_proba
# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
# Import imbalace technique algorithims

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter # counter takes values returns value_counts dictionary

from sklearn.datasets import make_classification
print('Original dataset shape %s' % Counter(y))



rus = RandomUnderSampler(random_state=42)

X_res, y_res = rus.fit_resample(X, y)



print('Resampled dataset shape %s' % Counter(y_res))
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, shuffle=True, random_state=0)



# Undersampling with Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)
print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))
# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
# Heatmap for Confusion Matrix



cnf_matrix = metrics.confusion_matrix(y_test , y_pred)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, annot_kws={"size": 25}, cmap="winter" ,fmt='g')



plt.title('Confusion matrix', y=1.1, fontsize = 22)

plt.xlabel('Predicted',fontsize = 18)

plt.ylabel('Actual',fontsize = 18)



# ax.xaxis.set_ticklabels(['Genuine', 'Fraud']); 

# ax.yaxis.set_ticklabels(['Genuine', 'Fraud']);



plt.show()
from imblearn.over_sampling import RandomOverSampler
print('Original dataset shape %s' % Counter(y))

random_state = 42



ros = RandomOverSampler(random_state=random_state)

X_res, y_res = ros.fit_resample(X, y)



print('Resampled dataset shape %s' % Counter(y_res))
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, shuffle=True, random_state=0)



# Oversampling with Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)
print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_test , y_pred))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))
# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a breast cancer classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
# Heatmap for Confusion Matrix



cnf_matrix = metrics.confusion_matrix(y_test , y_pred)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, annot_kws={"size": 25}, cmap="winter" ,fmt='g')



plt.title('Confusion matrix', y=1.1, fontsize = 22)

plt.xlabel('Predicted',fontsize = 18)

plt.ylabel('Actual',fontsize = 18)



# ax.xaxis.set_ticklabels(['Genuine', 'Fraud']); 

# ax.yaxis.set_ticklabels(['Genuine', 'Fraud']);



plt.show()
from imblearn.over_sampling import SMOTE, ADASYN
print('Original dataset shape %s' % Counter(y))



smote = SMOTE(random_state=42)

X_res, y_res = smote.fit_resample(X, y)



print('Resampled dataset shape %s' % Counter(y_res))
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, shuffle=True, random_state=0)



# SMOTE Sampling with Logistic Regression

logreg = LogisticRegression(max_iter=1000)

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)
print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_test , y_pred))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))
# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a breast cancer classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
# Heatmap for Confusion Matrix



cnf_matrix = metrics.confusion_matrix(y_test , y_pred)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, annot_kws={"size": 25}, cmap="winter" ,fmt='g')



plt.title('Confusion matrix', y=1.1, fontsize = 22)

plt.xlabel('Predicted',fontsize = 18)

plt.ylabel('Actual',fontsize = 18)



# ax.xaxis.set_ticklabels(['Genuine', 'Fraud']); 

# ax.yaxis.set_ticklabels(['Genuine', 'Fraud']);



plt.show()
print('Original dataset shape %s' % Counter(y))



adasyn = ADASYN(random_state=42)



X_res, y_res = adasyn.fit_resample(X, y)

print('Resampled dataset shape %s' % Counter(y_res))
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, shuffle=True, random_state=0)



#  ADASYN Sampling with Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)
print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))
# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a breast cancer classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
# Heatmap for Confusion Matrix



cnf_matrix = metrics.confusion_matrix(y_test , y_pred)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, annot_kws={"size": 25}, cmap="winter" ,fmt='g')



plt.title('Confusion matrix', y=1.1, fontsize = 22)

plt.xlabel('Predicted',fontsize = 18)

plt.ylabel('Actual',fontsize = 18)



# ax.xaxis.set_ticklabels(['Genuine', 'Fraud']); 

# ax.yaxis.set_ticklabels(['Genuine', 'Fraud']);



plt.show()
from sklearn.decomposition import PCA
X_reduced_pca_im = PCA(n_components=2, random_state=42).fit_transform(X)
# Generate and plot a synthetic imbalanced classification dataset

plt.figure(figsize=(12,8))



plt.scatter(X_reduced_pca_im[:,0], X_reduced_pca_im[:,1], c=(y == 0), label='No Fraud', cmap='coolwarm', linewidths=1)

plt.scatter(X_reduced_pca_im[:,0], X_reduced_pca_im[:,1], c=(y == 1), label='Fraud', cmap='coolwarm', linewidths=1)



plt.title("Scatter Plot of Imbalanced Dataset")

plt.legend()

plt.show()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X_res)
# Oversample and plot imbalanced dataset with ADASYN

plt.figure(figsize=(12,8))



plt.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_res == 0), cmap='coolwarm', label='No Fraud', linewidths=1)

plt.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_res == 1), cmap='coolwarm', label='Fraud', linewidths=1)



plt.title("Scatter Plot of Imbalanced Dataset With Adaptive Synthetic Sampling \(ADASYN\)")

plt.legend()

plt.show()
print('Original dataset shape %s' % Counter(y))



rus = RandomUnderSampler(random_state=42)

X_under, y_under = rus.fit_resample(X, y)

print('Resampled dataset shape %s' % Counter(y_under))



# Slit into train and test datasets

X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_under, y_under, shuffle=True, test_size=0.3, random_state=0)
print('Original dataset shape %s' % Counter(y))



ros = RandomOverSampler(random_state=42)

X_over, y_over = ros.fit_resample(X, y)

print('Resampled dataset shape %s' % Counter(y_over))



# Slit into train and test datasets

X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X_over, y_over, test_size=0.3, shuffle=True, random_state=0)
print('Original dataset shape %s' % Counter(y))



smote = SMOTE(random_state=42)

X_smote, y_smote = smote.fit_resample(X, y)

print('Resampled dataset shape %s' % Counter(y_smote))



# Slit into train and test datasets

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.3, shuffle=True, random_state=0)
print('Original dataset shape %s' % Counter(y))



adasyn = ADASYN(random_state=42)

X_adasyn, y_adasyn = adasyn.fit_resample(X, y)

print('Resampled dataset shape %s' % Counter(y_adasyn))



# Slit into train and test datasets

X_train_adasyn, X_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(X_adasyn, y_adasyn, test_size=0.3, shuffle=True, random_state=0)
# from sklearn.model_selection import cross_val_score

# from sklearn.model_selection import StratifiedKFold

# from sklearn.linear_model import LogisticRegression

# from sklearn.tree import DecisionTreeClassifier

# # from sklearn.ensemble import RandomForestClassifier

# from sklearn.svm import SVC

# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.naive_bayes import GaussianNB



# # Build Models

# # Let’s test 5 different algorithms:



# # Spot Check Algorithms

# models = []



# #------------------ Logistic Regression (LR) ------------------#

# models.append(('LR imbalance', LogisticRegression(solver='liblinear', multi_class='ovr'),X,y))

# models.append(('LR Undersampling', LogisticRegression(solver='liblinear', multi_class='ovr'),X_under,y_under))

# models.append(('LR Oversampling', LogisticRegression(solver='liblinear', multi_class='ovr'),X_over,y_over))

# models.append(('LR SMOTE', LogisticRegression(solver='liblinear', multi_class='ovr'),X_smote,y_smote))

# # models.append(('LR ADASYN', LogisticRegression(solver='liblinear', multi_class='ovr'),X_adasyn,y_adasyn))



# #-----------------Decision Tree (DT)------------------#

# models.append(('DT imbalance', DecisionTreeClassifier(),X,y))

# models.append(('DT Undersampling', DecisionTreeClassifier(),X_under,y_under))

# models.append(('DT Oversampling', DecisionTreeClassifier(),X_over,y_over))

# models.append(('DT SMOTE', DecisionTreeClassifier(),X_smote,y_smote))

# # models.append(('DT ADASYN', DecisionTreeClassifier(),X_adasyn,y_adasyn))



# #------------------ K-Nearest Neighbors (KNN) ------------------#

# models.append(('KNN imbalance', KNeighborsClassifier(),X,y))

# models.append(('KNN Undersampling', KNeighborsClassifier(),X_under,y_under))

# models.append(('KNN Oversampling', KNeighborsClassifier(),X_over,y_over))

# models.append(('KNN SMOTE', KNeighborsClassifier(),X_smote,y_smote))

# # models.append(('DT ADASYN', KNeighborsClassifier(),X_adasyn,y_adasyn))



# #------------------ Support Vector Machines (SVM) ------------------#

# # models.append(('SVM imbalance', SVC(gamma='auto'),X,y))

# # models.append(('SVM Undersampling', SVC(gamma='auto'),X_under,y_under))

# # models.append(('SVM Oversampling', SVC(gamma='auto'),X_over,y_over))

# # models.append(('SVM SMOTE', SVC(gamma='auto'),X_smote,y_smote))

# # # models.append(('SVM ADASYN', SVC(gamma='auto'),X_adasyn,y_adasyn))



# #------------------ Gaussian Naive Bayes (NB) ------------------#

# models.append(('NB imbalance', GaussianNB(),X,y))

# models.append(('NB Undersampling', GaussianNB(),X_under,y_under))

# models.append(('NB Oversampling', GaussianNB(),X_over,y_over))

# models.append(('NB SMOTE', GaussianNB(),X_smote,y_smote))

# # models.append(('NB ADASYN', GaussianNB(),X_adasyn,y_adasyn))



# # evaluate each model in turn

# names_lst = []

# aucs_lst = []

# accuracy_lst = []

# precision_lst = []

# recall_lst = []

# f1_lst = []



# plt.figure(figsize=(14,8))



# for name, model,Xdata,ydata in models:

    

#     names_lst.append(name)

    

#     # split data in train test set

#     X_train, X_test, y_train, y_test = train_test_split(Xdata, ydata, test_size=0.3, random_state=0)

#     # Build model

#     model.fit(X_train, y_train)

#     # Predict

#     y_pred = model.predict(X_test)

    

#     # calculate accuracy

#     Accuracy = metrics.accuracy_score(y_pred , y_test)

#     accuracy_lst.append(Accuracy)

    

#     # calculate auc

#     Aucs = metrics.roc_auc_score(y_test , y_pred)

#     aucs_lst.append(Aucs)

    

#     # calculate precision

#     PrecisionScore = metrics.precision_score(y_test , y_pred)

#     precision_lst.append(PrecisionScore)

    

#     # calculate recall

#     RecallScore = metrics.recall_score(y_test , y_pred)

#     recall_lst.append(RecallScore)

    

#     # calculate f1 score

#     F1Score = metrics.f1_score(y_test , y_pred)

#     f1_lst.append(F1Score)

    

#     print('F1 Score of '+ name +' model : {0:0.5f}'.format(F1Score))

    

# #     draw confusion matrix

# #     cnf_matrix = metrics.confusion_matrix(y_test , y_pred)



# #     print("Model Name :", name)

# #     print('Accuracy :{0:0.5f}'.format(Accuracy)) 

# #     print('AUC : {0:0.5f}'.format(Aucs))

# #     print('Precision : {0:0.5f}'.format(PrecisionScore))

# #     print('Recall : {0:0.5f}'.format(RecallScore))

# #     print('F1 : {0:0.5f}'.format(F1Score))

# #     print('Confusion Matrix : \n', cnf_matrix)

# #     print("\n")



    

#     # plot ROC Curve

#     fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

#     auc = metrics.roc_auc_score(y_test, y_pred)

#     plt.plot(fpr,tpr,linewidth=2, label=name + ", auc="+str(auc))

#     #---------- For loops ends here--------#

    



# plt.legend(loc=4)

# plt.plot([0,1], [0,1], 'k--' )

# plt.rcParams['font.size'] = 12

# plt.title('ROC curve for Predicting a credit card fraud detection')

# plt.xlabel('False Positive Rate (1 - Specificity)')

# plt.ylabel('True Positive Rate (Sensitivity)')

# plt.show()



# data = {'Model':names_lst,

#        'Accuracy':accuracy_lst,

#        'AUC':aucs_lst,

#        'PrecisionScore':precision_lst,

#        'RecallScore':recall_lst,

#        'F1Score':f1_lst}



# print("Performance measures of various classifiers: \n")

# performance_df = pd.DataFrame(data) 

# performance_df.sort_values(['AUC','RecallScore','F1Score','PrecisionScore'],ascending=False)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
names_lst = []



# Empty list to capture performance matrix for train set

aucs_train_lst = []

accuracy_train_lst = []

precision_train_lst = []

recall_train_lst = []

f1_train_lst = []



# Empty list to capture performance matrix for test set

aucs_test_lst = []

accuracy_test_lst = []

precision_test_lst = []

recall_test_lst = []

f1_test_lst = []



# Function for model building and performance measure



def build_measure_model(models):

    plt.figure(figsize=(12,6))



    for name, model,Xdata,ydata in models:

        

        names_lst.append(name)



        # split data in train test set

        X_train, X_test, y_train, y_test = train_test_split(Xdata, ydata, test_size=0.3, shuffle=True, random_state=0)

        

        # Build model

        model.fit(X_train, y_train)

        

        # Predict

        y_train_pred = model.predict(X_train)

        y_test_pred = model.predict(X_test)



        # calculate accuracy

        Accuracy_train = metrics.accuracy_score(y_train, y_train_pred)

        accuracy_train_lst.append(Accuracy_train)

        

        Accuracy_test = metrics.accuracy_score(y_test, y_test_pred)

        accuracy_test_lst.append(Accuracy_test)



        # calculate auc

        Aucs_train = metrics.roc_auc_score(y_train, y_train_pred)

        aucs_train_lst.append(Aucs_train)

        

        Aucs_test = metrics.roc_auc_score(y_test , y_test_pred)

        aucs_test_lst.append(Aucs_test)



        # calculate precision

        PrecisionScore_train = metrics.precision_score(y_train , y_train_pred)

        precision_train_lst.append(PrecisionScore_train)

        

        PrecisionScore_test = metrics.precision_score(y_test , y_test_pred)

        precision_test_lst.append(PrecisionScore_test)



        # calculate recall

        RecallScore_train = metrics.recall_score(y_train , y_train_pred)

        recall_train_lst.append(RecallScore_train)

        

        RecallScore_test = metrics.recall_score(y_test , y_test_pred)

        recall_test_lst.append(RecallScore_test)



        # calculate f1 score

        F1Score_train = metrics.f1_score(y_train , y_train_pred)

        f1_train_lst.append(F1Score_train)

        

        F1Score_test = metrics.f1_score(y_test , y_test_pred)

        f1_test_lst.append(F1Score_test)



        #print('F1 Score of '+ name +' model : {0:0.5f}'.format(F1Score_test))



        # draw confusion matrix

        cnf_matrix = metrics.confusion_matrix(y_test , y_test_pred)



        print("Model Name :", name)

        

        print('Train Accuracy :{0:0.5f}'.format(Accuracy_train)) 

        print('Test Accuracy :{0:0.5f}'.format(Accuracy_test))

        

        print('Train AUC : {0:0.5f}'.format(Aucs_train))

        print('Test AUC : {0:0.5f}'.format(Aucs_test))

        

        print('Train Precision : {0:0.5f}'.format(PrecisionScore_train))

        print('Test Precision : {0:0.5f}'.format(PrecisionScore_test))

        

        print('Train Recall : {0:0.5f}'.format(RecallScore_train))

        print('Test Recall : {0:0.5f}'.format(RecallScore_test))

        

        print('Train F1 : {0:0.5f}'.format(F1Score_train))

        print('Test F1 : {0:0.5f}'.format(F1Score_test))

        

        print('Confusion Matrix : \n', cnf_matrix)

        print("\n")





        # plot ROC Curve

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred)

        auc = metrics.roc_auc_score(y_test, y_test_pred)

        plt.plot(fpr,tpr,linewidth=2, label=name + ", auc="+str(auc))

    

        #---------- For loops ends here--------#





    plt.legend(loc=4)

    plt.plot([0,1], [0,1], 'k--' )

    plt.rcParams['font.size'] = 12

    plt.title('ROC curve for Predicting a credit card fraud detection')

    plt.xlabel('False Positive Rate (1 - Specificity)')

    plt.ylabel('True Positive Rate (Sensitivity)')

    plt.show()
#------------------ Logistic Regression (LR) ------------------#

LRmodels = []



LRmodels.append(('LR imbalance', LogisticRegression(solver='liblinear', multi_class='ovr'),X,y))

LRmodels.append(('LR Undersampling', LogisticRegression(solver='liblinear', multi_class='ovr'),X_under,y_under))

LRmodels.append(('LR Oversampling', LogisticRegression(solver='liblinear', multi_class='ovr'),X_over,y_over))

LRmodels.append(('LR SMOTE', LogisticRegression(solver='liblinear', multi_class='ovr'),X_smote,y_smote))

LRmodels.append(('LR ADASYN', LogisticRegression(solver='liblinear', multi_class='ovr'),X_adasyn,y_adasyn))



# Call function to create model and measure its performance

build_measure_model(LRmodels)
#-----------------Decision Tree (DT)------------------#

DTmodels = []



dt = DecisionTreeClassifier()



DTmodels.append(('DT imbalance', dt,X,y))

DTmodels.append(('DT Undersampling', dt,X_under,y_under))

DTmodels.append(('DT Oversampling', dt,X_over,y_over))

DTmodels.append(('DT SMOTE', dt,X_smote,y_smote))

DTmodels.append(('DT ADASYN', dt,X_adasyn,y_adasyn))



# Call function to create model and measure its performance

build_measure_model(DTmodels)
#-----------------Random Forest (RF) ------------------#

RFmodels = []



RFmodels.append(('RF imbalance', RandomForestClassifier(),X,y))

RFmodels.append(('RF Undersampling', RandomForestClassifier(),X_under,y_under))

RFmodels.append(('RF Oversampling', RandomForestClassifier(),X_over,y_over))

RFmodels.append(('RF SMOTE', RandomForestClassifier(),X_smote,y_smote))

RFmodels.append(('RF ADASYN', RandomForestClassifier(),X_adasyn,y_adasyn))



# Call function to create model and measure its performance

build_measure_model(RFmodels)
# #------------------ K-Nearest Neighbors (KNN) ------------------#

# KNNmodels = []



# KNNmodels.append(('KNN imbalance', KNeighborsClassifier(),X,y))

# KNNmodels.append(('KNN Undersampling', KNeighborsClassifier(),X_under,y_under))

# KNNmodels.append(('KNN Oversampling', KNeighborsClassifier(),X_over,y_over))

# KNNmodels.append(('KNN SMOTE', KNeighborsClassifier(),X_smote,y_smote))

# KNNmodels.append(('KNN ADASYN', KNeighborsClassifier(),X_adasyn,y_adasyn))



# Call function to create model and measure its performance

# build_measure_model(KNNmodels)
# #------------------ Support Vector Machines (SVM) ------------------#

# SVMmodels = []



# SVMmodels.append(('SVM imbalance', SVC(gamma='auto'),X,y))

# SVMmodels.append(('SVM Undersampling', SVC(gamma='auto'),X_under,y_under))

# SVMmodels.append(('SVM Oversampling', SVC(gamma='auto'),X_over,y_over))

# SVMmodels.append(('SVM SMOTE', SVC(gamma='auto'),X_smote,y_smote))

# SVMmodels.append(('SVM ADASYN', SVC(gamma='auto'),X_adasyn,y_adasyn))



# Call function to create model and measure its performance

# build_measure_model(SVMmodels)
#------------------ Gaussian Naive Bayes (NB) ------------------#

NBmodels = []



NBmodels.append(('NB imbalance', GaussianNB(),X,y))

NBmodels.append(('NB Undersampling', GaussianNB(),X_under,y_under))

NBmodels.append(('NB Oversampling', GaussianNB(),X_over,y_over))

NBmodels.append(('NB SMOTE', GaussianNB(),X_smote,y_smote))

NBmodels.append(('NB ADASYN', GaussianNB(),X_adasyn,y_adasyn))



# Call function to create model and measure its performance

build_measure_model(NBmodels)
data = {'Model':names_lst,

       'Accuracy_Train':accuracy_train_lst,

       'Accuracy_Test':accuracy_test_lst,

       'AUC_Train':aucs_train_lst,

       'AUC_Test':aucs_test_lst,

       'PrecisionScore_Train':precision_train_lst,

       'PrecisionScore_Test':precision_test_lst,

       'RecallScore_Train':recall_train_lst,

       'RecallScore_Test':recall_test_lst,

       'F1Score_Train':f1_train_lst,

       'F1Score_Test':f1_test_lst}



print("Performance measures of various classifiers: \n")

performance_df = pd.DataFrame(data) 

performance_df.sort_values(['AUC_Test','RecallScore_Test','F1Score_Test'],ascending=False)
YouTubeVideo('Gol_qOgRqfA', width=800, height=400)
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV
#------------ Logistic Regression ------------#

log_reg_params = {"solver": ['saga'],

                  "penalty": ['l1', 'l2'], 

                  'C':  [0.01, 0.1, 1, 10, 100], 

                  "max_iter" : [100000]},



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train_under,y_train_under)



# Logistic Regression best estimator

print("Logistic Regression best estimator : \n",grid_log_reg.best_estimator_)



# predict test dataset

y_pred_lr = grid_log_reg.predict(X_test_under)



# f1 score

print('\nLogistic Regression f1 Score : {0:0.5f}'.format(metrics.f1_score(y_test_under , y_pred_lr)))
#------------ K Nearest Neighbour ------------#

knears_params = {"n_neighbors": list(range(2,60,1)), 

                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)



grid_knears.fit(X_train_under,y_train_under)



# KNears best estimator

print("KNN best estimator : \n",grid_knears.best_estimator_)



# predict test dataset

y_pred_knn = grid_knears.predict(X_test_under)



# f1 score

print('\nKNN f1 Score : {0:0.5f}'.format(metrics.f1_score(y_test_under , y_pred_knn)))
#------------ Support Vector Classifier ------------#

svc_params = {'C': [0.5, 0.7, 0.9, 1], 

              'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}



grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train_under,y_train_under)



# SVC best estimator

print("SVC best estimator : \n",grid_svc.best_estimator_)



# predict test dataset

y_pred_svc = grid_svc.predict(X_test_under)



# f1 score

print('\nSVC f1 Score : {0:0.5f}'.format(metrics.f1_score(y_test_under , y_pred_svc)))
#------------ DecisionTree Classifier ------------#

tree_params = {"criterion": ["gini", "entropy"], 

               "max_depth": list(range(2,4,1)), 

               "min_samples_leaf": list(range(5,7,1))}



grid_tree = GridSearchCV(estimator = DecisionTreeClassifier(),

                        param_grid = tree_params,

                        scoring = 'accuracy', 

                        cv = 5, 

                        verbose = 1,

                        n_jobs = -1)





grid_tree.fit(X_train_under,y_train_under)



# tree best estimator

print("Decision Tree best estimator : \n",grid_tree.best_estimator_)



# predict test dataset

y_pred_dt = grid_tree.predict(X_test_under)





# f1 score

print('\nf1 Score : {0:0.5f}'.format(metrics.f1_score(y_test_under , y_pred_dt)))