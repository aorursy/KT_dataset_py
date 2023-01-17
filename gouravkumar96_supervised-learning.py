import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# importing file

bank_df = pd.read_csv("../input/Bank_Personal_Loan_Modelling.csv")
# Looking the first five observations

bank_df.head()
original_data = bank_df.copy()
# Shape of dataset

bank_df.shape



# 5000 observations with 14 columns
bank_df.columns
# Renaming columns having spaces in their title

bank_df.rename(columns = {'ZIP Code':'ZIP_Code', 'Personal Loan':'Personal_Loan', 'Securities Account':'Securities_Account', 'CD Account':'CD_Account'}, inplace = True)
bank_df.columns
# five point summary

bank_df.describe().T
# checking for any null values 

bank_df.isnull().any()
bank_df.info()
sns.boxplot(bank_df.Experience)
sns.boxplot(bank_df.Age)
sns.boxplot(bank_df.Income)
# Finding howmany outliers are in Income column

from numpy import percentile

q25, q75 = bank_df['Income'].quantile(0.25), bank_df['Income'].quantile(0.75)

# Finding IQR

IQR = q75 - q25

# calculating the outliers cutoff

cutoff = IQR * 1.5

lower, upper = q25 - cutoff, q75 + cutoff

#identifying outliers 

outliers = [x for x in bank_df.Income if x < lower or x > upper]

print("Total outliers in Income column are : ",len(outliers))
sns.boxplot(bank_df.CCAvg)
# Finding howmany outliers are in CCAvg column

from numpy import percentile

q25, q75 = bank_df['CCAvg'].quantile(0.25), bank_df['CCAvg'].quantile(0.75)

# Finding IQR

IQR = q75 - q25

# calculating the outliers cutoff

cutoff = IQR * 1.5

lower, upper = q25 - cutoff, q75 + cutoff

#identifying outliers 

outliers = [x for x in bank_df.CCAvg if x < lower or x > upper]

print("Total outliers in CCAvg column are : ",len(outliers))
sns.boxplot(bank_df.Mortgage)
# Finding howmany outliers are in Mortgage column

from numpy import percentile

q25, q75 = bank_df['Mortgage'].quantile(0.25), bank_df['Mortgage'].quantile(0.75)

# Finding IQR

IQR = q75 - q25

# calculating the outliers cutoff

cutoff = IQR * 1.5

lower, upper = q25 - cutoff, q75 + cutoff

#identifying outliers 

outliers = [x for x in bank_df.Mortgage if x < lower or x > upper]

print("Total outliers in Mortgage column are : ",len(outliers))
bank_df.drop(columns = {'ID','ZIP_Code'}, axis =1, inplace = True)
bank_df.columns
sns.countplot(bank_df.Family)
print(bank_df.Education.value_counts())

sns.countplot(bank_df.Education, palette='Set1')
print(bank_df.Securities_Account.value_counts())

sns.countplot(bank_df.Securities_Account, palette='Set1')
# 0: doesnot have certificate of deposit 1: have certificate of deposit

print(bank_df.CD_Account.value_counts())

sns.countplot(bank_df.CD_Account, palette='Set1')
# 0: does not use Online mode,  1: use online mode

print(bank_df.Online.value_counts())

sns.countplot(bank_df.Online, palette='Set1')
# 0: doesn't have a credit card, 1: have a credit card

print(bank_df.CreditCard.value_counts())

sns.countplot(bank_df.CreditCard, palette='Set1')
# 0: didnot accepted loan, 1: accepeted loan in last campaign

print(bank_df.Personal_Loan.value_counts())

sns.countplot(bank_df.Personal_Loan, palette='Set1')
sns.boxplot(x = 'Personal_Loan', y = 'Income', data = bank_df)
sns.boxplot(x = 'Personal_Loan', y = 'CCAvg', data = bank_df)
corr = bank_df.corr()

plt.figure(figsize = (15,10))

sns.heatmap(corr, annot = True, cmap = plt.cm.viridis)
# Let's see correlation w.r.t dependent variable - Personal_Loan

bank_df.corr().loc['Personal_Loan']
# checking the variance

bank_df.var()
# checking for multicollinearity

# using Variance Inflation Factor

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['VIF Factor'] = [variance_inflation_factor(bank_df.values,i) for i in range(bank_df.shape[1])]

vif['Features'] = bank_df.columns

vif
bank_df =  bank_df.drop(columns={'Age'})

bank_df.columns
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['VIF Factor'] = [variance_inflation_factor(bank_df.values,i) for i in range(bank_df.shape[1])]

vif['Features'] = bank_df.columns

vif
# Making dummy variables for Education

df_with_dummies = pd.get_dummies(bank_df, prefix = 'Category', columns=['Education'])

df_with_dummies.head()
# dropping 'Category_3' to prevent the model from dummy trap

df_with_dummies.drop(columns={'Category_3'}, inplace = True)
# renaming Category_1 and Category_2 with Undergraduate and Graduate respectively

df_with_dummies.rename(columns = {'Category_1':'Undergraduate','Category_2':'Graduate'}, inplace = True)
df_with_dummies.columns
X = df_with_dummies.drop(columns = {'Personal_Loan'}).values

y = df_with_dummies['Personal_Loan'].values
# splitting the data into training and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
# shape of (Y_test and  X_test) and (X_train and X_test) must be equal

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting the model on the training data 

from sklearn.linear_model import LogisticRegression

logisticregression = LogisticRegression(solver='liblinear')

logisticregression.fit(X_train, y_train)
# Predicting the values

y_pred = logisticregression.predict(X_test)
print(logisticregression.coef_)

print(logisticregression.intercept_)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True, fmt='g')

logistic_accuracy = round(accuracy_score(y_test,y_pred),2)

print("Accuracy for Logistic Regression Model : ", logistic_accuracy)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
# plotting auc roc curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



logistic_roc_auc = roc_auc_score(y_test, logisticregression.predict(X_test))

fpr1, tpr1, thresholds1 = roc_curve(y_test, logisticregression.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr1, tpr1, label='Logistic Regression (area = %0.3f)' % logistic_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")

plt.show()
# splitting data in training and testing set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
# shape of (Y_test and  X_test) and (X_train and X_test) must be equal

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# fitting the model on the training data 

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors = 3,)

knn_classifier.fit(X_train,y_train)
# predicting the values

y_pred = knn_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True, fmt='g')
knn_accuracy = round(accuracy_score(y_test,y_pred),2)

print("Accuracy score for KNN is: {0:.2f}".format(accuracy_score(y_test,y_pred)))
# plotting auc roc curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



knn_roc_auc = roc_auc_score(y_test, knn_classifier.predict(X_test))

fpr2, tpr2, thresholds2 = roc_curve(y_test, knn_classifier.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr2, tpr2, label='K-Nearest Neighbors (area = %0.3f)' % knn_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")

plt.show()
# splitting data in training and testing set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
# shape of (Y_test and  X_test) and (X_train and X_test) must be equal

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# fitting the model on the training data

from sklearn.naive_bayes import GaussianNB

naive_classifier = GaussianNB()

naive_classifier.fit(X_train, y_train)
# precditing the values

y_pred = naive_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True, fmt='g')
naive_bayes_accuracy = round(accuracy_score(y_test,y_pred), 2)

print("Accuracy score for Naive Bayes is: {0:.2f}".format(accuracy_score(y_test,y_pred)))
# plotting auc roc curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



naive_roc_auc = roc_auc_score(y_test, naive_classifier.predict(X_test))

fpr3, tpr3, thresholds3 = roc_curve(y_test, naive_classifier.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr3, tpr3, label='Naive Bayes (area = %0.3f)' % naive_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")

plt.show()
# splitting data in training and testing set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
# shape of (Y_test and  X_test) and (X_train and X_test) must be equal

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#fitting kernel SVM to training test

from sklearn.svm import SVC

kernel_svm = SVC(kernel='rbf', random_state = 0)

kernel_svm.fit(X_train, y_train)
#predicting the test result

y_pred = kernel_svm.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True, fmt='g')
svm_accuracy = round(accuracy_score(y_test,y_pred), 3)

print("Accuracy score for SVM is: {0:.3f}".format(accuracy_score(y_test,y_pred)))

# splitting data in training and testing set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
# shape of (Y_test and  X_test) and (X_train and X_test) must be equal

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting classifier to the Training set

from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier(criterion = 'gini', max_depth = 3 ,random_state=0)

# criterion default = "gini"

decisiontree.fit(X_train,y_train)
# Predicting the Test set results

y_pred = decisiontree.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True, fmt='g')
print("Accuracy score for training data is: {0:.3f}".format(decisiontree.score(X_train,y_train)))
decision_tree_accuracy = round(accuracy_score(y_test,y_pred), 3)

print("Accuracy score for test data (Decision Tree) is: {0:.3f}".format(accuracy_score(y_test,y_pred)))
# plotting auc roc curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



decision_tree_roc_auc = roc_auc_score(y_test, decisiontree.predict(X_test))

fpr5, tpr5, thresholds5 = roc_curve(y_test, decisiontree.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr5, tpr5, label='Decision Tree (area = %0.3f)' % decision_tree_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")

plt.show()
#comparing auc roc curve of all the models 

plt.figure(figsize=(10,7))

plt.plot(fpr1, tpr1, label='Logistic Regression (area = %0.3f)' % logistic_roc_auc)

plt.plot(fpr2, tpr2, label='K-Nearest Neighbors (area = %0.3f)' % knn_roc_auc)

plt.plot(fpr3, tpr3, label='Naive Bayes (area = %0.3f)' % naive_roc_auc)

plt.plot(fpr5, tpr5, label='Decision Tree (area = %0.3f)' % decision_tree_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Comparison')

plt.legend(loc="lower right")

plt.show()
#comparing accuracy_score of all the models 

print("Logistic Regression Model : ",logistic_accuracy)

print("KNN Model : ",knn_accuracy)

print("Naive Bayes Model : ", naive_bayes_accuracy)

print("SVM : ", svm_accuracy)

print("Decision Tree : ",decision_tree_accuracy)