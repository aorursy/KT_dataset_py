# Loads Python libraries used into the Kaggle Kernel.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
# The data is read in from a CSV file format into a Pandas data frame and is stored in a variable loan_data.

# Data is stored on Kaggle Kernel by default.

loan_data = pd.read_csv("../input/loan_data.csv")
# The data frame structure and format is displayed as well as the formatting of each feature.

# The loan_data data frame has zero null values and one cateforical feature(purpose) which will be transformed into dummy variables in a later step.

loan_data.info()
# The first 10 rows of the loan_data data frame are displayed.

loan_data.head(10)
loan_data.describe()
# A histogram of two FICO distributions on top of each other, one for each credit.policy outcome is displayed.

# A credit policy is extended from businesses to cutomers when credit is issued. We can see below that when credit.policy=0

# no credit was extended to customers. These cutomers tend to have lower FICO scores when looking at the graph below.

plt.figure(figsize=(10,6))

loan_data[loan_data['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',

                                              bins=30,label='Has an open credit policy')

loan_data[loan_data['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',

                                              bins=30,label='Does not have a credit policy')

plt.legend()

plt.xlabel('FICO')
# A histogram of two FICO distributions on top of each other, one for each loan outcome outcome is displayed as a histogram.

# An interesting trend is those with lower credit scores appear to not pay back the loan balance in full.

plt.figure(figsize=(10,6))

loan_data[loan_data['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='red',

                                              bins=30,label='Loan not paid in full')

loan_data[loan_data['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='blue',

                                              bins=30,label='Loan paid in full')

plt.legend()

plt.xlabel('FICO')
# A barplot was created to compare loans that were paid to those which weren't paid back to the described purpose of the loan.

chart = sns.countplot(data=loan_data,x="purpose",hue="not.fully.paid",palette='Set1',orient='v')

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
# A linear plot is created to show how the number of delinquency payments within the last 2 years have affected interest rates for cusotmers

# based on FICO score. 

# Customers with no or a few delinquencies that have a higher FICO score get lower interest rates according to the graph below.

sns.lmplot(y='int.rate',x='fico',data=loan_data,hue='delinq.2yrs',palette="cubehelix")
# A histogram of each feature is created and displayed diagonally. Additionally a scatter plot of each feature is generated to 

# compare the distribution of each features data points for finding relationships between them.

sns.pairplot(loan_data);
# The purpose feature is a categorical feature. Using the pandas get dummies function it is turned into dummy features.

cat_feat=['purpose']

loan_data = pd.get_dummies(loan_data,columns=cat_feat,drop_first=True)
# A statistical summary of the loan_data data frame is generated.

loan_data.describe()
# There are features from the above statistical description which have highly variable scales. 

# In order for a machine learning algorithm to work properly the scales of each variable should be normalized.

# Scikit MinMaxScaler is used to normalize the features in the function below.

scaler = MinMaxScaler() 

loan_data[['installment','log.annual.inc','dti','fico','days.with.cr.line','revol.bal','revol.util','inq.last.6mths','delinq.2yrs','pub.rec']] = scaler.fit_transform(loan_data[['installment','log.annual.inc','dti','fico','days.with.cr.line','revol.bal','revol.util','inq.last.6mths','delinq.2yrs','pub.rec']])



# loan_data is stored as a scaled_data variable

scaled_data = loan_data
# The features within the data frame are now normalized. They each range from 0 to 1 which will allow the machine learning algorithms

# to work properly and effectively.

# A statistical summary of scaled_data is generated.

scaled_data.describe()
scaled_data.head()
# Displays the count of each class for the not.fully.paid feature. It is apparent that this target feature is imbalance after running the code below.

scaled_data['not.fully.paid'].value_counts()
# The scaled_data is split into training and validation sets. 80% is for training and 20% for testing.

# To create the X variable the not.fully.paid variable is dropped from the data frame.

# To create the y variable only the not.fully.paid variable is selected.

X = scaled_data.drop('not.fully.paid',axis=1)

y = scaled_data['not.fully.paid']

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# The imbalanced not.fully.paid feature has its classes balanced using SMOTE in the code below.



print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 



# import SMOTE module from imblearn library 

# Loads the imblearn library which has SMOTE functionality 

from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state = 2) 

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 



print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 



print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 

# A grid Search is preformed for the best parameters for Logistic Regression

parameters = {

    'C': np.linspace(1, 50, 50)

             }

lr = LogisticRegression(solver='lbfgs', multi_class='auto')

clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)

clf.fit(X_train_res, y_train_res.ravel())
# The best parameters are shown for LR

clf.best_params_
# Six algorithm models will be created using the training data.

models = []

models.append(('RFC', RandomForestClassifier()))

models.append(('LR', LogisticRegression(C=47,solver='lbfgs', multi_class='auto',max_iter=1000)))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('ADB', AdaBoostClassifier()))

models.append(('SVM', SVC(gamma='auto')))

# Each model is evaluated using cross validation with 10 splits. 

results = []

names = []

for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1)

    cv_results = cross_val_score(model, X_train_res, y_train_res, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# A box & whisker plot is created to compare the six algorithms accuracies against each other.

plt.boxplot(results, labels=names)

plt.title('Algorithm Comparison')

plt.show() #Both logistic regression and support vector algorithms performed the best.
# Using the Logistic Regression algorithm the code below will make predictions using the validation dataset.

model = LogisticRegression(C=47,solver='lbfgs', multi_class='auto',max_iter=1000)

model.fit(X_train_res, y_train_res)

predictions = model.predict(X_validation)
# The accuracy score, confusion matrix, and classification_report is displayed for the validation data using the Logistic Regression algorithm.

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))



# the model has a 65% accuracy rate in predicting whether or not someone will repay their loan.