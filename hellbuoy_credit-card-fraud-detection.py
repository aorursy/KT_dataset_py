# import all the required libraries and dependencies for dataframe



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta



# import all libraries and dependencies for data visualization

pd.options.display.float_format='{:.4f}'.format

plt.rcParams['figure.figsize'] = [8,8]

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', -1) 

sns.set(style='darkgrid')

import matplotlib.ticker as plticker

%matplotlib inline



# import all the required libraries and dependencies for machine learning



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn import metrics

import statsmodels.api as sm

import pickle

import gc 

from sklearn import svm

from xgboost import XGBClassifier

import xgboost as xgb

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, average_precision_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
# Local file path.Please change the path accordingly.



path = '../input/credit-card-fraud/creditcard.csv'

# Reading the Credit Card file on which analysis needs to be done



df_card = pd.read_csv(path)

df_card.head()
# Shape of the Credit card dataframe



df_card.shape
# Data Description



df_card.describe()
# Data Information



df_card.info()
# Calculating the Missing Value% in the DF



df_null = df_card.isnull().mean()*100

df_null.sort_values(ascending=False).head()
# Datatype check for the dataframe



df_card.dtypes
plt.figure(figsize=(13,7))

plt.subplot(121)

plt.title('Fraudulent BarPlot', fontweight='bold',fontsize=14)

ax = df_card['Class'].value_counts().plot(kind='bar')

total = float(len(df_card))

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.5f}'.format(height/total),

            ha="center") 





plt.subplot(122)

df_card["Class"].value_counts().plot.pie(autopct = "%1.5f%%")

plt.show()
classes=df_card['Class'].value_counts()

normal_share=classes[0]/df_card['Class'].count()*100

fraud_share=classes[1]/df_card['Class'].count()*100

print(normal_share)

print(fraud_share)
# Box Plot of amount for both classes

plt.figure(figsize = (7, 6))

a=sns.boxplot(x = 'Class', y = 'Amount',hue='Class', data = df_card,showfliers=False) 

plt.setp(a.get_xticklabels(), rotation=45)
# KDE plot to visualize the distribution of Amount for both the classes

plt.rcParams['figure.figsize'] = [10,6]

sns.kdeplot(df_card.loc[df_card['Class'] == 0, 'Amount'], label = 'Non Fraud')

sns.kdeplot(df_card.loc[df_card['Class'] == 1, 'Amount'], label = 'Fraud')

plt.title('Distribution of Amount by Target Value')

plt.xlabel('Amount')

plt.ylabel('Density')
# Time Distribution plot for transactions 

plt.figure(figsize=(15,7))



plt.title('Distribution of Transaction Time')

sns.distplot(df_card['Time'].values/(60*60))
# Storing Fraud and non-Fraud transactions 



df_nonfraud = df_card[df_card.Class == 0]

df_fraud = df_card[df_card.Class == 1]
#Scatter plot between Time and Amount



fig = plt.figure(figsize = (8,8))

plt.scatter(df_nonfraud.Amount, df_nonfraud.Time.values/(60*60),alpha=0.5,label='Non Fraud')

plt.scatter(df_fraud.Amount, df_fraud.Time.values/(60*60),alpha=1,label='Fraud')

plt.xlabel('Amount')

plt.ylabel('Time')

plt.title('Scatter plot between Amount and Time ')

plt.show()
# Plot of high value transactions($200-$2000)



bins = np.linspace(200, 2000, 100)

plt.hist(df_nonfraud.Amount, bins, alpha=1, density=True, label='Non-Fraud')

plt.hist(df_fraud.Amount, bins, alpha=1, density=True, label='Fraud')

plt.legend(loc='upper right')

plt.title("Amount by percentage of transactions (transactions \$200-$2000)")

plt.xlabel("Transaction amount (USD)")

plt.ylabel("Percentage of transactions (%)")

plt.show()
# Plot of transactions in 48 hours



bins = np.linspace(0, 48, 48)

plt.hist((df_nonfraud.Time/(60*60)), bins, alpha=1,label='Non-Fraud')

plt.hist((df_fraud.Time/(60*60)), bins, alpha=0.6,label='Fraud')

plt.legend(loc='upper right')

plt.title("Percentage of transactions by hour")

plt.xlabel("Transaction time from first transaction in the dataset (hours)")

plt.ylabel("Percentage of transactions (%)")

plt.show()
# Putting the feature variable into X



X = df_card.drop(['Class'],axis = 1)

X.head(2)
# Putting the Target variable to y



y = df_card['Class']
from sklearn.model_selection import StratifiedShuffleSplit
# Splitting the data into Train and Test set

kfold = 4

sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.3, random_state=9487)

for train_index, test_index in sss.split(X, y):

        print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = X.iloc, X.iloc

        y_train, y_test = y[train_index], y[test_index]
# Checking Skewness of data



plt.rcParams['figure.figsize'] = [10,8]

plt.hist(df_card['Amount'],edgecolor='k',bins = 5)

plt.title('Transaction Amount')

plt.xlabel('Amount in USD') 

plt.ylabel('Count')
from sklearn import preprocessing

from sklearn.preprocessing import PowerTransformer
pt = preprocessing.PowerTransformer(copy=False)

PWTR_X = pt.fit_transform(X)
# Splitting dataset into test and train sets in 70:30 ratio after applying Power Transform



kfold = 4

sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.3, random_state=9487)

for train_index, test_index in sss.split(PWTR_X, y):

        print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = PWTR_X[train_index], PWTR_X[test_index]

        y_train, y_test = y[train_index], y[test_index]
from sklearn.linear_model import LogisticRegression



# Fit a logistic regression model to train data

model_lr = LogisticRegression()

model_lr.fit(X_train, y_train)

# Predict on test data

y_predicted = model_lr.predict(X_test)
# Evaluation Metrics



print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# Function for roc_curve

def plot_roc_curve(fpr,tpr,roc_auc):

    plt.plot(fpr, tpr, linewidth=5, label='AUC = %0.3f'% roc_auc)

    plt.plot([0,1],[0,1], linewidth=5)

    plt.xlim([-0.01, 1])

    plt.ylim([0, 1.01])

    plt.legend(loc='upper right')

    plt.title('Receiver operating characteristic curve (ROC)')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
# tpr and fpr

fpr, tpr, threshold = roc_curve(y_test, y_predicted)

roc_auc = roc_auc_score(y_test, y_predicted)
# Plotting the roc curve 

plt.rcParams['figure.figsize'] = [6,6]

plot_roc_curve(fpr,tpr,roc_auc)
from imblearn.metrics import sensitivity_specificity_support
# Number of folds



n_folds = 5

# parameters 

params ={'C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'penalty': ['l1', 'l2']}



lrh = LogisticRegression()



model_lrh = GridSearchCV(estimator=lrh, cv=n_folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)
# Fitting the model



model_lrh.fit(X_train,y_train)
pd.DataFrame(model_lrh.cv_results_)
print("Logistic Regression with PCA Best AUC : ", model_lrh.best_score_)

print("Logistic Regression with PCA Best hyperparameters: ", model_lrh.best_params_)
# Passing the best parameteres

model_lrh_tuned = LogisticRegression(penalty='l2',C=0.1)
# Predicting on test data



model_lrh_tuned.fit(X_train,y_train)

y_predicted = model_lrh_tuned.predict(X_test)
#Evaluation Metrices



print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# Create true and false positive rates

fpr, tpr, threshold = roc_curve(y_test, y_predicted)

roc_auc = roc_auc_score(y_test, y_predicted)

# Plot the roc curve 

plt.rcParams['figure.figsize'] = [6,6]

plot_roc_curve(fpr,tpr,roc_auc)
#Initializing Random forest and creating model



from sklearn.ensemble import RandomForestClassifier

model_rfc = RandomForestClassifier(n_jobs=-1, 

                             random_state=2018,

                             criterion='gini',

                             n_estimators=100,

                             verbose=False)
# Fitting the model on Train data and Predicting on Test data



model_rfc.fit(X_train,y_train)

y_predicted = model_rfc.predict(X_test)
# Evaluation Metrics



print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# Create true and false positive rates

fpr, tpr, threshold = roc_curve(y_test, y_predicted)

roc_auc = roc_auc_score(y_test, y_predicted)

# Plot the roc curve 

plt.rcParams['figure.figsize'] = [6,6]

plot_roc_curve(fpr,tpr,roc_auc)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV
# Defining Parameters

params = { 

    'n_estimators': [200, 400],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}
# Stratified K Fold

cross_val = StratifiedKFold(n_splits=3)

index_iterator = cross_val.split(X_train, y_train)

clf = RandomForestClassifier()

clf_random = RandomizedSearchCV(estimator = clf, param_distributions = params, n_iter = 50, cv = cross_val,

                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')

# Fitting the model on train data

# clf_random.fit(X_train, y_train)
# Scores of RandomizedSearchCV

#scores = clf_random.cv_results_

#pd.DataFrame(scores).head()
#print(clf_random.best_score_)

#print(clf_random.best_params_)

#print(clf_random.best_index_)
# Passing the best parameteres based on Randomized Search CV

model_rfc_tuned = RandomForestClassifier(bootstrap=True,

                               class_weight={0:1, 1:12}, # 0: non-fraud , 1:fraud

                               criterion='gini',

                               max_depth=5,

                               max_features='sqrt',

                               min_samples_leaf=10,

                               n_estimators=200,

                               n_jobs=-1, 

                               random_state=5)
# Fitting the model on Train data and Predicting on Test Data



model_rfc_tuned.fit(X_train,y_train)

y_predicted = model_rfc_tuned.predict(X_test)
# Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# Create true and false positive rates

fpr, tpr, threshold = roc_curve(y_test, y_predicted)

roc_auc = roc_auc_score(y_test, y_predicted)

# Plot the roc curve 

plt.rcParams['figure.figsize'] = [6,6]

plot_roc_curve(fpr,tpr,roc_auc)
#Initializing Random forest and creating model

model_xgb = XGBClassifier()
# Fitting the model on Train data and Predicting on Test data

model_xgb.fit(X_train,y_train)

y_predicted = model_xgb.predict(X_test)
# Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# Create true and false positive rates

fpr, tpr, threshold = roc_curve(y_test, y_predicted)

roc_auc = roc_auc_score(y_test, y_predicted)

# Plot the roc curve 

plt.rcParams['figure.figsize'] = [6,6]

plot_roc_curve(fpr,tpr,roc_auc)
# Defining parameters

params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }
# Stratified K Fold

cross_val = StratifiedKFold(n_splits=5)

index_iterator = cross_val.split(X_train, y_train)





xgb_cross = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic',

                    silent=True, nthread=1) 





xgb_random = RandomizedSearchCV(estimator = xgb_cross, param_distributions = params, n_iter =30 , cv = cross_val,

                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')
# Fitting the model on train data

#xgb_random.fit(X_train, y_train)
# scores of RandomizedSearchCV

#scores = xgb_random.cv_results_

#pd.DataFrame(scores).head()
#print(xgb_random.best_score_)

#print(xgb_random.best_params_)

#print(xgb_random.best_index_)
# Passing the best parameteres based on Randomized Search CV

model_xgb_tuned = XGBClassifier(min_child_weight= 5,

        gamma= 1.5,

        subsample= 1.0,

        colsample_bytree= 0.6,

        max_depth= 5)
# Fitting the model on Train data and Predicting on Test data

model_xgb_tuned.fit(X_train,y_train)

y_predicted = model_xgb_tuned.predict(X_test)
# Evaluation metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# Create true and false positive rates

fpr, tpr, threshold = roc_curve(y_test, y_predicted)

roc_auc = roc_auc_score(y_test, y_predicted)

# Plot the roc curve 

plt.rcParams['figure.figsize'] = [6,6]

plot_roc_curve(fpr,tpr,roc_auc)
from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import ADASYN



# Resample training data

ros = RandomOverSampler()

smote = SMOTE(random_state=5)

adasyn = ADASYN(random_state=5)



X_train_ros, y_train_ros = ros.fit_sample(X_train,y_train)

X_train_smote, y_train_smote = smote.fit_sample(X_train,y_train)

X_train_adasyn, y_train_adasyn =adasyn.fit_sample(X_train,y_train)
# Fit a logistic regression model to our data

from sklearn.linear_model import LogisticRegression



model_lr = LogisticRegression()

model_lr.fit(X_train_ros, y_train_ros)



# Obtain model predictions

y_predicted = model_lr.predict(X_test)
# Evaluation Metrics

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# Create true and false positive rates

fpr, tpr, threshold = roc_curve(y_test, y_predicted)

roc_auc = roc_auc_score(y_test, y_predicted)

# Plot the roc curve 

plt.rcParams['figure.figsize'] = [6,6]

plot_roc_curve(fpr,tpr,roc_auc)
# Fit a logistic regression model to our data

from sklearn.linear_model import LogisticRegression



model_lr = LogisticRegression()

model_lr.fit(X_train_smote, y_train_smote)



# Obtain model predictions

y_predicted = model_lr.predict(X_test)
# Evaluation Metrics

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# Create true and false positive rates

fpr, tpr, threshold = roc_curve(y_test, y_predicted)

roc_auc = roc_auc_score(y_test, y_predicted)

# Plot the roc curve 

plt.rcParams['figure.figsize'] = [6,6]

plot_roc_curve(fpr,tpr,roc_auc)
# Fit a logistic regression model to our data

from sklearn.linear_model import LogisticRegression



model_lr = LogisticRegression()

model_lr.fit(X_train_adasyn, y_train_adasyn)



# Obtain model predictions

y_predicted = model_lr.predict(X_test)
# Evaluation Metrics

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# Insantiate Model

model_rfc = RandomForestClassifier(bootstrap=True,

                               class_weight={0:1, 1:12}, # 0: non-fraud , 1:fraud

                               criterion='entropy',

                               max_depth=10, # Change depth of model

                               min_samples_leaf=10, # Change the number of samples in leaf nodes

                               n_estimators=20, # Change the number of trees to use

                               n_jobs=-1, 

                               random_state=5)
# Fit the model on train data and predict on test data 

model_rfc.fit(X_train_ros,y_train_ros)

y_predicted = model_rfc.predict(X_test)
# Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# Fit the model on train data and predict on test data 

model_rfc.fit(X_train_smote,y_train_smote)

y_predicted = model_rfc.predict(X_test)
# Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# Fit the model on train data and predict on test data 

model_rfc.fit(X_train_adasyn,y_train_adasyn)

y_predicted = model_rfc.predict(X_test)
# Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
params = { 

    'n_estimators': [200, 400],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}
cross_val = StratifiedKFold(n_splits=3)

index_iterator = cross_val.split(X_train_ros, y_train_ros)

clf = RandomForestClassifier()

clf_random = RandomizedSearchCV(estimator = clf, param_distributions = params, n_iter = 50, cv = cross_val,

                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')
#clf_random.fit(X_train_ros, y_train_ros)
# Scores of RandomizedSearchCV

#scores = clf_random.cv_results_

#pd.DataFrame(scores).head()
#print(clf_random.best_score_)

#print(clf_random.best_params_)

#print(clf_random.best_index_)
# Insanitiate Model on best params

model_rfc_tuned = RandomForestClassifier(bootstrap=True,

                               class_weight={0:1, 1:12}, 

                               criterion='entropy',

                               max_depth=8, 

                               max_features='auto',

                               n_estimators=200,

                               n_jobs=-1)
#Fit the model on train data and predict the model on test data

model_rfc_tuned.fit(X_train_ros,y_train_ros)

y_predicted = model_rfc_tuned.predict(X_test)
# Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
model_xgb_ros = XGBClassifier()
#Fit the model on train data and predict the model on test data

model_xgb_ros.fit(X_train_ros,y_train_ros)

y_predicted = model_xgb_ros.predict(X_test)
# Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# A parameter grid for XGBoost

params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }
cross_val = StratifiedKFold(n_splits=4)

index_iterator = cross_val.split(X_train_ros, y_train_ros)





xgb_cross = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic',

                    silent=True, nthread=1) 





xgb_random = RandomizedSearchCV(estimator = xgb_cross, param_distributions = params, n_iter =30 , cv = cross_val,

                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')
#xgb_random.fit(X_train_ros, y_train_ros)
# scores of RandomizedSearchCV

#scores = xgb_random.cv_results_

#pd.DataFrame(scores).head()
#print(xgb_random.best_score_)

#print(xgb_random.best_params_)

#print(xgb_random.best_index_)
model_xgb_tuned_ros = XGBClassifier(min_child_weight= 5,

        gamma= 1.5,

        subsample= 1.0,

        colsample_bytree= 0.6,

        max_depth= 5)
#Fit the model on train data and predict the model on test data

model_xgb_tuned_ros.fit(X_train_ros,y_train_ros)

y_predicted = model_xgb_tuned_ros.predict(X_test)
# Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
model_xgb_smote = XGBClassifier()
#Fit the model on train data and predict the model on test data

model_xgb_smote.fit(X_train_smote,y_train_smote)

y_predicted = model_xgb_smote.predict(X_test)
# Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# A parameter grid for XGBoost

params = {

        'min_child_weight': [1, 5, 10,15],

        'gamma': [0.5, 1, 1.5, 2, 5,8],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0,1.2],

        'max_depth': [3, 4, 5,6,7]

        }
cross_val = StratifiedKFold(n_splits=5)

index_iterator = cross_val.split(X_train_smote, y_train_smote)





xgb_cross = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic',

                    silent=True, nthread=1) 





xgb_random = RandomizedSearchCV(estimator = xgb_cross, param_distributions = params, n_iter =40 , cv = cross_val,

                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')
#xgb_random.fit(X_train_smote, y_train_smote)
# scores of RandomizedSearchCV

#scores = xgb_random.cv_results_

#pd.DataFrame(scores).head()
#print(xgb_random.best_score_)

#print(xgb_random.best_params_)

#print(xgb_random.best_index_)
model_xgb_tuned_smote = XGBClassifier(min_child_weight= 10,

        gamma= 1.5,

        subsample= 0.6,

        colsample_bytree= 0.6,

        max_depth= 5)
#Fit the model on train data and predict the model on test data

model_xgb_tuned_smote.fit(X_train_smote,y_train_smote)

y_predicted = model_xgb_tuned.predict(X_test)
#Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
model_xgb_adasyn = XGBClassifier()
#Fit the model on train data and predict the model on test data

model_xgb_adasyn.fit(X_train_adasyn,y_train_adasyn)

y_predicted = model_xgb_adasyn.predict(X_test)
#Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
# A parameter grid for XGBoost

params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }
cross_val = StratifiedKFold(n_splits=5)

index_iterator = cross_val.split(X_train_adasyn, y_train_adasyn)





xgb_cross = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic',

                    silent=True, nthread=1) 





xgb_random = RandomizedSearchCV(estimator = xgb_cross, param_distributions = params, n_iter =30 , cv = cross_val,

                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')
#xgb_random.fit(X_train_adasyn, y_train_adasyn)
# scores of RandomizedSearchCV

#scores = xgb_random.cv_results_

#pd.DataFrame(scores).head()
#print(xgb_random.best_score_)

#print(xgb_random.best_params_)

#print(xgb_random.best_index_)
model_xgb_tuned_adasyn = XGBClassifier(min_child_weight= 10,

        gamma= 1.5,

        subsample= 0.6,

        colsample_bytree= 0.6,

        max_depth= 5)
#Fit the model on train data and predict the model on test data

model_xgb_tuned_adasyn.fit(X_train_adasyn,y_train_adasyn)

y_predicted = model_xgb_tuned_adasyn.predict(X_test)
#Evaluation Metrices

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))
#Predicting on the test data using the best model

y_predicted = model_xgb_smote.predict(X_test)
# Create true and false positive rates

fpr, tpr, thresholds = roc_curve(y_test, y_predicted)

roc_auc = roc_auc_score(y_test, y_predicted)
# Printing Evaluation Metrices

print('Classification report for XGBoost Smote:\n', classification_report(y_test, y_predicted))

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted))

print('ROC AUC : ', roc_auc_score(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

threshold = thresholds[np.argmax(tpr-fpr)]

print("Threshold:",threshold)
# Plotting the roc curve 

plt.rcParams['figure.figsize'] = [6,6]

plot_roc_curve(fpr,tpr,roc_auc)
target = 'Class'

pca_comp = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\

       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\

       'Amount', 'Time']
tmp = pd.DataFrame({'Feature': pca_comp, 'Feature importance': model_xgb_smote.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (7,4))

plt.title('Features importance',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show()  