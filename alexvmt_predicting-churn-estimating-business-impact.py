# import libraries

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
# load the data

df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# get basic information on the data

df.info()
# get the number of unique customers

len(df.customerID.unique())
# checkout the first 5 rows of the data to get an impression of the data

df.head()
# drop customerID

df = df.drop(['customerID'], axis = 1)
# convert TotalCharges to float

df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
# checkout amount of missing values

df.isna().sum()
# drop instances with missing values

df = df.dropna()

# check whether there remain missing values

df.isna().sum()
# import visualization libraries

import matplotlib.pyplot as plt
import seaborn as sns

# settings

sns.set()
%matplotlib inline
# countplot of Churn

sns.countplot(df['Churn'])
plt.show()
# amount of customers who didn't churn and who churned

df.groupby('Churn').size()
# churn rate

churn_rate = df.groupby('Churn').size()[1]/df.Churn.count()
print('Churn rate: %.2f%%' % (churn_rate * 100.0))
# countplots of all categorical features

df_cat_features = df[['gender', 
                        'SeniorCitizen',
                        'Partner', 
                        'Dependents', 
                        'PhoneService', 
                        'MultipleLines', 
                        'InternetService', 
                        'OnlineSecurity', 
                        'OnlineBackup', 
                        'DeviceProtection',
                        'TechSupport',
                        'StreamingTV', 
                        'StreamingMovies',
                        'Contract',
                        'PaperlessBilling',
                        'PaymentMethod']].copy()

plt.figure(figsize=(16,16))
for i in range(0,16):
        plt.subplot(4,4,i+1)
        sns.countplot(df_cat_features.iloc[:,i])
        plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# countplots of all categorical features by Churn

plt.figure(figsize=(16,16))
for i in range(0,16):
        plt.subplot(4,4,i+1)
        sns.countplot(df_cat_features.iloc[:,i], hue=df['Churn'])
        plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# summary statistics for numerical features

df.describe().iloc[:,1:4]
# heatmap of correlations of the numeric features

corr = df.drop('SeniorCitizen', axis=1).corr() # SeniorCitizen is again excluded here
sns.heatmap(corr, annot=True)
plt.show()
# boxplots of MonthlyCharges and TotalCharges

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.boxplot(df['MonthlyCharges'], orient='vertical', color='blue')
plt.subplot(2,2,2)
sns.boxplot(df['TotalCharges'], orient='vertical', color='orange')
plt.show()
# histograms of MonthlyCharges and TotalCharges

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.distplot(df['MonthlyCharges'], kde=True, color='blue')
plt.subplot(2,2,2)
sns.distplot(df['TotalCharges'], kde=True, color='orange')
plt.show()
# boxplot of tenure

sns.boxplot(df['tenure'], orient='vertical', color='red')
plt.show()
# histogram of tenure

sns.distplot(df['tenure'], kde=True, color='red')
plt.show()
# function to map values of tenure to different groups based on quartiles

def monthly_charges_group(row):
    if row['MonthlyCharges'] <= df.MonthlyCharges.quantile(0.25):
        return 'low'
    elif row['MonthlyCharges'] <= df.MonthlyCharges.quantile(0.5):
        return 'lower medium'
    elif row['MonthlyCharges'] <= df.MonthlyCharges.quantile(0.75):
        return 'upper medium'
    else:
        return 'high'

# create new column containing the group information based on quartiles 

df['monthly charges group'] = df.apply(monthly_charges_group, axis=1)

# countplot of monthly charges groups by Churn

sns.countplot(df['monthly charges group'], hue=df['Churn'], order=['low', 'lower medium', 'upper medium', 'high'])
plt.show()
# function to map values of tenure to different groups based on quartiles

def total_charges_group(row):
    if row['TotalCharges'] <= df.TotalCharges.quantile(0.25):
        return 'low'
    elif row['TotalCharges'] <= df.TotalCharges.quantile(0.5):
        return 'lower medium'
    elif row['TotalCharges'] <= df.TotalCharges.quantile(0.75):
        return 'upper medium'
    else:
        return 'high'
    
# create new column containing the group information based on quartiles 
    
df['total charges group'] = df.apply(total_charges_group, axis=1)

# countplot of total charges groups by Churn

sns.countplot(df['total charges group'], hue=df['Churn'], order=['low', 'lower medium', 'upper medium', 'high'])
plt.show()
# function to map values of tenure to different groups based on quartiles

def tenure_group(row):
    if row['tenure'] <= df.tenure.quantile(0.25):
        return 'low'
    elif row['tenure'] <= df.tenure.quantile(0.5):
        return 'lower medium'
    elif row['tenure'] <= df.tenure.quantile(0.75):
        return 'upper medium'
    else:
        return 'high'
    
# create new column containing the group information based on quartiles 

df['tenure group'] = df.apply(tenure_group, axis=1)

# countplot of tenure groups by Churn

sns.countplot(df['tenure group'], hue=df['Churn'], order=['low', 'lower medium', 'upper medium', 'high'])
plt.show()
# drop columns created for exploratory analysis and check df before proceeding to modeling

df = df.drop(['tenure group', 'monthly charges group', 'total charges group'], axis=1)
df.head()
# copy df to create df_enc to encode categorical features and the target

df_enc = df.copy()
# encode target

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df_enc['Churn'])
df_enc['Churn'] = le.transform(df_enc['Churn'])
# encode categorical features

categorical_features = ['gender', 
                        'SeniorCitizen',
                        'Partner', 
                        'Dependents', 
                        'PhoneService', 
                        'MultipleLines', 
                        'InternetService', 
                        'OnlineSecurity', 
                        'OnlineBackup', 
                        'DeviceProtection',
                        'TechSupport',
                        'StreamingTV', 
                        'StreamingMovies',
                        'Contract',
                        'PaperlessBilling',
                        'PaymentMethod']

df_enc = pd.get_dummies(df_enc, columns=categorical_features, drop_first=True)
# split encoded categorical features and encoded target into X and y

X = df_enc.drop('Churn', axis=1)
y = df_enc['Churn']
# split X and y in training and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# copy numerical features from X_train and X_test to create X_train_scaled and X_test_scaled for scaled numerical features

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
# scale numerical features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train_scaled[['tenure', 'MonthlyCharges', 'TotalCharges']])
X_train_scaled[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_train_scaled[['tenure', 'MonthlyCharges', 'TotalCharges']])
X_test_scaled[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_test_scaled[['tenure', 'MonthlyCharges', 'TotalCharges']])
# encoded and scaled train features

X_train_scaled.head()
# encoded and scaled test features

X_test_scaled.head()
from sklearn.linear_model import LogisticRegression

# fit the model to the training set

LR = LogisticRegression()
LR.fit(X_train_scaled, y_train)

# make predictions on the test set

y_pred = LR.predict(X_test_scaled)
# evaluate model

from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,classification_report

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f%%' % (accuracy * 100.0))
print('\n')
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print('AUC: %.2f' % auc(fpr, tpr))
print('\n')
print('Confusion matrix')
print(confusion_matrix(y_test, y_pred))
print('\n')
print('Classification report')
print(classification_report(y_test, y_pred))
# 10-fold cross validation

from sklearn.model_selection import cross_val_score,KFold

cv = KFold(n_splits=10, random_state=42)
cv_results = cross_val_score(LR, X_train_scaled, y_train, cv=cv, scoring='accuracy')

print('Accuracy of all 10 runs: ', cv_results)
print('Mean: %.2f%%' % cv_results.mean())
# grid search to find optimal regularization parameter C

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.0001, 0.001, 0.01, 1, 10, 100, 1000, 10000]}
LR_grid = GridSearchCV(LR, param_grid, cv=cv)

# fit model with optimized parameter C

LR_grid.fit(X_train_scaled,y_train)
# best C

print('Optimal C found by grid search: C =', LR_grid.best_params_['C'])
# make predictions with C=0.01

y_pred_grid = LR_grid.predict(X_test_scaled)
# evaluate model

accuracy_grid = accuracy_score(y_test, y_pred_grid)
print('Accuracy: %.2f%%' % (accuracy_grid * 100.0))
print('\n')
fpr_grid, tpr_grid, thresholds_grid = roc_curve(y_test, y_pred_grid)
print('AUC: %.2f' % auc(fpr_grid, tpr_grid))
print('\n')
print('Confusion matrix')
print(confusion_matrix(y_test, y_pred_grid))
print('\n')
print('Classification report')
print(classification_report(y_test, y_pred_grid))
# set up list with different model thresholds and an empty cost variable

thresh = np.arange(0.1,1.1,0.1).tolist()
cost = [0] * 10
# get probabilities of belonging to the 'churn' class

probs = LR_grid.predict_proba(X_test_scaled)[:,0]
# set up objects to store predictions for different thresholds

thresh_1 = [0] * y_test.shape[0]
thresh_2 = [0] * y_test.shape[0]
thresh_3 = [0] * y_test.shape[0]
thresh_4 = [0] * y_test.shape[0]
thresh_5 = [0] * y_test.shape[0]
thresh_6 = [0] * y_test.shape[0]
thresh_7 = [0] * y_test.shape[0]
thresh_8 = [0] * y_test.shape[0]
thresh_9 = [0] * y_test.shape[0]
thresh_10 = [0] * y_test.shape[0]
# convert probabilities to binary predictions for different thresholds and store them in the respective lists

for i in range(0,y_test.shape[0]):
    if probs[i] > 0.1:
        thresh_1[i] = 1
    else:
        thresh_1[i] = 0

for i in range(0,y_test.shape[0]):
    if probs[i] > 0.2:
        thresh_2[i] = 1
    else:
        thresh_2[i] = 0

for i in range(0,y_test.shape[0]):
    if probs[i] > 0.3:
        thresh_3[i] = 1
    else:
        thresh_3[i] = 0

for i in range(0,y_test.shape[0]):
    if probs[i] > 0.4:
        thresh_4[i] = 1
    else:
        thresh_4[i] = 0

for i in range(0,y_test.shape[0]):
    if probs[i] > 0.5:
        thresh_5[i] = 1
    else:
        thresh_5[i] = 0
        
for i in range(0,y_test.shape[0]):
    if probs[i] > 0.6:
        thresh_6[i] = 1
    else:
        thresh_6[i] = 0
        
for i in range(0,y_test.shape[0]):
    if probs[i] > 0.7:
        thresh_7[i] = 1
    else:
        thresh_7[i] = 0
        
for i in range(0,y_test.shape[0]):
    if probs[i] > 0.8:
        thresh_8[i] = 1
    else:
        thresh_8[i] = 0
        
for i in range(0,y_test.shape[0]):
    if probs[i] > 0.9:
        thresh_9[i] = 1
    else:
        thresh_9[i] = 0
        
for i in range(0,y_test.shape[0]):
    if probs[i] > 1.0:
        thresh_10[i] = 1
    else:
        thresh_10[i] = 0
# calculate hypothetical cost per customer depending on the model threshold and store it in the cost list

cf = confusion_matrix(thresh_1,y_test)
TP = cf[0][0]/X_test_scaled.shape[0]
FP = cf[1][0]/X_test_scaled.shape[0]
FN = cf[0][1]/X_test_scaled.shape[0]
TN = cf[1][1]/X_test_scaled.shape[0]
cost[0] = FN*250 + TP*50 + FP*50 + TN*0

cf = confusion_matrix(thresh_2,y_test)
TP = cf[0][0]/X_test_scaled.shape[0]
FP = cf[1][0]/X_test_scaled.shape[0]
FN = cf[0][1]/X_test_scaled.shape[0]
TN = cf[1][1]/X_test_scaled.shape[0]
cost[1] = FN*250 + TP*50 + FP*50 + TN*0

cf = confusion_matrix(thresh_3,y_test)
TP = cf[0][0]/X_test_scaled.shape[0]
FP = cf[1][0]/X_test_scaled.shape[0]
FN = cf[0][1]/X_test_scaled.shape[0]
TN = cf[1][1]/X_test_scaled.shape[0]
cost[2] = FN*250 + TP*50 + FP*50 + TN*0

cf = confusion_matrix(thresh_4,y_test)
TP = cf[0][0]/X_test_scaled.shape[0]
FP = cf[1][0]/X_test_scaled.shape[0]
FN = cf[0][1]/X_test_scaled.shape[0]
TN = cf[1][1]/X_test_scaled.shape[0]
cost[3] = FN*250 + TP*50 + FP*50 + TN*0

cf = confusion_matrix(thresh_5,y_test)
TP = cf[0][0]/X_test_scaled.shape[0]
FP = cf[1][0]/X_test_scaled.shape[0]
FN = cf[0][1]/X_test_scaled.shape[0]
TN = cf[1][1]/X_test_scaled.shape[0]
cost[4] = FN*250 + TP*50 + FP*50 + TN*0

cf = confusion_matrix(thresh_6,y_test)
TP = cf[0][0]/X_test_scaled.shape[0]
FP = cf[1][0]/X_test_scaled.shape[0]
FN = cf[0][1]/X_test_scaled.shape[0]
TN = cf[1][1]/X_test_scaled.shape[0]
cost[5] = FN*250 + TP*50 + FP*50 + TN*0

cf = confusion_matrix(thresh_7,y_test)
TP = cf[0][0]/X_test_scaled.shape[0]
FP = cf[1][0]/X_test_scaled.shape[0]
FN = cf[0][1]/X_test_scaled.shape[0]
TN = cf[1][1]/X_test_scaled.shape[0]
cost[6] = FN*250 + TP*50 + FP*50 + TN*0

cf = confusion_matrix(thresh_8,y_test)
TP = cf[0][0]/X_test_scaled.shape[0]
FP = cf[1][0]/X_test_scaled.shape[0]
FN = cf[0][1]/X_test_scaled.shape[0]
TN = cf[1][1]/X_test_scaled.shape[0]
cost[7] = FN*250 + TP*50 + FP*50 + TN*0

cf = confusion_matrix(thresh_9,y_test)
TP = cf[0][0]/X_test_scaled.shape[0]
FP = cf[1][0]/X_test_scaled.shape[0]
FN = cf[0][1]/X_test_scaled.shape[0]
TN = cf[1][1]/X_test_scaled.shape[0]
cost[8] = FN*250 + TP*50 + FP*50 + TN*0

cf = confusion_matrix(thresh_10,y_test)
TP = cf[0][0]/X_test_scaled.shape[0]
FP = cf[1][0]/X_test_scaled.shape[0]
FN = cf[0][1]/X_test_scaled.shape[0]
TN = cf[1][1]/X_test_scaled.shape[0]
cost[9] = FN*250 + TP*50 + FP*50 + TN*0
print('cost per customer')
cost
# plot hypothetical cost for different thresholds

import matplotlib.pyplot as plt

plt.plot(thresh,cost)
plt.title('cost per customer for different model thresholds')
plt.xlabel('threshold')
plt.ylabel('cost per customer')
plt.show()
# assume that model with threshold=0.5 is currently used, calculate associated cost

cost_current_model = cost[5]
cost_current_model
# calculate hypothetical savings per customer as the difference between the currently used model and the optimal model

savings_per_customer = cost_current_model - min(cost)
savings_per_customer
# assume a customer base of 350000, multiply by savings per customer to get total savings
total_savings = 350000*savings_per_customer
total_savings