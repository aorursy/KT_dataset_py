# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

 # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import libraries

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
recruitment = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
recruitment.head(5)
recruitment.shape
recruitment.info()
recruitment.isnull().sum()
# Making null value as zero.
recruitment.fillna(0,inplace=True)
recruitment.head(5)
## Datatypes of columns
recruitment.dtypes
## Drop "sl.no." as it will not help in model learning.

recruitment.drop('sl_no', axis=1, inplace=True)
recruitment.drop(['hsc_b','ssc_b'], axis=1, inplace=True)
recruitment.head()
plt.figure(figsize=(15,10))

ax = plt.subplot(331)
plt.boxplot(recruitment['ssc_p'])
ax.set_title('Secondary School Percentage')

ax = plt.subplot(332)
plt.boxplot(recruitment['hsc_p'])
ax.set_title('Higher Secondary School Percentage')

ax = plt.subplot(333)
plt.boxplot(recruitment['degree_p'])
ax.set_title('Degree Percentage')

ax = plt.subplot(334)
plt.boxplot(recruitment['mba_p'])
ax.set_title('MBA Percentage')

ax = plt.subplot(335)
plt.boxplot(recruitment['etest_p'])
ax.set_title('Employibility Percentage')
Q1 = recruitment['hsc_p'].quantile(0.25)
Q3 = recruitment['hsc_p'].quantile(0.75)
IQR = Q3 - Q1

recruitment_processed= recruitment.loc[(recruitment['hsc_p'] >= Q1 - 1.5 * IQR) & (recruitment['hsc_p'] <= Q3 + 1.5 *IQR)]
plt.figure(figsize=(8,5))


plt.boxplot(recruitment_processed['hsc_p'])
plt.title('Higher Secondary School Percentage')

categorical_columns = recruitment_processed.select_dtypes("object").columns
categorical_columns
plt.figure(figsize = (15, 7))


#Gender
plt.subplot(231)
ax=sns.countplot(x="gender", data=recruitment_processed)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Higher secondary specialisation
plt.subplot(232)
ax=sns.countplot(x="hsc_s", data=recruitment_processed)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Degree type
plt.subplot(233)
ax=sns.countplot(x="degree_t", data=recruitment_processed)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Specialisation
plt.subplot(234)
ax=sns.countplot(x="specialisation", data=recruitment_processed)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Work experience
plt.subplot(235)
ax=sns.countplot(x="workex", data=recruitment_processed)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Status
plt.subplot(236)
ax=sns.countplot(x="status", data=recruitment_processed)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)
sns.pairplot(recruitment_processed,vars=['ssc_p','hsc_p','degree_p','mba_p','etest_p'],hue="status")
## Check categorical columns
categorical_columns
recruitment_processed[categorical_columns].head()
column_to_be_encoded = ['gender','workex','status']
# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in column_to_be_encoded:
    recruitment_processed[col] = label_encoder.fit_transform(recruitment_processed[col])
recruitment_processed.head()
## Creating dummies

dummies=pd.get_dummies(recruitment_processed[['hsc_s','degree_t','specialisation']])
recruitment_final = pd.concat([recruitment_processed,dummies],axis=1)
recruitment_final.drop(['hsc_s','degree_t','specialisation'],axis=1, inplace=True)
recruitment_final.head()
X = recruitment_final.drop(['status','salary'], axis=1)
y = recruitment_final['status']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, test_size=0.3, random_state=100)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# Import the StandardScaler()
from sklearn.preprocessing import StandardScaler

# Create a scaling object
scaler = StandardScaler()

# Create a list of the variables that you need to scale
varlist = ['ssc_p', 'hsc_p', 'degree_p','etest_p','mba_p']#, #'Asymmetrique Activity Score',
       #'Asymmetrique Profile Score']

# Scale these variables using 'fit_transform'
X_train[varlist] = scaler.fit_transform(X_train[varlist])

import statsmodels.api as sm

# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 10)             # running RFE with 13 variables as output
rfe = rfe.fit(X_train, y_train)
col = X_train.columns[rfe.support_]
col
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
## drop column whose VIF is more than 5
col =col.drop(['degree_t_Comm&Mgmt'])
col
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res2 = logm3.fit()
res2.summary()
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# Getting the predicted values on the train set
y_train_pred = res2.predict(X_train_sm)
y_train_pred[:10]

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'status':y_train.values, 'status_Prob':y_train_pred})
y_train_pred_final['ID'] = y_train.index
y_train_pred_final.head()
y_train_pred_final['Status_predicted'] = y_train_pred_final.status_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head(20)

from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.status, y_train_pred_final.Status_predicted )
print(confusion)
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.status, y_train_pred_final.Status_predicted))
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.status, y_train_pred_final.status_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.status, y_train_pred_final.status_Prob)

# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.status_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
 #Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.status, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
y_train_pred_final['Status_predicted'] = y_train_pred_final.status_Prob.map(lambda x: 1 if x > 0.7 else 0)

# Let's see the head
y_train_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.status, y_train_pred_final.Status_predicted)
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.status, y_train_pred_final.Status_predicted )
print(confusion)