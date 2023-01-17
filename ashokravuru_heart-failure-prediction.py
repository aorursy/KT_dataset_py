# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
heart_df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
heart_df.info()
heart_df.describe().transpose()
col = list(heart_df.columns)
heart_df.head()
sns.pairplot(vars=col, diag_kind = 'kde', data = heart_df, hue= 'DEATH_EVENT')
plt.show()
binary_vars = ['anaemia','diabetes','high_blood_pressure','sex','smoking','DEATH_EVENT']
plt.figure(figsize=(8,24))
for i in enumerate(binary_vars):
    plt.subplot(3,2,i[0]+1)
    sns.countplot(heart_df[i[1]],hue='DEATH_EVENT', data = heart_df)
plt.show()
cont_vars = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']
plt.figure(figsize=(10,24))
for i in enumerate(cont_vars):
    plt.subplot(4,2,i[0]+1)
    sns.boxplot(y=heart_df[i[1]], data = heart_df)
plt.show()

# plt.figure(figsize=(8,24))
# for i in enumerate(cont_vars):
#     plt.subplot(4,2,i[0]+1)
#     sns.boxplot(y=i[1],x='DEATH_EVENT', data = heart_df)
# plt.show()
# Data Preparation for Modeling
# outlier treatment
capping =['creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium']

for i in capping:
    Q3  = heart_df[i].quantile(0.75)
    Q1  = heart_df[i].quantile(0.25)
    IQR = Q3-Q1
    UW = Q3 + 1.5*IQR
    LW = Q1 - 1.5*IQR
    heart_df[i]= heart_df[i].apply(lambda x: x if x<=UW else UW)
    heart_df[i] = heart_df[i].apply(lambda x: x if x>=LW else LW)
    
cont_vars = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']
plt.figure(figsize=(10,24))
for i in enumerate(cont_vars):
    plt.subplot(4,2,i[0]+1)
    sns.boxplot(y=heart_df[i[1]], data = heart_df)
plt.show()
plt.figure(figsize=(15,12))
sns.heatmap(data=heart_df.corr(),cmap="YlGnBu",annot=True)
plt.show()
# Logistic Regression Model Preparation
# Train and Test split
import sklearn 
from sklearn.model_selection import train_test_split
train,test =train_test_split(heart_df, random_state=100, test_size=0.3)

train.info()
# Creating (X_train, y_train) and (X_test, y_test)

y_train = train.pop('DEATH_EVENT')
X_train = train

y_test = test.pop('DEATH_EVENT')
X_test = test

X_train.head()
# Performing Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[cont_vars] = scaler.fit_transform(X_train[cont_vars])
X_train.head()
# Buliding the model
import statsmodels.api as sm
# Constant to X_train
X_train_sm = sm.add_constant(X_train)

# Building a model
lgr0 =sm.GLM(y_train,X_train_sm, families = sm.families.Binomial())
lgr0.fit().summary()
# checking VIF for the X_train data set

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_sm
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# high_blood_pressure feature has P value = 0.97 can be dropped
X_train_sm.drop('high_blood_pressure', axis=1,inplace =True)
X_train_sm.head()
# Building a model 1
lgr1 = sm.GLM(y_train,X_train_sm, families = sm.families.Binomial())
res1 = lgr1.fit().summary()
res1
vif = pd.DataFrame()
X = X_train_sm
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# platelets feature has P value = 0.793 can be dropped
X_train_sm.drop('platelets', axis=1,inplace =True)
X_train_sm.head()

# Building a model 2
lgr2 = sm.GLM(y_train,X_train_sm, families = sm.families.Binomial())
res2 = lgr2.fit().summary()
res2
# diabetes feature has P value = 0.794 can be dropped
X_train_sm.drop('diabetes', axis=1,inplace =True)
X_train_sm.head()

# Building a model 3
lgr3 = sm.GLM(y_train,X_train_sm, families = sm.families.Binomial())
res3 = lgr3.fit().summary()
res3
# anaemia feature has P value = 0.784 can be dropped
X_train_sm.drop('anaemia', axis=1,inplace =True)
X_train_sm.head()

# Building a model 4
lgr4 = sm.GLM(y_train,X_train_sm, families = sm.families.Binomial())
res4 = lgr4.fit().summary()
res4
# smoking feature has P value = 0.614 can be dropped
X_train_sm.drop('smoking', axis=1,inplace =True)
X_train_sm.head()

# Building a model 5
lgr5 = sm.GLM(y_train,X_train_sm, families = sm.families.Binomial())
res5 = lgr5.fit().summary()
res5
# serum_sodium feature has P value = 0.49 can be dropped
X_train_sm.drop('serum_sodium', axis=1,inplace =True)
X_train_sm.head()

# Building a model 6
lgr6 = sm.GLM(y_train,X_train_sm, families = sm.families.Binomial())
res6 = lgr6.fit().summary()
res6
# creatinine_phosphokinase feature has P value = 0.18 can be dropped
X_train_sm.drop('creatinine_phosphokinase', axis=1,inplace =True)
X_train_sm.head()

# Building a model 7
lgr7 = sm.GLM(y_train,X_train_sm, families = sm.families.Binomial())
res7 = lgr7.fit().summary()
res7
# sex feature has P value = 0.172 can be dropped
X_train_sm.drop('sex', axis=1,inplace =True)
X_train_sm.head()

# Building a model 8
lgr8 = sm.GLM(y_train,X_train_sm, families = sm.families.Binomial())
res8 = lgr8.fit()
res8.summary()
vif = pd.DataFrame()
X = X_train_sm
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_train_pred = res8.predict(X_train_sm)
train_pred = pd.DataFrame(y_train_pred)
train_pred.columns = ["train_prob"]

train_pred['DEATH_EVENT']= y_train.values.reshape(-1)

train_pred.head()
# columns with different probability cutoffs 
numbers = [x for x in range(100)]
for i in numbers:
    k=train_pred.train_prob*100
    train_pred[i]= k.map(lambda x: 1 if x > i else 0)
train_pred.head()
# calculate accuracy sensitivity and specificity for various Score (probability*100) cutoffs.
# creating a cutoff dataframe 
cutoff_df = pd.DataFrame( columns = ['Score','accuracy','sensi','speci','Precision','F1_score'])

from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Actual/Predicted     N       P
        # N          TN        FP
        # P          FN        TP

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [x for x in range(100)]
for i in num:
    cm = metrics.confusion_matrix(train_pred.DEATH_EVENT, train_pred[i] )
    total=sum(sum(cm))
    accuracy = (cm[0,0]+cm[1,1])/total
    
    speci = cm[0,0]/(cm[0,0]+cm[0,1])
    sensi = cm[1,1]/(cm[1,0]+cm[1,1])
    Precision = cm[1,1]/(cm[1,1]+cm[0,1])
    F1_score = 2*Precision*  sensi/(Precision+sensi)
    Score = i
    cutoff_df.loc[i] =[Score,accuracy,sensi,speci,Precision,F1_score]
cutoff_df.iloc[20:45]
#  Plot accuracy, sensitivity and specificity for various probabilities
plt.figure(figsize=(12,8))
cutoff_df.plot.line(x='Score', y=['accuracy','sensi','speci'])
plt.show()

plt.figure(figsize=(12,8))
cutoff_df.plot.line(x='Score', y=['Precision','F1_score'])
plt.show()
# with cutoff at 39 , sensivity: 83 Accuracy : 0.81 , specificity : 0.81 and Precision 0.66, F1 = 73
# Model evaluation using test dataset

col = list(X_train_sm.columns)

# Performing Feature Scaling for Test 
X_test[cont_vars] = scaler.transform(X_test[cont_vars])
X_test.head()
# adding constant to X_test
X_test_sm = sm.add_constant(X_test)

# Model Prediction res8
test_prob = res8.predict(X_test_sm[col])

test_prob
final_df = pd.DataFrame(test_prob)
final_df.columns =['test_prob']
final_df['predicted_test'] = final_df['test_prob'].apply(lambda x: 1 if x> 0.39 else 0)
final_df['DEATH_EVENT']= y_test.values.reshape(-1)
final_df.head()
# Confusion matrix for test
ct = metrics.confusion_matrix(final_df['DEATH_EVENT'],final_df['predicted_test'])
TP = ct[1,1] # true positive 
TN = ct[0,0] # true negatives
FP = ct[0,1] # false positives
FN = ct[1,0] # false negative

sen = round(TP/(TP+FN),2)
spec = round(TN/(TN+FP),2)
accu = round((TP+TN)/sum(sum(ct)),2)
pre = round(TP/(TP+FP),2)
F1 = round(2*pre*sen/(pre+sen),2)
print('Model Evaluation Parameters: \n')
print('Sensitivity_Test: ',sen)
print('Specificity_Test: ',spec)
print('Accuracy_Test: ',accu)
print('Precision_Test: ',pre)
print('F1_Score: ',F1)
