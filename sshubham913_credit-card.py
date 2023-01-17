import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")  # To ignore warnings

sns.set(rc={"figure.figsize":(12,8)})  # Set figure size to 12,8



pd.options.display.max_columns=150 # to display all columns 
# to run the code line by line

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"#run single line code
from sklearn.preprocessing import StandardScaler as std

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

import pickle




from sklearn.tree import DecisionTreeClassifier

from imblearn.pipeline import Pipeline  # To build pipeline 
!kaggle datasets download -d mlg-ulb/creditcardfraud
#!unzip "creditcardfraud.csv.zip"
data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df=data.copy()
df.shape,data.shape
df.head()
df.dtypes
# no null values

df.isnull().sum()
df.describe()
round(df.corr(),2)
#sns.pairplot(df)
# highly imbalance class



df.Class.value_counts()

((df.Class.value_counts())/df.shape[0])*100
sns.countplot('Class', data=df)

fraud = df[df["Class"]==1]

Non_fraud= df[df["Class"]==0]
# try for other columns

plt.figure(figsize=(10,5))

plt.subplot(121)

fraud.Amount.plot.hist(title="Fraud Transacation")

plt.subplot(122)

Non_fraud.Amount.plot.hist(title="Non Fraud Transaction")
df.Time.value_counts()
sns.distplot(df.Time)
df['time']=std().fit_transform(df['Time'].values.reshape(-1, 1))
sns.distplot(df.time)
df.Amount.value_counts()
sns.distplot(df.Amount)
df['amount']=std().fit_transform(df['Amount'].values.reshape(-1, 1))
#sns.distplot(df.amount)
#sns.distplot(df.V1)
df.columns
final=df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Class', 'amount','time']]
final.columns
# convert target variable into x,y

y=final['Class']

X=final.drop(['Class'],axis=1)
from sklearn.model_selection import train_test_split
X_train,  X_val, y_train,y_val = train_test_split(X, y, stratify=y,test_size = 0.30, random_state = 222)



X_train.shape, X_val.shape, y_train.shape, y_val.shape

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
LRC = LogisticRegression(class_weight="balanced")#solver='newton-cg',max_iter=500



LRC.fit(X_train, y_train)

y_pred_LRC = LRC.predict(X_val)



print(classification_report(y_val, y_pred_LRC))
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()#criterion = 'entropy', max_features = 'sqrt', max_depth = 15, random_state = 0



DTC.fit(X_train, y_train)

y_pred_DT = DTC.predict(X_val)



print(classification_report(y_val, y_pred_DT))
from sklearn.ensemble import RandomForestClassifier#rfc_65
rfc11 = RandomForestClassifier(class_weight="balanced")#n_estimators = 1500, class_weight="balanced"



rfc11.fit(X_train, y_train)

y_pred_test_RF1 = rfc11.predict(X_val)



print(classification_report(y_val, y_pred_test_RF1))
# rfc111 = RandomForestClassifier(n_estimators = 2500, class_weight="balanced")#



# rfc111.fit(X_train, y_train)

# y_pred_test_RF11 = rfc111.predict(X_val)



# print(classification_report(y_val, y_pred_test_RF11))
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import MultiLabelBinarizer

import xgboost as xgb

from xgboost.sklearn import XGBClassifier  

from sklearn.model_selection import GridSearchCV
xgb = XGBClassifier()

xgb.fit(X_train, y_train)
y_pred_xg = xgb.predict(X_val)



print(classification_report(y_val, y_pred_xg))
## explained single line way

from collections import Counter

from numpy import mean

from sklearn.datasets import make_classification

from imblearn.under_sampling import RandomUnderSampler
## Find Number of samples which are Fraud

frauds = len(df[df['Class'] == 1])## 492
## Get indices of non fraud samples

non_frauds_index = df[df.Class == 0].index ##250000+
## Random sample non fraud indices and it changes 

random_indices = np.random.choice(non_frauds_index,frauds, replace=False)
## Find the indices of fraud samples

fraud_index = df[df.Class == 1].index
## Concat fraud indices with sample non-fraud ones

under_sample_indices = np.concatenate([fraud_index,random_indices])

## Get Balance Dataframe

under_sample = final.loc[under_sample_indices]##492+492
under_sample.columns
# you can implement directly by using this code

# undersamp = RandomUnderSampler(return_indices=True)

# X_us, y_us, id_us = rus.fit_sample(X, y)



# print('Removed indexes:', id_us)

sns.countplot('Class', data=under_sample)
# convert target variable into x,y

y=under_sample['Class']

X=under_sample.drop(['Class'],axis=1)
from sklearn.model_selection import train_test_split
X_train,  X_val, y_train,y_val = train_test_split(X, y, stratify=y,test_size = 0.30, random_state = 22122)



X_train.shape, X_val.shape, y_train.shape, y_val.shape

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
LRC = LogisticRegression(solver='newton-cg',max_iter=500)



LRC.fit(X_train, y_train)

y_pred_LRC = LRC.predict(X_val)



print(classification_report(y_val, y_pred_LRC))
LRC1 = LogisticRegression(class_weight = 'balanced')



LRC1.fit(X_train, y_train)

y_pred_LRC1 = LRC1.predict(X_val)



print(classification_report(y_val, y_pred_LRC1))
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(criterion = 'entropy', max_features = 'sqrt', max_depth = 15, random_state = 0)



DTC.fit(X_train, y_train)

y_pred_DT = DTC.predict(X_val)



print(classification_report(y_val, y_pred_DT))
from sklearn.ensemble import RandomForestClassifier#rfc_65
rfc11 = RandomForestClassifier()#n_estimators = 1500, class_weight="balanced"



rfc11.fit(X_train, y_train)

y_pred_test_RF1 = rfc11.predict(X_val)



print(classification_report(y_val, y_pred_test_RF1))
rfc111 = RandomForestClassifier(n_estimators = 2500, class_weight="balanced")#



rfc111.fit(X_train, y_train)

y_pred_test_RF11 = rfc111.predict(X_val)



print(classification_report(y_val, y_pred_test_RF11))
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(learning_rate=0.1, n_estimators = 500, max_features="sqrt")

GBC.fit(X_train, y_train)



y_pred_GBC = GBC.predict(X_val)

print(classification_report(y_val, y_pred_GBC))
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import MultiLabelBinarizer

import xgboost as xgb

from xgboost.sklearn import XGBClassifier  

from sklearn.model_selection import GridSearchCV
xgb = XGBClassifier()

xgb.fit(X_train, y_train)
y_pred_xg = xgb.predict(X_val)



print(classification_report(y_val, y_pred_xg))
from sklearn.metrics import confusion_matrix



conf_mat = confusion_matrix(y_true=y_val, y_pred=y_pred_xg)

print('Confusion matrix:\n', conf_mat)



labels = ['Class 0', 'Class 1']

fig = plt.figure()

sns.heatmap(conf_mat,cmap="coolwarm_r",annot=True)

#ax.set_xticklabels([''] + labels)

#ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import cross_val_score
# convert target variable into x,y

y=final['Class']

X=final.drop(['Class'],axis=1)
from sklearn.model_selection import train_test_split
X_train,  X_val, y_train,y_val = train_test_split(X, y, stratify=y,test_size = 0.30, random_state = 222)



X_train.shape, X_val.shape, y_train.shape, y_val.shape

oversample = RandomOverSampler(sampling_strategy='minority')#not majority, all,not minority



X_over, y_over = oversample.fit_resample(X_train, y_train)



# summarize class distribution

print("Distribution of y_over variable:",Counter(y_over))
steps_log= [#('over', RandomOverSampler(sampling_strategy="auto")),

           ('model', LogisticRegression(random_state=12))]



pipeline_log = Pipeline(steps=steps_log)



print(pipeline_log)
pipeline_log.fit(X_over, y_over)
y_pred_pipe_log = pipeline_log.predict(X_val)



print(classification_report(y_val, y_pred_pipe_log))
# evaluate pipeline

scores_log = cross_val_score(pipeline_log, X_over, y_over,  cv=3, n_jobs=-1)  ## K-fold cross validation

print("f1 scores for each fold: ",np.round(scores_log,decimals=3))



## Get the mean score 

score = mean(scores_log)

print('Mean F1 Score: %.3f' % score)
#print(classification_report(y_train, y_pred))
steps_dt= [('over', RandomOverSampler(sampling_strategy="minority",random_state=121)),

           ('model', DecisionTreeClassifier(random_state=121))]



pipeline_dt = Pipeline(steps=steps_dt)



print(pipeline_dt)
pipeline_dt.fit(X_train, y_train)



y_pred_pipe_dt = pipeline_dt.predict(X_val)



print(classification_report(y_val, y_pred_pipe_dt))
cross_val_score(pipeline_dt, X_over, y_over, cv=3, n_jobs=-1)
from imblearn.over_sampling import SMOTE



smote = SMOTE(sampling_strategy=0.5,k_neighbors=10,random_state=12)



X_smote, y_smote = smote.fit_resample(X_train, y_train)



print("Distribution of y_smote variable:",Counter(y_smote))
steps_sm= [('smote', SMOTE(sampling_strategy="all",k_neighbors=10,random_state=12)),

           ('model', DecisionTreeClassifier(random_state=12))]



pipeline_sm = Pipeline(steps=steps_sm)



print(pipeline_sm)
pipeline_sm.fit(X_train, y_train)



y_pred_pipe_sm = pipeline_sm.predict(X_val)



print(classification_report(y_val, y_pred_pipe_sm))
scores = cross_val_score(pipeline_sm, X_smote, y_smote,  cv=3, n_jobs=-1)  ## K-fold cross validation

print("f1 scores for each fold: ",np.round(scores,decimals=3))



## Get the mean score 

score = mean(scores)

print('Mean F1 Score: %.3f' % score)