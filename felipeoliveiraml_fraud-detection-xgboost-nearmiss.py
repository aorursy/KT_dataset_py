# packages
!pip install shap 
!pip install yellowbrick
!pip install imblearn
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from tqdm import tqdm_notebook

from sklearn.metrics import classification_report, recall_score, precision_score ,average_precision_score, plot_precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from yellowbrick.classifier import PrecisionRecallCurve, ConfusionMatrix
from sklearn.model_selection import train_test_split, cross_validate ,KFold, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
import shap
shap.initjs()

%matplotlib inline 
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
# import data
path = '../input/creditcardfraud/creditcard.csv' 
credit = pd.read_csv(path)
credit.head()
print('Rows: {} | Columns: {} '.format(credit.shape[0], credit.shape[1]))
# Descritive stats 
credit.describe()
# class
sns.countplot(x=credit['Class'], palette='coolwarm')
print('Normal: {} |  Fraud: {}'.format(credit[credit['Class']==0].shape[0] , credit[credit['Class']==1].shape[0]))
def missing_values(data):

    """ Summary of null data
        contained in the dataset """

    # total nulls     
    missing = data.isnull().sum()
    total = missing.sort_values(ascending=True)
    
    # percentage  
    percent = (missing / len(data.index ) * 100).round(2).sort_values(ascending=True)

    # concatenation 
    table_missing = pd.concat([total, percent], axis=1, keys=['NA numbers', 'NA percentage'])

    return table_missing
missing_values(credit)
# data types
credit.dtypes
# Compare Distributions

plt.figure(figsize=(16,12))

plt.subplot(2,1,1)
plt.title('Normal Transaction Seconds')
sns.distplot(credit[credit['Class']==0]['Time'], color='purple')

print('\n')
print('\n')

plt.subplot(2,1,2)
plt.title('Seconds Fraudulent Transaction')
sns.distplot(credit[credit['Class']==1]['Time'], color='red')
# Repeated fraud?

fraud = credit[credit['Class']==1].loc[credit.duplicated()]
print('Repeated scams: {} '.format(len(fraud)))
print('\n')
fraud
# cmap
cmap = sns.diverging_palette(120, 40, sep=20, as_cmap=True, center='dark')
# Correlation
corr = credit.corr(method='pearson')

fig, ax = plt.subplots(figsize=(23,15))

# cmap=Greys

plt.title('Correlation matrix', fontsize=16)
print('\n')
correlacao = sns.heatmap(corr, annot=True, cmap='Blues', ax=ax, lw=3.3, linecolor='lightgray')
correlacao
# cols plot  
cols_names = credit.drop(['Class', 'Amount', 'Time'], axis=1)
idx = 0

# Spliting classes
fraud = credit[credit['Class']==1]
normal = credit[credit['Class']==0]

# figure plot  
fig, ax = plt.subplots(nrows=7, ncols=4, figsize=(18,18))
fig.subplots_adjust(hspace=1, wspace=1)

for col in cols_names:
    idx += 1
    plt.subplot(7, 4, idx)
    sns.kdeplot(fraud[col], label="Normal", color='blue', shade=True)
    sns.kdeplot(normal[col], label="Fraud", color='orange', shade=True)
    plt.title(col, fontsize=11)
    plt.tight_layout()
# Range of fraud values 
credit[credit['Class']==1]['Amount'].value_counts
# What is the average value of the fraudulent transaction?
print('Average Fraud: {} | Average Normal: {}'.format(credit[credit['Class']==1]['Amount'].mean() , credit[credit['Class']==0]['Amount'].mean()))
# What is the highest fraud value?
print('Higher fraud value: {}  | Higher normal value: {}'.format(credit[credit['Class']==1]['Amount'].max(), credit[credit['Class']==0]['Amount'].max()))
# Transaction Amount

plt.figure(figsize=(11,7))
plt.title('Value of Transactions by Class (Normal | Fraud)', fontsize=15)
sns.barplot(x='Class', y='Amount', data=credit, palette='GnBu')
# Distribution of Transaction amounts
plt.figure(figsize=(10,5))
plt.title('Transactions', fontsize=14)
plt.grid(False)
sns.kdeplot(credit['Amount'], color='lightblue', shade=True)
# Separating feature | class

X = credit.drop('Class', axis=1)
y = credit['Class']

# Train|Test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)


# StandardScaler 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Encoder 
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
# Baseline model

baseline = LogisticRegression(random_state=42)
baseline.fit(X_train, y_train)
y_baseline = baseline.predict(X_test)

# Probabilities 
y_proba_baseline = baseline.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_baseline))
print('\n')
print('AUC: {}%'.format(roc_auc_score(y_test, y_proba_baseline)))
print('Precision-Recall: {}'.format(average_precision_score(y_test, y_proba_baseline)))
X = credit.drop('Class', axis=1)
y = credit['Class']

# Validation 
KFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


fold = 0
for train_index, test_index in KFold.split(X,y):
      fold += 1 
      print('Fold: ', fold)
      print('Train: ',train_index.shape[0])
      print('Test: ', test_index[0])

      # Split
      X = credit.drop('Class', axis=1)
      y = credit['Class']

      # OverSampling SMOTE 
      smote = SMOTE(random_state=42)
      X, y = smote.fit_sample(X, y)
      print('Normal: {}  |  Fraud: {}'.format(np.bincount(y)[0], np.bincount(y)[1]))

      # spliting data  
      X_train, X_test = X.loc[train_index], X.loc[test_index]
      y_train, y_test = y[train_index], y[test_index] 

      
      # pre-processing 
      scaler = QuantileTransformer(random_state=42)
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)

      # build model 
      forest = RandomForestClassifier(n_estimators=200, max_depth=13, min_samples_split=9,
                                    random_state=42)
      forest.fit(X_train, y_train)
      y_pred_forest = forest.predict(X_test)
      y_proba_forest = forest.predict_proba(X_test)[:,1]


      # metrics 
      print('\n')
      print(classification_report(y_test, y_pred_forest))
      print('--------------'*5)
      print('\n')
      auc_forest = roc_auc_score(y_test, y_proba_forest)
      precision_forest = precision_score(y_test, y_pred_forest)
      recall_forest = recall_score(y_test, y_pred_forest)
      auprc_forest = average_precision_score(y_test, y_proba_forest)
# Random Forest Validation  
print('Random Forest')
print('\n')

print('AUC: ', np.mean(auc_forest))
print('Precision: ', np.mean(precision_forest))
print('Recall: ', np.mean(recall_forest))
print('Precision-Recall: ', np.mean(auprc_forest))


print('\n')
print('\n')

# Curva ROC random forest 
auc_forest = np.mean(auc_forest)
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_test, y_proba_forest)

# plot 
plt.figure(figsize=(12,7))
plt.plot(fpr_forest, tpr_forest, color='blue', label='AUC: {}'.format(auc_forest))
plt.fill_between(fpr_forest, tpr_forest, color='skyblue', alpha=0.3)
plt.plot([0,1], [0,1], color='black', ls='--', label='Reference line')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Random Forest', fontsize=16)
plt.legend(loc=4, fontsize=14)
plt.grid(False)
plt.show()
# Precision-Recall Random Forest 

plt.figure(figsize=(10,5))
plt.title('Precision-Recall Random Forest')
viz = PrecisionRecallCurve(forest)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
# SVM 
X = credit.drop('Class', axis=1)
y = credit['Class']

# UnderSampling  
under = NearMiss()
X, y = under.fit_sample(X, y)


# Validation
KFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


fold = 0
for train_index, test_index in KFold.split(X,y):
      fold += 1 
      print('Fold: ', fold)
      print('Train: ', train_index.shape[0])
      print('Test: ', test_index[0])

      # Unbalanced class
      print('Normal: {}  |  Fraud {}'.format(np.bincount(y)[0], np.bincount(y)[1]))

      # spliting data  
      X_train, X_test = X.loc[train_index], X.loc[test_index]
      y_train, y_test = y[train_index], y[test_index] 

      
      # pre-processing  
      scaler = RobustScaler()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)

      # build model 
      svm = SVC(C=1.0, gamma=0.5, random_state=42, probability=True)
      svm.fit(X_train, y_train)
      y_pred_svm = svm.predict(X_test)
      y_proba_svm = svm.predict_proba(X_test)[:,1]


      # metrics 
      print('\n')
      print(classification_report(y_test, y_pred_svm))
      print('-------------'*5)
      print('\n')
      auc_svm = roc_auc_score(y_test, y_proba_svm)
      precision_svm = precision_score(y_test, y_pred_svm)
      recall_svm = recall_score(y_test, y_pred_svm)
      auprc_svm = average_precision_score(y_test, y_proba_svm)
# SVM Validation 
print('SVM')
print('\n')

print('AUC: ', np.mean(auc_svm))
print('Precision: ', np.mean(precision_svm))
print('Recall: ', np.mean(recall_svm))
print('Precision-Recall: ', np.mean(auprc_svm))


print('\n')
print('\n')

# Curva ROC random forest 
auc_svm = np.mean(auc_svm)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_proba_svm)

# plot 
plt.figure(figsize=(12,7))
plt.plot(fpr_svm, tpr_svm, color='blue', label='AUC: {}'.format(auc_svm))
plt.fill_between(fpr_svm, tpr_svm, color='skyblue', alpha=0.3)
plt.plot([0,1], [0,1], color='black', ls='--', label='Reference line')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC SVM', fontsize=16)
plt.legend(loc=4, fontsize=14)
plt.grid(False)
plt.show()
# Precision-Recall SVM 

plt.figure(figsize=(10,5))
plt.title('Precision-Recall SVM', fontsize=16)
viz = PrecisionRecallCurve(svm)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
X = credit.drop('Class', axis=1)
y = credit['Class']


# Validation
KFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
precision_xgboost = []
recall_xgboost = []
auc_xgboost = []
precision_recall_xgboost = []


fold = 0
for train_index, test_index in KFold.split(X,y):
      fold += 1 
      print('Fold: ', fold)
      print('Train: ',train_index.shape[0])
      print('Test: ', test_index[0])

      # OverSampling SMOTE 
      smt = SMOTE(random_state=42)
      X, y = smt.fit_sample(X, y)
      print('Normal: {}  |  Fraud: {}'.format(np.bincount(y)[0], np.bincount(y)[1]))

      # spliting data  
      X_train, X_test = X.loc[train_index], X.loc[test_index]
      y_train, y_test = y[train_index], y[test_index] 

      
      # pre-processing  
      scaler = QuantileTransformer()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)

      # XGboost 
      xgb = XGBClassifier(n_estimators=300, max_delta_step=1 ,eval_metric='aucpr', 
                          cpu_history='gpu', random_state=42)
      xgb.fit(X_train, y_train)
      y_pred = xgb.predict(X_test)
  

      # metrics 
      precision_recall_xgboost = average_precision_score(y_test, y_pred)
      precision_xgboost = precision_score(y_test, y_pred)
      recall_xgboost = recall_score(y_test, y_pred)
      auc_xgboost  = roc_auc_score(y_test, y_pred)
      print('Precision-Recall: ', average_precision_score(y_test, y_pred))
      print('\n')
      print('\n')



# Final Validation 
print('Precision-Recall: ', np.mean(precision_recall_xgboost))
print('Recall: ', np.mean(recall_xgboost))
print('Precision: ', np.mean(precision_xgboost))
print('AUC: ', np.mean(auc_xgboost))
# Validation XGboost + SMOTE
print('XGboost')
print('\n')

print('AUC: ', np.mean(auc_xgboost))
print('Precision: ', np.mean(precision_xgboost))
print('Recall: ', np.mean(recall_xgboost))
print('Precision-Recall: ', np.mean(precision_recall_xgboost))


print('\n')
print('\n')

# Curva ROC random forest 
roc_auc_xgboost = np.mean(auc_xgboost)
fpr_xgboost, tpr_xgboost, thresholds_xgboost = roc_curve(y_test, y_pred)

# plot 
plt.figure(figsize=(12,7))
plt.plot(fpr_xgboost, tpr_xgboost, color='blue', label='AUC: {}'.format(roc_auc_xgboost))
plt.fill_between(fpr_xgboost, tpr_xgboost, color='skyblue', alpha=0.3)
plt.plot([0,1], [0,1], color='black', ls='--', label='Reference line')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC XGboost', fontsize=16)
plt.legend(loc=4, fontsize=14)
plt.grid(False)
plt.show()
# Precision-Recall XGboost + SMOTE 

plt.figure(figsize=(10,5))
plt.title('Precision-Recall XGboost + SMOTE')
viz = PrecisionRecallCurve(xgb)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
X = credit.drop('Class', axis=1)
y = credit['Class']

# UnderSampling NearMiss
under = NearMiss()
X,y = under.fit_sample(X, y)

# Validation
KFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

resultados = []
fold = 0
for train_index, test_index in KFold.split(X,y):
      fold += 1 
      print('Fold: ', fold)
      print('Train: ',train_index.shape[0])
      print('Test: ', test_index[0])

      # Unbalanced class 
      print('Normal: {} | Fraud: {}'.format(np.bincount(y)[0], np.bincount(y)[1]))


      # spliting data 
      X_train, X_test = X.loc[train_index], X.loc[test_index]
      y_train, y_test = y[train_index], y[test_index] 

      
      # pre-processing 
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)


      # Encoder 
      encoder = LabelEncoder()
      y_train = encoder.fit_transform(y_train)
      y_test = encoder.transform(y_test)


      # XGboost 
      xgb = XGBClassifier(n_estimators=300, max_delta_step=1 ,eval_metric='aucpr', 
                          cpu_history='gpu', random_state=42)
      xgb.fit(X_train, y_train)
      y_pred = xgb.predict(X_test)

    # Metrics
      precision_recall_xgboost = average_precision_score(y_test, y_pred)
      precision_xgboost = precision_score(y_test, y_pred)
      recall_xgboost = recall_score(y_test, y_pred)
      auc_xgboost  = roc_auc_score(y_test, y_pred)
      print('Precision-Recall: ', average_precision_score(y_test, y_pred))
      print('\n')
      print('\n')



# Final Validation   
print('Precision-Recall: ', np.mean(precision_recall_xgboost))
print('Recall: ', np.mean(recall_xgboost))
print('Precision: ', np.mean(precision_xgboost))
print('AUC: ', np.mean(auc_xgboost))
# Validation XGboost + NearMiss  
print('XGboost')
print('\n')

print('AUC: ', np.mean(auc_xgboost))
print('Precision: ', np.mean(precision_xgboost))
print('Recall: ', np.mean(recall_xgboost))
print('Precision-Recall: ', np.mean(precision_recall_xgboost))


print('\n')
print('\n')

# Curva ROC random forest 
roc_auc_xgboost = np.mean(auc_xgboost)
fpr_xgboost, tpr_xgboost, thresholds_xgboost = roc_curve(y_test, y_pred)

# plot 
plt.figure(figsize=(12,7))
plt.plot(fpr_xgboost, tpr_xgboost, color='blue', label='AUC: {}'.format(roc_auc_xgboost))
plt.fill_between(fpr_xgboost, tpr_xgboost, color='skyblue', alpha=0.3)
plt.plot([0,1], [0,1], color='black', ls='--', label='Reference line')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC XGboost', fontsize=16)
plt.legend(loc=4, fontsize=14)
plt.grid(False)
plt.show()
# Precision-Recall XGboost + NearMiss 

plt.figure(figsize=(10,5))
plt.title('Precision-Recall XGboost + NearMiss')
viz = PrecisionRecallCurve(xgb)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
# Confusion Matrix  


plt.figure(figsize=(15,10))

plt.subplot(2, 1, 1)
plt.title('Confusion Matrix XGboost', fontsize=15)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='BuPu')

print('\n')
print('\n')
# Features Importance 
from xgboost import plot_importance



fig, ax = plt.subplots(figsize=(14,8))
plot_importance(xgb, ax=ax)
plt.title('Feature importance | XGboost + NearMiss')
plt.show()