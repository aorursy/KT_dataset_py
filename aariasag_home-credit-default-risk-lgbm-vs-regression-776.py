%matplotlib inline  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

data_raw = pd.read_csv('../input/application_train.csv')
data_bureau = pd.read_csv('../input/bureau.csv')
data_credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
data_previous_application = pd.read_csv('../input/previous_application.csv')
data_installments_payments = pd.read_csv('../input/installments_payments.csv')
data_POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
# Group data by SK_ID_CURR to join all data in just one dataset
data_bureau_agg=data_bureau.groupby(by='SK_ID_CURR').mean()
data_credit_card_balance_agg=data_credit_card_balance.groupby(by='SK_ID_CURR').mean()
data_previous_application_agg=data_previous_application.groupby(by='SK_ID_CURR').mean()
data_installments_payments_agg=data_installments_payments.groupby(by='SK_ID_CURR').mean()
data_POS_CASH_balance_agg=data_POS_CASH_balance.groupby(by='SK_ID_CURR').mean()
data_raw = data_raw.join(data_bureau_agg, how='inner', on='SK_ID_CURR', lsuffix='1', rsuffix='2')                
data_raw = data_raw.join(data_credit_card_balance_agg, how='inner', on='SK_ID_CURR', lsuffix='1', rsuffix='2')    
data_raw = data_raw.join(data_previous_application_agg, how='inner', on='SK_ID_CURR', lsuffix='1', rsuffix='2')   
data_raw = data_raw.join(data_installments_payments_agg, how='inner', on='SK_ID_CURR', lsuffix='1', rsuffix='2') 

# keep sk_id_curr column and delete it from dataset
sk_id_curr = data_raw['SK_ID_CURR']
data_raw.drop('SK_ID_CURR',axis=1,inplace=True)
#Probability of scoring 0 is at least 91.51% 
#This means that our model must, at least, improve this percentage, otherwise 
#we don`t need a model and the bank will just assume that almost 9 of 100 will default.
100*(1 - data_raw[data_raw['TARGET']==1].TARGET.count()/data_raw.TARGET.count())
# Differentiate numerical features (minus the target) and categorical features
categorical_features = data_raw.select_dtypes(include = ["object"]).columns
numerical_features = data_raw.select_dtypes(exclude = ["object"]).columns

print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
data_raw_num = data_raw[numerical_features]
data_raw_cat = data_raw[categorical_features]
data_raw.replace({'NAME_CONTRACT_TYPE': {'Cash loans': 1, 'Revolving loans': 2}},  inplace = True)
data_raw.replace({'CODE_GENDER': {'M': 1, 'F': 2, 'XNA': 0}},  inplace = True)
data_raw.replace({'FLAG_OWN_CAR': {'Y': 1, 'N': 0}},  inplace = True)
data_raw.replace({'FLAG_OWN_REALTY': {'Y': 1, 'N': 0}},  inplace = True)
data_raw.replace({'NAME_INCOME_TYPE': {'Working': 0, 'State servant': 1, 'Commercial associate': 2,
                                           'Pensioner': 3, 'Unemployed': 4, 'Student': 5, 'Businessman': 6}},  inplace = True)
data_raw.replace({'NAME_EDUCATION_TYPE': {'Secondary / secondary special': 0, 'Higher education': 1, 'Incomplete higher': 2,
                                           'Lower secondary': 3, 'Academic degree': 4}},  inplace = True)
data_raw.replace({'NAME_FAMILY_STATUS': {'Single / not married': 0, 'Married': 1, 'Widow': 2,
                                           'Civil marriage': 3, 'Separated': 4}},  inplace = True)
data_raw.replace({'NAME_HOUSING_TYPE': {'House / apartment': 0, 'Rented apartment': 1, 'Municipal apartment': 2,
                                           'With parents': 3, 'Office apartment': 4, 'Co-op apartment': 5}},  inplace = True)

# convert rest of categorical variable into dummy
data_raw = pd.get_dummies(data_raw)
from sklearn.preprocessing import MinMaxScaler, Imputer
imputer = Imputer(strategy = 'median') # Median imputation of missing values
scaler = MinMaxScaler(feature_range = (0, 1)) # Scale each feature to 0-1
for column in data_raw_num.columns:
    data_raw[[column]] = imputer.fit_transform(data_raw[[column]])
    data_raw[[column]] = scaler.fit_transform(data_raw[[column]])
# Find correlations with the target and sort
# correlations = data_raw.corr()['TARGET'].sort_values()
# print('Most Positive Correlations: \n', correlations.tail(6))
# print('\nMost Negative Correlations: \n', correlations.head(5))
train=data_raw
# Partition the dataset in train + validation sets
X = train.iloc[:,1:]
y = train.iloc[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("Y_train : " + str(Y_train.shape))
print("Y_test : " + str(Y_test.shape))
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

Y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(Y_test, Y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(Y_test, Y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(Y_test, Y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
#Light GBM 

LGB_clf = LGBMClassifier(n_estimators=100, 
                         boosting_type='gbdt', 
                         objective='binary', 
                         metric='binary_logloss')
LGB_clf.fit(X_train, Y_train)
Y_pred_proba = LGB_clf.predict_proba(X_test)[:, 1]

roc_auc_score(Y_test, Y_pred_proba)