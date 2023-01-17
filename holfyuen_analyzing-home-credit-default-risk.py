# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
print(os.listdir("../input"))
# Load data
# submit = pd.read_csv('../input/sample_submission.csv')
cash = pd.read_csv('../input/POS_CASH_balance.csv')
bureau_bal = pd.read_csv('../input/bureau_balance.csv')
card = pd.read_csv('../input/credit_card_balance.csv')
bureau = pd.read_csv('../input/bureau.csv')
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
previous = pd.read_csv('../input/previous_application.csv')
installment = pd.read_csv('../input/installments_payments.csv')
print ('Done!')
print (cash.shape, bureau_bal.shape, card.shape, bureau.shape, train.shape, test.shape, previous.shape, installment.shape)
train.head()
mean_def = train['TARGET'].mean()
print ("%.2f%% of the loans in the training data have repayment difficulties." %(mean_def*100))
na = train.isnull().sum()/len(train)
na.sort_values(ascending=False).head(5)
plt.figure(figsize=(6,4))

def plot_bar(var):
    train.groupby([var]).mean()['TARGET'].sort_values().plot.barh(color='blue', alpha=0.7)
    plt.axvline(mean_def, color='orange', linewidth=4)
        
plot_bar('CODE_GENDER')
plt.figure(figsize=(7,5))
plot_bar('OCCUPATION_TYPE')
plt.figure(figsize=(7,5))
plot_bar('NAME_EDUCATION_TYPE')
def plot_kde(var, annualize=False):
    sns.kdeplot(train.loc[train['TARGET'] == 0, var] / (-365 if annualize else 1), label = 'No repayment difficulties')
    sns.kdeplot(train.loc[train['TARGET'] == 1, var] / (-365 if annualize else 1), label = 'With repayment difficulties')

plot_kde('DAYS_BIRTH', annualize=True)
plt.figure(figsize=(13,10))
plt.subplot(221)
plot_kde('AMT_INCOME_TOTAL')
plt.title('Income')
plt.subplot(222)
plot_kde('AMT_CREDIT')
plt.title('Credit')
plt.subplot(223)
plot_kde('AMT_ANNUITY')
plt.title('Annuity')
plt.subplot(224)
plot_kde('AMT_GOODS_PRICE')
plt.title('Price of Goods')
plt.show()
train.loc[:,['SK_ID_CURR','TARGET','AMT_INCOME_TOTAL']].sort_values('AMT_INCOME_TOTAL', ascending=False).head(10)
train['AMT_INCOME_TOTAL'].replace({117000000: train['AMT_INCOME_TOTAL'].median()}, inplace = True)
train.loc[train.TARGET==1, ['SK_ID_CURR','TARGET','AMT_INCOME_TOTAL']].sort_values('AMT_INCOME_TOTAL').head()
plt.figure(figsize=(13,10))
plt.subplot(221)
plot_kde('DAYS_BIRTH', annualize = True)
plt.title('Birth')
plt.subplot(222)
plot_kde('DAYS_EMPLOYED', annualize = True)
plt.title('Employed')
plt.subplot(223)
plot_kde('DAYS_REGISTRATION', annualize = True)
plt.title('Registration')
plt.subplot(224)
plot_kde('DAYS_ID_PUBLISH', annualize = True)
plt.title('ID Publish')
plt.show()
train['DAYS_EMPLOYED'].value_counts().head()
train['DAYS_EMPLOYED'].replace({365243: 0}, inplace = True)
test['DAYS_EMPLOYED'].replace({365243: 0}, inplace = True)
plot_kde('DAYS_EMPLOYED', annualize = True)
bureau.head()
bureau_bal.head()
bureau_bal.STATUS.value_counts()
cash.head()
card.head()
previous.head()
installment.head()
house_train = train.loc[:,'APARTMENTS_AVG':'EMERGENCYSTATE_MODE']
house_test = test.loc[:,'APARTMENTS_AVG':'EMERGENCYSTATE_MODE']
house_var = house_train.columns.tolist()
house_train = pd.get_dummies(house_train)
house_test = pd.get_dummies(house_test)
house_train.fillna(0, inplace=True)
house_test.fillna(0, inplace=True)
house_train.head()
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
house_train2 = pca.fit_transform(house_train)
house_test2 = pca.transform(house_test)
house_train2.shape, house_test2.shape
house_train2 = pd.DataFrame(house_train2, columns=['house_pc1','house_pc2','house_pc3'])
house_test2 = pd.DataFrame(house_test, columns=['house_pc1','house_pc2','house_pc3'])
# Merge back to trian and test datasets
train = pd.concat([train.drop(columns=house_var), house_train2], axis=1)
test = pd.concat([test.drop(columns=house_var), house_test2], axis=1)
train.shape, test.shape
# Add a few domain variables
train['CREDIT_INCOME_PERCENT'] = train['AMT_CREDIT']/train['AMT_INCOME_TOTAL']
train['ANNUITY_INCOME_PERCENT'] = train['AMT_ANNUITY']/train['AMT_INCOME_TOTAL']
train['DAYS_EMPLOYED_PERCENT'] = train['DAYS_EMPLOYED']/train['DAYS_BIRTH']
train['INCOME_PER_PERSON'] = train['AMT_INCOME_TOTAL'] / train['CNT_FAM_MEMBERS']
train['PAYMENT_RATE'] = train['AMT_ANNUITY'] / train['AMT_CREDIT']
test['CREDIT_INCOME_PERCENT'] = test['AMT_CREDIT']/test['AMT_INCOME_TOTAL']
test['ANNUITY_INCOME_PERCENT'] = test['AMT_ANNUITY']/test['AMT_INCOME_TOTAL']
test['DAYS_EMPLOYED_PERCENT'] = test['DAYS_EMPLOYED']/test['DAYS_BIRTH']
test['INCOME_PER_PERSON'] = test['AMT_INCOME_TOTAL'] / test['CNT_FAM_MEMBERS']
test['PAYMENT_RATE'] = test['AMT_ANNUITY'] / test['AMT_CREDIT']
# Features to be log-transformed
log_features = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION']
# For days, they need to be changed back to positive. For the sake of consistency, all days columns are transformed
days_features = ['DAYS_BIRTH', 'DAYS_EMPLOYED','DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']
train[days_features]=-train[days_features]
test[days_features]=-test[days_features]
train[log_features]=np.log1p(train[log_features])
test[log_features]=np.log1p(test[log_features])
train['na_col'] = train.isnull().sum(axis=1)
test['na_col'] = test.isnull().sum(axis=1)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
train_labels = train['TARGET']
train, test = train.align(test, join = 'inner', axis = 1)
train['TARGET'] = train_labels
# Create features from bureau 
safe = bureau.loc[(bureau.CREDIT_DAY_OVERDUE==0)&(bureau.CNT_CREDIT_PROLONG==0),['SK_ID_CURR','SK_ID_BUREAU']]
safe = safe.groupby('SK_ID_CURR').count().reset_index()
safe.columns = ['SK_ID_CURR','past_loan_np']
prob = bureau[['SK_ID_CURR','CREDIT_DAY_OVERDUE','CNT_CREDIT_PROLONG']].groupby('SK_ID_CURR').max()
prob.columns=['overdue_max','prolong_max']
prob=prob.reset_index()
loan_type=pd.get_dummies(bureau.CREDIT_TYPE)
loan_type['SK_ID_CURR'] = bureau['SK_ID_CURR']
loan_type = loan_type.groupby('SK_ID_CURR').mean().reset_index()
# Create features from bureau balance data; They need to be merged with bureau data first
_ = bureau_bal.drop(columns=['STATUS']).groupby('SK_ID_BUREAU').max().reset_index()
final_status = _.merge(bureau_bal, on=['SK_ID_BUREAU','MONTHS_BALANCE'], how='left')
final_status = pd.get_dummies(final_status).drop(columns='MONTHS_BALANCE')
final_status2 = bureau[['SK_ID_CURR','SK_ID_BUREAU']].merge(final_status, on='SK_ID_BUREAU', how='left')
final_status2.STATUS_X.fillna(1, inplace=True) # Mark all cases without final status as unknown
final_status2.fillna(0, inplace=True)
final_status2 = final_status2.drop(columns='SK_ID_BUREAU').groupby('SK_ID_CURR').mean().reset_index()
# Keep only status 1-5 and X
final_status2 = final_status2.drop(columns=['STATUS_0','STATUS_C'])
# Free up memory
del bureau, bureau_bal
# Create features from cash and card balance data;
cash_dpd = cash.loc[:,['SK_ID_CURR','SK_DPD_DEF']].groupby('SK_ID_CURR').agg(['max','median']).reset_index()
cash_dpd.columns = ['SK_ID_CURR','Cash_SK_DPD_DEF_max', 'Cash_SK_DPD_DEF_median']
card_dpd = card.loc[:,['SK_ID_CURR','SK_DPD_DEF']].groupby('SK_ID_CURR').agg(['max','median']).reset_index()
card_dpd.columns = ['SK_ID_CURR','Card_SK_DPD_DEF_max', 'Card_SK_DPD_DEF_median']
del cash, card
# Create features from previous loans data
p_good = previous.loc[(previous.NAME_CONTRACT_STATUS=='Approved') | (previous.NAME_CONTRACT_STATUS=='Unused offer'), ['SK_ID_CURR','AMT_CREDIT']]
p_bad = previous.loc[(previous.NAME_CONTRACT_STATUS=='Canceled') | (previous.NAME_CONTRACT_STATUS=='Refused'), ['SK_ID_CURR','AMT_CREDIT']]
p_good = p_good.dropna()
p_good2 = p_good.groupby('SK_ID_CURR').sum().reset_index()
p_good2.columns = ['SK_ID_CURR','good_credit']
p_bad2 = p_bad.groupby('SK_ID_CURR').sum().reset_index()
p_bad2.columns = ['SK_ID_CURR','bad_credit']
installment['diff'] = installment.AMT_PAYMENT - installment.AMT_INSTALMENT
ins_comp = installment.dropna(subset=['AMT_PAYMENT']).loc[:,['SK_ID_CURR','AMT_INSTALMENT','AMT_PAYMENT','diff']].groupby('SK_ID_CURR').sum().reset_index()
ins_comp['repay']=ins_comp.AMT_PAYMENT / ins_comp.AMT_INSTALMENT
ins_comp = ins_comp.loc[ins_comp.AMT_INSTALMENT!=0,:]
del previous, installment
# Function to merge dataframes
def merge_data(train, test, dfs):
    for df in dfs:
        train = train.merge(df, on='SK_ID_CURR', how='left')
        test = test.merge(df, on='SK_ID_CURR', how='left')
    return train, test
train, test = merge_data(train, test, [safe, prob, loan_type, final_status2, cash_dpd, card_dpd, p_good2, p_bad2, ins_comp])
train.shape, test.shape
train.repay.fillna(1, inplace=True)
test.repay.fillna(1, inplace=True)
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
# Remove collinear variables
def rm_collinear(train, test):
    threshold = 0.9
    corr_matrix = train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold) and column != 'SK_ID_CURR']
    train = train.drop(columns = to_drop)
    test = test.drop(columns = to_drop)
    return train, test, to_drop

train, test, collin_feat = rm_collinear(train, test)
collin_feat
train.shape, test.shape
# Test of models
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from time import time
# Min-max scaling and imputing
x_train = train.drop(columns = ['TARGET','SK_ID_CURR'])
x_test = test.drop(columns=['SK_ID_CURR'])

features = list(x_train.columns)
scaler = MinMaxScaler(feature_range = (0, 1))

# Min-max scale
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# Defining candidate models
gb = GradientBoostingClassifier(n_estimators=30, learning_rate=0.1, min_samples_split=4, random_state=10, verbose=1)
ab = AdaBoostClassifier(n_estimators=300, learning_rate=1, random_state=123)
rf0 = RandomForestClassifier(n_estimators=49, max_depth = 5, random_state=100)
lr = LogisticRegression(C=1, random_state=100)
nn = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, learning_rate_init=0.01, max_iter=100, batch_size=4082)
dt = DecisionTreeClassifier(max_features=100)
lgb = LGBMClassifier(n_estimators=2000, learning_rate=0.02, objective='binary', reg_lambda=0.1, num_leaves=34)
x_train_x, x_train_v, y_train_x, y_train_v = train_test_split(x_train, train_labels, test_size=0.1, random_state=168)

def run_model_try(ml):
    start = time()
    model = ml
    model.fit(x_train_x, y_train_x)
    train_pred = model.predict_proba(x_train_x)[:,1]
    v_pred = model.predict_proba(x_train_v)[:,1]
    train_auc = roc_auc_score(y_train_x, train_pred)
    v_auc = roc_auc_score(y_train_v, v_pred)
    train_acc = accuracy_score(y_train_x, model.predict(x_train_x))
    v_acc = accuracy_score(y_train_v, model.predict(x_train_v))
    end = time()
    print ("Training and validation auc: %.4f, %.4f" %(train_auc, v_auc))
    print ("Training and validation accuracy: %.4f, %.4f" %(train_acc, v_acc))
    print ("Time used: %.2f" %(end-start))
    return model

def run_model_real(ml):
    model = ml
    model.fit(x_train, train_labels)
    return model
# model = run_model_try(lgb)
# The two lines below are for running models with all training data to submit for competition
model = run_model_real(lgb)
test_pred = model.predict_proba(x_test)[:,1]
plt.figure(figsize=(8,8))
# feats = np.array(features)[cols].tolist()
feats = np.array(features).tolist()
fi = pd.DataFrame()
fi['feature'] = feats
fi['importance'] = model.feature_importances_
sns.barplot(x="importance", y="feature", data=fi.sort_values(by="importance", ascending=False).head(50));
submit = test[['SK_ID_CURR']]
submit['TARGET'] = test_pred
submit.to_csv('lgb0825.csv', index = False)