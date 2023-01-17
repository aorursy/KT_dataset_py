%matplotlib inline
import os 
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import datetime
import seaborn as sns
import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib
from xgboost import XGBClassifier


ROOT_PATH = '../'
accepted = pd.read_csv(ROOT_PATH + './input/accepted_2007_to_2017Q3.csv.gz', compression='gzip')
accepted.shape
missing_data = accepted.isnull().sum().sort_values(ascending= False)
drop_columns = list(missing_data[missing_data > accepted.shape[0] *0.1].index) # drop columns where at least 10% of data is missing
accepted = accepted.drop(drop_columns, axis = 1)

missing_data
accepted.term = accepted.term.apply(str)
accepted['term'] = accepted['term'].apply(lambda x: x.strip().split(" ")[0])

accepted.issue_d = pd.to_datetime(accepted.issue_d)
accepted['issue_yr'] = accepted.issue_d.dt.year
accepted['issue_yr'].plot.hist()
accepted = accepted.drop(['title', 'funded_amnt'], axis = 1)
accepted = accepted.drop(['out_prncp_inv','total_rec_prncp','total_pymnt_inv'], axis = 1)
accepted = accepted.drop(['fico_range_low', 'last_fico_range_low',
                         'avg_cur_bal',
                         'addr_state', 'initial_list_status', 'pymnt_plan',
                         'application_type', 'hardship_flag', 'disbursement_method',
                          'debt_settlement_flag','sub_grade',
                         'zip_code', 'id','policy_code','tax_liens', 'tax_liens'], axis = 1)

accepted.home_ownership = accepted.home_ownership.replace(['ANY', 'NONE','OTHER'], 'RENT')

accepted['issue_yr'] = accepted.issue_d.dt.year
accepted['earliest_cr_line'] = pd.to_datetime(accepted.earliest_cr_line)
accepted['early_cr_yr'] = accepted.earliest_cr_line.dt.year

median_year = accepted.emp_length.value_counts(ascending = False).index[0]
accepted.loc[:, 'emp_length'] = accepted.loc[:, 'emp_length'].fillna(median_year)

accepted.emp_length = accepted.emp_length.replace(['10+ years'], '10 years')
accepted.emp_length = accepted.emp_length.replace(['< 1 year'], '0 years')

accepted.emp_length = accepted.emp_length.apply(lambda x: int(str(x).split(' ')[0]))
print(accepted.emp_length.value_counts())

accepted.loc[:, 'emp_title'] = accepted.loc[:, 'emp_title'].fillna('other')
accepted.emp_title = accepted.emp_title.apply(lambda x: x.lower())
accepted.emp_title = accepted.emp_title.replace(['lpn','registered nurse', 'rn'], 'nurse')
rate = pd.pivot_table(accepted[accepted['term'] == '36'],index=["grade","issue_yr"],values=["int_rate"], aggfunc=np.mean)
rate.shape # 77, 1
rate = rate.reset_index()
g = sns.FacetGrid(rate, col = 'grade', col_wrap = 4)
g = g.map(sns.pointplot, "issue_yr", "int_rate")

labels = np.arange(2007, 2018, 1)
labels = [str(i) for i in labels]
g = g.set_xticklabels(labels, rotation=70)
g = g.set_ylabels("3yr interest rate")

plt.subplots_adjust(top=0.9)
g.fig.suptitle('Interest Rate over time and grade')
# Number of observations for each grade
# to verify the variance of rates
rate_count = pd.pivot_table(accepted[accepted['term'] == '36'],index=["grade","issue_yr"],values=["int_rate"], aggfunc='count')
rate_count = rate_count.unstack('grade')
rate_count
accepted.purpose.value_counts().sort_values(ascending=False)
incomeVerified = accepted[accepted['verification_status'] != 'Not Verified'].dropna()
incomeVerified = incomeVerified[['grade','annual_inc']]
quantile_low = incomeVerified['annual_inc'].min()
quantile_high = incomeVerified['annual_inc'].quantile(0.95)
filtered = incomeVerified[(incomeVerified['annual_inc'] > quantile_low) & (incomeVerified['annual_inc'] <= quantile_high)]
grade_list = filtered['grade'].unique()
plt.figure(figsize=(10,10))
for i in range(len(grade_list)):
    data = filtered[filtered['grade'] == grade_list[i]]['annual_inc'].values
    sns.distplot(data, bins = 30)

plt.ylim(ymax = 0.00002)
# Median of incomes in each grade
income_median = pd.pivot_table(filtered, values = 'annual_inc', index = 'grade', aggfunc = np.median)
income_median
leq1mil = accepted['annual_inc'] <= 1e6
accepted = accepted[leq1mil]
accepted = accepted[accepted.dti < 100.0]
pd.options.display.float_format = '{:,.0f}'.format
salary_limit = 7e4

emp_annual_all = accepted.loc[((accepted['annual_inc'] >= 1.2e5) & (accepted['verification_status'] == 'Verified')) 
                              | ((accepted['annual_inc'] >= salary_limit) & (accepted['annual_inc'] < 1.2e5)) 
                              | ((accepted['annual_inc'] < salary_limit) & (accepted['verification_status'] == 'Verified')),
                              ['emp_title','annual_inc']].groupby('emp_title')


summ_inc = emp_annual_all.agg(['min','mean','median','max', 'count'])
summ_inc.columns = summ_inc.columns.levels[1]
summ_inc = summ_inc.sort_values(by = ['count','min'], ascending = False)

# Filter for professions with more than 500 observations
summ_inc = summ_inc[summ_inc['count'] >= 500].sort_values(by = ['count','min'], ascending = False)
summ_inc
accepted.boxplot(by = 'grade', column = 'fico_range_high')
accepted['verified'] = accepted['verification_status'] == 'Verified'
grade_yr_loanamnt = pd.pivot_table(accepted,index=["grade","verified"], values=['loan_amnt'], aggfunc=np.sum)

grade_yr_loanamnt_default = pd.pivot_table(accepted[(accepted.loan_status == 'Charged Off') | (accepted.loan_status == 'Default')],
                                           index=["grade","verified"], values=['loan_amnt'], aggfunc=np.sum)

grade_yr_loanamnt_default.columns = ['Charged_off']

loan_verified = pd.merge(grade_yr_loanamnt, grade_yr_loanamnt_default, left_index = True, right_index = True)
loan_verified['chargeoff_rate']  = loan_verified['Charged_off'] /  loan_verified['loan_amnt'] 

loan_verified_unstack = loan_verified.unstack("verified")
verified_chargedoff = loan_verified_unstack['chargeoff_rate']
verified_chargedoff.plot()

accepted.loan_status.value_counts()
accepted = accepted.loc[accepted.loan_status != 'Current', :]
accepted['target'] = 1
accepted.loc[(accepted.loan_status == 'Fully Paid') | (accepted.loan_status == 'In Grace Period') 
             | (accepted.loan_status == 'Does not meet the credit policy. Status:Fully Paid')
                , 'target'] = 0
accepted.columns
accepted.isnull().sum().sort_values(ascending=False)
# I will be a bit lazy here 
accepted = accepted.dropna()
accepted.shape
# Assuming that if borrowing for debt consolidation, they actually have more disposable income
# because they reduce their debt servicing by half
accepted.loc[:, 'temp'] = np.where((accepted.purpose == 'debt_consolidation') | (accepted.purpose == 'credit_card'),0.5, -1.0)
accepted.loc[:, 'disposable_inc'] = (1 - accepted['dti']/100)*accepted['annual_inc']/12 + accepted['temp'] * accepted['installment']    
accepted.loc[:, 'disposable_ratio']= accepted['disposable_inc']*12/ accepted['annual_inc']

accepted.drop('temp', axis = 1)

accepted.loc[:, 'cr_yr_before_loan'] = accepted['issue_yr'] - accepted['early_cr_yr']

accepted['log_annual_inc'] = np.log(accepted.annual_inc)
accepted['log_installment'] = np.log(accepted.installment)

accepted['install_loan_ratio'] = accepted['installment'] / accepted['loan_amnt']
accepted['balance_annual_inc'] = accepted['loan_amnt'] / accepted['annual_inc']
accepted['install_annual'] = accepted['installment'] / accepted['annual_inc']
selected_features = ['loan_amnt', 'term', 'int_rate', 'installment'
            # , 'grade' - grade is dropped because it is highly correlated with int_rate
            ,'emp_length', 'home_ownership', 'annual_inc', 'verification_status'
            # ,'issue_d'
            # , 'purpose'
            ,'dti', 'delinq_2yrs'
            ,'fico_range_high', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal'
            ,'revol_util', 'total_acc'
            #, 'out_prncp', 'total_pymnt', 'total_rec_int','total_rec_late_fee', 'recoveries', 'collection_recovery_fee'
            #,'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d',
            #,'last_fico_range_high', 'collections_12_mths_ex_med', 'acc_now_delinq'
            #,'tot_coll_amt', 
            ,'tot_cur_bal', 'total_rev_hi_lim','acc_open_past_24mths', 'bc_open_to_buy', 'bc_util'
            ,'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct'
            ,'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl'
            ,'mort_acc', 'mths_since_recent_bc', 'num_accts_ever_120_pd'
            ,'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl'
            ,'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0'
            ,'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m'
            ,'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75'
            ,'pub_rec_bankruptcies', 'tot_hi_cred_lim'
            ,'total_bal_ex_mort'
            ,'total_bc_limit', 'total_il_high_credit_limit'
            #,'issue_yr'
            #,'earliest_cr_yr'
            ,'loan_status','target'
            ,'cr_yr_before_loan'
            ,'log_annual_inc', 'log_installment'
            ,'disposable_inc', 'install_loan_ratio', 'disposable_ratio', 'balance_annual_inc'
            ,'install_annual']
accepted_train_clean = accepted.loc[(accepted.issue_yr != 2017) | (accepted.issue_yr != 2016), :]
accepted2017_clean = accepted.loc[(accepted.issue_yr == 2017) | (accepted.issue_yr == 2016), :]
print("Training size", accepted_train_clean.shape)
print("Testing size", accepted2017_clean.shape)
## In my experiments, I actually output these 2 datasets so that 
## I can come back to them with ease instead of having to run the whole workbook again
# accepted_train_clean.to_csv(ROOT_PATH + './input_clean/train.csv', index = False)
# accepted2017_clean.to_csv(ROOT_PATH + './input_clean/test.csv', index = False)
# accepted_train_clean = pd.read_csv(ROOT_PATH + './input_clean/train.csv', encoding = "ISO-8859-1")
# accepted2017_clean = pd.read_csv(ROOT_PATH + './input_clean/test.csv', encoding = "ISO-8859-1")
def GetAUC(model, X_train, y_train, X_test, y_test):
    '''
    To quickly get the AUC of model on the training and testing set
    '''
    res = [0.0, 0.0]
    y_train_score = model.predict_proba(X_train)[:, 1]
    res[0] = metrics.roc_auc_score(y_train, y_train_score)
    print("In sample", res[0])
    
    y_test_score = model.predict_proba(X_test)[:, 1]
    res[1] = metrics.roc_auc_score(y_test, y_test_score)
    print("Out of sample", res[1])
    
def GetXY(df, features):
    '''
    Select the subset of features
    Create dummy variables if needed
    '''
    df = df.loc[:, features]

    categorical_features = ['term', 'home_ownership', 'verification_status']
    
    for cat_feature in categorical_features:
        if cat_feature in df.columns:
            df = pd.get_dummies(df, prefix = [cat_feature], columns = [cat_feature], drop_first = True)
            
    X = df.drop(['loan_status', 'target'], axis = 1)
    y = df.target
    
    return X, y
X, y = GetXY(accepted_train_clean, selected_features)
X_test, y_test = GetXY(accepted2017_clean, selected_features)

print(X.shape)
print(X_test.shape)

# The difference in shape is fixed later
# X has an extra column: home_ownership_OTHER
set(X.columns.values)-set(X_test.columns.values)
# X_test['home_ownership_OTHER'] = 0
gbm = GradientBoostingClassifier(max_depth = 6, n_estimators= 300, max_features = 0.3)
gbm.fit(X, y)
GetAUC(gbm, X, y, X_test, y_test)
rfc = RandomForestClassifier(max_depth = 6, n_estimators= 300, class_weight = {0: 1, 1:10})
rfc.fit(X, y)
GetAUC(rfc, X, y, X_test, y_test)
xgb = XGBClassifier(max_depth = 6, n_estimators= 200, class_weight = {0: 1, 1:5})
xgb.fit(X, y)
GetAUC(xgb, X, y, X_test, y_test)
# joblib.dump(gbm, ROOT_PATH + './model_run/gbm.pkl')
# joblib.dump(rfc, ROOT_PATH + './model_run/rfc.pkl')
pd.options.display.float_format = '{:,.2f}'.format
feature_imp = pd.DataFrame({'name': X.columns, 'imp': gbm.feature_importances_}).sort_values(by = 'imp', ascending = False)

feature_imp['mult_gbm'] = feature_imp.imp.max() / feature_imp['imp']
feature_imp['mult_rfc'] = rfc.feature_importances_.max() / rfc.feature_importances_
feature_imp['mult_xgb'] = xgb.feature_importances_.max() / xgb.feature_importances_

# feature_imp.to_csv(ROOT_PATH + './model_run/feature_importance.csv')
feature_imp
selected = feature_imp.loc[feature_imp.mult_gbm < 6, ['name','mult_gbm']]
X_reduced = X[selected.name.values]
X_test_reduced = X_test[selected.name.values]

X_reduced.shape
X_reduced.columns
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

class MasOrMenos(BaseEstimator):
    def __init__(self):
        self.NumFeatures = 0
        self.coeffs = None
        self.model = LogisticRegression()
        
    def fit(self, X_train, y_train):
        self.NumFeatures = X_train.shape[1]
        sqrtNumFeatures = np.sqrt(self.NumFeatures)
        self.coeffs = [0.0] * self.NumFeatures
        for i in range(self.NumFeatures):
            self.model.fit(X_train.iloc[:, i:(i+1)], y_train)
            self.coeffs[i] = np.sign(self.model.coef_[0]) / sqrtNumFeatures
        
        self.coeffs = np.array(self.coeffs).reshape(-1,1)
    
    def sigmoid(self, x):
        return (1/ (1 + np.exp(-x)))

    def predict_proba(self, X_test):
        tmp = np.zeros((X_test.shape[0], 2))
        tmp[:, 1] = self.sigmoid(np.matmul(X_test, self.coeffs))[:,0]
    
        return tmp
    
    def predict(self, X_test):
        return self.predict_proba(X_test)[:, 1]
# I have to comment out some models because it will result in TimeOutError on Kaggle.
modelList = {'mm': MasOrMenos()
             ,'rfc': RandomForestClassifier(n_estimators= 300, max_depth = 6, class_weight = {0: 1, 1:10})
             ,'logistic': LogisticRegression()
             ,'logistic_l1_05': LogisticRegression(penalty='l1')
             ,'logistic_l1_4': LogisticRegression(penalty='l1', C = 4.0)
             #,'gbm': GradientBoostingClassifier(max_depth = 6, n_estimators= 300, subsample = 0.8, max_features = 0.5)
             #,'gbm_overfit': GradientBoostingClassifier(max_depth = 10, n_estimators= 300,
             #subsample = 0.8, max_features = 0.5)
             ,'xgb': XGBClassifier(max_depth = 10, n_estimators = 200)
            }
### Standard scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_standardScaled = scaler.fit_transform(X_reduced)
X_test_standardScaled = scaler.transform(X_test_reduced)
### Airbnb Scaler
X_reshaped_target = X_reduced[y == 1]
print("Shape of the referenced dataset", X_reshaped_target.shape)

def Transform_to_CDF(data_, compare):
    data_sorted = data_.sort_values()
    data = data_sorted.values
    
    compare = compare.sort_values().values
    output = [0] * len(data)
    idx_data = 0
    idx_compare = 0
    loop = True
    
    tmp = 0
    
    while loop:
        if idx_compare == len(compare):
            if idx_data < len(data):
                for i in range(idx_data, len(data)):
                    output[i] = tmp    
            break
        if idx_data == len(data):
            break
        if data[idx_data] < compare[idx_compare]:
            output[idx_data] = tmp
            idx_data += 1
        else:
            tmp += 1
            idx_compare += 1
    
    output = pd.Series(output, index = data_sorted.index)
    return output[data_.index]

X_bnbScaled = X_reduced.copy()
for i in range(X_reduced.shape[1]):
    X_bnbScaled.iloc[:, i] = Transform_to_CDF(X_reduced.iloc[:, i], X_reshaped_target.iloc[:, i])
X_bnbScaled = X_bnbScaled / X_reshaped_target.shape[0]

X_test_bnbScaled = X_test_reduced.copy()
for i in range(X_test_reduced.shape[1]):
    X_test_bnbScaled.iloc[:, i] = Transform_to_CDF(X_test_reduced.iloc[:, i], X_reshaped_target.iloc[:, i])
X_test_bnbScaled = X_test_bnbScaled / X_reshaped_target.shape[0]
### Apply sampling method
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE takes a lot of time to create
# I suggest not using smote at all
# smote = SMOTE(random_state=1234)
underSampler = RandomUnderSampler(random_state=1234)

start = time.time()
# X_standardScaled_smote, y_standardScaled_smote = smote.fit_sample(X_standardScaled, y)
# X_bnbScaled_smote, y_bnbScaled_smote = smote.fit_sample(X_bnbScaled, y)
# print("Smote time", (time.time()-start)/ 60)

X_standardScaled_underSample, y_standardScaled_underSample = underSampler.fit_sample(X_standardScaled, y)
X_bnbScaled_underSample, y_bnbScaled_underSample = underSampler.fit_sample(X_bnbScaled, y)
# X_standardScaled_smote = pd.DataFrame(X_standardScaled_smote)
# X_bnbScaled_smote = pd.DataFrame(X_bnbScaled_smote)
X_standardScaled_underSample = pd.DataFrame(X_standardScaled_underSample)
X_bnbScaled_underSample = pd.DataFrame(X_bnbScaled_underSample)

X_test_standardScaled = pd.DataFrame(X_test_standardScaled)
X_test_bnbScaled = pd.DataFrame(X_test_bnbScaled)

X_standardScaled = pd.DataFrame(X_standardScaled)
X_bnbScaled = pd.DataFrame(X_bnbScaled)

X_standardScaled.columns = X_reduced.columns
X_standardScaled_underSample.columns = X_reduced.columns
# X_standardScaled_smote.columns = X_reduced.columns

X_bnbScaled.columns = X_reduced.columns
X_bnbScaled_underSample.columns = X_reduced.columns
# X_bnbScaled_smote.columns = X_reduced.columns

X_test_standardScaled.columns = X_reduced.columns
X_test_bnbScaled.columns = X_reduced.columns
# Run models
dataList = {
    'base': [X_reduced, y, X_test_reduced]
    ,'standardScaled': [X_standardScaled, y
                        , X_test_standardScaled]
    ,'bnbScaled': [X_bnbScaled, y
                   , X_test_bnbScaled]
    #,'standardScaled_smote': [X_standardScaled_smote, y_standardScaled_smote
    #                          , X_test_standardScaled]
    ,'standardScaled_underSample': [X_standardScaled_underSample, y_standardScaled_underSample
                                    , X_test_standardScaled]
    #,'bnbScaled_smote': [X_bnbScaled_smote, y_bnbScaled_smote
    #                     , X_test_bnbScaled]
    ,'bnbScaled_underSample':[X_bnbScaled_underSample, y_bnbScaled_underSample
                              , X_test_bnbScaled]
}

cols = ['model', 'In sample','Test sample','runtime', 'model_type']
models_report = pd.DataFrame(columns = cols)
for model in modelList: 
    for trainData in dataList:
        start = time.time()
        thisModel = modelList[model]
        thisModel.fit(dataList[trainData][0], dataList[trainData][1])
        runtime = (time.time() - start) / 60
        
        # joblib.dump(thisModel, ROOT_PATH + './model_run/' + model + '_' + trainData + '.pkl')
        
        y_score = thisModel.predict_proba(dataList[trainData][0])[:, 1]
        y_test_score = thisModel.predict_proba(dataList[trainData][2])[:, 1]
        
        tmp = pd.Series({'model': model + '_' + trainData,
                         'In sample': metrics.roc_auc_score(dataList[trainData][1], y_score),
                         'Test sample' : metrics.roc_auc_score(y_test, y_test_score),
                         'runtime': runtime,
                         'model_type': model
                         })

        models_report = models_report.append(tmp, ignore_index = True)

# models_report.to_csv(ROOT_PATH + './model_run/' + 'models_report.csv' )
models_report
accepted_train_clean_2 = accepted.loc[(accepted.issue_yr != 2017), :]
accepted2017_clean_2 = accepted.loc[(accepted.issue_yr == 2017), :]
X_2, y_2 = GetXY(accepted_train_clean_2, selected_features)
X_test_2, y_test_2= GetXY(accepted2017_clean_2, selected_features)

X_reduced_2 = X_2[selected.name.values]
X_test_reduced_2 = X_test_2[selected.name.values]

xgb_2 = XGBClassifier(max_depth = 10, n_estimators = 200)
xgb_2.fit(X_reduced_2, y_2)
res2 = GetAUC(xgb_2, X_reduced_2, y_2, X_test_reduced_2, y_test_2)