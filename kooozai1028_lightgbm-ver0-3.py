import numpy as np

import scipy as sp 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from pandas import DataFrame,Series

from sklearn.ensemble import GradientBoostingClassifier

from category_encoders import *

from sklearn import model_selection, metrics   #Additional scklearn functions

from sklearn.model_selection import GridSearchCV   #Perforing grid search

from matplotlib.pylab import rcParams

from sklearn.model_selection import StratifiedKFold

import seaborn as sns

import category_encoders as ce

from sklearn.preprocessing import Imputer

import lightgbm as lgb

from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from tqdm import tqdm_notebook as tqdm
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 40)
#提出時

df_train = pd.read_csv('../input/train.csv',index_col=0)

df_test = pd.read_csv('../input/test.csv',index_col=0)

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)

state_latlong = pd.read_csv('../input/statelatlong.csv')

state_gdp = pd.read_csv('../input/US_GDP_by_State.csv')

spi = pd.read_csv('../input/spi.csv')

# submission = pd.read_csv('sample_submission.csv')

# df_train = pd.read_csv('train.csv')

# df_test = pd.read_csv('test.csv')

# state_latlong = pd.read_csv('statelatlong.csv')

# state_gdp = pd.read_csv('US_GDP_by_State.csv')

# spi = pd.read_csv('spi.csv')



full_data = [df_train,df_test]

target = 'loan_condition'
df_train.head()
#欠損値処理

df_train = df_train.dropna(subset=['earliest_cr_line'])
#earliest_cr_line

df_train = pd.concat([df_train,df_train['earliest_cr_line'].str.split('-', expand=True).rename(columns={0: 'earliest_M', 1: 'earliest_Y'})], axis=1)

df_test = pd.concat([df_test,df_test['earliest_cr_line'].str.split('-', expand=True).rename(columns={0: 'earliest_M', 1: 'earliest_Y'})], axis=1)

df_train['earliest_M'] = df_train['earliest_M'].astype('category')

df_test['earliest_M'] = df_test['earliest_M'].astype('category')
#issue_d

df_train = pd.concat([df_train,df_train['issue_d'].str.split('-', expand=True).rename(columns={0: 'issue_M', 1: 'issue_Y'})], axis=1)

df_test = pd.concat([df_test,df_test['issue_d'].str.split('-', expand=True).rename(columns={0: 'issue_M', 1: 'issue_Y'})], axis=1)
print(df_train.shape,df_test.shape,submission.shape)
df_train = pd.merge(df_train,state_latlong,left_on='addr_state',right_on='State',how='left').drop(['State'],axis=1)

df_test = pd.merge(df_test,state_latlong,left_on='addr_state',right_on='State',how='left').drop(['State'],axis=1)
state_gdp_g = state_gdp.groupby('State',as_index=False).mean()

state_gdp_g = state_gdp_g.drop(['year'],axis=1)
df_train = pd.merge(df_train,state_gdp_g,left_on='City',right_on='State',how='left').drop(['City','State'],axis=1)

df_test = pd.merge(df_test,state_gdp_g,left_on='City',right_on='State',how='left').drop(['City','State'],axis=1)
df_train.head()
df_train.describe()
#ローン継続期間

df_train['loan_continue'] = df_train['issue_Y'].astype('int') - df_train['earliest_Y'].astype('int')

df_test['loan_continue'] = df_test['issue_Y'].astype('int') - df_test['earliest_Y'].astype('int')
#emp_title

df_train['unemployed_flg'] = df_train['emp_title'].notnull().replace({True:0,False:1})

df_train['unemployed_flg'] = df_train['unemployed_flg'].astype('category')

df_test['unemployed_flg'] = df_test['emp_title'].notnull().replace({True:0,False:1})

df_test['unemployed_flg'] = df_test['unemployed_flg'].astype('category')
#数値処理

#grade処理

df_train.loc[df_train['grade']=='G','grade'] = 'F'

df_test.loc[df_test['grade']=='G','grade'] = 'F'

#home_ownership処理

df_train.loc[df_train['home_ownership'] == 'NONE','home_ownership'] = 'OTHER'

df_train.loc[df_train['home_ownership'] == 'ANY','home_ownership'] = 'OTHER'

df_test.loc[df_test['home_ownership'] == 'NONE','home_ownership'] = 'OTHER'

df_test.loc[df_test['home_ownership'] == 'ANY','home_ownership'] = 'OTHER'

#home_ownership処理

df_train.loc[df_train['addr_state'] == 'ME','addr_state'] = 'OTHER'

df_train.loc[df_train['addr_state'] == 'ND','addr_state'] = 'OTHER'

df_train.loc[df_train['addr_state'] == 'IA','addr_state'] = 'OTHER'

df_train.loc[df_train['addr_state'] == 'ID','addr_state'] = 'OTHER'

df_test.loc[df_test['addr_state'] == 'ME','addr_state'] = 'OTHER'

df_test.loc[df_test['addr_state'] == 'ND','addr_state'] = 'OTHER'

df_test.loc[df_test['addr_state'] == 'IA','addr_state'] = 'OTHER'

df_test.loc[df_test['addr_state'] == 'ID','addr_state'] = 'OTHER'
#区分作成

def f_emp_length(x):

    if  x == '< 1 year':

        return 0.5

    if  x == '1 year':

        return 1

    if  x == '2 years':

        return 2

    if  x == '3 years':

        return 3

    if  x == '4 years':

        return 4

    if  x == '5 years':

        return 5

    if  x == '6 years':

        return 6

    if  x == '7 years':

        return 7

    if  x == '8 years':

        return 8

    if  x == '9 years':

        return 9

    if  x == '10+ years':

        return 10.5

    else:

        return -9999
df_train['emp_length'] = df_train['emp_length'].apply(f_emp_length)

df_test['emp_length'] = df_test['emp_length'].apply(f_emp_length)
df_train['t'] = 1

df_test['t'] = 0

df = pd.concat([df_train,df_test],axis=0,sort=False)
df.shape
#肩書きのテキスト処理

vec_tfidf = TfidfVectorizer()

TXT_emp = df.emp_title

TXT_emp.fillna('nan',inplace=True)
vec_tfidf = TfidfVectorizer(max_features=100)

TXT_emp_2 = vec_tfidf.fit_transform(TXT_emp)
df = pd.concat([df,pd.DataFrame(TXT_emp_2.toarray(), columns=vec_tfidf.get_feature_names(),index=df.index)],axis=1)
df.shape
df_train = df[df['t']==1]

df_train = df_train.drop(['t'],axis=1)

df_test = df[df['t']==0]

df_test = df_test.drop(['t','loan_condition'],axis=1)
print(df_train.shape,df_test.shape)
def get_duplicate_cols(df): 

    return pd.Series(df.columns).value_counts()[lambda x: x>1]
#重複削除

#学習データ

df_dup_cols = get_duplicate_cols(df_train)

list_dup_cols = df_dup_cols.reset_index().values.tolist()



dup_cols = []



for i in range(len(list_dup_cols)):

    dup_cols.append(list_dup_cols[i][0])



cols = []

count = 1

for column in df_train.columns:

    if column in dup_cols:

        cols.append(column+'_'+str(count))

        count+=1

    else:

        cols.append(column)

df_train.columns = cols



#テストデータ

df_dup_cols = get_duplicate_cols(df_test)

list_dup_cols = df_dup_cols.reset_index().values.tolist()



dup_cols = []



for i in range(len(list_dup_cols)):

    dup_cols.append(list_dup_cols[i][0])



cols = []

count = 1

for column in df_test.columns:

    if column in dup_cols:

        cols.append(column+'_'+str(count))

        count+=1

    else:

        cols.append(column)

df_test.columns = cols
#ローン返済期間

df_train['loan_span'] = df_train['loan_amnt']/df_train['installment']

df_test['loan_span'] = df_test['loan_amnt']/df_test['installment']

#残高合計

# df_train['all_bal'] = df_train['revol_bal'] + df_train['tot_cur_bal']

# df_test['all_bal'] = df_test['revol_bal'] + df_test['tot_cur_bal']
# #グレード

# #出現頻度

# df_train['grade_freq'] = df_train.groupby('grade')['grade'].transform('count')

# df_test['grade_freq'] = df_test.groupby('grade')['grade'].transform('count')

# #年収の中央値

# df_train['grade_inc_med'] = df_train.groupby('grade')['annual_inc'].transform('median')

# df_test['grade_inc_med'] = df_test.groupby('grade')['annual_inc'].transform('median')

# #相対的な年収

# df_train['grade_inc_med_diff'] = df_train['annual_inc'] - df_train['grade_inc_med']

# df_test['grade_inc_med_diff'] = df_test['annual_inc'] - df_test['grade_inc_med']

# #残高の中央値

# df_train['grade_bal_med'] = df_train.groupby('grade')['tot_cur_bal'].transform('median')

# df_test['grade_bal_med'] = df_test.groupby('grade')['tot_cur_bal'].transform('median')

# #相対的な残額

# df_train['grade_bal_med_diff'] = df_train['tot_cur_bal'] - df_train['grade_bal_med']

# df_test['grade_bal_med_diff'] = df_test['tot_cur_bal'] - df_test['grade_bal_med']

# #リボ残高の中央値

# df_train['grade_rev_bal_med'] = df_train.groupby('grade')['revol_bal'].transform('median')

# df_test['grade_rev_bal_med'] = df_test.groupby('grade')['revol_bal'].transform('median')

# #相対的なリボ残額

# df_train['grade_rev_bal_med_diff'] = df_train['revol_bal'] - df_train['grade_rev_bal_med']

# df_test['grade_rev_bal_med_diff'] = df_test['revol_bal'] - df_test['grade_rev_bal_med']
#サブグレード

#出現頻度

df_train['sub_grade_freq'] = df_train.groupby('sub_grade')['sub_grade'].transform('count')

df_test['sub_grade_freq'] = df_test.groupby('sub_grade')['sub_grade'].transform('count')

# #年収の中央値

df_train['sub_grade_inc_med'] = df_train.groupby('sub_grade')['annual_inc'].transform('median')

df_test['sub_grade_inc_med'] = df_test.groupby('sub_grade')['annual_inc'].transform('median')

# #相対的な年収

df_train['sub_grade_inc_med_diff'] = df_train['annual_inc'] - df_train['sub_grade_inc_med']

df_test['sub_grade_inc_med_diff'] = df_test['annual_inc'] - df_test['sub_grade_inc_med']

#残高の中央値

df_train['sub_grade_bal_med'] = df_train.groupby('sub_grade')['tot_cur_bal'].transform('median')

df_test['sub_grade_bal_med'] = df_test.groupby('sub_grade')['tot_cur_bal'].transform('median')

#相対的な残額

df_train['sub_grade_bal_med_diff'] = df_train['tot_cur_bal'] - df_train['sub_grade_bal_med']

df_test['sub_grade_bal_med_diff'] = df_test['tot_cur_bal'] - df_test['sub_grade_bal_med']

#リボ残高の中央値

df_train['sub_grade_rev_bal_med'] = df_train.groupby('sub_grade')['revol_bal'].transform('median')

df_test['sub_grade_rev_bal_med'] = df_test.groupby('sub_grade')['revol_bal'].transform('median')

#相対的なリボ残額

df_train['sub_grade_rev_bal_med_diff'] = df_train['revol_bal'] - df_train['sub_grade_rev_bal_med']

df_test['sub_grade_rev_bal_med_diff'] = df_test['revol_bal'] - df_test['sub_grade_rev_bal_med']
#目的

#出現頻度

df_train['purpose_freq'] = df_train.groupby('purpose')['purpose'].transform('count')

df_test['purpose_freq'] = df_test.groupby('purpose')['purpose'].transform('count')

#年収の中央値

df_train['purpose_inc_med'] = df_train.groupby('purpose')['annual_inc'].transform('median')

df_test['purpose_inc_med'] = df_test.groupby('purpose')['annual_inc'].transform('median')

#相対的な年収

df_train['purpose_inc_med_diff'] = df_train['annual_inc'] - df_train['purpose_inc_med']

df_test['purpose_inc_med_diff'] = df_test['annual_inc'] - df_test['purpose_inc_med']

#ローン額の中央値

df_train['purpose_loan_med'] = df_train.groupby('purpose')['loan_amnt'].transform('median')

df_test['purpose_loan_med'] = df_test.groupby('purpose')['loan_amnt'].transform('median')

#相対的なローン額

df_train['purpose_loan_med_diff'] = df_train['loan_amnt'] - df_train['purpose_loan_med']

df_test['purpose_loan_med_diff'] = df_test['loan_amnt'] - df_test['purpose_loan_med']

#返済期間の中央値

df_train['purpose_loan_span_med'] = df_train.groupby('purpose')['loan_amnt'].transform('median')

df_test['purpose_loan_span_med'] = df_test.groupby('purpose')['loan_amnt'].transform('median')

#相対的な返済期間

df_train['purpose_loan_span_med_diff'] = df_train['loan_span'] - df_train['purpose_loan_span_med']

df_test['purpose_loan_span_med_diff'] = df_test['loan_span'] - df_test['purpose_loan_span_med']
#各州

#出現頻度

df_train['addr_state_freq'] = df_train.groupby('addr_state')['addr_state'].transform('count')

df_test['addr_state_freq'] = df_test.groupby('addr_state')['addr_state'].transform('count')

#年収の中央値

df_train['addr_state_inc_med'] = df_train.groupby('addr_state')['annual_inc'].transform('median')

df_test['addr_state_inc_med'] = df_test.groupby('addr_state')['annual_inc'].transform('median')

#相対的な年収

df_train['addr_state_inc_med_diff'] = df_train['annual_inc'] - df_train['addr_state_inc_med']

df_test['addr_state_inc_med_diff'] = df_test['annual_inc'] - df_test['addr_state_inc_med']
#ローン発行年

#出現頻度

df_train['issue_Y_freq'] = df_train.groupby('issue_Y')['issue_Y'].transform('count')

df_test['issue_Y_freq'] = df_test.groupby('issue_Y')['issue_Y'].transform('count')

#年収の中央値

df_train['issue_Y_inc_med'] = df_train.groupby('issue_Y')['annual_inc'].transform('median')

df_test['issue_Y_inc_med'] = df_test.groupby('issue_Y')['annual_inc'].transform('median')

#相対的な年収

df_train['issue_Y_inc_med_diff'] = df_train['annual_inc'] - df_train['issue_Y_inc_med']

df_test['issue_Y_inc_med_diff'] = df_test['annual_inc'] - df_test['issue_Y_inc_med']
#ローン開始年

#出現頻度

df_train['earliest_Y_freq'] = df_train.groupby('earliest_Y')['earliest_Y'].transform('count')

df_test['earliest_Y_freq'] = df_test.groupby('earliest_Y')['earliest_Y'].transform('count')

#年収の中央値

df_train['earliest_Y_inc_med'] = df_train.groupby('earliest_Y')['annual_inc'].transform('median')

df_test['earliest_Y_inc_med'] = df_test.groupby('earliest_Y')['annual_inc'].transform('median')

#相対的な年収

df_train['earliest_Y_inc_med_diff'] = df_train['annual_inc'] - df_train['earliest_Y_inc_med']

df_test['earliest_Y_inc_med_diff'] = df_test['annual_inc'] - df_test['earliest_Y_inc_med']
#家保有

#出現頻度

df_train['home_ownership_freq'] = df_train.groupby('home_ownership')['home_ownership'].transform('count')

df_test['home_ownership_freq'] = df_test.groupby('home_ownership')['home_ownership'].transform('count')

#年収の中央値

df_train['home_ownership_inc_med'] = df_train.groupby('home_ownership')['annual_inc'].transform('median')

df_test['home_ownership_inc_med'] = df_test.groupby('home_ownership')['annual_inc'].transform('median')

#相対的な年収

df_train['home_ownership_inc_med_diff'] = df_train['annual_inc'] - df_train['home_ownership_inc_med']

df_test['home_ownership_inc_med_diff'] = df_test['annual_inc'] - df_test['home_ownership_inc_med']
#グレード別目的

df_train['grade_pur'] = df_train[['grade','purpose']].apply(lambda x:'{}_{}'.format(x[0],x[1]),axis=1)

df_test['grade_pur'] = df_test[['grade','purpose']].apply(lambda x:'{}_{}'.format(x[0],x[1]),axis=1)

#グレード別家保有

df_train['grade_home'] = df_train[['grade','home_ownership']].apply(lambda x:'{}_{}'.format(x[0],x[1]),axis=1)

df_test['grade_home'] = df_test[['grade','home_ownership']].apply(lambda x:'{}_{}'.format(x[0],x[1]),axis=1)
# #対数変換

df_train['installment'] = df_train.installment.apply(np.log1p)

df_test['installment'] = df_test.installment.apply(np.log1p)



df_train['annual_inc'] = df_train.annual_inc.apply(np.log1p)

df_test['annual_inc'] = df_test.annual_inc.apply(np.log1p)



df_train['dti'] = df_train.dti.apply(np.log1p)

df_test['dti'] = df_test.dti.apply(np.log1p)



df_train['delinq_2yrs'] = df_train.delinq_2yrs.apply(np.log1p)

df_test['delinq_2yrs'] = df_test.delinq_2yrs.apply(np.log1p)



df_train['revol_bal'] = df_train.revol_bal.apply(np.log1p)

df_test['revol_bal'] = df_test.revol_bal.apply(np.log1p)



df_train['revol_util'] = df_train.revol_util.apply(np.log1p)

df_test['revol_util'] = df_test.revol_util.apply(np.log1p)



df_train['total_acc'] = df_train.total_acc.apply(np.log1p)

df_test['total_acc'] = df_test.total_acc.apply(np.log1p)



df_train['collections_12_mths_ex_med'] = df_train.collections_12_mths_ex_med.apply(np.log1p)

df_test['collections_12_mths_ex_med'] = df_test.collections_12_mths_ex_med.apply(np.log1p)



df_train['acc_now_delinq'] = df_train.acc_now_delinq.apply(np.log1p)

df_test['acc_now_delinq'] = df_test.acc_now_delinq.apply(np.log1p)



df_train['tot_coll_amt'] = df_train.tot_coll_amt.apply(np.log1p)

df_test['tot_coll_amt'] = df_test.tot_coll_amt.apply(np.log1p)



df_train['tot_cur_bal'] = df_train.tot_cur_bal.apply(np.log1p)

df_test['tot_cur_bal'] = df_test.tot_cur_bal.apply(np.log1p)
#欠損値カウント

df_train['null_cnt'] = df_train.isnull().sum(axis=1)

df_test['null_cnt'] = df_test.isnull().sum(axis=1)
for data in [df_train,df_test]:

    for col in data.columns:

        try:

            if data[col].dtypes == 'object':

                data[col] = data[col].astype('category')

        except:

            pass
#変数の選択

predictors = [x for x in df_train.columns if x not in [target

                                                       ,'ID'

                                                       ,'grade'

                                                       ,'emp_title'

                                                       ,'issue_d'

                                                      ,'issue_M'

                                                       ,'issue_Y'

                                                       ,'issue_M_Y'

                                                       ,'earliest_M_Y'

                                                        ,'title'

                                                       ,'zip_code'

                                                       ,'earliest_cr_line'

                                                      ]]
df_train.sort_values(['issue_d'],ascending=True)
def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True):

    #Fit the algorithm on the data

    alg.fit(dtrain[predictors], dtrain[target])

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[predictors])

    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    

    #Perform cross-validation:

    

    cv_folds=model_selection.TimeSeriesSplit(n_splits=5)

    if performCV:

        cv_score = model_selection.cross_val_score(alg

                                                   ,dtrain[predictors]

                                                   ,dtrain[target]

                                                   ,cv=cv_folds

                                                   ,scoring='roc_auc'

                                                  )



    #Print model report:

    print ("Model Report")

    print ("Accuracy : {:.4f}".format(metrics.accuracy_score(dtrain[target].values, dtrain_predictions)))

    print ("AUC Score (Train): {:.4f}".format(metrics.roc_auc_score(dtrain[target], dtrain_predprob)))

    plt.style.use('ggplot')    



    %matplotlib inline

    plt.figure(figsize=[10,5])

    plt.hist(dtrain_predprob,bins=50,color='r',label='train',alpha=0.5,density=True)

    plt.legend()

    plt.show()

    

    if performCV:

        print ("CV Score : Mean - {:.6f} | Std - {:.6f} | Min - {:.6f} | Max - {:.6f}".format(np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

        print(cv_score)

    

    #Print Feature Importance:

    if printFeatureImportance:

        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)

        print(feat_imp)
predictors
#Choose all predictors except target

lgbm = LGBMClassifier(

    n_estimators=100

    ,random_state=10)

modelfit(lgbm, df_train, predictors)
df_train['prob'] = lgbm.predict_proba(df_train[predictors])[:,1]
p1 = df_train['prob'].quantile(.1)

p2 = df_train['prob'].quantile(.2)

p3 = df_train['prob'].quantile(.3)

p4 = df_train['prob'].quantile(.4)

p5 = df_train['prob'].quantile(.5)

p6 = df_train['prob'].quantile(.6)

p7 = df_train['prob'].quantile(.7)

p8 = df_train['prob'].quantile(.8)

p9 = df_train['prob'].quantile(.9)
def f_kbn_prob(x):

    if  x < p1:

        return '01'

    if  x < p2:

        return '02'

    if  x < p3:

        return '03'

    if  x < p4:

        return '04'

    if  x < p5:

        return '05'

    if  x < p6:

        return '06'

    if  x < p7:

        return '07'

    if  x < p8:

        return '08'

    if  x < p9:

        return '09'

    else:

        return '10'
df_train['prob_kbn'] = df_train['prob'].apply(f_kbn_prob)
df_train
df_train.describe()
pd.DataFrame(df_train.groupby(['prob_kbn','loan_condition']).mean())
# df_train.sort_values(['prob'],ascending=True).to_csv('train_check_asc.csv')

# df_train.sort_values(['prob'],ascending=False).to_csv('train_check_dsc.csv')
y_pred = lgbm.predict_proba(df_test[predictors])[:,1]
%matplotlib inline

plt.figure(figsize=[10,5])

plt.hist(pd.DataFrame(y_pred)[0],bins=50,color='r',label='test',alpha=0.5,density=True)

plt.legend()

plt.show()
submission.loan_condition = y_pred

submission.to_csv('submission.csv')
submission