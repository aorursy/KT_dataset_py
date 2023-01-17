import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm, notebook



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

from datetime import datetime

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold,KFold

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

import eli5

from eli5.sklearn import PermutationImportance

import xgboost as xgb

import seaborn as sns

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier

import catboost
xgb.XGBClassifier()
#ハンズオンのスムーズな進行のために全体の20分の1だけ読み込むことにします。

#実際に課題でモデリングする際には"skiprows=lambda x: x%20!=0"を削除してください。

df_train = pd.read_csv('../input/homework-for-students3/train.csv', index_col=0)

df_test = pd.read_csv('../input/homework-for-students3/test.csv', index_col=0)

print(len(df_test))

print(len(df_train))
gdp=pd.read_csv('../input/homework-for-students3/US_GDP_by_State.csv')

zipdata=pd.read_csv('../input/homework-for-students3/free-zipcode-database.csv')

drop_col = ['WorldRegion',

       'Country', 'LocationText', 'Location', 'Decommisioned',

       'TaxReturnsFiled', 'EstimatedPopulation', 'TotalWages', 'Notes']

zipdata=zipdata.drop(drop_col,axis=1)

state=pd.read_csv('../input/homework-for-students3/statelatlong.csv')

spi=pd.read_csv('../input/homework-for-students3/spi.csv')

spi['date']=pd.to_datetime(spi['date'])

spi= spi.set_index("date")

spi=spi.asfreq('d', method='ffill')

spi = spi.reset_index()
#はずれ値Null

df_train["issue_d"]=pd.to_datetime(df_train["issue_d"])

df_test["issue_d"]=pd.to_datetime(df_test["issue_d"])

df_train = df_train[df_train.issue_d.dt.year >= 2015]

#df_train["issue_d"]=df_train["issue_d"].astype(str)

df_train = df_train[df_train['annual_inc'] < df_train['annual_inc'].quantile(0.999)]

df_train['IDdami']=df_train.index

df_test['IDdami']=df_test.index
#元データから特徴量生成

df_train["earliest_cr_line"]=pd.to_datetime(df_train["earliest_cr_line"])

df_test["earliest_cr_line"]=pd.to_datetime(df_test["earliest_cr_line"])



df_train["issue_d_unix"] = df_train["issue_d"].view('int64') // 10**9

df_test["issue_d_unix"] = df_test["issue_d"].view('int64') // 10**9

df_train["earliest_cr_line_unix"] = df_train["earliest_cr_line"].view('int64') // 10**9

df_test["earliest_cr_line_unix"] = df_test["earliest_cr_line"].view('int64') // 10**9



df_train["period"]=df_train["issue_d_unix"]-df_train["earliest_cr_line_unix"]

df_test["period"]=df_test["issue_d_unix"]-df_test["earliest_cr_line_unix"]



df_train["period"]=df_train["period"].fillna(0)

df_test["period"]=df_test["period"].fillna(0) 



#ローンの合計/月々の支払＝どれくらいの期間支払うのか

df_train['aaa']=round(df_train['loan_amnt']/df_train['installment'],5)

df_test['aaa']=round(df_test['loan_amnt']/df_test['installment'],5)



#年収に対していくらがローン返済に回されるのか

df_train['bbb']=round(df_train['loan_amnt']/df_train['annual_inc'],5)

df_test['bbb']=round(df_test['loan_amnt']/df_test['annual_inc'],5)

#年収のうち月々ローン返済に回される額

#df_train['ccc']=round(df_train['installment']/df_train['annual_inc'],5)

#df_test['ccc']=round(df_test['installment']/df_test['annual_inc'],5)



#信用枠以内で何回借りたか

df_train['ddd']=round(df_train['revol_bal']/df_train['revol_util'],5)

df_test['ddd']=round(df_test['revol_bal']/df_test['revol_util'],5)

#総預金のうち、信用枠以内で借りた金額

df_train['eee']=round(df_train['revol_bal']/df_train['total_acc'],5)

df_test['eee']=round(df_test['revol_bal']/df_test['total_acc'],5)



#スコア出てない系

df_train['fff']=round(df_train['revol_util']/df_train['total_acc'],5)

df_test['fff']=round(df_test['revol_util']/df_test['total_acc'],5)



#ローン総額/新しい負債の本数

df_train['aaa_open_acc']=round(df_train['loan_amnt']/df_train['open_acc'],5)

df_test['aaa_open_acc']=round(df_test['loan_amnt']/df_test['open_acc'],5)



#精度15位だけど感覚的に理解困難

#df_train['annual_inc_open_acc']=round(df_train['annual_inc']/df_train['open_acc'],5)

#df_test['annual_inc_open_acc']=round(df_test['annual_inc']/df_test['open_acc'],5)





#df_train['remaining']=round(df_train['loan_amnt']/df_train['tot_coll_amt'],5)

#df_test['remaining']=round(df_test['loan_amnt']/df_test['tot_coll_amt'],5)
df_train.columns

print(len(df_test))

print(len(df_train))
#元データとstatelatlong結合

df_train = df_train.reset_index()

df_test = df_test.reset_index()

kari_df_train=pd.merge(df_train, state, how='left',left_on='addr_state',right_on='State')

kari_df_test=pd.merge(df_test, state, how='left',left_on='addr_state',right_on='State')

df_train = kari_df_train.set_index("ID")

df_test =kari_df_test.set_index("ID")

df_train=df_train.drop('State',axis=1)

df_test=df_test.drop('State',axis=1)
print(len(df_test))

print(len(df_train))
#元データとstatelatlongとgdp結合

df_train['dami_year']=df_train.issue_d.dt.year

df_test['dami_year']=int(2015)

df_train = df_train.reset_index()

df_test = df_test.reset_index()

kari_df_train=pd.merge(df_train, gdp, how='left',left_on=['City','dami_year'],right_on=['State','year'])

kari_df_test=pd.merge(df_test, gdp, how='left',left_on=['City','dami_year'],right_on=['State','year'])

df_train = kari_df_train.set_index("ID")

df_test =kari_df_test.set_index("ID")

df_train=df_train.drop(['State','dami_year','year'],axis=1)

df_test=df_test.drop(['State','dami_year','year'],axis=1)
#元データとstatelatlongとgdpとspi結合

df_train = df_train.reset_index()

df_test = df_test.reset_index()

kari_df_train=pd.merge(df_train, spi, how='left',left_on=['issue_d'],right_on=['date'])

kari_df_test=pd.merge(df_test, spi, how='left',left_on=['issue_d'],right_on=['date'])

df_train = kari_df_train.set_index("ID")

df_test =kari_df_test.set_index("ID")

df_train=df_train.drop(['date'],axis=1)

df_test=df_test.drop(['date'],axis=1)
print(len(df_test))

print(len(df_train))

zipdata.columns
#zipdatagroupby

zipdata["Zipcode"]=zipdata["Zipcode"].astype(str)

zipdata["Zipcode"]=zipdata["Zipcode"].str[:3]

zipdata=zipdata[['Zipcode','State','Xaxis', 'Yaxis', 'Zaxis']]

zipdata=zipdata.groupby(['Zipcode','State'],as_index=False).mean()

#zipdata = zipdata.reset_index()
#zipdatagroupby

df_train['zip_code']=df_train['zip_code'].str[:3]

df_test['zip_code']=df_test['zip_code'].str[:3]

df_train["zip_code"]=df_train["zip_code"].astype(str)

df_test["zip_code"]=df_test["zip_code"].astype(str)

#df_train["City"]=df_train["City"].str.upper()

#df_test["City"]=df_test["City"].str.upper()
zipdata[zipdata.duplicated()]
#元データとstatelatlongとgdpとspi結合とzipcode

df_train = df_train.reset_index()

df_test = df_test.reset_index()

kari_df_train=pd.merge(df_train, zipdata, how='left',left_on=['zip_code','addr_state'],right_on=['Zipcode','State'])

kari_df_test=pd.merge(df_test, zipdata, how='left',left_on=['zip_code','addr_state'],right_on=['Zipcode','State'])

df_train = kari_df_train.set_index("ID")

df_test =kari_df_test.set_index("ID")

df_train=df_train.drop(['Zipcode','State'],axis=1)

df_test=df_test.drop(['Zipcode','State'],axis=1)
#Labelエンコーディング

encoder = OrdinalEncoder()

enc_train = encoder.fit_transform(df_train['zip_code'].values)

enc_test = encoder.transform(df_test['zip_code'].values)

df_train = df_train.reset_index()

df_test = df_test.reset_index()

df_train['zip_code_la']=enc_train.iloc[:,0]

df_test['zip_code_la']=enc_test.iloc[:,0]

df_train = df_train.set_index("ID")

df_test =df_test.set_index("ID")
#カウントエンコーディング

zi_cal1='zip_code'

zi_summary1 = df_train[zi_cal1].value_counts()

# mapする。

df_train['zip_code_co'] = df_train[zi_cal1].map(zi_summary1)

df_test['zip_code_co'] = df_test[zi_cal1].map(zi_summary1)
#Labelエンコーディング

encoder = OrdinalEncoder()

enc_train = encoder.fit_transform(df_train['addr_state'].values)

enc_test = encoder.transform(df_test['addr_state'].values)

df_train = df_train.reset_index()

df_test = df_test.reset_index()

df_train['addr_state_la']=enc_train.iloc[:,0]

df_test['addr_state_la']=enc_test.iloc[:,0]

df_train = df_train.set_index("ID")

df_test =df_test.set_index("ID")
#カウントエンコーディング

zi_cal2='addr_state'

zi_summary2 = df_train[zi_cal2].value_counts()

# mapする。

df_train['addr_state_co'] = df_train[zi_cal2].map(zi_summary2)

df_test['addr_state_co'] = df_test[zi_cal2].map(zi_summary2)
df_train.columns
"""

df_train['Xaxis']=df_train['Xaxis'].replace([np.inf, -np.inf,np.nan], -9999)

df_test['Xaxis']=df_test['Xaxis'].replace([np.inf, -np.inf,np.nan], -9999)

df_train['Xaxis'].astype(str)

df_test['Xaxis'].astype(str)

#カウントエンコーディング

zi_cal2='Xaxis'

zi_summary2 = df_train[zi_cal2].value_counts()

# mapする。

df_train['Xaxis_co'] = df_train[zi_cal2].map(zi_summary2)

df_test['Xaxis_co'] = df_test[zi_cal2].map(zi_summary2)

df_train['Xaxis'].astype(int)

df_test['Xaxis'].astype(int)

"""
"""

df_train['Yaxis']=df_train['Yaxis'].replace([np.inf, -np.inf,np.nan], -9999)

df_test['Yaxis']=df_test['Yaxis'].replace([np.inf, -np.inf,np.nan], -9999)

df_train['Yaxis'].astype(str)

df_test['Yaxis'].astype(str)

#カウントエンコーディング

zi_cal2='Yaxis'

zi_summary2 = df_train[zi_cal2].value_counts()

# mapする。

df_train['Yaxis_co'] = df_train[zi_cal2].map(zi_summary2)

df_test['Yaxis_co'] = df_test[zi_cal2].map(zi_summary2)

df_train['Yaxis'].astype(int)

df_test['Yaxis'].astype(int)

"""
"""

df_train['Zaxis']=df_train['Zaxis'].replace([np.inf, -np.inf,np.nan], -9999)

df_test['Zaxis']=df_test['Zaxis'].replace([np.inf, -np.inf,np.nan], -9999)

df_train['Zaxis'].astype(str)

df_test['Zaxis'].astype(str)

#カウントエンコーディング

zi_cal2='Zaxis'

zi_summary2 = df_train[zi_cal2].value_counts()

# mapする。

df_train['Zaxis_co'] = df_train[zi_cal2].map(zi_summary2)

df_test['Zaxis_co'] = df_test[zi_cal2].map(zi_summary2)

df_train['Zaxis'].astype(int)

df_test['Zaxis'].astype(int)

"""
#特徴量の選択

#df_train=df_train.drop(['issue_d','earliest_cr_line','issue_d_unix','earliest_cr_line_unix'],axis=1)

#df_test=df_test.drop(['issue_d','earliest_cr_line','issue_d_unix','earliest_cr_line_unix'],axis=1)

df_train=df_train.drop(['issue_d','earliest_cr_line'],axis=1)

df_test=df_test.drop(['issue_d','earliest_cr_line'],axis=1)

drop_col=['City','acc_now_delinq']

df_train=df_train.drop(drop_col,axis=1)

df_test=df_test.drop(drop_col,axis=1)
"""

df_train=df_train.replace({'initial_list_status':{'w':1,'f':2}})

df_test=df_test.replace({'initial_list_status':{'w':1,'f':2}})

df_train["initial_list_status"]=df_train["initial_list_status"].astype(int)

df_test["initial_list_status"]=df_test["initial_list_status"].astype(int)

"""
"""

df_train=df_train.replace({'application_type':{'Individual':1,'Joint App':2}})

df_test=df_test.replace({'application_type':{'Individual':1,'Joint App':2}})

df_train["application_type"]=df_train["application_type"].astype(int)

df_test["application_type"]=df_test["application_type"].astype(int)

"""
#カウントエンコーディング

ce_cal2='initial_list_status'

ce_summary2 = df_train[ce_cal2].value_counts()

# mapする。

df_train['initial_list_status'] = df_train[ce_cal2].map(ce_summary2)

df_test['initial_list_status'] = df_test[ce_cal2].map(ce_summary2)
#カウントエンコーディング

ce_cal2='application_type'

ce_summary2 = df_train[ce_cal2].value_counts()

# mapする。

df_train['application_type'] = df_train[ce_cal2].map(ce_summary2)

df_test['application_type'] = df_test[ce_cal2].map(ce_summary2)
#カテゴリマッピング

df_train['grade'].unique()

df_train=df_train.replace({'grade':{'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}})

df_test=df_test.replace({'grade':{'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}})

df_train["grade"]=df_train["grade"].astype(int)

df_test["grade"]=df_test["grade"].astype(int)
df_train=df_train.replace({'sub_grade':{'A1':1,'A2':2,'A3':3,'A4':4,'A5':5,

                                    'B1':6,'B2':7,'B3':8,'B4':9,'B5':10,

                                    'C1':11,'C2':12,'C3':13,'C4':14,'C5':15,

                                    'D1':16,'D2':17,'D3':18,'D4':19,'D5':20,

                                    'E1':21,'E2':22,'E3':23,'E4':24,'E5':25,

                                    'F1':26,'F2':27,'F3':28,'F4':29,'F5':30,

                                   'G1':31,'G2':32,'G3':33,'G4':34,'G5':35}})

df_test=df_test.replace({'sub_grade':{'A1':1,'A2':2,'A3':3,'A4':4,'A5':5,

                                    'B1':6,'B2':7,'B3':8,'B4':9,'B5':10,

                                    'C1':11,'C2':12,'C3':13,'C4':14,'C5':15,

                                    'D1':16,'D2':17,'D3':18,'D4':19,'D5':20,

                                    'E1':21,'E2':22,'E3':23,'E4':24,'E5':25,

                                    'F1':26,'F2':27,'F3':28,'F4':29,'F5':30,

                                   'G1':31,'G2':32,'G3':33,'G4':34,'G5':35}})

df_train["sub_grade"]=df_train["sub_grade"].astype(int)

df_test["sub_grade"]=df_test["sub_grade"].astype(int)


in_0=df_train[df_train.loan_condition==0].installment.median()

df_train['in_0_sa'] =df_train['installment']-in_0

df_test['in_0_sa'] =df_test['installment']-in_0

lo_0=df_train[df_train.loan_condition==0].loan_amnt.median()

df_train['lo_0_sa'] =df_train['loan_amnt']-lo_0

df_test['lo_0_sa'] =df_test['loan_amnt']-lo_0



dti_0=df_train[df_train.loan_condition==0].dti.median()

df_train['dti_0_sa'] =df_train['dti']-dti_0

df_test['dti_0_sa'] =df_test['dti']-dti_0



tot_0=df_train[df_train.loan_condition==0].tot_cur_bal.median()

df_train['tot_0_sa'] =df_train['tot_cur_bal']-tot_0

df_test['tot_0_sa'] =df_test['tot_cur_bal']-tot_0



rev_0=df_train[df_train.loan_condition==0].revol_bal.median()

df_train['rev_0_sa'] =df_train['revol_bal']-rev_0

df_test['rev_0_sa'] =df_test['revol_bal']-rev_0



pe_0=df_train[df_train.loan_condition==0].period.median()

df_train['pe_0_sa'] =df_train['period']-pe_0

df_test['pe_0_sa'] =df_test['period']-pe_0



#sg_0=df_train[df_train.loan_condition==0].sub_grade.median()

#df_train['sg_0_sa'] =df_train['sub_grade']-sg_0

#df_test['sg_0_sa'] =df_test['sub_grade']-sg_0



in_1=df_train[df_train.loan_condition==1].installment.median()

df_train['in_1_sa'] =df_train['installment']-in_1

df_test['in_1_sa'] =df_test['installment']-in_1





lo_1=df_train[df_train.loan_condition==1].loan_amnt.median()

df_train['lo_1_sa'] =df_train['loan_amnt']-lo_1

df_test['lo_1_sa'] =df_test['loan_amnt']-lo_1



dti_1=df_train[df_train.loan_condition==1].dti.median()

df_train['dti_1_sa'] =df_train['dti']-dti_1

df_test['dti_1_sa'] =df_test['dti']-dti_1



tot_1=df_train[df_train.loan_condition==1].tot_cur_bal.median()

df_train['tot_1_sa'] =df_train['tot_cur_bal']-tot_1

df_test['tot_1_sa'] =df_test['tot_cur_bal']-tot_1



rev_1=df_train[df_train.loan_condition==1].revol_bal.median()

df_train['rev_1_sa'] =df_train['revol_bal']-rev_1

df_test['rev_1_sa'] =df_test['revol_bal']-rev_1



pe_1=df_train[df_train.loan_condition==1].period.median()

df_train['pe_0_sa'] =df_train['period']-pe_1

df_test['pe_0_sa'] =df_test['period']-pe_1





#sg_1=df_train[df_train.loan_condition==1].sub_grade.median()

#df_train['sg_1_sa'] =df_train['sub_grade']-sg_1

#df_test['sg_1_sa'] =df_test['sub_grade']-sg_1

df_train['home_ownership'].unique()

df_train=df_train.replace({'home_ownership':{'MORTGAGE':3,'RENT':2,'OWN':4,'ANY':1}})

df_test=df_test.replace({'home_ownership':{'MORTGAGE':3,'RENT':2,'OWN':4,'ANY':1}})

df_train["home_ownership"]=df_train["home_ownership"].astype(int)

df_test["home_ownership"]=df_test["home_ownership"].astype(int)

print(len(df_train.columns))

print(df_test.columns)
"""

df_train=df_train.replace({'purpose':{'debt_consolidation':1,'credit_card':2,'home_improvement':3,'other':4,

                                     'major_purchase':5,'medical':6,'small_business':7,'car':8,

                                     'moving':9,'vacation':10,'house':11,'renewable_energy':12,

                                     'wedding':13,'educational':14}})

df_test=df_test.replace({'purpose':{'debt_consolidation':1,'credit_card':2,'home_improvement':3,'other':4,

                                     'major_purchase':5,'medical':6,'small_business':7,'car':8,

                                     'moving':9,'vacation':10,'house':11,'renewable_energy':12,

                                     'wedding':13,'educational':14}})

df_train["purpose"]=df_train["purpose"].astype(int)

df_test["purpose"]=df_test["purpose"].astype(int)

print(len(df_test.columns))

print(df_test.columns)

"""
summary = df_train['purpose'].value_counts()

summary

df_train['purpose_co'] = df_train['purpose'].map(summary)

df_test['purpose_co'] = df_test['purpose'].map(summary)
df_train['purpose'].unique()
df_train=df_train.replace({'emp_length':{'< 1 year':0.5,'1 year':1,'2 years':2,'3 years':3,

                                        '4 years':4,'5 years':5,'6 years':6,'7 years':7,

                                        '8 years':8,'9 years':9,'10+ years':10}})  

df_test=df_test.replace({'emp_length':{'< 1 year':0.5,'1 year':1,'2 years':2,'3 years':3,

                                        '4 years':4,'5 years':5,'6 years':6,'7 years':7,

                                        '8 years':8,'9 years':9,'10+ years':10}})  

df_train["emp_length"].head()
#特徴の追加

#

#df_train['emp_length_ggg']=round(df_train['loan_amnt']*df_train['emp_length'],5)

#df_test['emp_length_ggg']=round(df_test['loan_amnt']*df_test['emp_length'],5)



#df_train['emp_length_hhh']=round(df_train['installment']*df_train['emp_length'],5)

#df_test['emp_length_hhh']=round(df_test['installment']*df_test['emp_length'],5)



df_train['emp_length_iii']=round(df_train['annual_inc']*df_train['emp_length'],5)

df_test['emp_length_iii']=round(df_test['annual_inc']*df_test['emp_length'],5)



#df_train['emp_length_jjj']=round(df_train['dti']*df_train['emp_length'],5)

#df_test['emp_length_jjj']=round(df_test['dti']*df_test['emp_length'],5)



#df_train['emp_length_kkk']=round(df_train['open_acc']*df_train['emp_length'],5)

#df_test['emp_length_kkk']=round(df_test['open_acc']*df_test['emp_length'],5)



#df_train['emp_length_lll']=round(df_train['revol_bal']*df_train['emp_length'],5)

#df_test['emp_length_lll']=round(df_test['revol_bal']*df_test['emp_length'],5)



#df_train['emp_length_mmm']=round(df_train['revol_util']*df_train['emp_length'],5)

#df_test['emp_length_mmm']=round(df_test['revol_util']*df_test['emp_length'],5)



#df_train['emp_length_nnn']=round(df_train['total_acc']*df_train['emp_length'],5)

#df_test['emp_length_nnn']=round(df_test['total_acc']*df_test['emp_length'],5)



#df_train['emp_length_ooo']=round(df_train['tot_cur_bal']*df_train['emp_length'],5)

#df_test['emp_length_ooo']=round(df_test['tot_cur_bal']*df_test['emp_length'],5)
#最後に滞納してからの影響

df_train['ggg_mths_since_last_delinq']=round(df_train['installment']*df_train['mths_since_last_delinq'],5)

df_test['ggg_mths_since_last_delinq']=round(df_test['installment']*df_test['mths_since_last_delinq'],5)
#特徴の追加

df_train['ggg']=round(df_train['loan_amnt']*df_train['sub_grade'],5)

df_test['ggg']=round(df_test['loan_amnt']*df_test['sub_grade'],5)



df_train['hhh']=round(df_train['installment']*df_train['sub_grade'],5)

df_test['hhh']=round(df_test['installment']*df_test['sub_grade'],5)



df_train['iii']=round(df_train['annual_inc']*df_train['sub_grade'],5)

df_test['iii']=round(df_test['annual_inc']*df_test['sub_grade'],5)



df_train['jjj']=round(df_train['dti']*df_train['sub_grade'],5)

df_test['jjj']=round(df_test['dti']*df_test['sub_grade'],5)



df_train['kkk']=round(df_train['open_acc']*df_train['sub_grade'],5)

df_test['kkk']=round(df_test['open_acc']*df_test['sub_grade'],5)



df_train['lll']=round(df_train['revol_bal']*df_train['sub_grade'],5)

df_test['lll']=round(df_test['revol_bal']*df_test['sub_grade'],5)



df_train['mmm']=round(df_train['revol_util']*df_train['sub_grade'],5)

df_test['mmm']=round(df_test['revol_util']*df_test['sub_grade'],5)



df_train['nnn']=round(df_train['total_acc']*df_train['sub_grade'],5)

df_test['nnn']=round(df_test['total_acc']*df_test['sub_grade'],5)



df_train['ooo']=round(df_train['tot_cur_bal']*df_train['sub_grade'],5)

df_test['ooo']=round(df_test['tot_cur_bal']*df_test['sub_grade'],5)
df_train[df_train.grade==1].loan_amnt.mean() 
df_train[df_train.loan_condition==1].loan_amnt.mean() # 貸し倒れたローンの平均額
df_train[df_train.loan_condition==0].loan_amnt.mean() # 貸し倒れていないローンの平均額
df_train.describe()
df_test.describe()
f = 'purpose'



df_train[f].value_counts() / len(df_train)
df_test[f].value_counts() / len(df_test)
# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

cats = []

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, df_train[col].nunique())

print(cats)
print(df_train['title'].unique())

#df_train['title'].value_counts() / len(df_train)

print(len(df_test.columns))

print(len(df_train.columns))


#labelエンコーディング

encoder = OrdinalEncoder()

enc_train = encoder.fit_transform(df_train['emp_title'].values)

enc_test = encoder.transform(df_test['emp_title'].values)

df_train = df_train.reset_index()

df_test = df_test.reset_index()

df_train['emp_title_lab']=enc_train.iloc[:,0]

df_test['emp_title_lab']=enc_test.iloc[:,0]

df_train = df_train.set_index("ID")

df_test =df_test.set_index("ID")
#カウントエンコーディング

ce_cal1='emp_title'

ce_summary1 = df_train[ce_cal1].value_counts()

# mapする。

df_train['emp_title_co'] = df_train[ce_cal1].map(ce_summary1)

df_test['emp_title_co'] = df_test[ce_cal1].map(ce_summary1)


#Labelエンコーディング

encoder = OrdinalEncoder()

enc_train = encoder.fit_transform(df_train['title'].values)

enc_test = encoder.transform(df_test['title'].values)

df_train = df_train.reset_index()

df_test = df_test.reset_index()

df_train['title_la']=enc_train.iloc[:,0]

df_test['title_la']=enc_test.iloc[:,0]

df_train = df_train.set_index("ID")

df_test =df_test.set_index("ID")

#カウントエンコーディング

ce_cal2='title'

ce_summary2 = df_train[ce_cal2].value_counts()

# mapする。

df_train['title_co'] = df_train[ce_cal2].map(ce_summary2)

df_test['title_co'] = df_test[ce_cal2].map(ce_summary2)
df_train['NaN']=df_train.isnull().sum(axis=1)

df_test['NaN']=df_test.isnull().sum(axis=1)

df_train['NaN']=df_train["NaN"].fillna(0) 

df_test['NaN']=df_test["NaN"].fillna(0)

#df_train['NaN_%']=df_train["NaN"]/(len(df_train.columns)-1)*100

#df_test['NaN_%']=df_test["NaN"]/(len(df_train.columns)-1)*100

#df_train['NaN_%']=df_train["NaN_%"].replace([np.inf, -np.inf,np.nan], 0)

#df_test['NaN_%']=df_test["NaN_%"].replace([np.inf, -np.inf,np.nan], 0)
#行の削除

"""

df_train=df_train[df_train['title']<= 13]

df_test=df_test[df_test['title']<= 13]

df_train=df_train[df_train['inq_last_6mths']<= 5]

df_test=df_test[df_test['inq_last_6mths']<= 5]

df_train=df_train[df_train['eee']<= 100000]

df_test=df_test[df_test['eee']<= 100000]



#df_train=df_train.drop('bbb',axis=1)

#df_test=df_test.drop('bbb',axis=1)

#df_train=df_train.drop('ccc',axis=1)

#df_test=df_test.drop('ccc',axis=1)

#df_train=df_train.drop('fff',axis=1)

#df_test=df_test.drop('fff',axis=1)

#df_train=df_train.drop('jjj',axis=1)

#df_test=df_test.drop('jjj',axis=1)

#df_train=df_train.drop('iii',axis=1)

#df_test=df_test.drop('iii',axis=1)

#df_train=df_train.drop('close',axis=1)

#df_test=df_test.drop('close',axis=1)

#df_train=df_train.drop('emp_length_iii',axis=1)

#df_test=df_test.drop('emp_length_iii',axis=1)



an_max=max(df_train['annual_inc'])

df_train.annual_inc.loc[(df_train['annual_inc']>an_max)]=an_max

df_test.annual_inc.loc[(df_test['annual_inc']>an_max)]=an_max

lo_max=max(df_train['loan_amnt'])

df_train.loan_amnt.loc[(df_train['loan_amnt']>lo_max)]=lo_max

df_test.loan_amnt.loc[(df_test['loan_amnt']>lo_max)]=lo_max

to_max=max(df_train['tot_coll_amt'])

df_train.tot_coll_amt.loc[(df_train['tot_coll_amt']>to_max)]=to_max

df_test.tot_coll_amt.loc[(df_test['tot_coll_amt']>to_max)]=to_max

df_train = df_train[df_train.revol_bal <= 1500000]

df_test = df_test[df_test.revol_bal <= 1500000]

df_train = df_train[df_train.lll <= 20000000]

df_test = df_test[df_test.lll <= 20000000]

"""

#lo_max=max(df_train['loan_amnt'])

#df_train.loan_amnt.loc[(df_train['loan_amnt']>lo_max)]=lo_max

#df_test.loan_amnt.loc[(df_test['loan_amnt']>lo_max)]=lo_max

df_train=df_train.drop("pub_rec",axis=1)

df_test=df_test.drop("pub_rec",axis=1)

df_train=df_train.drop("annual_inc",axis=1)

df_test=df_test.drop("annual_inc",axis=1)

#drop_col=['lll','tot_coll_amt','Population (million)','purpose','no_loan2','delinq_2yrs','ooo','annual_inc',

#         'no_loan4','installment','Gross State Product','ggg_mths_since_last_delinq','no_loan','nnn','eee',

#         'title','mths_since_last_major_derog','open_acc']

#drop_col=['no_loan2','delinq_2yrs','ooo','annual_inc',

#         'no_loan4','installment','Gross State Product','ggg_mths_since_last_delinq','no_loan','nnn','eee',

#         'title','mths_since_last_major_derog','open_acc']

#df_train=df_train.drop(drop_col,axis=1)

#df_test=df_test.drop(drop_col,axis=1)
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test

col='title'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

X_train['title']=enc_train

X_test['title']=enc_test
col='emp_title'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

X_train['emp_title']=enc_train

X_test['emp_title']=enc_test
col='zip_code'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

X_train['zip_code']=enc_train

X_test['zip_code']=enc_test
col='addr_state'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

X_train['addr_state']=enc_train

X_test['addr_state']=enc_test
col='purpose'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

X_train['purpose']=enc_train

X_test['purpose']=enc_test
col='application_type'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

X_train['application_type_ta']=enc_train

X_test['application_type_ta']=enc_test

X_train['Yaxis'].isnull().sum()
X_train['Yaxis']=X_train['Yaxis'].replace([np.inf, -np.inf,np.nan], -9999)

X_test['Yaxis']=X_test['Yaxis'].replace([np.inf, -np.inf,np.nan], -9999)

X_train['Yaxis'].astype(str)

X_test['Yaxis'].astype(str)

col='Yaxis'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

X_train['Yaxis_ta']=enc_train

X_test['Yaxis_ta']=enc_test

X_train['Yaxis'].astype(int)

X_test['Yaxis'].astype(int)
#無限大やNULL値の置換

X_train=X_train.replace([np.inf, -np.inf,np.nan], -9999)

X_test=X_test.replace([np.inf, -np.inf,np.nan], -9999)

#X_train.fillna(X_train.median(), inplace=True)

#X_test.fillna(X_train.median(), inplace=True)
## CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

scores = []

y_pred_test=np.zeros(len(X_test))

df = pd.DataFrame(index=[], columns=[])

df['feature']=X_train.columns

n=10

for i in range(n):

    X_train_,X_val,y_train_,y_val=train_test_split(X_train,y_train,test_size=0.3,random_state=i*10)   



    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1,

              importance_type='split', learning_rate=0.05, max_depth=-1,

               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

               n_estimators=100, n_jobs=-1, num_leaves=50, objective=None,

               random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,

               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])



    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    df[i]=Series(clf.booster_.feature_importance(importance_type='gain'))

    y_pred_test+=clf.predict_proba(X_test)[:,1]



df['ave']=df.mean(axis=1)

df['std']=df.std(axis=1)

df=df.sort_values('ave',ascending=False)

ykai=y_pred_test/n

    
print(scores)

print(df)

print(ykai)


## CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

scores_xg=[]

y_pred_test_xg=np.zeros(len(X_test))

df_xg = pd.DataFrame(index=[], columns=[])

df_xg['feature']=X_train.columns

n=10

for i in range(n):

    X_train_,X_val,y_train_,y_val=train_test_split(X_train,y_train,test_size=0.3,random_state=i*10)  



    xg=xgb.XGBClassifier()

    

    xg.fit(X_train_, y_train_,early_stopping_rounds=100, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred_xg = xg.predict_proba(X_val)[:,1]

    score_xg = roc_auc_score(y_val, y_pred_xg)

    print(score_xg)

    scores_xg.append(score_xg)

    y_pred_test_xg+=xg.predict_proba(X_test)[:,1]



ykai_xg=y_pred_test_xg/n

## CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

scores_cb = []

y_pred_test_cb=np.zeros(len(X_test))

df_cb = pd.DataFrame(index=[], columns=[])

df_cb['feature']=X_train.columns

n=10

for i in range(n):

    X_train_,X_val,y_train_,y_val=train_test_split(X_train,y_train,test_size=0.3,random_state=i*10)   



    cb =  catboost.CatBoostClassifier(eval_metric='AUC')    

    cb.fit(X_train_, y_train_, early_stopping_rounds=200,eval_set=[(X_val, y_val)])



    y_pred_cb = cb.predict_proba(X_val)[:,1]

    #score_cb = metrics.roc_auc_score(y_val, y_pred)

    #scores_cb.append(score_cb)

    y_pred_test_cb+=cb.predict_proba(X_test)[:,1]



ykai_cb=y_pred_test_cb/n
"""

## CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

scores_gd = []

y_pred_test_gd=np.zeros(len(X_test))

df_gd = pd.DataFrame(index=[], columns=[])

df_gd['feature']=X_train.columns

n=5

for i in range(n):

    X_train_,X_val,y_train_,y_val=train_test_split(X_train,y_train,test_size=0.3,random_state=i*10)   



    gd =   GradientBoostingClassifier()  

    gd.fit(X_train_, y_train_)



    y_pred_gd = gd.predict_proba(X_val)[:,1]

    score_gd = roc_auc_score(y_val, y_pred_gd)

    print(score_gd)

    scores_gd.append(score_gd)

    y_pred_test_gd+=gd.predict_proba(X_test)[:,1]



ykai_gd=y_pred_test_gd/n



"""
print(ykai)

print(ykai_xg)

print(ykai_cb)

#print(ykai_gd)
#all=(ykai+ykai_xg+ykai_cb)

y_pred=(ykai+ykai_xg+ykai_cb)/3

#y_pred
# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv('../input/homework-for-students3/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')
submission.head()