# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from ycimpute.imputer import EM

import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# get the number of missing data points per column
def missing_data(df):
    total = df.isnull().sum()
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    types = df.dtypes
    unieq = df.select_dtypes(include = [object,float,int]).apply(pd.Series.nunique, axis = 0)
    df_mis = pd.concat([total, percent ,unieq,types], axis=1, keys=['Sum_Mis', 'Per_Mis','Unieq', 'Types'])
    
    return df_mis.head(len(df_mis))
def iqr(df):
    IQR = df.describe().T
    IQR['lower'] = IQR['25%']-1.5*(IQR['75%']-IQR['25%'])
    IQR['upper'] = IQR['75%']+1.5*(IQR['75%']-IQR['25%'])
    upper_count = []
    i = 0
    for col in df:
        upper_count.append(len(df[df[col] > IQR.iloc[i][-1]]))
        i +=1
    
    lower_count = []
    i = 0
    for col in df:
        lower_count.append(len(df[df[col] < IQR.iloc[i][-2]]))
        i +=1
    
    IQR['lower_count'] = lower_count
    IQR['upper_count'] = upper_count
    IQR['out_percent'] = (IQR['upper_count'] + IQR['lower_count'])/IQR['count']
    return IQR
#Read the installments_payments.csv
ins = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')
ins['NEW_DAYS_PAID_EARLIER'] = ins['DAYS_INSTALMENT']-ins['DAYS_ENTRY_PAYMENT']

# Her bir taksit ödemesinin gec olup olmama durumu 1: gec ödedi 0: erken ödemeyi temsil eder
ins['NEW_NUM_PAID_LATER'] = ins['NEW_DAYS_PAID_EARLIER'].map(lambda x: 1 if x<0 else 0)

# Agrregation ve degisken tekillestirme
agg_list = {'NUM_INSTALMENT_VERSION':['nunique'],
               'NUM_INSTALMENT_NUMBER':'max',
               'DAYS_INSTALMENT':['min','max'],
               'DAYS_ENTRY_PAYMENT':['min','max'],
               'AMT_INSTALMENT':['min','max','sum','mean'],
               'AMT_PAYMENT':['min','max','sum','mean'],
               'NEW_DAYS_PAID_EARLIER':'mean',
               'NEW_NUM_PAID_LATER':'sum'}


ins_agg = ins.groupby('SK_ID_PREV').agg(agg_list)


# Multi index problemi cözümü
ins_agg.columns = pd.Index(["INS_" + e[0] + '_' + e[1].upper() for e in ins_agg.columns.tolist()])

# drop variables 
ins_agg.drop(['INS_DAYS_INSTALMENT_MIN','INS_DAYS_INSTALMENT_MAX','INS_DAYS_ENTRY_PAYMENT_MIN','INS_DAYS_ENTRY_PAYMENT_MAX'],axis=1,inplace=True)

# Kredi ödeme yüzdesi ve toplam kalan borc
ins_agg['INS_NEW_PAYMENT_PERC'] = ins_agg['INS_AMT_PAYMENT_SUM'] / ins_agg['INS_AMT_INSTALMENT_SUM']
ins_agg['INS_NEW_PAYMENT_DIFF'] = ins_agg['INS_AMT_INSTALMENT_SUM'] - ins_agg['INS_AMT_PAYMENT_SUM']
    
ins_agg.reset_index(inplace = True)


# NaN values fill with EM()
# def mlearn(df):   
#     df_num = df.select_dtypes(include=["float64", "int64"])
#     from ycimpute.imputer import EM
#     df_new = EM().complete(np.array(df_num))
#     df_new = pd.DataFrame(pos_new, columns=list(df_num))
#     return df_new

# ins_agg = mlearn(ins_agg)

# missing_data(ins_agg)
pos = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')
# Kategorik Degiskenimizi Dummy Degiskenine Dönüstürme
pos = pd.get_dummies(pos, columns=['NAME_CONTRACT_STATUS'], dummy_na = True)
# Aggregation Islemi - Tekillestirme
agg_list = {
    'MONTHS_BALANCE':['min','max'],
    'CNT_INSTALMENT':['min','max'],
    'CNT_INSTALMENT_FUTURE':['min','max'],
    'SK_DPD':['max','mean'],
    'SK_DPD_DEF':['max','mean'],
    'NAME_CONTRACT_STATUS_Active':'sum',
    'NAME_CONTRACT_STATUS_Amortized debt':'sum',
    'NAME_CONTRACT_STATUS_Approved':'sum',
    'NAME_CONTRACT_STATUS_Canceled':'sum',
    'NAME_CONTRACT_STATUS_Completed':'sum',
    'NAME_CONTRACT_STATUS_Demand':'sum',
    'NAME_CONTRACT_STATUS_Returned to the store':'sum',
    'NAME_CONTRACT_STATUS_Signed':'sum',
    'NAME_CONTRACT_STATUS_XNA':'sum',
    'NAME_CONTRACT_STATUS_nan':'sum'
}

pos_agg = pos.groupby('SK_ID_PREV').agg(agg_list)

# Multilayer index'i tek boyutlu index'e dönüstürme
pos_agg.columns= pd.Index(["POS_" + e[0] + '_' + e[1].upper() for e in pos_agg.columns.tolist()])

# SK_DPD kac kredide 0 olma durumu (SK_DPD MAX alacagiz 0 durumunu veriyor) 
# SK_DPD_DEF (SK_DPD_DEF_MAX sifir olma durumunu veriyor)
# CNT_INSTALMENT_FUTURE_MIN==0 oldugunda NAME_CONTRACT_STATUS_Completed_SUM==0 olma durumu 

pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME']= (pos_agg['POS_CNT_INSTALMENT_FUTURE_MIN']==0) & (pos_agg['POS_NAME_CONTRACT_STATUS_Completed_SUM']==0)


# 1:kredi zamaninda kapanmamis 0:kredi zamaninda kapanmis

pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME']=pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME'].astype(int)

pos_agg.drop(['POS_NAME_CONTRACT_STATUS_Approved_SUM',
   'POS_NAME_CONTRACT_STATUS_Amortized debt_SUM',
   'POS_NAME_CONTRACT_STATUS_Canceled_SUM',
   'POS_NAME_CONTRACT_STATUS_Returned to the store_SUM',
   'POS_NAME_CONTRACT_STATUS_Signed_SUM',
   'POS_NAME_CONTRACT_STATUS_XNA_SUM',
   'POS_NAME_CONTRACT_STATUS_nan_SUM'],axis=1,inplace=True)

pos_agg.reset_index(inplace = True)


# # NaN values fill with EM()
# def mlearn(df):   
#     df_num = df.select_dtypes(include=["float64", "int64"])
#     from ycimpute.imputer import EM
#     df_new = EM().complete(np.array(df_num))
#     df_new = pd.DataFrame(pos_new, columns=list(df_num))
#     return df_new

# pos_agg = mlearn(pos_agg)

# missing_data(pos_agg)

CCB = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv')

CCB = pd.get_dummies(CCB, columns= ['NAME_CONTRACT_STATUS'] )  # artik tumu sayisal 

dropthis = ['NAME_CONTRACT_STATUS_Approved', 'NAME_CONTRACT_STATUS_Demand',
    'NAME_CONTRACT_STATUS_Refused', 'NAME_CONTRACT_STATUS_Sent proposal','NAME_CONTRACT_STATUS_Signed' ]

CCB['NAME_CONTRACT_STATUS_Active']=CCB['NAME_CONTRACT_STATUS_Active'].astype(int)
CCB['NAME_CONTRACT_STATUS_Completed']=CCB['NAME_CONTRACT_STATUS_Completed'].astype(int)

CCB = CCB.drop(dropthis, axis=1)

agg_list = {
'SK_ID_CURR' :['mean'],
'MONTHS_BALANCE':["sum","mean"], 
'AMT_BALANCE':["sum","mean","min","max"],
'AMT_CREDIT_LIMIT_ACTUAL':["sum","mean"],
'AMT_DRAWINGS_ATM_CURRENT':["sum","mean","min","max"],
'AMT_DRAWINGS_CURRENT':["sum","mean","min","max"], 
'AMT_DRAWINGS_OTHER_CURRENT':["sum","mean","min","max"],
'AMT_DRAWINGS_POS_CURRENT':["sum","mean","min","max"], 
'AMT_INST_MIN_REGULARITY':["sum","mean","min","max"],
'AMT_PAYMENT_CURRENT':["sum","mean","min","max"], 
'AMT_PAYMENT_TOTAL_CURRENT':["sum","mean","min","max"],
'AMT_RECEIVABLE_PRINCIPAL':["sum","mean","min","max"], 
'AMT_RECIVABLE':["sum","mean","min","max"], 
'AMT_TOTAL_RECEIVABLE':["sum","mean","min","max"],
'CNT_DRAWINGS_ATM_CURRENT':["sum","mean"], 
'CNT_DRAWINGS_CURRENT':["sum","mean","max"],
'CNT_DRAWINGS_OTHER_CURRENT':["mean","max"], 
'CNT_DRAWINGS_POS_CURRENT':["sum","mean","max"],
'CNT_INSTALMENT_MATURE_CUM':["sum","mean","max","min"],
'SK_DPD':["sum","mean","max"], 
'SK_DPD_DEF':["sum","mean","max"],
'NAME_CONTRACT_STATUS_Active':["sum","mean","min","max"],
'NAME_CONTRACT_STATUS_Completed':["sum","mean","min","max"], 
}


CCB_agg = CCB.groupby('SK_ID_PREV').agg(agg_list)


CCB_agg.columns = pd.Index(['CCB_' + e[0] + "_" + e[1].upper() for e in CCB_agg.columns.tolist()])

CCB_agg.reset_index(inplace = True)


# NaN values fill with EM()

# missing_data(CCB_agg)
pre_app = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')

#% 99 olan iki stunu (RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED) drop ediyoruz
pre_app.drop(pre_app[['RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED']],axis=1,inplace=True) 


#pre_app missin data nan to 0
df1 = pre_app[(pre_app.AMT_ANNUITY.isna() == True) & (pre_app.AMT_APPLICATION == 0 )].fillna(0)
df2 = pre_app[(pre_app.AMT_ANNUITY.isna() == True) & (pre_app.AMT_APPLICATION != 0 )]
df3 = pre_app[(pre_app.AMT_ANNUITY.isna() == False) & (pre_app.AMT_APPLICATION == 0 )]
df4 = pre_app[(pre_app.AMT_ANNUITY.isna() == False) & (pre_app.AMT_APPLICATION != 0 )]
pre_app = pd.concat([df1, df2, df3, df4])


pre_app['NAME_TYPE_SUITE'].replace(0 , 'Unaccompre_appnied', inplace = True)
pre_app['NAME_TYPE_SUITE'].replace(np.nan , 'Unaccompre_appnied', inplace = True)
pre_app['NAME_TYPE_SUITE'].value_counts()

# yanlis degeri nan ile doldurma
pre_app['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
pre_app['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
pre_app['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
pre_app['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
pre_app['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

# hafta gunleri ve calisma saatleri degistirildi
pre_app["WEEKDAY_APPR_PROCESS_START"] = pre_app["WEEKDAY_APPR_PROCESS_START"].replace(['MONDAY','TUESDAY', 'WEDNESDAY','THURSDAY','FRIDAY'], 'WEEK_DAY')
pre_app["WEEKDAY_APPR_PROCESS_START"] = pre_app["WEEKDAY_APPR_PROCESS_START"].replace(['SATURDAY', 'SUNDAY'], 'WEEKEND')
pre_app["HOUR_APPR_PROCESS_START"] = pre_app["HOUR_APPR_PROCESS_START"].replace([8,9,10,11,12,13,14,15,16,17], 'working_hours')
pre_app["HOUR_APPR_PROCESS_START"] = pre_app["HOUR_APPR_PROCESS_START"].replace([18,19,20,21,22,23,0,1,2,3,4,5,6,7], 'off_hours')


# DAYS_DECISION değeri 1 yıldan küçük olanlara 1, büyük olanlara 0 değeri verildi.
pre_app["DAYS_DECISION"] = [1 if abs(i/(12*30)) <=1 else 0 for i in pre_app.DAYS_DECISION]


# "NAME_TYPE_SUITE"  değişkeninin alone ve not_alone olarak iki kategoriye ayrılması
pre_app["NAME_TYPE_SUITE"] = pre_app["NAME_TYPE_SUITE"].replace('Unaccompanied', 'alone')
pre_app["NAME_TYPE_SUITE"] = pre_app["NAME_TYPE_SUITE"].replace('Unaccompre_appnied', 'alone')
b = ['Family', 'Spouse, partner', 'Children', 'Other_B', 'Other_A', 'Group of people']
pre_app["NAME_TYPE_SUITE"] = pre_app["NAME_TYPE_SUITE"].replace(b, 'not_alone')

# "NAME_GOODS_CATEGORY"  değişkenindeki bu değerler others olarak kategorize edilecek
a = [
    'Auto Accessories', 'Jewelry', 'Homewares', 'Medical Supplies', 'Vehicles', 'Sport and Leisure', 
    'Gardening', 'Other', 'Office Appliances', 'Tourism', 'Medicine', 'Direct Sales', 'Fitness',
    'Additional Service','Education', 'Weapon', 'Insurance', 'House Construction', 'Animals'
]
pre_app["NAME_GOODS_CATEGORY"] = pre_app["NAME_GOODS_CATEGORY"].replace(a, 'others')

# "NAME_SELLER_INDUSTRY"  değişkenindeki bu değerler others olarak kategorize edilecek
a = ['Auto technology', 'Jewelry', 'MLM partners', 'Tourism'] 
pre_app["NAME_SELLER_INDUSTRY"] = pre_app["NAME_SELLER_INDUSTRY"].replace(a, 'other')

# İstenilen krecinin verilen krediye oranı içeren değişkeni türetir
pre_app["NEW_LOAN_RATE"] = pre_app.AMT_APPLICATION/pre_app.AMT_CREDIT

# Ödeme gününü geciktirmiş mi bunu gösteren churn_prev değişkeni türetilir.
# 1= geciktirmiş, 0 = geciktirmemiş, NaN = boş değer
k = pre_app.DAYS_LAST_DUE_1ST_VERSION - pre_app.DAYS_LAST_DUE
pre_app["DAYS_DUE"] = [1 if i >= 0 else (0 if i < 0  else "NaN") for i in k]

drop_list = ['AMT_DOWN_PAYMENT', 'SELLERPLACE_AREA', 'CNT_PAYMENT', 'PRODUCT_COMBINATION',
             'DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE',
             'DAYS_TERMINATION','NFLAG_INSURED_ON_APPROVAL']
pre_app.drop(drop_list, axis = 1, inplace = True)


category_columns = pre_app.select_dtypes(include = [object]).columns

pre_app = pd.get_dummies(pre_app, columns = category_columns )

agg_list =     {"SK_ID_CURR":['mean'], 
                "AMT_ANNUITY":["max"],
                "AMT_APPLICATION":["min","mean","max"],
                "AMT_CREDIT":["max"], 
                "AMT_GOODS_PRICE":["sum", "mean"],
                "NFLAG_LAST_APPL_IN_DAY":["sum","mean"], 
                "RATE_DOWN_PAYMENT":["sum", "mean"],
                "DAYS_DECISION":["sum"],
                'DAYS_DUE_0' : ["sum", "mean"],
                'DAYS_DUE_1' : ["sum", "mean"],
                'DAYS_DUE_NaN' : ["sum", "mean"],
                "NEW_LOAN_RATE":["sum", "mean", "min", "max"],
                "NAME_CONTRACT_TYPE_Cash loans":["sum", "mean"],
                "NAME_CONTRACT_TYPE_Consumer loans":["sum", "mean"],
                "NAME_CONTRACT_TYPE_Revolving loans":["sum", "mean"],
                "NAME_CONTRACT_TYPE_XNA":["sum", "mean"],
                "WEEKDAY_APPR_PROCESS_START_WEEKEND":["sum", "mean"],
                "WEEKDAY_APPR_PROCESS_START_WEEK_DAY":["sum", "mean"],
                "HOUR_APPR_PROCESS_START_off_hours":["sum", "mean"],
                "HOUR_APPR_PROCESS_START_working_hours":["sum", "mean"],
                "FLAG_LAST_APPL_PER_CONTRACT_N":["sum", "mean"],
                "FLAG_LAST_APPL_PER_CONTRACT_Y":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Building a house or an annex":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Business development":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Buying a garage":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Buying a home":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Buying a new car":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Buying a used car":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Car repairs":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Education":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Everyday expenses":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Furniture":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Gasification / water supply":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Hobby":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Journey":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Medicine":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Money for a third person":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Other":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Payments on other loans":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Refusal to name the goal":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Repairs":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Urgent needs":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_XAP":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_XNA":["sum", "mean"],
                "NAME_CONTRACT_STATUS_Approved":["sum", "mean"],
                "NAME_CONTRACT_STATUS_Canceled":["sum", "mean"],
                "NAME_CONTRACT_STATUS_Refused":["sum", "mean"],
                "NAME_CONTRACT_STATUS_Unused offer":["sum", "mean"],
                "NAME_PAYMENT_TYPE_Cash through the bank":["sum", "mean"],
                "NAME_PAYMENT_TYPE_Cashless from the account of the employer":["sum", "mean"],
                "NAME_PAYMENT_TYPE_Non-cash from your account":["sum", "mean"],
                "NAME_PAYMENT_TYPE_XNA":["sum", "mean"],
                "CODE_REJECT_REASON_CLIENT":["sum", "mean"],
                "CODE_REJECT_REASON_HC":["sum", "mean"],
                "CODE_REJECT_REASON_LIMIT":["sum", "mean"],
                "CODE_REJECT_REASON_SCO":["sum", "mean"],
                "CODE_REJECT_REASON_SCOFR":["sum", "mean"],
                "CODE_REJECT_REASON_SYSTEM":["sum", "mean"],
                "CODE_REJECT_REASON_VERIF":["sum", "mean"],
                "CODE_REJECT_REASON_XAP":["sum", "mean"],
                "CODE_REJECT_REASON_XNA":["sum", "mean"],
                "NAME_TYPE_SUITE_alone":["sum", "mean"],
                "NAME_TYPE_SUITE_not_alone":["sum", "mean"],
                "NAME_CLIENT_TYPE_New":["sum", "mean"],
                "NAME_CLIENT_TYPE_Refreshed":["sum", "mean"],
                "NAME_CLIENT_TYPE_Repeater":["sum", "mean"],
                "NAME_CLIENT_TYPE_XNA":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Audio/Video":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Clothing and Accessories":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Computers":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Construction Materials":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Consumer Electronics":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Furniture":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Mobile":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Photo / Cinema Equipment":["sum", "mean"],
                "NAME_GOODS_CATEGORY_XNA":["sum", "mean"],
                "NAME_GOODS_CATEGORY_others":["sum", "mean"],
                "NAME_PORTFOLIO_Cards":["sum", "mean"],
                "NAME_PORTFOLIO_Cars":["sum", "mean"],
                "NAME_PORTFOLIO_Cash":["sum", "mean"],
                "NAME_PORTFOLIO_POS":["sum", "mean"],
                "NAME_PORTFOLIO_XNA":["sum", "mean"],
                "NAME_PRODUCT_TYPE_XNA":["sum", "mean"],
                "NAME_PRODUCT_TYPE_walk-in":["sum", "mean"],
                "NAME_PRODUCT_TYPE_x-sell":["sum", "mean"],
                "CHANNEL_TYPE_AP+ (Cash loan)":["sum", "mean"],
                "CHANNEL_TYPE_Car dealer":["sum", "mean"],
                "CHANNEL_TYPE_Channel of corporate sales":["sum", "mean"],
                "CHANNEL_TYPE_Contact center":["sum", "mean"],
                "CHANNEL_TYPE_Country-wide":["sum", "mean"],
                "CHANNEL_TYPE_Credit and cash offices":["sum", "mean"],
                "CHANNEL_TYPE_Regional / Local":["sum", "mean"],
                "CHANNEL_TYPE_Stone":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Clothing":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Connectivity":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Construction":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Consumer electronics":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Furniture":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Industry":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_XNA":["sum", "mean"],
                'NAME_SELLER_INDUSTRY_other':["sum", "mean"],
                "NAME_YIELD_GROUP_XNA":["sum", "mean"],
                "NAME_YIELD_GROUP_high":["sum", "mean"],
                "NAME_YIELD_GROUP_low_action":["sum", "mean"],
                "NAME_YIELD_GROUP_low_normal":["sum", "mean"],
                "NAME_YIELD_GROUP_middle":["sum", "mean"],}

pre_agg = pre_app.groupby('SK_ID_PREV').agg(agg_list)

pre_agg.columns = pd.Index(["PREV_" + e[0] + '_' + e[1].upper() for e in pre_agg.columns.tolist()])

pre_agg.reset_index(inplace = True)

display(missing_data(ins_agg))

display(missing_data(pos_agg))

display(missing_data(CCB_agg))

display(missing_data(pre_agg))

display(iqr(ins_agg).round(2))
display(iqr(ins_agg).round(2))
display(iqr(pos_agg).round(2))
display(iqr(CCB_agg).round(2))
display(iqr(pre_agg).round(2))
df_ins_pos = ins_agg.merge(pos_agg, how = 'left', on = 'SK_ID_PREV')
df_ins_pos
df_ins_pos_ccb = df_ins_pos.merge(CCB_agg ,how = 'left', on = 'SK_ID_PREV')
df_ins_pos_ccb
df_all = pre_agg.merge(df_ins_pos_ccb, how = 'left', on = 'SK_ID_PREV' )
df_all