# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ins = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')
ins["NEW_DELAY"] = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"] 
ins.head()
a=ins.groupby('SK_ID_CURR').NEW_DELAY.apply(lambda x: pd.Series([(x < 0).sum() / (x >= 0).sum()])).unstack()
a.head()
#df = df.replace([np.nan],'XNA')
#df = df.replace([np.inf, -np.inf], np.nan)
ins.head()
ins[ins["SK_ID_CURR"] == 100001]

a.isnull().sum()
#df.columns = df.columns.astype(str)
b=ins.groupby('SK_ID_CURR').NEW_DELAY.apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack()
ins["SK_ID_CURR"].nunique()
c = a.reset_index(col_level=None)
c.drop('SK_ID_CURR',axis=1,inplace=True)
d=pd.concat([ins,c],axis=1)
d.head()

pd.set_option('display.max_columns', None);
#pd.set_option('display.max_rows', None);
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_colwidth', None)

df=pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')



df.head()
a=ins.groupby('SK_ID_CURR').NEW_DELAY.apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack()
a.head()
def installments(num_rows=None):
    pd.options.mode.chained_assignment = None
    df = pd.read_csv("../input/home-credit-default-risk/installments_payments.csv")
    df["NEW_DELAY"] = df["DAYS_INSTALMENT"] - df["DAYS_ENTRY_PAYMENT"] # pozitif ise erken ödemiş negatif ise geç ödemiş ( <0 ise 0, >= 0 ise 1 diyelim.)
    
    df['NEW_FLAG_DELAY'] = df['NEW_DELAY'].apply(lambda x : 1 if x < 0 else 0)
    df['NEW_RATIO_DELAY'] = df[['SK_ID_PREV','NEW_FLAG_DELAY']].groupby('SK_ID_PREV')['NEW_FLAG_DELAY'].transform(lambda x : x.sum() / x.count())
    
    df["NEW_PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]
    
    #df["NEW_"] = df.groupby('SK_ID_CURR').NEW_DELAY.apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack()
    
    df["NUM_INSTALMENT_VERSION"] = df["NUM_INSTALMENT_VERSION"].astype("object")
    df[(df["NUM_INSTALMENT_VERSION"] != 1) & (df["NUM_INSTALMENT_VERSION"] != 0) & (df["NUM_INSTALMENT_VERSION"] != 2) & (df["NUM_INSTALMENT_VERSION"] != 3)]['NUM_INSTALMENT_VERSION'] = 4
    
    cat_features = list(df.select_dtypes(['object']).columns)
    df = pd.get_dummies(df, columns= cat_features,drop_first=True)
    
    
    agg1 = {'SK_ID_CURR': ['count','max'],#how many monts does a customer paid.
           'NEW_DELAY': ['max', 'min', 'mean','std', 'sum'],
           'NUM_INSTALMENT_NUMBER':['min','max'], #her bir eski kredi için min ve max değerleri
           'DAYS_INSTALMENT':['max','min','std'], # bu aggden sonra belli bir sürenin üstündekileri 1 ve 0 olarak 2 ye ayırabiliriz.
           'NEW_PAYMENT_DIFF': ['max', 'mean', 'std', 'min','sum'],
           'AMT_INSTALMENT': ['max', 'mean', 'sum', 'min', 'std'],
           'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
           'DAYS_ENTRY_PAYMENT': ['max', 'min', 'std'],
           "NUM_INSTALMENT_VERSION_1.0":["sum"],
           "NUM_INSTALMENT_VERSION_2.0":["sum"],
           "NUM_INSTALMENT_VERSION_3.0":["sum"],
           "NUM_INSTALMENT_VERSION_4.0":["sum"]
            }
    
    
    
    Installments_agg = df.groupby(['SK_ID_PREV']).agg(agg1)
    
    Installments_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in Installments_agg.columns.tolist()])
    
    Installments_agg['NEW_DAYS_INSTALMENT_NUMBER']=Installments_agg['DAYS_INSTALMENT_MAX']-Installments_agg['DAYS_INSTALMENT_MIN'] #toplam kaç günlük ödemesi var bilgisi. 
    
    Installments_agg['NEW_AMT_INSTALMENT_DIFF']=Installments_agg['AMT_INSTALMENT_MAX']-Installments_agg['AMT_INSTALMENT_MIN']
    
    
    
    agg2= {'SK_ID_CURR_COUNT':['min', 'max'],#Farklı kredilerde kaçar ay ödeme yapılmış.
           'SK_ID_CURR_MAX':['min', 'max'],
           'NEW_DELAY_MAX':['min', 'max', 'mean'],#Farklı kredilerde maximum geciktirme miktarının minimumu maximumu ve ortalaması.
           'NEW_DELAY_MIN':['min', 'max', 'mean'],# aynısının min için olanı
           'NEW_DELAY_MEAN':['min', 'max', 'mean'],# xd
           'NEW_DELAY_STD':['min', 'max', 'mean'],
           'NEW_DELAY_SUM':['min', 'max', 'mean', 'sum', 'std'],
           'NUM_INSTALMENT_NUMBER_MIN':['min','max','mean'], # 1 olanlar bitmiş krediler YENİ DEĞİŞKEN OLUŞTUR.
           'NUM_INSTALMENT_NUMBER_MAX':['min','max','mean','sum'],#farklı krediler için maksimum taksit sayısı. sum=toplam kaç aylık kredi almış.
           'NEW_DAYS_INSTALMENT_NUMBER':['min','max','std'], #farklı krediler için kaç günlük krediler ödenmiş bilgisi.
           'DAYS_INSTALMENT_STD':['min','max','std'], # burdan sonrası için düşünemedim beynim çalışmıyor.
           'DAYS_INSTALMENT_MIN':['std','min','max'],
           'DAYS_INSTALMENT_MAX':['std','min','max'],
           'NEW_PAYMENT_DIFF_MAX':['min', 'max', 'mean',"std"],
           'NEW_PAYMENT_DIFF_MEAN':['min', 'max', 'mean',"std"],
           'NEW_PAYMENT_DIFF_SUM':['min', 'max', 'mean',"std"],
           'NEW_PAYMENT_DIFF_STD':['min', 'max', 'mean',"std"],
           'NEW_PAYMENT_DIFF_MIN':['min', 'max', 'mean',"std"],
           'AMT_INSTALMENT_MAX':['min', 'max', 'mean',"sum"],
           'AMT_INSTALMENT_MEAN':['min', 'max', 'mean',"sum"],
           'AMT_INSTALMENT_SUM':['min', 'max', 'mean',"sum"],
           'AMT_INSTALMENT_STD':['min', 'max', 'mean',"sum"],
           'AMT_INSTALMENT_MIN':['min', 'max', 'mean',"sum"],
           'NEW_AMT_INSTALMENT_DIFF':['min','max','mean',"sum"],
           'AMT_PAYMENT_MIN':['min', 'max', 'mean',"std","sum"],
           'AMT_PAYMENT_MAX':['min', 'max', 'mean',"std","sum"],
           'AMT_PAYMENT_MEAN':['min', 'max', 'mean',"std","sum"],
           'AMT_PAYMENT_STD':['min', 'max', 'mean',"std","sum"],
           'AMT_PAYMENT_SUM':['min', 'max', 'mean',"std","sum"],
           'DAYS_ENTRY_PAYMENT_MIN':['min', 'max', 'mean'],
           'DAYS_ENTRY_PAYMENT_STD':['min', 'max', 'mean'],
           'DAYS_ENTRY_PAYMENT_MAX':['min', 'max', 'mean'],
           'NUM_INSTALMENT_VERSION_1.0_SUM':['sum'],
           'NUM_INSTALMENT_VERSION_2.0_SUM':['sum'],
           'NUM_INSTALMENT_VERSION_3.0_SUM':['sum'],
           'NUM_INSTALMENT_VERSION_4.0_SUM':['sum']
    }
    
    Installments_agg2=Installments_agg.groupby('SK_ID_CURR_MAX').agg(agg2)
    Installments_agg2.index.names = ['SK_ID_CURR']
    
    """
    a = df.groupby('SK_ID_CURR').NEW_DELAY.apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack()
    
    a = a.reset_index(col_level=None)
    a.drop('SK_ID_CURR',axis=1,inplace=True)
    a = a[]
    Installments_agg2 =pd.concat([Installments_agg2,a],axis=1)
    
    
    """
    
    Installments_agg2.columns = pd.Index("INSTAL_" + e[0] + "_" + e[1].upper() for e in Installments_agg2.columns.tolist())
    return(df)
ins.head()
newDf=installments()

newDf.head()
newDf[newDf["SK_ID_PREV"] == 2085231]
newDf.isnull().sum()
newDf.shape
450953-339587

pd.options.mode.chained_assignment = None
pos = pd.read_csv("../input/home-credit-default-risk/POS_CASH_balance.csv")
pos["NAME_CONTRACT_STATUS"].value_counts()


b = ["Demand","Returned to the store","Approved","Amortized debt","Canceled","XNA"]
pos["NAME_CONTRACT_STATUS"] = pos["NAME_CONTRACT_STATUS"].replace(b, 'Others')
pos["NAME_CONTRACT_STATUS"].head()
pos.info()
def Pos_Cash(num_rows=None):
    df=pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')
    
    df['NEW_ADJOURNMENT']=df['SK_DPD']-df['SK_DPD_DEF']
    
    
    b = ["Demand","Returned to the store","Approved","Amortized debt","Canceled","XNA"]
    df["NAME_CONTRACT_STATUS"].replace(b, 'Others',inplace=True)
    
    
    
    #Ohe 
    cat_features = list(df.select_dtypes(['object']).columns)
    df = pd.get_dummies(df, columns= cat_features, dummy_na= True)
    
    
    agg={
    'MONTHS_BALANCE': ['max',"min"],
    'SK_DPD': ['max', 'mean',"std"],
    'SK_DPD_DEF': ['max', 'mean',"std"],
    'CNT_INSTALMENT':['min','mean','max'],
    'CNT_INSTALMENT_FUTURE':['mean','min','max'],
    'SK_ID_CURR':['max','size'],
    'NEW_ADJOURNMENT':['max','mean',"std"],
    'NAME_CONTRACT_STATUS_Active':['sum'],
    'NAME_CONTRACT_STATUS_Completed':['sum'],
    'NAME_CONTRACT_STATUS_Signed':['sum'],
    'NAME_CONTRACT_STATUS_Others':['sum']
    
    }
    
    
    pos_agg = df.groupby(['SK_ID_PREV']).agg(agg)
    
    
    pos_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    
    pos_agg["NEW_PAID_MONTH"] = pos_agg["CNT_INSTALMENT_MAX"] - pos_agg["CNT_INSTALMENT_FUTURE_MIN"]#ödenen ay sayısı
    
    agg2={
        "MONTHS_BALANCE_MAX":["min","max","mean"],
        "MONTHS_BALANCE_MIN":["min","max","mean"],
        "SK_DPD_MAX":["max","mean","min"],
        "SK_DPD_MEAN":["max","mean","min"],
        "SK_DPD_STD":["max","mean","min","std"],
        "SK_DPD_DEF_MAX":["max","mean","min"],
        "SK_DPD_DEF_MEAN":["max","mean","min"],
        "SK_DPD_DEF_STD":["max","mean","min"],
        "CNT_INSTALMENT_MIN":["max","mean","min"],
        "CNT_INSTALMENT_MEAN":["max","mean","min"],
        "CNT_INSTALMENT_MAX":["max","mean","min"],
        "CNT_INSTALMENT_FUTURE_MEAN":["max","mean","min"],
        "CNT_INSTALMENT_FUTURE_MIN":["max","mean","min"],
        "CNT_INSTALMENT_FUTURE_MAX":["max","mean","min"],
        "SK_ID_CURR_MAX":["max","min"],
        "SK_ID_CURR_SIZE":["max","min"],
        "NEW_ADJOURNMENT_MAX":["max","mean","min"],
        "NEW_ADJOURNMENT_MEAN":["max","mean","min"],
        "NEW_ADJOURNMENT_STD":["max","mean","min"],
        "NAME_CONTRACT_STATUS_Active_SUM":["max","min","sum"],
        'NAME_CONTRACT_STATUS_Signed_SUM':["max","min","sum"],
        'NAME_CONTRACT_STATUS_Completed_SUM':["max","min","sum"],
        'NAME_CONTRACT_STATUS_Others_SUM':["max","min","sum"]
        
    }
    
    pos_agg2 = pos_agg.groupby(["SK_ID_CURR_MAX"]).agg(agg2)
    pos_agg2.index.names = ['SK_ID_CURR']
    
    pos_agg2.columns = pd.Index(["POS" + "_" + e[0] + "_" + e[1].upper() for e in pos_agg2.columns.tolist()])
    
    return (pos_agg2)

Pos_Cash().head()




