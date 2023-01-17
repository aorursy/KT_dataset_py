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
import seaborn as sns
import matplotlib.pyplot as plt

payment2=pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
pos_cash= pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')

balance=pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
balance.head()

balance.info()

balance.isnull().sum()
liste1=['AMT_BALANCE',
'AMT_DRAWINGS_ATM_CURRENT',
'AMT_DRAWINGS_CURRENT',
'AMT_DRAWINGS_OTHER_CURRENT',
'AMT_DRAWINGS_POS_CURRENT',
'AMT_INST_MIN_REGULARITY',
'AMT_PAYMENT_CURRENT',
'AMT_PAYMENT_TOTAL_CURRENT',
'AMT_RECEIVABLE_PRINCIPAL',
'AMT_RECIVABLE',
'AMT_TOTAL_RECEIVABLE',
'CNT_DRAWINGS_ATM_CURRENT',
'CNT_DRAWINGS_OTHER_CURRENT',
'CNT_DRAWINGS_POS_CURRENT',
'CNT_INSTALMENT_MATURE_CUM',]
plt.figure(figsize = (16,10))
sns.heatmap(balance[liste1].corr(),annot=True,linewidths=0.5)
plt.show()
amt_balance=balance[['AMT_BALANCE','AMT_RECEIVABLE_PRINCIPAL','AMT_RECIVABLE','AMT_TOTAL_RECEIVABLE']]
amt_balance.head(10)

payment2.head()
payment2.info()
payment2.isnull().sum()
payment_li=['NUM_INSTALMENT_VERSION',
'DAYS_INSTALMENT',
'DAYS_ENTRY_PAYMENT',
'AMT_INSTALMENT',
'AMT_PAYMENT']
plt.figure(figsize = (16,10))
sns.heatmap(payment2[payment_li].corr(),annot=True,linewidths=0.5)
plt.show() 
payment=payment2.copy()
payment= payment.dropna()
#payment=payment[['DAYS_INSTALMENT','DAYS_ENTRY_PAYMENT']]
payment.head(10)



from sklearn.preprocessing import StandardScaler
payment.index = payment.iloc[:,0]
payment = payment.iloc[:,0:8]
payment.head()

payment = StandardScaler().fit_transform(payment)
payment[0:5,0:5]
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)  
pca_fit = pca.fit_transform(payment)
b_df = pd.DataFrame(data = pca_fit, 
                          columns = ["bir","ik","uc",'dort','bes'])
b_df.head()
pca.explained_variance_ratio_  

# aciklanan varyans orani. her bilesenin varyansi. yani verinin aciklanma orani, 
# eger yuksekse,  bilesen sayisi arttirilabilir. kac degisken olacagini bu yorumlanaarak yapilabilir.
# genelde 2 yada 3 bilesen olmasi daha uygundur,
pca = PCA().fit(payment)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
#1
payment1=pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
payment1['NEW_DAYS_PAID_EARLIER'] = payment1['DAYS_INSTALMENT']-payment1['DAYS_ENTRY_PAYMENT']  
    # Her bir taksit ödemesinin gec olup olmama durumu 1:gec ödedi    0: erken ödemeyi temsil eder
payment1['NEW_NUM_PAID_LATER'] = payment1['NEW_DAYS_PAID_EARLIER'].map(lambda x: 1 if x<0 else 0)
# Agrregation ve degisken tekillestirme
agg_list = {'NUM_INSTALMENT_VERSION':['nunique'],
               'NUM_INSTALMENT_NUMBER':'max',
               'DAYS_INSTALMENT':['min','max'],
               'DAYS_ENTRY_PAYMENT':['min','max'],
               'AMT_INSTALMENT':['min','max','sum','mean'],
               'AMT_PAYMENT':['min','max','sum','mean'],
               'NEW_DAYS_PAID_EARLIER':'mean',
               'NEW_NUM_PAID_LATER':'sum'}


# Multi index problemi cözümü
payment1_agg = payment1.groupby('SK_ID_PREV').agg(agg_list)
payment1_agg.columns = pd.Index(["INS_" + e[0] + '_' + e[1].upper() for e in payment1_agg.columns.tolist()])
 # drop variables 
payment1_agg.drop(['INS_DAYS_INSTALMENT_MIN',
                   'INS_DAYS_INSTALMENT_MAX',
                   'INS_DAYS_ENTRY_PAYMENT_MIN',
                   'INS_DAYS_ENTRY_PAYMENT_MAX'],axis=1,inplace=True)
 # Kredi ödeme yüzdesi ve toplam kalan borc
payment1_agg['INS_NEW_PAYMENT_PERC'] = payment1_agg['INS_AMT_PAYMENT_SUM'] / payment1_agg['INS_AMT_INSTALMENT_SUM']
payment1_agg['INS_NEW_PAYMENT_DIFF'] = payment1_agg['INS_AMT_INSTALMENT_SUM'] - payment1_agg['INS_AMT_PAYMENT_SUM']
    
agg_list_previous_application = {}
for col in payment1_agg.columns:
    agg_list_previous_application[col] = ['mean',"min","max","sum"]
    
payment1_agg.reset_index(inplace = True) 
agg_list_previous_application
payment1_agg
#2
balance1=pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
balance1 = pd.get_dummies(balance1, columns= ['NAME_CONTRACT_STATUS'] )  # artik tumu sayisal 

dropthis = ['NAME_CONTRACT_STATUS_Approved', 'NAME_CONTRACT_STATUS_Demand',
           'NAME_CONTRACT_STATUS_Refused', 'NAME_CONTRACT_STATUS_Sent proposal',
           'NAME_CONTRACT_STATUS_Signed' ]

balance1 = balance1.drop(dropthis, axis=1)

grp = balance1.groupby(by = ['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index().rename(index = str, columns = {'SK_ID_PREV': 'NUMBER_OF_LOANS_PER_CUSTOMER'})
balance1= balance1.merge(grp, on = ['SK_ID_CURR'], how = 'left')

grp = balance1.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].max().reset_index().rename(index = str, columns = {'CNT_INSTALMENT_MATURE_CUM': 'NUMBER_OF_INSTALMENTS'})
grp1 = grp.groupby(by = ['SK_ID_CURR'])['NUMBER_OF_INSTALMENTS'].sum().reset_index().rename(index = str, columns = {'NUMBER_OF_INSTALMENTS': 'TOTAL_INSTALMENTS_OF_ALL_LOANS'})
balance1 = balance1.merge(grp1, on = ['SK_ID_CURR'], how = 'left')

balance1['INSTALLMENTS_PER_LOAN'] = (balance1['TOTAL_INSTALMENTS_OF_ALL_LOANS']/balance1['NUMBER_OF_LOANS_PER_CUSTOMER']).astype('uint32')


    # Bu fonksiyon, kac defa odemelerin geciktigini hesaplar   # Function to calculate number of times Days Past Due occurred 
def geciken_gun_hesapla(DPD):

     # DPD ile beklenen bir seri: SK_DPD degiskeninin her bir prev_app daki gecmis kredi icin olan degerleri  # DPD is a series of values of SK_DPD for each of the groupby combination 
     # We convert it to a list to get the number of SK_DPD values NOT EQUALS ZERO
    x = DPD.tolist()
    c = 0
    for i,j in enumerate(x):
        if j != 0:
             c += 1  
    return c 

grp = balance1.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV']).apply(lambda x: geciken_gun_hesapla(x.SK_DPD)).reset_index().rename(index = str, columns = {0: 'NUMBER_OF_DPD'})
grp1 = grp.groupby(by = ['SK_ID_CURR'])['NUMBER_OF_DPD'].mean().reset_index().rename(index = str, columns = {'NUMBER_OF_DPD' : 'DPD_COUNT'})

balance1=balance1.merge(grp1, on = ['SK_ID_CURR'], how = 'left')


def f(min_pay, total_pay):

    M = min_pay.tolist()
    T = total_pay.tolist()
    P = len(M)        # P: taksit sayisi
    c = 0 
     # Find the count of transactions when Payment made is less than Minimum Payment 
    for i in range(len(M)):
         if T[i] < M[i]:
            c += 1  
    return (100*c)/P

grp = balance1.groupby(by = ['SK_ID_CURR']).apply(lambda x: f(x.AMT_INST_MIN_REGULARITY, x.AMT_PAYMENT_CURRENT)).reset_index().rename(index = str, columns = { 0 : 'PERCENTAGE_MIN_MISSED_PAYMENTS'})
balance1 = balance1.merge(grp, on = ['SK_ID_CURR'], how = 'left')

grp = balance1.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_ATM_CURRENT' : 'DRAWINGS_ATM'})
balance1 = balance1.merge(grp, on = ['SK_ID_CURR'], how = 'left')


grp = balance1.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_CURRENT' : 'DRAWINGS_TOTAL'})
balance1 = balance1.merge(grp, on = ['SK_ID_CURR'], how = 'left')


balance1['CASH_CARD_RATIO1'] = (balance1['DRAWINGS_ATM']/balance1['DRAWINGS_TOTAL'])*100  # ATM den cektigi nakit / toplam cektigi
del balance1['DRAWINGS_ATM']
del balance1['DRAWINGS_TOTAL']

grp = balance1.groupby(by = ['SK_ID_CURR'])['CASH_CARD_RATIO1'].mean().reset_index().rename(index = str, columns ={ 'CASH_CARD_RATIO1' : 'CASH_CARD_RATIO'})
balance1 = balance1.merge(grp, on = ['SK_ID_CURR'], how = 'left')


grp = balance1.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_CURRENT' : 'TOTAL_DRAWINGS'})
balance1 = balance1.merge(grp, on = ['SK_ID_CURR'], how = 'left')


grp = balance1.groupby(by = ['SK_ID_CURR'])['CNT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'CNT_DRAWINGS_CURRENT' : 'NUMBER_OF_DRAWINGS'})
balance1 = balance1.merge(grp, on = ['SK_ID_CURR'], how = 'left')


balance1['DRAWINGS_RATIO1'] = (balance1['TOTAL_DRAWINGS']/balance1['NUMBER_OF_DRAWINGS'])*100     # yuzdelik degil, genisleme yapmis
del balance1['TOTAL_DRAWINGS']
del balance1['NUMBER_OF_DRAWINGS']


grp = balance1.groupby(by = ['SK_ID_CURR'])['DRAWINGS_RATIO1'].mean().reset_index().rename(index = str, columns ={ 'DRAWINGS_RATIO1' : 'DRAWINGS_RATIO'})
balance1 = balance1.merge(grp, on = ['SK_ID_CURR'], how = 'left')

del balance1['DRAWINGS_RATIO1']

balance1['CC_COUNT'] = balance1.groupby('SK_ID_CURR').size()

balance1_agg = balance1.groupby('SK_ID_CURR').agg({
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
        'INSTALLMENTS_PER_LOAN':["sum","mean","min","max"],

        'NUMBER_OF_LOANS_PER_CUSTOMER':["mean"], 
        'DPD_COUNT':["mean"],
        'PERCENTAGE_MIN_MISSED_PAYMENTS':["mean"], 
        'CASH_CARD_RATIO':["mean"], 
        'DRAWINGS_RATIO':["mean"]})


balance1_agg.columns = pd.Index(['balance1_' + e[0] + "_" + e[1].upper() for e in balance1_agg.columns.tolist()])

balance1_agg.reset_index(inplace = True)
    
balance1_agg
#3
pos_cash1=pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')

    
# Kategorik Degiskenimizi Dummy Degiskenine Dönüstürme
pos_cash1= pd.get_dummies(pos_cash1, columns=['NAME_CONTRACT_STATUS'], dummy_na = True)
# Aggregation Islemi - Tekillestirme
agg_list = {'MONTHS_BALANCE':['min','max'],
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

pos_agg = pos_cash1.groupby('SK_ID_PREV').agg(agg_list)

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

for col in pos_agg.columns:
    agg_list_previous_application[col] = ['mean',"min","max","sum"]

pos_agg.reset_index(inplace = True)     

agg_list_previous_application
pos_agg
#4
df_prev = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')

# "WEEKDAY_APPR_PROCESS_START"  değişkeninin  WEEK_DAY ve WEEKEND olarak iki kategoriye ayrılması

df_prev["WEEKDAY_APPR_PROCESS_START"] = df_prev["WEEKDAY_APPR_PROCESS_START"].replace(['MONDAY','TUESDAY', 'WEDNESDAY','THURSDAY','FRIDAY'], 'WEEK_DAY')
df_prev["WEEKDAY_APPR_PROCESS_START"] = df_prev["WEEKDAY_APPR_PROCESS_START"].replace(['SATURDAY', 'SUNDAY'], 'WEEKEND')

# "HOUR_APPR_PROCESS_START"  değişkeninin working_hours ve off_hours olarak iki kategoriye ayrılması
a = [8,9,10,11,12,13,14,15,16,17]
df_prev["HOUR_APPR_PROCESS_START"] = df_prev["HOUR_APPR_PROCESS_START"].replace(a, 'working_hours')

b = [18,19,20,21,22,23,0,1,2,3,4,5,6,7]
df_prev["HOUR_APPR_PROCESS_START"] = df_prev["HOUR_APPR_PROCESS_START"].replace(b, 'off_hours')


# DAYS_DECISION değeri 1 yıldan küçük olanlara 1, büyük olanlara 0 değeri verildi.
df_prev["DAYS_DECISION"] = [1 if abs(i/(12*30)) <=1 else 0 for i in df_prev.DAYS_DECISION]

# "NAME_TYPE_SUITE"  değişkeninin alone ve not_alone olarak iki kategoriye ayrılması

df_prev["NAME_TYPE_SUITE"] = df_prev["NAME_TYPE_SUITE"].replace('Unaccompanied', 'alone')

b = ['Family', 'Spouse, partner', 'Children', 'Other_B', 'Other_A', 'Group of people']
df_prev["NAME_TYPE_SUITE"] = df_prev["NAME_TYPE_SUITE"].replace(b, 'not_alone')



# "NAME_GOODS_CATEGORY"  değişkenindeki bu değerler others olarak kategorize edilecek
a = ['Auto Accessories', 'Jewelry', 'Homewares', 'Medical Supplies', 'Vehicles', 'Sport and Leisure', 
     'Gardening', 'Other', 'Office Appliances', 'Tourism', 'Medicine', 'Direct Sales', 'Fitness', 'Additional Service', 
     'Education', 'Weapon', 'Insurance', 'House Construction', 'Animals'] 
df_prev["NAME_GOODS_CATEGORY"] = df_prev["NAME_GOODS_CATEGORY"].replace(a, 'others')

# "NAME_SELLER_INDUSTRY"  değişkenindeki bu değerler others olarak kategorize edilecek
a = ['Auto technology', 'Jewelry', 'MLM partners', 'Tourism'] 
df_prev["NAME_SELLER_INDUSTRY"] = df_prev["NAME_SELLER_INDUSTRY"].replace(a, 'others')
# İstenilen krecinin verilen krediye oranı içeren değişkeni türetir
df_prev["LOAN_RATE"] = df_prev.AMT_APPLICATION/df_prev.AMT_CREDIT

#YENI DEGISKENLER

# İstenilen krecinin verilen krediye oranı içeren değişkeni türetir
df_prev["NEW_LOAN_RATE"] = df_prev.AMT_APPLICATION/df_prev.AMT_CREDIT

# Ödeme gününü geciktirmiş mi bunu gösteren churn_prev değişkeni türetilir.
# 1= geciktirmiş, 0 = geciktirmemiş, NaN = boş değer
k = df_prev.DAYS_LAST_DUE_1ST_VERSION - df_prev.DAYS_LAST_DUE
df_prev["NEW_CHURN_PREV"] = [1 if i >= 0 else (0 if i < 0  else "NaN") for i in k]


# NFLAG_INSURED_ON_APPROVAL değişkeni yerine kullanılmak izere NEW_INSURANCE değişkeni tanımlandı.
df_prev[(df_prev['AMT_CREDIT'] == 0) | (df_prev['AMT_GOODS_PRICE'] == 0)]['NEW_INSURANCE'] = np.nan
df_prev['sigorta_miktari'] = df_prev['AMT_CREDIT'] - df_prev['AMT_GOODS_PRICE']
df_prev["NEW_INSURANCE"] = df_prev['sigorta_miktari'].apply(lambda x: 1 if x > 0 else (0 if x <= 0 else np.nan))
df_prev.drop('sigorta_miktari', axis=1, inplace=True)

# INTEREST_RATE değişkenini oluşturur.
#df_prev['INTEREST_RATE'] = (df_prev.AMT_ANNUITY*df_prev.CNT_PAYMENT/df_prev.AMT_CREDIT)**(12/df_prev.CNT_PAYMENT)-1
#df_prev[df_prev['INTEREST_RATE']==-1]=np.nan


drop_list = ['AMT_DOWN_PAYMENT', 'SELLERPLACE_AREA', 'CNT_PAYMENT', 'PRODUCT_COMBINATION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
            'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE','DAYS_TERMINATION','NFLAG_INSURED_ON_APPROVAL']
df_prev.drop(drop_list, axis = 1, inplace = True)

# Previous tablosundaki kategorik değişkenlerin isimlerini tutar.
category_columns=[]
for i in df_prev.columns:
    if df_prev[i].dtypes == "O":
        category_columns.append(i)

df_prev = pd.get_dummies(df_prev, columns = category_columns )

prev_agg_list = {"SK_ID_CURR":["count"], 
            "AMT_ANNUITY":["max"],
            "AMT_APPLICATION":["min","mean","max"],
            "AMT_CREDIT":["max"], 
            "AMT_GOODS_PRICE":["sum", "mean"],
            "NFLAG_LAST_APPL_IN_DAY":["sum","mean"], 
            "RATE_DOWN_PAYMENT":["sum", "mean"],
            "RATE_INTEREST_PRIMARY":["sum", "mean"],
            "RATE_INTEREST_PRIVILEGED":["sum", "mean"],
            "DAYS_DECISION":["sum"],
            "NEW_LOAN_RATE":["sum", "mean", "min", "max"],
            "NEW_INSURANCE":["sum", "mean"],
            #"INTEREST_RATE":["sum", "mean", "min", "max"],
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
            "NAME_SELLER_INDUSTRY_others":["sum", "mean"],
            "NAME_YIELD_GROUP_XNA":["sum", "mean"],
            "NAME_YIELD_GROUP_high":["sum", "mean"],
            "NAME_YIELD_GROUP_low_action":["sum", "mean"],
            "NAME_YIELD_GROUP_low_normal":["sum", "mean"],
            "NAME_YIELD_GROUP_middle":["sum", "mean"],
            "NEW_CHURN_PREV_0":["sum", "mean"],
            "NEW_CHURN_PREV_1":["sum", "mean"],
            "NEW_CHURN_PREV_NaN":["sum", "mean"]}

prev_agg_list.update(agg_list_previous_application)
prev_agg_list
df_prev
len(df_prev.columns)
len(balance.columns)

len(payment1.columns)
len(pos_cash1.columns)
len(df_prev.columns)
#5
df_prev_ins = df_prev.merge(payment1_agg, how = 'left', on = 'SK_ID_PREV')
df_prev_ins_pos = df_prev_ins.merge(pos_agg, how = 'left', on = 'SK_ID_PREV')
df_prev_ins_pos_agg = df_prev_ins_pos.groupby("SK_ID_CURR").agg(prev_agg_list).reset_index()
df_prev_ins_pos_agg.columns = pd.Index(["PREV_" + col[0] + "_" + col[1].upper() for col in df_prev_ins_pos_agg.columns.tolist()])
