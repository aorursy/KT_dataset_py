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
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
#1
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
#2
bureau_balance=pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
bureau.info()
bureau_balance.info()
print('Size of bureau data', bureau.shape)
print('Size of bureau_balance data', bureau_balance.shape)
bureau.isnull().sum()
bureau_balance.isnull().sum()
#KAc farkli unique deger var

print("SK_ID_CURR", len(bureau.SK_ID_CURR.unique()))
print("SK_ID_BUREAU",len(bureau.SK_ID_BUREAU.unique()))
print("CREDIT_ACTIVE",len(bureau.CREDIT_ACTIVE.unique()))
print("CREDIT_CURRENCY",len(bureau.CREDIT_CURRENCY.unique()))
print("DAYS_CREDIT",len(bureau.DAYS_CREDIT.unique()))
print("CREDIT_DAY_OVERDUE",len(bureau.CREDIT_DAY_OVERDUE.unique()))
print("DAYS_CREDIT_ENDDATE",len(bureau.DAYS_CREDIT_ENDDATE.unique()))
print("DAYS_ENDDATE_FACT",len(bureau.DAYS_ENDDATE_FACT.unique()))
print("AMT_CREDIT_MAX_OVERDUE",len(bureau.AMT_CREDIT_MAX_OVERDUE.unique()))
print("CNT_CREDIT_PROLONG",len(bureau.CNT_CREDIT_PROLONG.unique()))
print("AMT_CREDIT_SUM",len(bureau.AMT_CREDIT_SUM.unique()))
print("AMT_CREDIT_SUM_DEBT",len(bureau.AMT_CREDIT_SUM_DEBT.unique()))
print("AMT_CREDIT_SUM_LIMIT",len(bureau.AMT_CREDIT_SUM_LIMIT.unique()))
print("AMT_CREDIT_SUM_OVERDUE",len(bureau.AMT_CREDIT_SUM_OVERDUE.unique()))
print("CREDIT_TYPE",len(bureau.CREDIT_TYPE.unique()))
print("DAYS_CREDIT_UPDATE",len(bureau.DAYS_CREDIT_UPDATE.unique()))
print("AMT_ANNUITY",len(bureau.AMT_ANNUITY.unique()))


#KAc farkli unique deger var
print("SK_ID_BUREAU", len(bureau_balance.SK_ID_BUREAU.unique()))
print("MONTHS_BALANCE",len(bureau_balance.MONTHS_BALANCE.unique()))
print("STATUS",len(bureau_balance.STATUS.unique()))
#unutulan bureau data yuzdesi
total = bureau.isnull().sum().sort_values(ascending = False)
percent = (bureau.isnull().sum()/bureau.isnull().count()*100).sort_values(ascending = False)
missing_application_bureau_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_bureau_data
bureau.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
bureau['CREDIT_ACTIVE'].value_counts()
print(bureau.CREDIT_ACTIVE.unique())
bureau['CREDIT_CURRENCY'].value_counts()
bureau['CREDIT_TYPE'].value_counts()
bureau['SK_ID_BUREAU'].value_counts()
bureau['SK_ID_CURR'].value_counts()
#silinebilinir (sil)
bureau['CREDIT_CURRENCY'].value_counts()
#krediye kac gun once basvurmus daha onceki krediler ile ilgili olarak
bureau['DAYS_CREDIT'].value_counts()
#daha onceki kredilerinde kac gun vadesi gecmis 
bureau['CREDIT_DAY_OVERDUE'].value_counts()
#home crediye basvurdugi anda diger kredinin bitmesine kac gun var 
bureau['DAYS_CREDIT_ENDDATE'].value_counts()
#home credisine basvurdugunda kapatilan krediler kac gun once kapatilmis
bureau['DAYS_ENDDATE_FACT'].value_counts()

#geciktirilmis en yuksek meblag
bureau['AMT_CREDIT_MAX_OVERDUE'].value_counts()
# kredi kac defa uzatildi
bureau['CNT_CREDIT_PROLONG'].value_counts()
#mevcut kredi miktari
bureau['AMT_CREDIT_SUM'].value_counts()
#m3vcut borc miktari
bureau['AMT_CREDIT_SUM_DEBT'].value_counts()
#kredi kardi limiti
bureau['AMT_CREDIT_SUM_LIMIT'].value_counts()
#mevcut odenmemis meblag
bureau['AMT_CREDIT_SUM_OVERDUE'].value_counts()
bureau['CREDIT_TYPE'].value_counts()
#kredi basvurusundan kac gun once kredi basvuranin en son bilgileri geldi kredi basvuru merkezine
bureau['DAYS_CREDIT_UPDATE'].value_counts()
#kredi burosu kredisinin yillik geliri
bureau['AMT_ANNUITY'].value_counts()
bureau_balance["SK_ID_BUREAU"].value_counts()
descriptioncolumns=pd.read_csv('/kaggle/input/description/HomeCredit_columns_descriptionn.csv', encoding= 'unicode_escape')
descriptioncolumns.info()
descriptioncolumns.head()
descriptioncolumns.Table.unique()


descclmnsbureau=descriptioncolumns[descriptioncolumns['Table'] == 'bureau.csv']
descclmnsbureaublnc=descriptioncolumns[descriptioncolumns['Table'] == 'bureau_balance.csv']
descclmnsbureau
descclmnsbureaublnc
bureau
test=bureau[bureau["SK_ID_CURR"]==360228]
test

#vadesi gecmis tum krediler
test1=bureau[bureau["CREDIT_DAY_OVERDUE"]!=0]
test1
test1['SK_ID_CURR'].value_counts()
test2=test1[test1["SK_ID_CURR"]==441392]
test2
#vadesi gecen kredilerin mevcut durumu
test1['CREDIT_ACTIVE'].value_counts()
#vadesi gecmis olanlardan kapatilan krediler
test3=test1[test1["CREDIT_ACTIVE"]=="Closed"]
test3
#krediye basvurdugunda diger kredilerinin son gunu gecmis tarihli olanlar
test4=bureau[bureau["DAYS_CREDIT_ENDDATE"]<0]
#burda days credit end degeri eksi oldugu halde ayrica overdue 0 oldugu halde nasil credi active olabilir
test4
test4["CREDIT_ACTIVE"].value_counts()
test4["CREDIT_DAY_OVERDUE"].value_counts()
test5=bureau[bureau["DAYS_CREDIT_ENDDATE"]>0]
test5
bureau
test6=bureau[bureau["AMT_ANNUITY"]<0]
#3
bureau.drop("CREDIT_CURRENCY", axis=1, inplace=True)
bureau.corr()
#4
bureau.drop("AMT_ANNUITY", axis=1, inplace=True)
bureacopy=bureau.copy()
bureacopy.AMT_CREDIT_MAX_OVERDUE.fillna("leyla", inplace=True)
bureacopy
test7=bureacopy[bureacopy["AMT_CREDIT_MAX_OVERDUE"]=="leyla"]
test7
test8=bureau[bureau["AMT_CREDIT_SUM_OVERDUE"]!=0]
test8
test8[test8["AMT_CREDIT_MAX_OVERDUE"]==0]
#5
bureaucp=bureau.copy()
#6
bureaucp.AMT_CREDIT_MAX_OVERDUE.fillna(bureaucp.AMT_CREDIT_SUM_OVERDUE, inplace=True)
bureaucp.isnull().sum()
#7
bureaucp.drop("DAYS_ENDDATE_FACT", axis=1, inplace=True)
bureaucp.isnull().sum()
bureaucp
desc = bureaucp.AMT_CREDIT_SUM_LIMIT.describe()
desc
Q1 = desc[4]
Q3 = desc[6]
IQR = Q3-Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
print("Anything outside this range is an outlier for desc: (", lower_bound ,",", upper_bound,")")
outlier=bureaucp[(bureaucp.AMT_CREDIT_SUM_LIMIT < lower_bound) | (bureaucp.AMT_CREDIT_SUM_LIMIT > upper_bound)].AMT_CREDIT_SUM_LIMIT.values
print("Outliers: ",outlier)
len(outlier)
rate=(74506*100)/(1716428*0.65)
print(rate)
#8
bureaucpp=bureaucp.copy()
#9
bureaucpp.AMT_CREDIT_SUM_LIMIT.fillna(bureaucp.AMT_CREDIT_SUM_LIMIT.mean(), inplace=True)
bureaucpp.isnull().sum()
bureaucpp
bureaucpp[bureaucpp["CREDIT_TYPE"]=="Credit card"]
#10
bureaucpp.DAYS_CREDIT_ENDDATE.fillna(bureaucp.DAYS_CREDIT_ENDDATE.mean(), inplace=True)
bureaucpp.isnull().sum()
test9=bureaucpp.copy()
test9.AMT_CREDIT_SUM.fillna("ayse", inplace=True)

test10=test9[test9["AMT_CREDIT_SUM"]=="ayse"]
test10
k=0  
i=0
liste=list(bureaucpp["SK_ID_CURR"])
liste1=list(test10["SK_ID_CURR"])
liste3=[]
while i < len(liste1):
    for each in liste:
        if each==liste1[i]:
            k=k+1
    liste3.append(k)
    i=i+1
print(liste3)
    
#11
bureau_cp=bureaucpp.copy()
#12
bureau_cp.dropna(subset=['AMT_CREDIT_SUM'],inplace=True)

bureau_cp
bureau_cp.isnull().sum()
#13
bureau_cp['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)
bureau_cp.isnull().sum()
#14
bureau_cp.drop("DAYS_CREDIT_UPDATE", axis=1, inplace=True)
bureau_cp
bureau_cp.groupby('SK_ID_BUREAU')['SK_ID_CURR'].nunique()
bureau_balance.isnull().sum()
bureau_balance
#15
b_a_u_blance = bureau_cp.SK_ID_BUREAU.unique()
#16
bureau_blnctest = bureau_balance[bureau_balance.SK_ID_BUREAU.isin(b_a_u_blance)]
bureau_blnctest
descclmnsbureaublnc
print("STATUS", bureau_blnctest.STATUS.unique())
bureau_blnctest.info()
print("STATUS", bureau_blnctest.MONTHS_BALANCE.unique())
print(bureau_blnctest.SK_ID_BUREAU.unique())
bureau_blnctest['SK_ID_BUREAU'].value_counts()
#17
bureau_balance_cp=bureau_blnctest.copy()
#18
bureau_c_p=bureau_cp.copy()
#19
def bureau_bb():

    #bureau_balance tablosunun okutulması

   
    bb = pd.get_dummies(bureau_balance_cp, dummy_na = True)

    agg_list = {"MONTHS_BALANCE":"count",
                "STATUS_0":["sum","mean"],
                "STATUS_1":["sum"],
                "STATUS_2":["sum"],
                "STATUS_3":["sum"],
                "STATUS_4":["sum"],
                "STATUS_5":["sum"],
                "STATUS_C":["sum","mean"],
                "STATUS_X":["sum","mean"] }

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(agg_list)

    # Degisken isimlerinin yeniden adlandirilmasi 
    bb_agg.columns = pd.Index([col[0] + "_" + col[1].upper() for col in bb_agg.columns.tolist()])

    # Status_sum ile ilgili yeni bir degisken olusturma
    bb_agg['NEW_STATUS_SCORE'] = bb_agg['STATUS_1_SUM'] + bb_agg['STATUS_2_SUM']^2 + bb_agg['STATUS_3_SUM']^3 + bb_agg['STATUS_4_SUM']^4 + bb_agg['STATUS_5_SUM']^5

    bb_agg.drop(['STATUS_1_SUM','STATUS_2_SUM','STATUS_3_SUM','STATUS_4_SUM','STATUS_5_SUM'], axis=1,inplace=True)

    
    bureau_and_bb = bureau_c_p.join(bb_agg, how='left', on='SK_ID_BUREAU')

    #BUREAU BALANCE VE BUREAU ORTAK TABLO

    #CREDIT_TYPE degiskeninin sinif sayisini 3'e düsürmek 
    bureau_and_bb['CREDIT_TYPE'] = bureau_and_bb['CREDIT_TYPE'].replace(['Car loan',
              'Mortgage',
              'Microloan',
              'Loan for business development', 
              'Another type of loan',
              'Unknown type of loan', 
              'Loan for working capital replenishment',
              "Loan for purchase of shares (margin lending)",                                                
              'Cash loan (non-earmarked)', 
              'Real estate loan',
              "Loan for the purchase of equipment", 
              "Interbank credit", 
              "Mobile operator loan"], 'Rare')


    #CREDIT_ACTIVE degiskeninin sinif sayisini 2'ye düsürmek (Sold' u Closed a dahil etmek daha mi uygun olur ???)
    bureau_and_bb['CREDIT_ACTIVE'] = bureau_and_bb['CREDIT_ACTIVE'].replace(['Bad debt','Sold'], 'Active')

    # bureau_bb tablosundaki kategorik degiskenlere One Hot Encoding uygulanmasi
    bureau_and_bb = pd.get_dummies(bureau_and_bb, columns = ["CREDIT_TYPE","CREDIT_ACTIVE"])

    #  SK_ID_BUREAU sildik  
    bureau_and_bb.drop(["SK_ID_BUREAU"], inplace = True, axis = 1)


    #NEW FEATURES

    #ortalama kac aylık kredi aldıgını gösteren yeni degisken
    bureau_and_bb["NEW_MONTHS_CREDIT"]= round((bureau_and_bb.DAYS_CREDIT_ENDDATE - bureau_and_bb.DAYS_CREDIT)/30)

    agg_list = {
          "SK_ID_CURR":["count"],
          "DAYS_CREDIT":["min","max"],
          "CREDIT_DAY_OVERDUE":["sum","mean","max"],     
          "DAYS_CREDIT_ENDDATE":["max","min"],
          "AMT_CREDIT_MAX_OVERDUE":["mean","max","min"],
          "CNT_CREDIT_PROLONG":["sum","mean","max","min"],
          "AMT_CREDIT_SUM":["mean","max","min"],            
          "AMT_CREDIT_SUM_DEBT":["sum","mean","max"],
          "AMT_CREDIT_SUM_LIMIT":["sum","mean","max"],
          'AMT_CREDIT_SUM_OVERDUE':["sum","mean","max"], 
          'MONTHS_BALANCE_COUNT':["sum"], 
          'STATUS_0_SUM':["sum"],         
          'STATUS_0_MEAN':["mean"], 
          'STATUS_C_SUM':["sum"], 
          'STATUS_C_MEAN':["mean"],
          'CREDIT_ACTIVE_Active':["sum","mean"], 
          'CREDIT_ACTIVE_Closed':["sum","mean"], 
          'CREDIT_TYPE_Rare':["sum","mean"],      
          'CREDIT_TYPE_Consumer credit':["sum","mean"], 
          'CREDIT_TYPE_Credit card':["sum","mean"],
          "NEW_MONTHS_CREDIT":["count","sum","mean","max","min"]}


    # bureau_bb_agg tablosuna aggreagation islemlerinin uygulanamasi  
    bureau_and_bb_agg = bureau_and_bb.groupby("SK_ID_CURR").agg(agg_list).reset_index()


    # Degisken isimlerinin yeniden adlandirilmasi 
    bureau_and_bb_agg.columns = pd.Index(["BB_" + col[0] + "_" + col[1].upper() for col in bureau_and_bb_agg.columns.tolist()])

    # kisinin aldıgı en yuksek ve en dusuk kredinin farkını gösteren yeni degisken
    bureau_and_bb_agg["BB_NEW_AMT_CREDIT_SUM_RANGE"] = bureau_and_bb_agg["BB_AMT_CREDIT_SUM_MAX"] - bureau_and_bb_agg["BB_AMT_CREDIT_SUM_MIN"]

    # ortalama kac ayda bir kredi cektigini ifade eden  yeni degisken
    bureau_and_bb_agg["BB_NEW_DAYS_CREDIT_RANGE"]= round((bureau_and_bb_agg["BB_DAYS_CREDIT_MAX"] - bureau_and_bb_agg["BB_DAYS_CREDIT_MIN"])/(30 * bureau_and_bb_agg["BB_SK_ID_CURR_COUNT"]))
    
    return bureau_and_bb_agg
#20
bure=bureau_bb()
bure
bure.columns
bure.isnull().sum()
print(bure.BB_STATUS_0_MEAN_MEAN.unique())
#21
bure['BB_STATUS_0_MEAN_MEAN'].fillna(0, inplace=True)
#22
bure['BB_STATUS_C_MEAN_MEAN'].fillna(0, inplace=True)
bure.isnull().sum()
#23
bureau_application = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
b_a_u = bureau_application.SK_ID_CURR.unique()
b_a_u
#24
bureau_and_bb_lastform = bure[bure.BB_SK_ID_CURR_.isin(b_a_u)]
#sendeki train datasiyle merge etmen gereken nihai data
bureau_and_bb_lastform
bureau_and_bb_lastform.isnull().sum()
bureau_and_bb_lastform.info()