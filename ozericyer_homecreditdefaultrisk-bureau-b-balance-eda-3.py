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
#oz1
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
bureau = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
bureau_balance = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau_balance.csv")
bureau.head(5)
bureau.info()
bureau_balance.head(100)
bureau_balance.info()
bureau.head()
##Kayıp Veri Analizi bureau
mis_value = bureau.isnull().sum()
mis_value_percent = 100*bureau.isnull().sum()/len(bureau)
mis_value_table = pd.concat([mis_value,mis_value_percent], axis = 1)
mis_value_table.columns=['count', 'percent']
mis_value_table = mis_value_table.sort_values('percent',ascending=False)
mis_value_table_1 = mis_value_table[mis_value_table['percent']>0]
pd.set_option('display.max_rows', None)
mis_value_table_1
bureau["AMT_CREDIT_SUM_LIMIT"].describe()
##Kayıp Veri Analizi burea_ balance
mis_value = bureau_balance.isnull().sum()
mis_value_percent = 100*bureau_balance.isnull().sum()/len(bureau)
mis_value_table = pd.concat([mis_value,mis_value_percent], axis = 1)
mis_value_table.columns=['count', 'percent']
mis_value_table = mis_value_table.sort_values('percent',ascending=False)
mis_value_table = mis_value_table[mis_value_table['percent']>0]
pd.set_option('display.max_rows', None)
mis_value_table
#bureau.dtypes.value_counts()
#bureau.select_dtypes(include = [object]).apply(pd.Series.nunique, axis = 0)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,6))
ax = fig.add_axes([0,0,1,1])
ax.bar(mis_value_table.index,mis_value_table.percent, color = 'purple')
plt.xticks(rotation =90,fontsize =10)
plt.title('Missing Data')
plt.xlabel('Feature')
plt.ylabel('% Percentage')
plt.show()
bureau.describe().T
import missingno as msno
msno.matrix(bureau)

import missingno as msno
msno.bar(bureau);
msno.heatmap(bureau);
# Heatmap of correlations
plt.figure(figsize = (28, 26))
sns.heatmap(bureau.corr(), cmap = plt.cm.RdYlBu_r, vmin = -0.25,fmt='.0g', annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
# coralasyon ve degerler
# DAYS_CREDIT','DAYS_ENDDATE_FACT arasindaki corolasyona bakildiginda DAYS_ENDDATE_FACT" makineye tahmini
# dogru olacagi dusunulmustur.
bureau[['DAYS_CREDIT','DAYS_ENDDATE_FACT']].sample(20)
bureau.info()
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(bureau.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
bureau.isnull().sum()
# 1- DAYS_ENDDATE_FACT %36 MISSING DATA ICERIYOR. CREDIT_ACTIVE ILE ARASINDAKI COROLASYONA
#BAKILDIGINDA DAYS_ENDDATE_FACT'IN MAKINEYE TAHMIN ETTIRILECEK KARAR VERILDI.


bureau.isnull().sum()
# 2
#bureau.DAYS_CREDIT_ENDDATE.fillna(bureacopy.DAYS_CREDIT_ENDDATE.mean(), inplace=True)


#oz2
# 3- max deger en fazla toplama esit olabileceginden birbiri yerine dolduruldu.
bureau.AMT_CREDIT_MAX_OVERDUE.fillna(bureau.AMT_CREDIT_SUM_OVERDUE, inplace=True)
bureau.isnull().sum()
#oz3
# 4- 1
bureau.AMT_CREDIT_SUM_LIMIT.fillna(bureau.AMT_CREDIT_SUM_LIMIT.mean(), inplace=True)

bureau.isnull().sum()
#oz4
# 4- 2 subset 13 adet kayip veriyi alt kumeleri ile doldurdu.
bureau.dropna(subset=['AMT_CREDIT_SUM'],inplace=True)


bureau.isnull().sum()
# sifirlar NAN'larla dolduruldu ama %94 missing data olustu ve bazi sifirlarin gercek bazilarinin nan oldugu goruldu.
# bureau.loc[bureau['AMT_CREDIT_SUM_DEBT']==0.0,'AMT_CREDIT_SUM_DEBT']=np.nan
# bureau.loc[bureau['AMT_CREDIT_SUM_LIMIT']==0.0,'AMT_CREDIT_SUM_LIMIT']=np.nan
# mis_value = bureau.isnull().sum()
# mis_value_percent = 100*bureau.isnull().sum()/len(bureau)
# mis_value_table = pd.concat([mis_value,mis_value_percent], axis = 1)
# mis_value_table.columns=['count', 'percent']
# mis_value_table = mis_value_table.sort_values('percent',ascending=False)
# mis_value_table_2 = mis_value_table[mis_value_table['percent']>0]
# pd.set_option('display.max_rows', None)
# mis_value_table_2
# 4-3 makineye tahmin ettirilecek
#bureau['AMT_CREDIT_SUM_DEBT']
bureau.isnull().sum()
#oz5
# 5- %71 missing data mevcut, test(data) kisminda missing data yok o yuzden droplandi
bureau.drop("AMT_ANNUITY", axis=1, inplace=True)
bureau.isnull().sum()
#oz6
df_nume = bureau.select_dtypes(include = ['float64','int64'])
df_nume.drop(['SK_ID_CURR','SK_ID_BUREAU'],axis=1,inplace=True)
#oz7
# numerikler makine yontemiyle tahmin ettirildi.
#!pip install ycimpute
#from ycimpute.imputer import EM
#var_names = list(df_nume)
#np_buro_num =np.array(df_nume)
#brr = EM().complete(np_buro_num)
#brr = pd.DataFrame(brr, columns = var_names)
#brr.isnull().sum()
# tahminden sonra corolasyonlardaki benzerliklerin korundugu gozlendi.
# brr_corr = brr.corr()
# plt.figure(figsize = (28, 26))

# # Heatmap of correlations
# sns.heatmap(brr_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25,fmt='.0g', annot = True, vmax = 0.6)
# plt.title('Correlation Heatmap')
bureau.dtypes
#oz8
# katagosrikler secildi
brr_kat=bureau[["CREDIT_ACTIVE","CREDIT_CURRENCY","CREDIT_TYPE"]] 
brr_kat.head()
bureau_balance.head(5)

#oz9
# tek tek kontrol edildi . %10'dan fazla olan kisimlar baskilandi
IQR =brr.describe().T
IQR['lower'] = IQR['25%']-1.5*(IQR['75%']-IQR['25%'])
IQR['upper'] = IQR['75%']+1.5*(IQR['75%']-IQR['25%'])
IQR.T
len(brr[brr['DAYS_CREDIT']<-3.454000e+03])
len(brr[brr['CREDIT_DAY_OVERDUE']>0.000000e+00])
len(brr[brr['DAYS_CREDIT_ENDDATE']<-3.268500e+03])
len(brr[brr['DAYS_CREDIT_ENDDATE']>2.583500e+03])
len(brr[brr['DAYS_ENDDATE_FACT']<-2.722500e+03])
brr.info()
len(brr[brr['AMT_CREDIT_MAX_OVERDUE']>0.000000e+00])
len(brr[brr['AMT_CREDIT_SUM']>7.105500e+05])
#oz10
brr.loc[brr['AMT_CREDIT_SUM'] > 7.105500e+05,'AMT_CREDIT_SUM'] = 7.105500e+05
len(brr[brr['AMT_CREDIT_SUM']<-3.442500e+05])
#oz11
brr.loc[brr['AMT_CREDIT_SUM_DEBT'] > 4.927500e+03,'AMT_CREDIT_SUM_DEBT'] = 4.927500e+03
len(brr[brr['AMT_CREDIT_SUM_LIMIT']>1.557379e+04])
len(brr[brr['AMT_CREDIT_SUM_OVERDUE']<0.000000e+00])
len(brr[brr['DAYS_CREDIT_UPDATE']>1.279500e+03])
#oz12
bureau = pd.concat([brr_kat,brr],axis=1)
bureau.head(100)
#from sklearn.neighbors import LocalOutlierFactor
# n_neighbors = 10 komşuluk sayısı, contamination = 0.1 saydamlık
#clf = LocalOutlierFactor(n_neighbors = 5, contamination = 0.1)
#clf.fit_predict(brr)
# negatif skorlar 
#brr_scores = clf.negative_outlier_factor_
#np.sort(brr_scores)[0:1000]
#esik_deger = np.sort(brr_scores)[7]
#esik_deger
#len(brr[brr_scores<esik_deger])
#brr[brr_scores==esik_deger]
# eşik skora sahip gözlem baskılama verisi olarak belirleniyor
#baskılama_deg = brr[brr_scores==esik_deger]
# esik skordan daha küçük skora sahip gözlemler için True-False şeklinde ARRAY oluşturulyor
#outlier_array = brr_scores<esik_deger
#outlier_array
# outlier_array'ın döndürdüğü True-False değerler ile filtreleme yapılarak Outlier gözlemler ile DATAFRAME oluşturuluyor
# outlier_br = brr[outlier_array]
# len(outlier_br)
#outlier_br
# outlier_df indexlerinden arındırılarak ARRAY'a dönüştürülüyor.
#outlier_br.to_records(index=False)
# Bu array res olarak tutuluyor.
#res = outlier_br.to_records(index=False)
# res'deki tüm veriler yerine baskılama dergerleri atanıyor
#res[:] = baskılama_deg.to_records(index=False)
#res
#brr[outlier_array]
# Bir array olan res aykırı gözlemlerin indexleri kullanılarak DATAFRAME dönüştürülüyor ve dff deki aykırı gözlemlerin yerine atanyor
#brr[outlier_array] = pd.DataFrame(res, index = brr[outlier_array].index)
#brr[outlier_array]
#brr.columns
#oz13
bakim = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
bakim.head()
#oz14
bureau.insert(0,'SK_ID_CURR',bakim['SK_ID_CURR'].values)#1.kolona ekledik
bureau.insert(1,'SK_ID_BUREAU',bakim['SK_ID_BUREAU'].values)  #2.kolona ekledik

bureau.head()
#oz15
bureau.groupby('SK_ID_BUREAU')['SK_ID_CURR'].nunique()

bureau_balance.isnull().sum()
bureau_balance.head()
#oz15
b_a_u_blance = bureau.SK_ID_BUREAU.unique()
b_a_u_blance
#oz16
bureau_blnctest = bureau_balance[bureau_balance.SK_ID_BUREAU.isin(b_a_u_blance)]

bureau_blnctest.head()
#oz16
def bureau_bb():

    #bureau_balance tablosunun okutulması

   
    bb = pd.get_dummies(bureau_blnctest, dummy_na = True)

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

    
    bureau_and_bb = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')

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

    #  SK_ID_BUREAU ve CREDIT_CURRENCY(degiskeninin %99u currency1, bu sebeple ayirt ediciligi olmadigindan silindi)
    bureau_and_bb.drop(["SK_ID_BUREAU","CREDIT_CURRENCY"], inplace = True, axis = 1)


    #NEW FEATURES

    #ortalama kac aylık kredi aldıgını gösteren yeni degisken
    bureau_and_bb["NEW_MONTHS_CREDIT"]= round((bureau_and_bb.DAYS_CREDIT_ENDDATE - bureau_and_bb.DAYS_CREDIT)/30)

    agg_list = {
          "SK_ID_CURR":["count"],
          "DAYS_CREDIT":["min","max"],
          "CREDIT_DAY_OVERDUE":["sum","mean","max"],     
          "DAYS_CREDIT_ENDDATE":["max","min"],
          "DAYS_ENDDATE_FACT": ["max","min"],
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
#oz17
bure=bureau_bb()
bure.head()
bure.columns
bure.isnull().sum()
#oz18
bure['BB_STATUS_0_MEAN_MEAN'].fillna(0, inplace=True)
#oz19
bure['BB_STATUS_C_MEAN_MEAN'].fillna(0, inplace=True)
bure.isnull().sum()
#oz20
bureau_application = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
b_a_u = bureau_application.SK_ID_CURR.unique()
#oz21
bureau_and_bb_lastform = bure[bure.BB_SK_ID_CURR_.isin(b_a_u)]
#sendeki train datasiyle merge etmen gereken nihai data
bureau_and_bb_lastform.head()
bureau_and_bb_lastform.info()
