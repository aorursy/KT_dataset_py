# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# IMPORTING LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud,STOPWORDS
import io
import base64
from matplotlib import rc,animation
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
print(os.listdir("../input"))

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data1=pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
df_train=data1.copy()
df_train.tail()
data2=pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")
df_test=data2.copy()
df_test.tail()
len(df_test.columns)
len(df_train.columns)
df_train['data_type'] = 'train'
df_test['data_type'] = 'test'
df = pd.concat([df_train,df_test])
df.tail()
df["TARGET"].value_counts()
df["TARGET"].isnull().sum()  #test datasinin Nan degerleri
df_train['TARGET'].astype(int).plot.hist();
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(df_train).head(20)
missing_data(df_test).head(20)
fig = plt.figure(figsize=(18,6))
miss_train = pd.DataFrame((df_train.isnull().sum())*100/df_train.shape[0]).reset_index()
miss_test = pd.DataFrame((df_test.isnull().sum())*100/df_test.shape[0]).reset_index()
miss_train["type"] = "train"
miss_test["type"]  =  "test"
missing = pd.concat([miss_train,miss_test],axis=0)
ax = sns.pointplot("index",0,data=missing,hue="type")
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values in application train and test data")
plt.ylabel("PERCENTAGE")
plt.xlabel("COLUMNS")
ax.set_facecolor("k")
fig.set_facecolor("lightgrey")
import seaborn as sns
sns.heatmap(df.isnull(), cbar = False);
import missingno as msno
msno.matrix(df.sample(500), inline=True, sparkline=True, figsize=(20,10), sort=None, color=(0.25, 0.45, 0.6))
msno.matrix(df.iloc[0:100, 40:94], inline=True, sparkline=True, figsize=(20,10), sort='ascending', fontsize=12, labels=True, color=(0.25, 0.45, 0.6))
#We can see amount of missing values in bar grafic
import missingno as msno
msno.bar(df);
msno.heatmap(df); 
#We can see relation between missing values in dendrogram
msno.dendrogram(df)
plt.show()
df.dtypes.value_counts()
# Number of unique classes in each object column 
#WE USE LEBEL ENCODER FOR 2 VALUES AND ONE-HOT FOR MORE THAN 2.We will use it later
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
cor = df.corr()['TARGET'].sort_values()
print('Most Positive Correlations:\n', cor.tail(15))
print('\nMost Negative Correlations:\n', cor.head(15))
df.columns.values
df.describe().T
df.head()
df['PHONE'] = df[['FLAG_MOBIL','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL']].sum(axis = 1)
df.drop(df[['FLAG_MOBIL','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL']],axis=1,inplace=True)
df['WRONG_ADRESS'] = df[['REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY']].sum(axis = 1)
df.drop(df[['REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY']],axis=1,inplace=True)
df['WRONG_ADRESS'].corr(df['TARGET'])
print(df['FLAG_DOCUMENT_2'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_3'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_4'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_5'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_6'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_7'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_8'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_8'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_9'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_10'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_11'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_12'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_13'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_14'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_15'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_16'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_17'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_18'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_19'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_20'].corr(df['TARGET']))
print(df['FLAG_DOCUMENT_21'].corr(df['TARGET']))
df['DOCUMENT_SUM_1'] = (df.loc[:,'FLAG_DOCUMENT_4':'FLAG_DOCUMENT_19'].sum(axis = 1))/(16)
df['DOCUMENT_SUM_1'].corr(df['TARGET'])
df['DOCUMENT_SUM_2'] = (df[['FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']].sum(axis = 1))/(4)
df['DOCUMENT_SUM_2'].corr(df['TARGET'])
df.drop(df.loc[:,'FLAG_DOCUMENT_2':'FLAG_DOCUMENT_21'],axis=1,inplace=True)
df.columns
df['DOCUMENT_SUM_1'].sample(20)
df_train.loc[:,'APARTMENTS_AVG':'EMERGENCYSTATE_MODE']
crr=df_train.loc[:,'APARTMENTS_AVG':'EMERGENCYSTATE_MODE'].corr() 
df_train.loc[:,'APARTMENTS_AVG':'EMERGENCYSTATE_MODE'].corr()
# Heatmap of correlations
plt.figure(figsize = (28, 26))
sns.heatmap(crr, cmap = plt.cm.RdYlBu_r, vmin = -0.25,fmt='.0g', annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
#print(df_train['EMERGENCYSTATE_MODE'].corr(df['TARGET'])#kate
#print(df_train['WALLSMATERIAL_MODE'].corr(df['TARGET']))#kate          
print(df_train['TOTALAREA_MODE'].corr(df['TARGET']))
#print(df_train['HOUSETYPE_MODE'].corr(df['TARGET']))#kategorik          
#print(df_train['FONDKAPREMONT_MODE'].corr(df['TARGET']))#kategorik
print("------------------------------------------------------------------")
print("1",df_train['NONLIVINGAREA_MEDI'].corr(df['TARGET']))          
print("2",df_train['NONLIVINGAPARTMENTS_MEDI'].corr(df['TARGET']))
print("3",df_train['LIVINGAREA_MEDI'].corr(df['TARGET']))
print("4",df_train['LIVINGAPARTMENTS_MEDI'].corr(df['TARGET']))
print("5",df_train['LANDAREA_MEDI'].corr(df['TARGET']))
print("6",df_train['FLOORSMIN_MEDI'].corr(df['TARGET']))
print("7",df_train['FLOORSMAX_MEDI'].corr(df['TARGET']))
print("8",df_train['ENTRANCES_MEDI'].corr(df['TARGET']))
print("9",df_train['ELEVATORS_MEDI'].corr(df['TARGET']))
print("10",df_train['COMMONAREA_MEDI'].corr(df['TARGET']))
print("11",df_train['YEARS_BUILD_MEDI'].corr(df['TARGET']))
print("12",df_train['YEARS_BEGINEXPLUATATION_MEDI'].corr(df['TARGET']))
print("13",df_train['BASEMENTAREA_MEDI'].corr(df['TARGET']))
print("14",df_train['APARTMENTS_MEDI'].corr(df['TARGET']))
print("---------------------------------------------------------")
print("1",df_train['NONLIVINGAREA_MODE'].corr(df['TARGET']))
print("2",df_train['NONLIVINGAPARTMENTS_MODE'].corr(df['TARGET']))
print("3",df_train['LIVINGAREA_MODE'].corr(df['TARGET']))
print("4",df_train['LIVINGAPARTMENTS_MODE'].corr(df['TARGET']))
print("5",df_train['LANDAREA_MODE'].corr(df['TARGET']))
print("6",df_train['FLOORSMIN_MODE'].corr(df['TARGET']))
print("7",df_train['FLOORSMAX_MODE'].corr(df['TARGET']))
print("8",df_train['ENTRANCES_MODE'].corr(df['TARGET']))
print("9",df_train['ELEVATORS_MODE'].corr(df['TARGET']))
print("10",df_train['COMMONAREA_MODE'].corr(df['TARGET']))
print("11",df_train['YEARS_BUILD_MODE'].corr(df['TARGET']))
print("12",df_train['YEARS_BEGINEXPLUATATION_MODE'].corr(df['TARGET']))
print("13",df_train['BASEMENTAREA_MODE'].corr(df['TARGET']))
print("14",df_train['APARTMENTS_MODE'].corr(df['TARGET']))
print("-------------------------------------------------------------------")
print("1",df_train['NONLIVINGAREA_AVG'].corr(df['TARGET']))
print("2",df_train['NONLIVINGAPARTMENTS_AVG'].corr(df['TARGET']))
print("3",df_train['LIVINGAREA_AVG'].corr(df['TARGET']))
print("4",df_train['LIVINGAPARTMENTS_AVG'].corr(df['TARGET']))
print("5",df_train['LANDAREA_AVG'].corr(df['TARGET']))
print("6",df_train['FLOORSMIN_AVG'].corr(df['TARGET']))
print("7",df_train['FLOORSMAX_AVG'].corr(df['TARGET']))
print("8",df_train['ENTRANCES_AVG'].corr(df['TARGET']))
print("9",df_train['ELEVATORS_AVG'].corr(df['TARGET']))
print("10",df_train['COMMONAREA_AVG'].corr(df['TARGET']))
print("11",df_train['YEARS_BUILD_AVG'].corr(df['TARGET']))
print("12",df_train['YEARS_BEGINEXPLUATATION_AVG'].corr(df['TARGET']))
print("13",df_train['BASEMENTAREA_AVG'].corr(df['TARGET']))
print("14",df_train['APARTMENTS_AVG'].corr(df['TARGET']))

df.loc[:,'APARTMENTS_AVG':'EMERGENCYSTATE_MODE'].dtypes.value_counts()
df.loc[:,'APARTMENTS_AVG':'EMERGENCYSTATE_MODE'].columns  # columns number 45 to 91
df.loc[:,'APARTMENTS_AVG':'EMERGENCYSTATE_MODE'].sample(10)

#1
df["EMERGENCYSTATE_MODE"].unique()
df["EMERGENCYSTATE_MODE"].value_counts()
df["EMERGENCYSTATE_MODE"].isnull().sum()  #bu nan value ler unknown yapilabilir mi?/

#2
df["WALLSMATERIAL_MODE"].unique()   
#nan lar mod medyan mantiksiz cok nan value var,one-hot yapildiktan sonra makine ile doldurulabilir
df["WALLSMATERIAL_MODE"].value_counts()
df["WALLSMATERIAL_MODE"].isnull().sum()

#3
df["FONDKAPREMONT_MODE"].unique()
df["FONDKAPREMONT_MODE"].value_counts()  #nan value cok.One-hot dan sonra makine ile doldurmak mantikli.
df["FONDKAPREMONT_MODE"].isnull().sum()

#4
df["HOUSETYPE_MODE"].unique()
df["HOUSETYPE_MODE"].value_counts()#mod ile doldurulabilir veya makine ile
df["HOUSETYPE_MODE"].isnull().sum()

df.loc[:,'APARTMENTS_AVG':'EMERGENCYSTATE_MODE'].dtypes.value_counts()
df.loc[:,'APARTMENTS_AVG':'EMERGENCYSTATE_MODE'].dtypes
df['TARGET'].corr(df.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_MEDI'].sum(axis = 1))
for col in df.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_MEDI'].columns:
    df[col] = df[col].fillna(df[col].median())
df['TARGET'].corr((df.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_MEDI'].sum(axis = 1))*(3-df['REGION_RATING_CLIENT_W_CITY']))
# BUILDING_FEATURES değişkeni veriye ekleniyor
df['BUILDING_FEATURES'] = (df.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_AVG'].sum(axis = 1))*(3-df['REGION_RATING_CLIENT_W_CITY'])
df.drop(df.loc[:,'APARTMENTS_AVG':'NONLIVINGAREA_MEDI'],axis=1,inplace=True)
list(df.columns)

df['CODE_GENDER'].value_counts()
#df.drop(df[df['CODE_GENDER']=='XNA'].index, inplace=True)
df['CODE_GENDER'].replace('XNA', 'F',inplace=True)  
df['CODE_GENDER'].value_counts()
df['DAYS_BIRTH'].sample(20)
df['DAYS_BIRTH']=df['DAYS_BIRTH']/ -365
df['DAYS_BIRTH'].sample(20)
df.head()
df.iloc[:,37:].head()
display(df['FLAG_OWN_CAR'].value_counts())
display(df['FLAG_OWN_REALTY'].value_counts())
display(df['NAME_TYPE_SUITE'].value_counts())
display(df['NAME_INCOME_TYPE'].value_counts())
display(df['NAME_EDUCATION_TYPE'].value_counts())
display(df['NAME_FAMILY_STATUS'].value_counts())
display(df['NAME_HOUSING_TYPE'].value_counts())
display(df['OCCUPATION_TYPE'].value_counts())
display(df['WEEKDAY_APPR_PROCESS_START'].value_counts())
display(df['ORGANIZATION_TYPE'].value_counts())
display(df['FONDKAPREMONT_MODE'].value_counts())
display(df['HOUSETYPE_MODE'].value_counts())
display(df['WALLSMATERIAL_MODE'].value_counts())
display(df['EMERGENCYSTATE_MODE'].value_counts())

df['DAYS_EMPLOYED'].value_counts()
df['DAYS_EMPLOYED'].describe()
df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
df['DAYS_EMPLOYED'].isnull().sum()
# Heatmap of correlations
plt.figure(figsize = (28, 26))
sns.heatmap(df.corr(), cmap = plt.cm.RdYlBu_r, vmin = -0.25,fmt='.0g', annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');

df.loc[:,'AMT_REQ_CREDIT_BUREAU_HOUR':'AMT_REQ_CREDIT_BUREAU_YEAR'].sample(15)
df.loc[:,'AMT_REQ_CREDIT_BUREAU_HOUR':'AMT_REQ_CREDIT_BUREAU_YEAR'].isnull().sum()
df['AMT_REQ_CREDIT_BUREAU_YEAR'].value_counts()
df['AMT_REQ_CREDIT_BUREAU_QRT'].value_counts()
df['AMT_REQ_CREDIT_BUREAU_MON'].value_counts()
df['AMT_REQ_CREDIT_BUREAU_WEEK'].value_counts()
df['AMT_REQ_CREDIT_BUREAU_DAY'].value_counts()
df['AMT_REQ_CREDIT_BUREAU_HOUR'].value_counts()
list(df.columns)
msno.bar(df);
df.info()
df.isnull().sum()
msno.matrix(df.sample(200));
msno.matrix(df);
msno.heatmap(df);
df['OWN_CAR_AGE'].isnull().sum()
#df['OWN_CAR_AGE'].value_counts()
df['FLAG_OWN_CAR'].value_counts()
df['OWN_CAR_AGE'].corr(df['TARGET'])
df[df['OWN_CAR_AGE']==0.0].sample(5)
df[['OWN_CAR_AGE','FLAG_OWN_CAR','TARGET']].sample(20)
df[['OCCUPATION_TYPE','NAME_INCOME_TYPE','TARGET']].sample(20)
amt = df[[ 'AMT_INCOME_TOTAL','AMT_CREDIT',
                         'AMT_ANNUITY', 'AMT_GOODS_PRICE',"TARGET"]]
amt = amt[(amt["AMT_GOODS_PRICE"].notnull()) & (amt["AMT_ANNUITY"].notnull())]
sns.pairplot(amt,hue="TARGET",palette=["b","r"])
plt.show()
df.isnull().sum()

df_cat = df.select_dtypes(include = [object])
del df_cat['data_type']
df_cat.select_dtypes(include = [object]).apply(pd.Series.nunique, axis = 0)
for col in list(df_cat.columns):
    df_cat[col].fillna(method = "ffill")
df_cat.isnull().sum()
print('df_cat shape:',df_cat.shape)
from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in df_cat:
    if df_cat[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(df_cat[col].unique())) <= 2:
            # Train on the training data
            le.fit(df_cat[col])
            # Transform both training and testing data
            df_cat[col] = le.transform(df_cat[col])
                        
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
df_cat.head()
df_cat = pd.get_dummies(df_cat)
print('df_cat shape:',df_cat.shape)
df_cat.head()
df_num = df.select_dtypes(include = ['float64','int64'])
df_num.drop(['SK_ID_CURR','TARGET'],axis=1,inplace=True)
IQR = df_num.describe().T
IQR['lower'] = IQR['25%']-1.5*(IQR['75%']-IQR['25%'])
IQR['upper'] = IQR['75%']+1.5*(IQR['75%']-IQR['25%'])
IQR.T
IQR.T.iloc[:,9:]
IQR.T.iloc[:,18:]
IQR.T.iloc[:,27:]
len(df[df['OBS_30_CNT_SOCIAL_CIRCLE']>5.000000])
len(df[df['TOTALAREA_MODE']>0.259500])
len(df[df['OWN_CAR_AGE']>30.000000])
len(df[df['AMT_GOODS_PRICE']>1.336500e+06])  ####################There are many outlier 
len(df[df['AMT_CREDIT']> 1.588894e+06])      
len(df[df['AMT_ANNUITY']>62304.750000])
df.loc[df['AMT_GOODS_PRICE'] > 1.336500e+06,'AMT_GOODS_PRICE']=np.nan
df['AMT_GOODS_PRICE'].isnull().sum()
df_num = df.select_dtypes(include = ['float64','int64'])
df_num.drop(['SK_ID_CURR','TARGET'],axis=1,inplace=True)
df_num.isnull().sum()
!pip install ycimpute
from ycimpute.imputer import EM
var_names = list(df_num) 
var_names
np_df_num =np.array(df_num)
dff = EM().complete(np_df_num)
dff = pd.DataFrame(dff, columns = var_names)
dff.isnull().sum()
dff.sample(10)
dff_corr = dff.corr()
plt.figure(figsize = (28, 26))

# Heatmap of correlations
sns.heatmap(dff_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25,fmt='.0g', annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')
from sklearn.neighbors import LocalOutlierFactor
# n_neighbors = 10 komşuluk sayısı, contamination = 0.1 saydamlık
clf = LocalOutlierFactor(n_neighbors = 10, contamination = 0.1)
clf.fit_predict(dff)
# negatif skorlar 
dff_scores = clf.negative_outlier_factor_
np.sort(dff_scores)[0:1000]
esik_deger = np.sort(dff_scores)[7]
esik_deger
len(dff[dff_scores<esik_deger])
dff[dff_scores==esik_deger]
# eşik skora sahip gözlem baskılama verisi olarak belirleniyor
baskılama_deg = dff[dff_scores==esik_deger]
# esik skordan daha küçük skora sahip gözlemler için True-False şeklinde ARRAY oluşturulyor
outlier_array = dff_scores<esik_deger
outlier_array
# outlier_array'ın döndürdüğü True-False değerler ile filtreleme yapılarak Outlier gözlemler ile DATAFRAME oluşturuluyor
outlier_df = dff[outlier_array]
len(outlier_df)
outlier_df 
# outlier_df indexlerinden arındırılarak ARRAY'a dönüştürülüyor.
outlier_df.to_records(index=False)
# Bu array res olarak tutuluyor.
res = outlier_df.to_records(index=False)
# res'deki tüm veriler yerine baskılama dergerleri atanıyor
res[:] = baskılama_deg.to_records(index=False)
res
dff[outlier_array]
# Bir array olan res aykırı gözlemlerin indexleri kullanılarak DATAFRAME dönüştürülüyor ve dff deki aykırı gözlemlerin yerine atanyor
dff[outlier_array] = pd.DataFrame(res, index = dff[outlier_array].index)
dff[outlier_array]
df_app = df[['data_type','SK_ID_CURR','TARGET']]
print(df_app.shape,dff.shape,df_cat.shape)
list(dff.columns)
list(df_cat.columns)
display(dff.sample(10))
display(df_cat.sample(10))
display(df.head())
dff.insert(0,'data_type',df['data_type'].values)#1.kolona ekledik
dff.insert(1,'SK_ID_CURR',df['SK_ID_CURR'].values)  #2.kolona ekledik
dff.insert(2,'TARGET',df['TARGET'].values)      #3.kolona ekledik
df_cat.insert(0,'SK_ID_CURR',df['SK_ID_CURR'].values)   #kategorik dataya id ekledik merge yapmak icin
df_cat.head()
df1 = pd.merge( dff, df_cat, on='SK_ID_CURR')  #kategorik ve numerik ler birlestirildi
df1.head()
df1.isnull().sum()
df1.to_csv (r'C:\Users\Dell\Desktop\proje\traintest.csv', index = False, header=True)

df1.head()
#df1.to_csv(r'C:\Users\Dell\Desktop\proje\traintest.csv', index = False)
