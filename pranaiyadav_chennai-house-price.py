import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/chennai-house-pricing-/train.csv')

df.head()
df.info()
df.describe()
df.isna().sum()
df.corr()['SALES_PRICE']
def numerical(feature):

    sns.scatterplot(x=df[feature],y=df.SALES_PRICE,hue=df.REG_FEE,palette='Spectral')

def categorical(feature):

    plt.figure(figsize=(15,5))

    sns.barplot(x=df[feature],y=df.SALES_PRICE)
numerical('INT_SQFT')
numerical('REG_FEE')
sns.barplot(x=df.N_BEDROOM,y=df.SALES_PRICE)
sns.barplot(x=df.N_ROOM,y=df.SALES_PRICE)
numerical('COMMIS')
df.DATE_BUILD = pd.to_datetime(df.DATE_BUILD)

df.DATE_SALE = pd.to_datetime(df.DATE_SALE)

df['build_to_sale']= (df.DATE_SALE - df.DATE_BUILD).dt.days

df.head()
li = df.corr()['SALES_PRICE'].index

for i in df.columns:

    if i not in li:

        print(i)
categorical('AREA')
categorical('SALE_COND')
categorical('BUILDTYPE')
categorical('PARK_FACIL')
categorical('UTILITY_AVAIL')
categorical('STREET')
categorical('MZZONE')
df.head()
df.corr()['SALES_PRICE'].sort_values(ascending=False)
sns.jointplot(df.REG_FEE,df.SALES_PRICE,kind='reg')
sns.jointplot(df.COMMIS,df.SALES_PRICE,kind='reg')
sns.jointplot(df.INT_SQFT,df.SALES_PRICE,kind='reg')
sns.boxplot(df.N_ROOM,df.SALES_PRICE)
df[df.N_ROOM<6].corr()['SALES_PRICE'].sort_values(ascending=False)
sns.boxplot(df.N_BEDROOM,df.SALES_PRICE)
df.info()
df.AREA.value_counts()
area = {'Chrompt':'Chrompet','Chormpet':'Chrompet','Chrmpet':'Chrompet','TNagar':'T Nagar','Ana Nagar':'Anna Nagar','Karapakam':'Karapakkam','Ann Nagar':'Anna Nagar','Velchery':'Velachery','KKNagar':'KK Nagar','Adyr':'Adyar'}

df.AREA = df.AREA.replace(area)
df.SALE_COND.value_counts()
cond = {'Adj Land':'AdjLand','Ab Normal':'AbNormal','Partiall':'Partial','PartiaLl':'Partial'}

df.SALE_COND = df.SALE_COND.replace(cond)
df.PARK_FACIL.value_counts()
park = {'Noo':'No'}

df.PARK_FACIL = df.PARK_FACIL.replace(park)
df.BUILDTYPE.value_counts()
df.BUILDTYPE = df.BUILDTYPE.replace({'Comercial':'Commercial','Other':'Others'})
df.UTILITY_AVAIL.value_counts()
df.UTILITY_AVAIL = df.UTILITY_AVAIL.replace({'All Pub':'AllPub'})
df.STREET.value_counts()
df.STREET = df.STREET.replace({'NoAccess':'No Access','Pavd':'Paved'})
df.MZZONE.value_counts()
df.QS_OVERALL = (df.QS_BATHROOM + df.QS_BEDROOM + df.QS_ROOMS)/3

df[df.QS_OVERALL.isna()]
df[df.N_BEDROOM.isna()]
df[(df.AREA=='Anna Nagar') & (df.N_ROOM==4) & (df.N_BATHROOM==1)]
df.N_BEDROOM.fillna(1.0,inplace=True)
df[df.N_BATHROOM.isna()]
df[(df.AREA=='Chrompet')&(df.N_BEDROOM==1)&(df.N_ROOM==3)].N_BATHROOM.value_counts()
df.N_BATHROOM.fillna(1.0,inplace=True)
df.drop(['PRT_ID','DATE_SALE','DATE_BUILD'],axis=1,inplace=True)

df.head()
df = pd.get_dummies(df,drop_first=True)

df.head()
df.corr()['SALES_PRICE'].sort_values(ascending=False)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score, KFold

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
X = df.drop('SALES_PRICE',axis=1)

y = df.SALES_PRICE
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,random_state=0,test_size=0.2)
kf = KFold(n_splits=10,shuffle=True)
lr = LinearRegression().fit(X_train,y_train)

cs = cross_val_score(lr,X_train,y_train,cv=kf)
cs.mean()
lr.score(X_train,y_train), lr.score(X_valid,y_valid)
rr = RandomForestRegressor(n_estimators=500).fit(X_train,y_train)

rr.score(X_train,y_train), rr.score(X_valid,y_valid)
xx = XGBRegressor().fit(X_train,y_train)

xx.score(X_train,y_train), xx.score(X_valid,y_valid)