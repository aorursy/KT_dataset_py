# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression



from sklearn.linear_model import Ridge



import matplotlib.pyplot as plt



from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings('ignore')



from sklearn.ensemble import RandomForestRegressor



from sklearn.preprocessing import OneHotEncoder



from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split



import plotly.express as px

import plotly.graph_objects as go



from matplotlib.ticker import FormatStrFormatter

zillow_data_dictionary = pd.read_excel('../input/zillow-prize-1/zillow_data_dictionary.xlsx')





train_2016_v2 = pd.read_csv('../input/zillow-prize-1/train_2016_v2.csv')

properties_2016 = pd.read_csv('../input/zillow-prize-1/properties_2016.csv')

sample_submission = pd.read_csv('../input/zillow-prize-1/sample_submission.csv')

train_2017 = pd.read_csv('../input/zillow-prize-1/train_2017.csv')

properties_2017 = pd.read_csv('../input/zillow-prize-1/properties_2017.csv')

zillow_data_dictionary.head()
properties_2016.shape
train_2016_v2.head()
properties_2016.head()
plt.figure(figsize = (6,6))

sns.countplot(x="bedroomcnt", data=properties_2016.loc[properties_2016['bedroomcnt']<=7]).set_title('Houses with less than SEVEN bedrooms in 2016')

plt.show()
plt.figure(figsize = (5,5))

sns.countplot(x="bedroomcnt", data=properties_2016.loc[properties_2016['bedroomcnt']>7]).set_title('Houses with more than SEVEN bedrooms in 2016')

plt.legend()

plt.show()
ax_list = train_2016_v2.hist(figsize = (7,7), column = 'logerror',range=[-.5,.5], color='blue', bins=50)

#Rename features 

properties_2016.rename(columns=

{

 'yearbuilt' :'build_year' ,

 'basementsqft': 'area_basement' ,

 'yardbuildingsqft17': 'area_patio'  ,

 'yardbuildingsqft26': 'area_shed'  , 

  'poolsizesum':'area_pool'  ,  

 'lotsizesquarefeet': 'area_lot'  , 

  'garagetotalsqft':'area_garage'  ,

 'finishedfloor1squarefeet': 'area_firstfloor_finished'  ,

 'calculatedfinishedsquarefeet' :'area_total_calc'  ,

 'finishedsquarefeet6' :'area_base'  ,

 'finishedsquarefeet12': 'area_live_finished'  ,

 'finishedsquarefeet13': 'area_liveperi_finished'  ,

'finishedsquarefeet15' : 'area_total_finished'  ,  

'finishedsquarefeet50':  'area_unknown'  ,

 'unitcnt' : 'num_unit' , 

 'numberofstories': 'num_story'  ,  

'roomcnt' : 'num_room' ,

 'bathroomcnt' :'num_bathroom' ,

'bedroomcnt' : 'num_bedroom' ,

 'calculatedbathnbr': 'num_bathroom_calc' ,

   'fullbathcnt':'num_bath' ,  

  'threequarterbathnbr':'num_75_bath'  , 

 'fireplacecnt': 'num_fireplace'  ,

'poolcnt' : 'num_pool'  ,  

 'garagecarcnt' :'num_garage'  ,  

 'regionidcounty': 'region_county' ,

 'regionidcity' :'region_city'  ,

 'regionidzip' :'region_zip'  ,

 'regionidneighborhood': 'region_neighbor'  ,  

  'taxvaluedollarcnt':'tax_total'  ,

 'structuretaxvaluedollarcnt': 'tax_building'  ,

'landtaxvaluedollarcnt' : 'tax_land'  ,

 'taxamount' :'tax_property' ,

  'assessmentyear':'tax_year'  ,

  'taxdelinquencyflag':'tax_delinquency'  ,

'taxdelinquencyyear' : 'tax_delinquency_year' ,

'propertyzoningdesc':  'zoning_property' ,

'propertylandusetypeid'  :'zoning_landuse'  ,

 'propertycountylandusecode': 'zoning_landuse_county'  ,

'fireplaceflag'  :'flag_fireplace'  , 

  'hashottuborspa': 'flag_tub' ,

'buildingqualitytypeid' : 'quality'  ,

 'buildingclasstypeid' :'framing'  ,

'typeconstructiontypeid':  'material'  ,

 'decktypeid'  :'deck' ,

 'storytypeid': 'story' ,

 'heatingorsystemtypeid' :'heating' ,

'airconditioningtypeid' : 'aircon' ,

'architecturalstyletypeid' : 'architectural_style' }, inplace=True)
plt.figure(figsize=(10,10))

sns.countplot(x="num_bathroom", data=properties_2016)

formatter = FormatStrFormatter("%.0f")

plt.gca().xaxis.set_major_formatter(formatter)
properties_2016.head()
train_2016_v2['absolute value of logerror']=train_2016_v2['logerror'].abs()
train_2016_v2['date']=pd.to_datetime(train_2016_v2['transactiondate'])
train_2016_v2['month']=train_2016_v2['date'].dt.month
train_2016_v2.head()
missing_values = pd.isnull(properties_2016).sum()/len(properties_2016)
miss_values = pd.DataFrame(data = missing_values,index = properties_2016.columns, columns =['missing values per column'])

miss_values.head()
miss_val = miss_values.sort_values(by='missing values per column', ascending=False)
miss_val.head()
plt.figure(figsize=(15,15))

sns.barplot(x='missing values per column',y=miss_val.index, data =miss_val)

plt.show()
#When were the houses built?

#Let’s plot the distribution of build year for the houses. 

#Most houses were built around 1950. There are not many older houses, neither many new houses >2000.

plt.figure(figsize= (6,6))

plt.xlim(1880, 2016)



sns.distplot(properties_2016.build_year, kde=False, color='red')

plt.show()
#When were the houses built?



plt.figure(figsize= (6,6))



sns.kdeplot(properties_2016.build_year,  color='red',linewidth=3)

plt.xlim(1870, 2016)

plt.legend()

plt.show()
print("Oldest house built in:",int(properties_2016.build_year.min()))
train_2016_v2['year']= train_2016_v2['date'].dt.year
train_2016_v2.head()
#How does absolute log error change with time

abs_log_error_month = train_2016_v2[['absolute value of logerror', 'month']].groupby('month').mean()

abs_log_error_month
#How does absolute log error change with time

plt.figure(figsize=(6.5,6.5))

log_error_month = train_2016_v2[['logerror', 'month']].groupby('month').mean()

sns.pointplot(x=log_error_month.index,y=log_error_month['logerror'], data=log_error_month,color='red')

plt.xlabel('Months in 2016')

plt.show()
train_2016_v2.head()
#Distribution of transaction dates



sns.countplot(x="month", data=train_2016_v2,color='red')

plt.xlabel('transaction in each month in 2016')

plt.show()
new_merge = pd.merge(train_2016_v2, properties_2016, on='parcelid')

new_merge.head()
new_merge.info()
new_merge.describe()
#How does the absolute logerror change with build_year?



#First, let’s join the two tables together on the parcelid column 

# such that we only include properties that have a target value in train_2016.



logerror_buildyear = new_merge[['absolute value of logerror', 'build_year']].groupby('build_year').mean()



#sns.pointplot(x=log_error_month.index,y=log_error_month['logerror'], data=logerror_buildyear,color='red')

logerror_buildyear.head()
plt.figure(figsize=(10,10))

sns.relplot(x=logerror_buildyear.index, y="absolute value of logerror", ci="sd",kind="line", data=logerror_buildyear)

plt.xlabel('build_year')

plt.show()
plt.figure(figsize=(8,8))

sns.scatterplot(x=logerror_buildyear.index, y="absolute value of logerror",  data=logerror_buildyear, color='red',s=100)

plt.xlim(1880,2016)

logerror_buildyear.reset_index(inplace=True)
logerror_buildyear.head()
sns.lmplot(x="build_year", y="absolute value of logerror",size=8, data=logerror_buildyear,line_kws={'color': 'red'}, scatter_kws={'color': 'red'})

plt.xlim(1880,2016);

reallogerror_buildyear = new_merge[['logerror', 'build_year']].groupby('build_year').mean()

reallogerror_buildyear.reset_index(inplace=True)
reallogerror_buildyear.head()
sns.lmplot(x="build_year", y="logerror",size=8, data=reallogerror_buildyear,line_kws={'color': 'red'}, scatter_kws={'color': 'red'})

plt.xlim(1880,2016);

plt.ylim(-.02,0.07);

sns.relplot(x="latitude", y="absolute value of logerror", data=new_merge,kind='line')

plt.show()
sns.countplot(x="num_bedroom" ,data=new_merge )

formatter = FormatStrFormatter("%.0f")

plt.gca().xaxis.set_major_formatter(formatter)
plt.figure(figsize=(6,6))

sns.boxplot(x="num_bedroom" ,y="logerror", data=new_merge, )

plt.ylim(-.4,.4)



formatter = FormatStrFormatter("%.0f")

plt.gca().xaxis.set_major_formatter(formatter)

fig = px.scatter(new_merge,  x='longitude',y='latitude')

fig.show()
train_2016_v2.head()
fig = px.line(new_merge, x='date', y='absolute value of logerror')

fig.show()
plt.figure(figsize=(9,9))

sns.jointplot(x='longitude',y='latitude', data=new_merge, color ='red', alpha=0.1);

plt.show()
corr_list = new_merge.corr()['logerror'].sort_values(ascending=False)
corr = pd.DataFrame(data= corr_list, index = corr_list.index)
corr[corr.index[0]]
corr[1:].plot(kind='barh',y='logerror', color='red',fontsize=14,  figsize=(15,15), width=0.8)

plt.show()
#Using XGBoost

import xgboost as xgb
new_merge.head()
pd.set_option('display.max_rows', 500)



(pd.isnull(new_merge).sum()/len(new_merge) *100).sort_values(ascending=False)
A = (pd.isnull(new_merge).sum()/len(new_merge)<0.3).reset_index()

new_merge = new_merge[A[A[0]==True]['index']]
new_merge.columns
y = new_merge['logerror']
new_merge.drop(columns=['parcelid', 'transactiondate','absolute value of logerror'], inplace=True)
new_merge.drop(columns=['date'], inplace=True)
new_merge.fillna(0, inplace=True)
new_merge.info()
#Convert categorical variable into dummy/indicator variables.

X_dummy = pd.get_dummies(new_merge.drop(['logerror'], axis=1), drop_first=True)
#Standardize features by removing the mean and scaling to unit variance

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_dummy)
data_dmatrix = xgb.DMatrix(data=X_scaled,label=y)



params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}



cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,

                    num_boost_round=50, early_stopping_rounds=10, metrics="mae", as_pandas=True, seed=42, verbose_eval =True)