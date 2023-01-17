import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#flag missing data
na_sentinels = {'SALEDATE':['1900-01-01T00:00:00.000Z'],'PRICE':[0,''],'AYB':[0],'EYB':[0]}

#three data sources
residential = pd.read_csv("../input/raw_residential_data.csv", index_col='SSL', na_values=na_sentinels).drop(columns=["OBJECTID"])
condo = pd.read_csv("../input/raw_condominium_data.csv", index_col='SSL', na_values=na_sentinels).drop(columns=["OBJECTID"])
address = pd.read_csv("../input/raw_address_points.csv")

residential["SOURCE"] = "Residential"
condo["SOURCE"] = "Condominium"

df = pd.concat([residential,condo], sort=False)
df.info()
#Identify all categorical variables
categories = [['CNDTN_D','CNDTN'],['HEAT_D','HEAT'],['STYLE_D','STYLE'],['STRUCT_D','STRUCT'],['GRADE_D','GRADE'],['ROOF_D','ROOF'],['EXTWALL_D','EXTWALL'],['INTWALL_D','INTWALL']]
cat_drop = []
for c in categories:
    df[c[1]] = df[c[0]].astype('category')
    cat_drop.append(c[0])

df['SOURCE'] = df['SOURCE'].astype('category')    
#eliminate redundant dummy variables
df.drop(cat_drop, inplace=True, axis=1)
print(df.isnull().sum())
df.dropna(subset=['ROOMS','BEDRM','BATHRM','HF_BATHRM','FIREPLACES','EYB','QUALIFIED'], inplace=True)

print(df.isnull().sum())
int_col = ['BATHRM','HF_BATHRM','ROOMS','BEDRM','EYB','SALE_NUM','BLDG_NUM','FIREPLACES','LANDAREA']
#con_col = ['BATHRM','HF_BATHRM','NUM_UNITS','ROOMS','BEDRM','EYB','STORIES','SALE_NUM','KITCHENS','FIREPLACES','LANDAREA']

for i in int_col:
    df[i] = df[i].astype('int64')
print(df["SALEDATE"].sort_values(ascending=True).head(5))
print(df["SALEDATE"].sort_values(ascending=False).head(5))
import re

def valid_datestring(x):
    if re.match(r'(19|20)\d{2}\-',str(x)):
        return x
    else:
        return None

df["SALEDATE"] = df['SALEDATE'].map(valid_datestring)
df['GIS_LAST_MOD_DTTM'] =  df['GIS_LAST_MOD_DTTM'].map(valid_datestring)
df['SALEDATE'] = pd.to_datetime(df['SALEDATE'], dayfirst=False)
df['GIS_LAST_MOD_DTTM'] = pd.to_datetime(df['GIS_LAST_MOD_DTTM'], dayfirst=False)
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-colorblind')
# Basic correlogram
sns.pairplot(df[['ROOMS','BATHRM','HF_BATHRM','BEDRM']], kind="scatter", diag_kind = 'kde', plot_kws = {'alpha': 0.33, 's': 80, 'edgecolor': 'k'}, size = 4)
plt.show()
df = df[( (df["ROOMS"]<100) & (df["ROOMS"]>=df["BEDRM"]) & (df["BATHRM"]<24) )]
#df.head()
address.head(5)
address_subset = address.drop_duplicates(['SSL'], keep='last').set_index("SSL")[["FULLADDRESS","CITY","STATE","ZIPCODE","NATIONALGRID","LATITUDE","LONGITUDE","ASSESSMENT_NBHD","ASSESSMENT_SUBNBHD","CENSUS_TRACT","CENSUS_BLOCK","WARD"]]
premaster = pd.merge(df,address_subset,how="left",on="SSL")
address["SQUARE"] = address["SQUARE"].apply(lambda x: str(x)[0:4])

address_impute = address[((address["SQUARE"]!="0000") & (address["SQUARE"].str.match(r'\d+')) )] \
    .groupby("SQUARE") \
    .agg({'X':'median','Y':'median','QUADRANT':'first','ASSESSMENT_NBHD':'first','ASSESSMENT_SUBNBHD':'first','CENSUS_TRACT': 'median','WARD':'first','ZIPCODE':'median','LATITUDE':'median','LONGITUDE':'median'})  

print(address_impute.head())
#create a SQUARE key on premaster
premaster["SQUARE"] = df.apply(axis=1, func=lambda x: str(x.name)[0:4]) 
master = pd.merge(premaster,address_impute,how="left",on="SQUARE", suffixes=('', '_impute')) \

cols_to_impute = ["CENSUS_TRACT","LATITUDE","LONGITUDE","ZIPCODE","WARD","ASSESSMENT_NBHD","ASSESSMENT_SUBNBHD"]
for c in cols_to_impute:
    master[c] = master[c].fillna(master[(c + "_impute")])

master.drop(["CENSUS_TRACT_impute","LATITUDE_impute","LONGITUDE_impute","ZIPCODE_impute","WARD_impute","ASSESSMENT_NBHD_impute","ASSESSMENT_SUBNBHD_impute"],axis=1,inplace=True)
master.describe()
master.to_csv("DC_Properties.csv", header=True)