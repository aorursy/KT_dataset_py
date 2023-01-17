# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import re

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, pairwise_distances

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from pandas.tools.plotting import scatter_matrix

from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split as split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import MaxAbsScaler, LabelEncoder



import warnings



warnings.filterwarnings('ignore')

%matplotlib inline
# use error_bad_lines to get rid of error in line 821

phones = pd.read_csv("../input/phone_dataset .csv",error_bad_lines=False)
phones.head()
phones.info()
# number of nulls in each feature

phones.isnull().sum()
# number of unique values in each feature

phones.nunique()
# remove the column img_url (a url for the image of the model)

phones.drop(columns=['img_url'],inplace=True)
# switch weight columns into floats

phones.weight_g = phones.weight_g.convert_objects(convert_numeric=True)

phones.weight_oz = phones.weight_oz.convert_objects(convert_numeric=True)
# check for mismatch in weight conversion from oz to gr up to 0.03

# how should we treat mismatched weight data ???

phones[abs(0.03527396 * (phones.weight_g) - (phones.weight_oz)) >= 0.03]
# check what type of strings show up in battery column

phones.battery.head()
# a function to turn removable battery data into a boolean feature

def removable(battery):

    if battery[0] == 'Removable':

        return True

    elif battery[0] == 'Non-removable':

        return False

    else:

        return np.nan
# looking for regular expressions in battery column strings

# the method 'findall' returns a list with a single element. that element is a tuple of the regular expressions found

a = phones.battery.astype(str).apply(lambda x: 

                                        (re.findall('(Non-removable|Removable|)\s*(Li-Ion|Li-Po|Li-ion|Li-po|NiMH|)\s*(\d*)',x))) 



phones['battery_removable'] = a.apply(lambda x: removable(x[0])) # the first regex indicates 'removable' / 'non-removable'

phones['battery_type'] = a.apply(lambda x: (x[0])[1]) # the second regex indicates 'Li-Ion' / 'Li-Po' / 'Li-ion' / 'Li-po' / 'NiMH'

phones['battery_mah'] = a.apply(lambda x: ((x[0])[2])).convert_objects(convert_numeric=True) # the third regex gives the numerical data for mah

# the battery column was translated into three new features:

# 1. battery_removable (boolean)

# 2. battery_type (string): Li-Ion / Li-Po / Li-ion / Li-po / NiMH

# 3. battery_mah (float)

phones.head()
# now we can remove the original battery column...

phones.drop(columns=['battery'],inplace=True)
# begin work on display_resolution feature

phones.display_resolution.head()
# extract regex from display_resolution feature

a = phones.display_resolution.astype(str).apply(lambda x: 

                                        (re.findall('(\d*\.\d*)\s*inches(\s*\(.(\d*\.\d*)|)',x))) 



# look at the empty regexes we got (1233 in total)

a[a.apply(lambda x: len(x) == 0)]
# there are origianlly 1214 NaNs in the display_resolution column 

# regex did not work for additional 19 values

# let's look at those values ...

phones.display_resolution.isnull().sum()
# here are the 19 "weird" display_resolution values that are not NaNs

# conclusion: we're not missing anything by getting the empty regex for these

phones.display_resolution[~phones.display_resolution.isnull() & a.apply(lambda x: len(x) == 0)]
# functions to parse data from display_resolution regex



def display_resolution_inches(regex):

    if regex == []:

        return np.nan

    else:

        return float((regex[0])[0])

    

def screen_to_body_ratio(regex):

    if regex == []:

        return np.nan

    elif regex[0][-1] is None:

        return np.nan

    else:

        return (regex[0][-1])
phones['display_resolution_inches'] = a.apply(lambda x: display_resolution_inches(x))

phones['screen_to_body_ratio_%'] = a.apply(lambda x: screen_to_body_ratio(x)).convert_objects(convert_numeric=True)
# the display_resolution was translated into two new features:

# 1. display_resolution_inches (float)

# 2. screen_to_body_ratio_% (float)

phones.head()
# now we can remove the original display_resolution column...

phones.drop(columns=['display_resolution'],inplace=True)
phones.dimentions.isnull().sum()
# weird '-' as a value in 'dimentions'

# there are 331 '-' and 19 nulls, that leaves us with 8628-331-19=8278 valid values

phones.dimentions[phones.dimentions == '-'].value_counts()
# extract regex from "dimentions" feature

# eliminate 'mm' to include min value of thickness, we get two tuples, one for mm and one for inches

a = phones.dimentions.astype(str).apply(lambda x: 

                                        (re.findall('(\d*\.\d*|\d*)\sx\s(\d*\.\d*|\d*)\sx\s(\d*\.\d*|\d*)',x))) 

# a = phones.dimentions.astype(str).apply(lambda x: 

#                                         (re.findall('(\d*\.\d*|\d*)\sx\s(\d*\.\d*|\d*)\sx\s(\d*\.\d*|\d*)\smm',x))) 

a.head()
phones.dimentions.head()
# check which regex reads failed (but not because there was a NaN or '-')

phones.dimentions[a.apply(lambda x: len(x) == 0) & ~phones.dimentions.isnull() & (phones.dimentions != '-')]
# process mm thickness-only info

b = phones.dimentions.astype(str).apply(lambda x: 

                                        (re.findall('(\d*\.\d*|\d*)\smm\sthickness',x)))
# convert regex data into numerical data

# return value as a list

def dimension_conversion(regex):

    if len(regex) == 0: # empty regex

        return []

    elif len(regex) == 1 & isinstance(regex[0],str): # regex coming from thickneww only data

        new_regex = []

        try:            

            new_regex.append(float(regex[0]))

        except:

            pass

        return new_regex

    elif (len(regex) >= 1) & isinstance(regex[0],tuple): # regex coming from full dimension data

        new_regex = list(regex[0])

        for i in range(len(new_regex)):

            try:

                new_regex[i] = float(regex[0][i])

            except:

                pass

        return new_regex

    else:

        return []
# combine dimensions data

# data that did not match one of the following (but wasn't NaN) was converted into NaN

# 1. number x number x number mm

# 2. number mm thickness

# 3. '-'

c = a.apply(lambda x: dimension_conversion(x)) + b.apply(lambda x: dimension_conversion(x))
# define functions to extract data on each dimension

def length(x):

    if len(x) == 3:

        if x[0] is not None:

            return x[0]

        else:

            return np.nan

    else:

        return np.nan

    

def width(x):

    if len(x) == 3:

        if x[1] is not None:

            return x[1]

        else:

            return np.nan

    else:

        return np.nan

    

def thickness(x):

    if len(x) == 3:

        if x[2] is not None:

            return x[2]

        else:

            return np.nan

    elif len(x) == 1:

        if x[0] is not None:

            return x[0]

        else:

            return np.nan

    else:

        return np.nan
length = c.apply(lambda x: length(x))
width = c.apply(lambda x: width(x))
thickness = c.apply(lambda x: thickness(x))
# test thickness function against original b values

thickness[c.apply(lambda x: len(x) == 1)]
# test thickness function against NaNs

thickness[c.apply(lambda x: len(x) == 0)]
phones['length_mm'] = length

phones['width_mm'] = width

phones['thickness_mm'] = thickness
phones.drop(columns=['dimentions'],inplace=True)
phones.internal_memory.value_counts()
# process internal_memory info

a = phones.internal_memory.astype(str).apply(lambda x: 

                                        (re.findall('(?i)(\d*|\d*\.\d*)\/*(\d*)\/*(\d*)\s(?=GB)|(\d*|\d*\.\d*)\/*(\d*)\/*(\d*)\s(?=MB)|(\d*|\d*\.\d*)\/*(\d*)\/*(\d*)\s(?=KB)',x,re.IGNORECASE)))
a.head(10)
(a.iloc[3][0][3]=='')
# functions for extracting internal_memory string data broken into types: GB / MB / KB



def GB(regex):

    gb = []

    for i in range(3):

        for j in range(2):

            try:

                gb.append(float(regex[j][i])) # GB found in regex groups 1, 2, and 3

            except:

                pass

    return gb



def MB(regex):

    mb = []

    for i in range(3):

        for j in range(2):

            try:

                mb.append(float(regex[j][i+3])) # MB found in regex groups 4,5,6

            except:

                pass

    return mb



def KB(regex):

    kb = []

    for i in range(3):

        for j in range(2):

            try:

                kb.append(float(regex[j][i+6])) # KB found in regex groups 7,8,9

            except:

                pass

    return kb



def GB_MB_KB(regex):

    return [GB(regex),MB(regex),KB(regex)]
a.apply(lambda x: GB_MB_KB(x))
# test internal_memory with mixed types (eg, GB and MB)

a[a.apply(lambda x: len(x) == 2)]
# apply BG_MB_KB to mixed type internal_memory strings to check validity

a[a.apply(lambda x: len(x) == 2)].apply(lambda x: GB_MB_KB(x))
a_GB = a.apply(lambda x: GB_MB_KB(x)[0])

a_MB = a.apply(lambda x: GB_MB_KB(x)[1])

a_KB = a.apply(lambda x: GB_MB_KB(x)[2])
a_GB.head()
a_MB.iloc[[180,1354,7351,7381]]
# test KB types

a_KB[a.apply(lambda x: GB_MB_KB(x)[2] != [])]
# generate three seperate columns including all internal_memory options of a given type (GB / MB / KB) as floats

# in the next step, turn categorical...

phones["internal_memory_BG"] = a_GB 

phones["internal_memory_MB"] = a_MB

phones["internal_memory_KB"] = a_KB
phones.RAM.value_counts()
phones_new=phones[phones['approx_price_EUR'].isnull()==0]
sns.heatmap(phones_new.isnull(),yticklabels=False,cbar=False,cmap='viridis')
phones_new.drop('NFC',axis=1,inplace=True)
phones_new.drop('4G_bands',axis=1,inplace=True)
# GPU  (graphics processing unit) column will be categorial type



a_regex = '(""|Adreno|Intel|Mali|Broadcom|PowerVR|ULP|Vivante|Nvidia|Kepler|3D|Mediatek|SGX531u|VideoCore)'

GPU_replacements = {'':0, 'Adreno':1,'Intel':2,'Mali':3,'Broadcom':4,'PowerVR':5,'ULP':6,'Vivante':7,'Nvidia':8,'Kepler':9

                   ,'3D':10,'Mediatek':11,'SGX531u':12,'VideoCore':13}



#Extract Using Regex

phones_new['GPU_new'] = phones_new['GPU'].str.extract(a_regex).fillna('')

#Look up values from dictionary

phones_new['GPU_new'] = phones_new['GPU_new'].apply(lambda x: GPU_replacements.get(x,''))

#Use default value from other coumn if no other value

phones_new['GPU_new'].value_counts()
# indexing brands

brands=[]

phones_new['brand_idx']=999

x=1

brands.append(phones_new['brand'].iloc[0])

phones_new['brand_idx'].iloc[0]=1

i=1

for i in range(1,len(phones_new['brand'])-1):

    if phones_new['brand'].iloc[i] in brands:

        phones_new['brand_idx'].iloc[i]=brands.index(phones_new['brand'].iloc[i])+1

    else:

        x=x+1

        brands.append(phones_new['brand'].iloc[i])

        phones_new['brand_idx'].iloc[i]=x
# indexing models - irrelavant

# models=[]

# phones_new['model_idx']=999

# x=1

# models.append(phones_new['model'].iloc[0])

# phones_new['model_idx'].iloc[0]=1

# i=1

# for i in range(1,len(df_new['model'])-1):

#     if phones_new['model'].iloc[i] in models:

#         phones_new['model_idx'].iloc[i]=models.index(phones_new['model'].iloc[i])+1

#     else:

#         x=x+1

#         models.append(phones_new['model'].iloc[i])

#         phones_new['model_idx'].iloc[i]=x
phones_new['year'] = phones_new['announced'].str.extract('(\d\d\d\d)', expand=True)

phones_new['year'].fillna(phones_new['year'].value_counts().index[0],inplace=True)

phones_new['year'].value_counts()
a_regex = '(January|February|March|April|May|June|July|August|September|October|November|December)'

month_replacements = {'January':1, 'February':2,

            'March':3,'April':4 ,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11, 'December':12}



#Extract Using Regex

phones_new['Month'] = phones_new['announced'].str.extract(a_regex).fillna('')

#Look up values from dictionary

phones_new['Month'] = phones_new['Month'].apply(lambda x: month_replacements.get(x,''))

#Use default value from other coumn if no other value
# create a list of ratio of each month in data to complete missing data

ratio=[]

for i in range (1,13):

    a=(phones_new['Month']==i).sum()/(sum(phones_new['Month'].value_counts())-(phones_new['Month']=="").sum())*(phones_new['Month']=="").sum()

    ratio.append(round(a))

ratio

# sum(df_new['Month'].value_counts())
#creating new column-Month_new to replace "" with the number of month by ratio

phones_new['Month_new']=999

j=0 # place in list

k=1

l=1

for i in range(0, len(phones_new)):

    if phones_new['Month'].iloc[i]=="":

        if k<=ratio[j]:

            phones_new['Month_new'].iloc[i]=l

            k=k+1

        else:

            k=1

            l=l+1

            j=j+1

    else:

         phones_new['Month_new'].iloc[i]=phones_new['Month'].iloc[i]
phones_new['Month_new'].replace(999,12,inplace=True)

phones_new['Month_new'].value_counts()
#creating a binary column - GPS_new if there is a GPS in model

check_list = ['Yes', 'GPS']

regstr = '|'.join(check_list)

phones_new['GPS_new']=phones_new['GPS']

# df_new['GPS_new'] = np.where(df_new['GPS'].isin(check_list),'YES','NO')

phones_new['GPS_new']=phones_new['GPS_new'].str.contains(regstr, case=False, na=False)
phones_new['primary_cam_MP']=phones_new["primary_camera"].str.split(' ',expand=True)[0]

phones_new['primary_cam_MP'].fillna(0,inplace=True)

phones_new['primary_cam_MP'].replace('2MP',2,inplace=True)

phones_new['primary_cam_MP'].replace('5MP|',5,inplace=True)

phones_new['primary_cam_MP'].replace('600',0,inplace=True)

phones_new['primary_cam_MP'].replace(['SVGA','Yes.','QVGA','Yes|','CIF','No','VGA|','Yes','Dual','VGA'],0,inplace=True)



phones_new['secondary_cam_MP']=phones_new["secondary_camera"].str.split(' ',expand=True)[0]

phones_new['secondary_cam_MP'].fillna(0,inplace=True)
phones_new['secondary_cam_MP'].replace('8MP',8,inplace=True)

phones_new['secondary_cam_MP'].replace('1.3MP',1.3,inplace=True)

phones_new['secondary_cam_MP'].replace(['No','Yes','QCIF','0.','QVGA','Videocalling','CIF','0','VGA|','HD','Dual','VGA','Spy','VGA@15fps','QCIF@15fps','VGA/','720p','Videocall',],0,inplace=True)
# OS  (Operation System) column will be int type



a_regex = '(""|Android|Windows|Firefox|iOS|BlackBerry|Linux|webOS|Sailfish|Nokia|Symbian|Tizen)'

OS_replacements = {'':0, 'Android':1,'Windows':2,'Firefox':3,'iOS':4,'BlackBerry':5,'Linux':6,'webOS':7,'Sailfish':8,'Nokia':9

                   ,'Symbian':10,'Tizen':11}



#Extract Using Regex

phones_new['OS'] = phones_new['OS'].str.extract(a_regex).fillna('')

#Look up values from dictionary

phones_new['OS'] = phones_new['OS'].apply(lambda x: OS_replacements.get(x,''))

#Use default value from other coumn if no other value

# drop rows with NAN and 0

phones_new=phones_new[phones_new['length_mm']!=0]

phones_new=phones_new[phones_new['length_mm']!=""]
#change data types (length_mm,Year, primary_cam_MP, secondary_cam_MP) to numeric

phones_new['year'] = phones_new['year'].apply(lambda x: int(x))

phones_new['primary_cam_MP'] = phones_new['primary_cam_MP'].apply(lambda x: float(x))

phones_new['secondary_cam_MP'] = phones_new['secondary_cam_MP'].apply(lambda x: float(x))

phones_new['length_mm']=phones_new['length_mm'].apply(lambda x: float(x))

# drop columns with over 1,000 empty cells

phones_new.drop(['3G_bands','CPU','network_speed','Chipset','sensors','GPU','RAM'], axis=1, inplace=True)

sns.heatmap(phones_new.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# SUM PRICES OVER THE YEARS

group_year_prices=phones_new.groupby('year')['approx_price_EUR'].sum()

group_year_prices.plot.bar()
# MEAN PRICES PER YEAR

group_year_prices_mean=phones_new.groupby('year')['approx_price_EUR'].mean()

group_year_prices_mean.plot.bar()



# 2003 AND 2017 SEEMS TO BE WITH OUTLIERS - NEED TO BE CHECKED


ax = sns.boxplot(x='year', y='approx_price_EUR', data=phones_new[phones_new['year']==2017])

# OVER 5000 EURO'S - not reasonable
# Checking outliers by months

phones_new_month=phones_new[['Month_new','approx_price_EUR']]

plt.figure(figsize=(12,8))

ax = sns.boxplot(x="Month_new", y="approx_price_EUR", data=phones_new_month,palette="Set3")

plt.show()



#Several samples over ~1,000 euro, to be dropped

# DECLARE RANGES AND CHECK NUMBER OF SAMPLES IN EACH RANGE

bins = [0,100, 200, 300, 400, 500,600,700,800,900,1000, np.inf]

names = ['0-100','101-200', '201-300', '301-400', '401-500', '501-600','601-700','701-800','801-900','901-1000','1001+']



phones_new['Range'] = pd.cut(phones_new['approx_price_EUR'], bins, labels=names)

phones_new['Range'].value_counts(dropna=False)
# drop rows where price is over 1000 EURO

phones_new = phones_new.drop(phones_new[(phones_new['approx_price_EUR'] >=1000)].index)
# Check prices after dropping outliers - yearly

phones_new_year=phones_new[['year','approx_price_EUR']]

plt.figure(figsize=(22,8))

ax = sns.boxplot(x="year", y="approx_price_EUR", data=phones_new_year,palette="Set3")

plt.show()
# Check prices after dropping outliers - monthly

phones_new_month=phones_new[['Month_new','approx_price_EUR']]

plt.figure(figsize=(12,8))

ax = sns.boxplot(x="Month_new", y="approx_price_EUR", data=phones_new_month,palette="Set3")

plt.show()
# creating subdata of numeric and bool features to see corr

phones_corr=phones_new[['approx_price_EUR','OS','weight_g','weight_oz', 'GPS_new', 'primary_cam_MP', 'secondary_cam_MP','GPU_new', 'battery_mah','battery_removable',

       'display_resolution_inches', 'screen_to_body_ratio_%', 'length_mm','brand_idx']]

phones_corr.corr(method ='kendall')
# Heatmap of corr - check important features

corr=phones_corr.corr()

fig=plt.figure(figsize=(10,6))

ax=fig.add_subplot(111)

cax=ax.matshow(corr,cmap='coolwarm',vmin=-1, vmax=1)

fig.colorbar(cax)

ticks=np.arange(0,len(phones_corr.columns)-1,1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(phones_corr.columns)

ax.set_yticklabels(phones_corr.columns)

plt.show()
phones_ready=phones_new[['approx_price_EUR','OS','weight_g','weight_oz', 'GPS_new', 'primary_cam_MP', 'secondary_cam_MP','GPU_new', 'battery_mah','battery_removable',

       'display_resolution_inches', 'screen_to_body_ratio_%', 'length_mm','width_mm','brand_idx']]

phones_ready=phones_ready.dropna(subset=['screen_to_body_ratio_%', 'battery_mah','length_mm','width_mm','battery_removable'])



phones_ready['weight_g'].fillna(phones_ready['weight_g'].mean(),inplace=True)

phones_ready['weight_oz'].fillna(phones_ready['weight_oz'].mean(),inplace=True)

phones_ready['battery_removable'] = phones_ready['battery_removable'].apply(lambda x: bool(x))
# Define ranges to each group for classification

bins = [0,50, 100, 150, 200, 250,300,350,400,450,500, np.inf]

names = [50,100, 150, 200, 250,300,350,400,450,500,800]



phones_ready['price_Range'] = pd.cut(phones_ready['approx_price_EUR'], bins, labels=names)

phones_ready['price_Range'].value_counts(dropna=False)


phones_ready.info()