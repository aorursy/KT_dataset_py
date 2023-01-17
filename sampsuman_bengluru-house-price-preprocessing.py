# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Data Loading

df = pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
# Making a copy pf dataframe as data

data=df.copy()
# Data Info

data.info()
data.isna().sum()/data.shape[0]*100
# Dealing Missing Values.

data.isna().sum()
# Society Column-

data['society'].fillna("Info Not available",inplace = True)
# Location Column--

data[data.location.isna()]
# Looking for same society

data[data.society=='Grare S']
data.location.fillna('Anantapura',inplace=True)
# Size Column--

data[data['size'].isna()]
data[data.society=='Orana N']
data.area_type.unique()
# Will look in missing Value later.
# Total Sqft column- dealing with range value data and other sq. unit data.
def preprocess_total_sqft(my_list):

    if len(my_list) == 1:

        

        try:

            return float(my_list[0])

        except:

            strings = ['Sq. Meter', 'Sq. Yards', 'Perch', 'Acres', 'Cents', 'Guntha', 'Grounds']

            split_list = re.split('(\d*.*\d)', my_list[0])[1:]

            area = float(split_list[0])

            type_of_area = split_list[1]

            

            if type_of_area == 'Sq. Meter':

                area_in_sqft = area * 10.7639

            elif type_of_area == 'Sq. Yards':

                area_in_sqft = area * 9.0

            elif type_of_area == 'Perch':

                area_in_sqft = area * 272.25

            elif type_of_area == 'Acres':

                area_in_sqft = area * 43560.0

            elif type_of_area == 'Cents':

                area_in_sqft = area * 435.61545

            elif type_of_area == 'Guntha':

                area_in_sqft = area * 1089.0

            elif type_of_area == 'Grounds':

                area_in_sqft = area * 2400.0

            return float(area_in_sqft)

        

    else:

        return (float(my_list[0]) + float(my_list[1]))/2.0
data['total_sqft'] = data.total_sqft.str.split('-').apply(preprocess_total_sqft)
data.groupby('size')['total_sqft'].mean()
data['size'].unique()
data['size']
data.head()
data.isnull().sum()
data[data['size'].isna()]
sqft_list = data.groupby('size')['total_sqft'].mean()
type(sqft_list)
def closest_size(K):  

    idx = (np.abs(sqft_list - K)).argmin() 

    return sqft_list.index[idx]
missing_size = data[data['size'].isna()]['total_sqft'].apply(closest_size)
missing_size.shape
missing_size
data.iloc[missing_size.index,3]=missing_size.values
# for i,j in zip(data['size'],data['total_sqft']):

#     if ~(type(i)==str):

#         i = closest(j)
data.isnull().sum()
# making three column-- bedroom, hall, kitchen

# Bedroom-

data['bedroom'] = data['size'].apply(lambda x: x.split(' ')[0] if isinstance(x,str) else np.nan)

data['bedroom'] = data['bedroom'].astype(float)





# Hall-

def isHall(x):

    if type(x)==str:

        tokens = x.split(' ')

        if tokens[1]=='BHK':

            return 1

        else:

            return 0

        

        

# Kitchen

def isKitchen(x):

    if type(x)==str:

        tokens = x.split(' ')

        if (tokens[1]=='BHK')| (tokens[1]=='RK'):

            return 1

        else:

            return 0
data['hall']=data['size'].apply(isHall,)

data['kitchen']=data['size'].apply(isKitchen)
# Two column missing value- Bath and balcony.

# can be filled with mode or as the above apraoch for size(based on sqft available)
import matplotlib.pyplot as plt
data.groupby('bath')['bedroom'].mean()
# Bath

bedroom_list_bath = data.groupby('bath')['bedroom'].mean()
df[df.bath==18]
bedroom_list_bath
data[data.bath==18]
def closest_bath(K):  

    idx = (np.abs(bedroom_list_bath - K)).argmin() 

    return bedroom_list_bath.index[idx]
missing_bath = data[data['bath'].isna()]['bedroom'].apply(closest_bath)
missing_bath.shape
missing_bath
data.iloc[missing_bath.index,6]=missing_bath.values
# Balcony

bedroom_list_balcony = data.groupby('balcony')['bedroom'].mean()
def closest_balcony(K):  

    idx = (np.abs(bedroom_list_balcony - K)).argmin() 

    return bedroom_list_balcony.index[idx]
missing_balcony = data[data['balcony'].isna()]['bedroom'].apply(closest_balcony)
data.iloc[missing_balcony.index, 7]=missing_balcony.values
data.isnull().sum()
data.isnull().sum()
data.head()
data.describe()
data[data.bath>10]
data.iloc[344]
df.loc[344]
df[df.bath>10]
data.isnull().sum()
data.info()
# No need to have size- as all data has been extracted in bedroom hall and kitchen column.

data.drop('size',axis=1 ,inplace=True)
data.info()
data.area_type.unique()
data.society.nunique()
data.location.nunique()
plt.plot(data.location,data.price)
data
data.availability.unique()
data.area_type.value_counts()
replace_area_type = {'Super built-up  Area': 0, 'Built-up  Area': 1, 'Plot  Area': 2, 'Carpet  Area': 3}

data['area_type'] = data.area_type.map(replace_area_type)
def replacing_availabilty(string):

    if string == 'Ready To Move':

        return 0

    elif string == 'Immediate Possession':

        return 1

    else:

        return 2
data['availability'] = data.availability.apply(replacing_availabilty)
data.head()
from sklearn.preprocessing import LabelEncoder
loc_encoder = LabelEncoder()

loc_encoder.fit(data['location'])

data['location'] = loc_encoder.transform(data['location'])
data.head()
loc_encoder = LabelEncoder()

loc_encoder.fit(data['society'])

data['society'] = loc_encoder.transform(data['society'])
data.head()
data.info()