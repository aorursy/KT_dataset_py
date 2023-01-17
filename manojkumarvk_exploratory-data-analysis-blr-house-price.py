
import numpy as np 
import pandas as pd
import re

import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('../input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
data.head()
data.describe()
data.info()
data.isnull().sum()
data['society'].shape
data['size'].unique()
data.corr()
sns.pairplot(data)
sns.distplot(data['price'])
data.select_dtypes(exclude=['object']).describe()
corr=data.corr()
sns.heatmap(corr)
from collections import Counter
Counter(data['total_sqft'])
data.shape
#preprocessing the total sqft cols as it has vivid entries
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
#converting the categorical to numerical data - area_type
data.area_type.value_counts()
replace_area_type = {'Super built-up  Area': 0, 'Built-up  Area': 1, 'Plot  Area': 2, 'Carpet  Area': 3}
data['area_type'] = data.area_type.map(replace_area_type)
#converting the categorical to numerical data - availabilty
data.availability.value_counts()
def replace_availabilty(my_string):
    if my_string == 'Ready To Move':
        return 0
    elif my_string == 'Immediate Possession':
        return 1
    else:
        return 2
data['availability'] = data.availability.apply(replace_availabilty)
#converting NaN in location
data['location'].isnull().sum()
data['location'] = data['location'].fillna('No Location')
#converting the categorical to numerical data - size
Counter(data['size'])
col_names = ['balcony','bath', 'price']

fig, ax = plt.subplots(len(col_names), figsize=(8,40))

for i, col_val in enumerate(col_names):
        
    sns.boxplot(y=data[col_val], ax=ax[i])
    ax[i].set_title('Box plot - '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    
plt.show()
data.isnull().sum()
data['size'].fillna('ffill',inplace=True)

data['society'].fillna('ffill',inplace=True)
data['bath'].fillna('ffill',inplace=True)
data['balcony'].fillna('ffill',inplace=True)
data.isnull().sum()