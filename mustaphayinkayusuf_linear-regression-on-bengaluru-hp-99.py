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
import pandas as pd

import numpy as np

import seaborn as sns

from pandas_profiling import ProfileReport
#Load the data

df = pd.read_csv('../input/bengaluru-house-price-data/Bengaluru_House_Data.csv')

df.head()
df.shape
report = ProfileReport(df, title ='Bengaluru HouseData Profile Report')

report
#Area type available

df['area_type'].unique()
#Number of locations

df.location.nunique()
#Total amount of rows each area type is found

df.groupby(['area_type'])['area_type'].agg('count')
df.balcony.value_counts()
#Dropping of empty cells

df = df.dropna()

df.isna().sum()
#Shape of data after droping empty cells

df.shape
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

df.head()

df.bhk.min(),df.bhk.max()
df.loc[df['bhk'] == 1, 'area_type'].value_counts()
df.loc[df['bhk'] == 11, 'area_type'].value_counts()
df.price.max(), df.price.min(), df.price.mean()
#Locations with the lowest price

df.loc[df['price'] == 8, 'location'].value_counts()
#Area type with the lowest price

df.loc[df['price'] == 8, 'area_type'].value_counts()

#Area type with the highest price

df.loc[df['price'] == 360, 'area_type'].value_counts()



#Super built up area has the highest price and the lowest price
df.society.nunique()
df.total_sqft.unique()
def is_float(x):

    try:

        float(x)

    except:

        return False

    return True

        
df[-df['total_sqft'].apply(is_float)].tail()
#convert total_sqft to average of the two numbers

def convert_sqft_to_num(x):

    tokens = x.split('-')

    if len(tokens) == 2:

        return (float(tokens[0]) + float(tokens[1]))/2

    try:

        return float(x)

    except:

        return None
#To check...

convert_sqft_to_num('2100')
#To check...

convert_sqft_to_num('2100 - 2850')
convert_sqft_to_num('34.465q.Meter')

#returns no output
df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

df.head()
df.isna().sum()
df.head()
df.location.nunique()
df.isna().sum()
df.loc[df.total_sqft.isna() == True]
df = df.dropna().reset_index( drop = True)

df.shape
df.isna().sum()
#One hot encoding for location column

dummies = pd.get_dummies(df.location)

dummies.head()
#Transform area type to number

df.area_type.unique()
#Transform area_type to number

df['area_type_no'] = ''

df.loc[df.area_type == 'Super built-up  Area', 'area_type_no'] = 0

df.loc[df.area_type == 'Plot  Area', 'area_type_no'] = 1

df.loc[df.area_type == 'Built-up  Area', 'area_type_no'] = 2

df.loc[df.area_type == 'Carpet  Area', 'area_type_no'] = 3

df.head()
df1 = pd.concat([df, dummies.drop('Banaswadi', axis = 1)], axis = 1)
df1.head()
df1.shape
df1.isna().sum()
#Setting features and target

X = df1.drop(['area_type', 'availability', 'location', 'size', 'society'], axis = 1)

y = df1['price']
#Spliting the data into training and testing set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)
model.score(x_test, y_test)
y_pred = model.predict(x_test)
len(y_pred)
#First 20 predictions

y_pred[:20]
#To check...

y_test[:20]