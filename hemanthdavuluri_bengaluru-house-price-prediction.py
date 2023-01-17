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
import numpy as np 
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize']=(20,10)
data = pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
data.info()
data.head()
data.shape
data.groupby('area_type')['area_type'].agg('count')
data['area_type'].value_counts()
data2 = data.drop(columns=['area_type','society','balcony','availability'])
data2.head()
data2.isnull().sum()
data2=data2.dropna()
data2.isnull().sum()
data2.corr()
data2['bhk']=data2['size'].apply(lambda x: int(x.split(" ")[0]))
data2.head()
data2[data2.bhk>20]
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
data2[~data2['total_sqft'].apply(is_float)].head()
def con_sqrt_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

data2['sqrt']=data2['total_sqft'].apply(con_sqrt_num)
data2 = data2.drop(columns=['size','total_sqft'])
data2.head()
data2.isnull().sum()
data2['sqrt'].dropna(inplace=True)
data2.head()
data2.isnull().sum()
data2.info()
data2.corr()
data3 = data2.copy()
data3['price_per_sqrt']= data2['price']*100000/data2['sqrt']
data3.head()
data3.corr()
data4 = data3.copy()
data4.location = data4.location.apply(lambda x: x.strip())
location_stats = data4.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats
len(location_stats[location_stats<=10])
location_stats_less_10 = location_stats[location_stats<=10]
location_stats_less_10
data4.location = data4.location.apply(lambda x: 'other' if x in location_stats_less_10 else x )
data4.head()
data4[data4.sqrt/data4.bhk<300].head()
data5 = data4[~(data4.sqrt/data4.bhk<300)]
data5.corr()
data5.price_per_sqrt.describe()
def remove_outliers(df):
    df_out = pd.DataFrame()
    for key , subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqrt)
        st = np.std(subdf.price_per_sqrt)
        reduced_df = subdf[(subdf.price_per_sqrt>(m-st))& (subdf.price_per_sqrt<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
        
    return df_out    
clean = remove_outliers(data5)
clean.shape
clean.corr()
clean.location.unique()
dummies = pd.get_dummies(clean.location)
final = pd.concat([clean,dummies.drop('other',axis='columns')],axis='columns')
final.corr()
final = final.drop(columns='location')
final.head(2)
X = final.drop(columns='price')
X
y = clean.price
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5 , test_size=0.2 , random_state=0)
cross_val_score(LinearRegression(),X,y,cv=cv)
from sklearn.ensemble import RandomForestRegressor

model2  = RandomForestRegressor(n_estimators = 1000, random_state = 42)
model2.fit(X_train,y_train)
model.score(X_test,y_test)
model2.score(X_test,y_test)
