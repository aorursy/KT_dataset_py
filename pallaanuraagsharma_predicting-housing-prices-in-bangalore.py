import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd 

import numpy as np 

import matplotlib

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline 

matplotlib.rcParams['figure.figsize'] = (20,10)
df = pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
df.sample(5,random_state=23)
df.info()
df.shape
df.groupby('area_type')['area_type'].agg('count')
df = df.drop(columns=['availability','area_type','balcony','society'])
df.sample(5,random_state=23)
df.info()
df.isnull().sum()
df = df.dropna()
df.sample(5,random_state=23)
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
df.sample(5,random_state=23)
df.drop(columns=['size'])
df[df.bhk>20]
df['total_sqft'].unique()
def is_float(x):

    try:

        float(x)

    except:

        return False

    return True
df[~df['total_sqft'].apply(is_float)]
def convert_range_to_num(x):

    tokens = x.split('-')

    if len(tokens) == 2:

        return (float(tokens[0])+float(tokens[1]))/2

    try:

        return float(x)

    except:

        return None 
df['total_sqft'] = df['total_sqft'].apply(convert_range_to_num)
df.sample(5,random_state=23)
df = df.dropna(how='any')
df.info()
df.sample(7)
df['Price_per_sqft'] = (df['price']*100000)/df['total_sqft']
df['Price_per_sqft'] = df['Price_per_sqft'].round(1)
df.sample(5)
len(df.location.unique())
df.location = df.location.apply(lambda x : x.strip())
location_stats = df.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats
location_less_than_10 = location_stats[location_stats<=10]
df.location = df.location.apply(lambda x: 'Other' if x in location_less_than_10 else x)
df.sample(20)
from pandas_profiling import ProfileReport
report = ProfileReport(df,title='REPORT OF BANGLORE HOUSING')

report
df['sqft_per_bedroom'] = df['total_sqft']/df['bhk']
df['sqft_per_bedroom'] = df['sqft_per_bedroom'].apply(lambda x: None if x < 300 else x)
df = df.dropna()
df['Price_per_sqft'] = df['Price_per_sqft'].apply(lambda x: None if x > 16000 else x)
df = df.dropna()
df['Price_per_sqft'].describe()
def remove_pps_outliers(df):

    df_out = pd.DataFrame()

    for key, subdf in df.groupby('location'):

        m = np.mean(subdf.Price_per_sqft)

        st = np.std(subdf.Price_per_sqft)

        reduced_df = subdf[(subdf.Price_per_sqft>(m-st)) & (subdf.Price_per_sqft<=(m+st))]

        df_out = pd.concat([df_out,reduced_df],ignore_index=True)

    return df_out

df = remove_pps_outliers(df)

df.shape
def plot_scatter_chart(df,location):

    bhk2 = df[(df.location==location) & (df.bhk==2)]

    bhk3 = df[(df.location==location) & (df.bhk==3)]

    matplotlib.rcParams['figure.figsize'] = (15,10)

    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)

    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)

    plt.xlabel("Total Square Feet Area")

    plt.ylabel("Price (Lakh Indian Rupees)")

    plt.title(location)

    plt.legend()

    

plot_scatter_chart(df,"Hebbal")
def remove_bhk_outliers(df):

    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):

        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby('bhk'):

            bhk_stats[bhk] = {

                'mean': np.mean(bhk_df.Price_per_sqft),

                'std': np.std(bhk_df.Price_per_sqft),

                'count': bhk_df.shape[0]

            }

        for bhk, bhk_df in location_df.groupby('bhk'):

            stats = bhk_stats.get(bhk-1)

            if stats and stats['count']>5:

                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.Price_per_sqft<(stats['mean'])].index.values)

    return df.drop(exclude_indices,axis='index')

df = remove_bhk_outliers(df)

df.shape
plot_scatter_chart(df,"Hebbal")
plt.hist(df.Price_per_sqft,rwidth=0.8)

plt.xlabel("Price Per Square Feet")

plt.ylabel("Count")
df.bath.unique()
plt.hist(df.bath,rwidth=0.8)

plt.xlabel("Number of bathrooms")

plt.ylabel("Count")
df = df[df.bath<df.bhk+2]
df = df.drop(['size','Price_per_sqft'],axis='columns')
df = df.drop(['sqft_per_bedroom'],axis='columns')
df.sample(10)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['location_cat'] = le.fit_transform(df.location)
df.sample(10,random_state=33)
from sklearn.model_selection import train_test_split as tts



x = df[['total_sqft','bath','bhk','location_cat']]

y = df['price']



x_train,x_test,y_train,y_test = tts(x,y,test_size=0.3,random_state=36)
from sklearn.linear_model import LinearRegression

lr_clf = LinearRegression()

lr_clf.fit(x_train,y_train)

score_lr = 100*lr_clf.score(x_test,y_test)

print(f'LR Model score = {score_lr:4.3f}%')
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

dtr.fit(x_train,y_train)

score_dtr = 100*dtr.score(x_test,y_test)

print(f'DTR Model score = {score_dtr:4.3f}%')
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(x_train,y_train)

score_rfr = 100*rfr.score(x_test,y_test)

print(f'RFR Model score = {score_rfr:4.3f}%')