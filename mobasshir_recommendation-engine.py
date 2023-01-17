import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 100) # Setting pandas to display a N number of columns

pd.set_option('display.max_rows', 10) # Setting pandas to display a N number rows

pd.set_option('display.width', 1000) # Setting pandas dataframe display width to N

# from scipy import stats # statistical library

# from statsmodels.stats.weightstats import ztest # statistical library for hypothesis testing

# import plotly.graph_objs as go # interactive plotting library

# import pandas_profiling # library for automatic EDA

# %pip install autoviz # installing and importing autoviz, another library for automatic data visualization

# from autoviz.AutoViz_Class import AutoViz_Class

# from itertools import cycle # function used for cycling over values



from IPython.display import display # display from IPython.display



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def dataframe_split_between_null_and_not_null(df):

    not_null_df = df.dropna()

    null_df = df.drop(not_null_df.index)

    percentage = 100 * null_df.shape[0] / df.shape[0]

    print("Actual dataset ", df.shape)

    print("Null dataset ", null_df.shape)

    print("Not null dataset ", not_null_df.shape)

    print(f'Null percentage {percentage}%')

    return df, null_df, not_null_df,percentage



def column_missing_state(df):

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * mis_val / len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values(1, ascending=False).round(1)

    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table.shape[0]) +

              " columns that have missing values.")

    return mis_val_table
    xlsx = pd.ExcelFile('/kaggle/input/khaodao/data.xlsx')

    df = dict()

    for index, sheet in enumerate(xlsx.sheet_names):

        df[sheet] = xlsx.parse(sheet, index_col=None)

        df[sheet] = df[sheet].replace("", np.nan)
df_r = dict()

import os

for dirname, _, filenames in os.walk('/kaggle/input/restaurant-recommendation-challenge'):

    for filename in filenames:

        if filename.endswith('.csv'):

            name = filename.split('.')[0]

            df_r[name] = pd.read_csv(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))
raw_customer = df_r['train_customers']

raw_orders = df_r['orders']

raw_locations = df_r['train_locations']

raw_vendors = df_r['vendors']

train_df = df_r['train_full']

for k,v in df.items():

    print(f"Table Name: {k} ---- Shape: {v.shape}")
for k,v in df.items():

    print(f"Table Name: {k}\n")

    for i, column in enumerate(v.columns):

        print(f"{i+1}. {column}")

    print('\n')
for k,v in df_r.items():

    print(f"Table Name: {k}\n")

    for i, column in enumerate(v.columns):

        print(f"{i+1}. {column}")

    print('\n')
display(df['customer'].head(2))
display(df_r['train_customers'].head(2))
customer = pd.DataFrame({

    'customer_id': raw_customer.akeed_customer_id,

    'gender': raw_customer.gender,

    'dob': raw_customer.dob,

    'status': raw_customer.status,

    'verified': raw_customer.verified,

    'language': raw_customer. language,

    'location_service_preference': True,

    'notification_preference': True,

    

})
def clean_string(string):

    string = str(string)

    if '?' in string or string.lower()=='nan' or string.strip(' ')=='':

        return np.nan

    string = string.strip(' ').lower()

    return string



customer.loc[:, 'gender'] = customer['gender'].apply(clean_string)





from datetime import date

def calc_age(year):

    if len(str(year))==2:

        if str(year).startswith('0'):

            year = '20'+str(year)

        else:

            year = '19'+str(year)

        year = int(year)

    if year is None:

        return np.nan

    return date.today().year-year



customer.loc[:, 'age'] = customer['dob'].apply(calc_age)
display(customer.head(4))
customers = customer
customers.to_csv('final_customer.csv', index= False)
import geopandas as gpd

import re 



def check_num(string):

    if string is None:

        return float(0)

    string = str(string)

    regex = r'-?[0-9]*.[0-9]*'

    m = re.match(regex, string)

    if m is None:

        return float(0)

    return float(string[:6])



raw_locations.loc[:,'latitude'] = raw_locations['latitude'].apply(check_num)

raw_locations.loc[:,'longitude'] = raw_locations['longitude'].apply(check_num)

# raw_locations.loc[:,'geometry'] = gpd.points_from_xy(raw_locations['longitude'], raw_locations['latitude'])



display(raw_locations.head(4))
raw_locations['location_type'].value_counts()
locations = raw_locations
locations.to_csv('final_locations.csv', index= False)
# sorted(df['order'].columns.to_list())
# df['order'].head()
# df['order_items'].head()
# df['order_line_item_modifier'].head()
# df['order_status'].head()
raw_orders.rename(columns={'customer_id': 'customer_id', 'vendor_discount_amount': 'discounts', 'akeed_order_id':'order_id','created_at':'order_placed_time','vendor_id':'restaurant_id'}, inplace=True)
raw_orders.drop(['CID X LOC_NUM X VENDOR'], axis=1, inplace=True)
display(raw_orders.head())
raw_orders.to_csv('final_orders.csv', index= False)
# df['menu'].head()
# df['menu_category'].head()
# df['menu_item'].head()
# df['modifier_type'].head()
# df['modifier_item'].head()
# df['payment'].head()
raw_vendors.rename(columns={'id':'restaurant_id','is_akeed_delivering':'delivery_available','vendor_rating':'restaurent_rating','vendor_tag_name':'restaurent_tag_name'}, inplace=True)
raw_vendors = raw_vendors[['restaurant_id','latitude','longitude','vendor_category_en','delivery_available','delivery_charge','serving_distance','is_open','prepration_time','commission','discount_percentage','status','verified','language','restaurent_rating','open_close_flags','restaurent_tag_name','country_id','city_id']]
display(raw_vendors.head(4))
raw_vendors.to_csv('final_restaurents.csv', index= False)
# display(df['restaurant_detail'].head(4))
# display(df['restaurant'].head(4))
# df['restaurant_admin'].columns
# display(df['restaurant_menu'].head(4))
train_df.shape
train_df.drop(['authentication_id','vendor_category_id','primary_tags','vendor_tag','one_click_vendor','device_type','display_orders','CID X LOC_NUM X VENDOR'], axis=1, inplace=True)
train_df.rename(columns={'is_akeed_delivering':'delivery_available','vendor_rating':'restaurent_rating','vendor_tag_name':'restaurent_tag_name'}, inplace=True)
def clean_string(string):

    string = str(string)

    if '?' in string or string.lower()=='nan' or string.strip(' ')=='':

        return np.nan

    string = string.strip(' ').lower()

    return string



train_df.loc[:, 'gender'] = train_df['gender'].apply(clean_string)
import re

def check_num(string):

    if string is None:

        return float(0)

    string = str(string)

    regex = r'-?[0-9]*.[0-9]*'

    m = re.match(regex, string)

    if m is None:

        return float(0)

    try:

        return float(string[:6])

    except:

        return float(0)



train_df.loc[:,'latitude_x'] = train_df['latitude_x'].apply(check_num)

train_df.loc[:,'longitude_x'] = train_df['longitude_x'].apply(check_num)



train_df.loc[:,'latitude_y'] = train_df['latitude_y'].apply(check_num)

train_df.loc[:,'longitude_y'] = train_df['longitude_y'].apply(check_num)
age_df = pd.DataFrame({

    'customer_id': customers['customer_id'],

    'age': customers['age']

})
train_df = pd.merge(train_df, age_df, on='customer_id', how='inner')
display(train_df.head(4))
train_df.to_csv('final_train_df.csv', index= False)
# !pip3 install numpy

# !pip3 install scikit-surprise
from surprise import Reader

from surprise import Dataset

from surprise.model_selection import cross_validate

from surprise import NormalPredictor

from surprise import KNNBasic

from surprise import KNNWithMeans

from surprise import KNNWithZScore

from surprise import KNNBaseline

from surprise import SVD

from surprise import BaselineOnly

from surprise import SVDpp

from surprise import NMF

from surprise import SlopeOne

from surprise import CoClustering

from surprise.accuracy import rmse

from surprise import accuracy

from surprise.model_selection import train_test_split

from surprise.model_selection import GridSearchCV
# df = pd.read_csv ("/kaggle/working/final_train_df.csv")
df = train_df
df = df.sample(100000)


df.head()


df.tail()
train_df.columns
explicit_df = df[['customer_id','id','restaurent_rating']]

explicit_df.columns = ['customer_id','restaurent_id','restaurent_rating']
explicit_df.head()
explicit_df.shape
explicit_df.info()
print('Dataset shape: {}'.format(explicit_df.shape))

print('-Dataset examples-')

print(explicit_df.iloc[::20000, :])
from plotly.offline import init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



data = explicit_df['restaurent_rating'].value_counts().sort_index(ascending=False)

trace = go.Bar(x = data.index,

               text = ['{:.1f} %'.format(val) for val in (data.values / explicit_df.shape[0] * 100)],

               textposition = 'auto',

               textfont = dict(color = '#000000'),

               y = data.values,

               )

# Create layout

layout = dict(title = 'Distribution Of {} ratings'.format(explicit_df.shape[0]),

              xaxis = dict(title = 'Rating'),

              yaxis = dict(title = 'Count'))

# Create plot

fig = go.Figure(data=[trace], layout=layout)

fig.show()
# Number of ratings per book

data = explicit_df.groupby('restaurent_id')['restaurent_rating'].count().clip(upper=100)



# Create trace

trace = go.Histogram(x = data.values,

                     name = 'Ratings',

                     xbins = dict(start = 0,

                                  end = 100,

                                  size = 2))

# Create layout

layout = go.Layout(title = 'Distribution Of Number of Ratings Per Restaurent (Clipped at 100)',

                   xaxis = dict(title = 'Number of Ratings Per Restaurent'),

                   yaxis = dict(title = 'Count'),

                   bargap = 0.2)



# Create plot

fig = go.Figure(data=[trace], layout=layout)

fig.show()
explicit_df.groupby('restaurent_id')['restaurent_rating'].count().reset_index().sort_values('restaurent_rating', ascending=False)[:10]


# Number of ratings per user

data = explicit_df.groupby('customer_id')['restaurent_rating'].count().clip(upper=100)



# Create trace

trace = go.Histogram(x = data.values,

                     name = 'Ratings',

                     xbins = dict(start = 0,

                                  end = 100,

                                  size = 2))

# Create layout

layout = go.Layout(title = 'Distribution Of Number of Ratings Per User (Clipped at 100)',

                   xaxis = dict(title = 'Ratings Per User'),

                   yaxis = dict(title = 'Count'),

                   bargap = 0.2)



# Create plot

fig = go.Figure(data=[trace], layout=layout)

fig.show()
explicit_df.groupby('customer_id')['restaurent_rating'].count().reset_index().sort_values('restaurent_rating', ascending=False)[:10]
explicit_df.restaurent_rating.unique()
df = explicit_df


min_ratings = 5

filter_items = df['restaurent_id'].value_counts() > min_ratings

filter_items = filter_items[filter_items].index.tolist()



min_user_ratings = 5

filter_users = df['customer_id'].value_counts() > min_user_ratings

filter_users = filter_users[filter_users].index.tolist()



df_new = df[(df['restaurent_id'].isin(filter_items)) & (df['customer_id'].isin(filter_users))]

print('The original data frame shape:\t{}'.format(df.shape))

print('The new data frame shape:\t{}'.format(df_new.shape))
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df_new[['customer_id', 'restaurent_id', 'restaurent_rating']], reader)
benchmark = []



algorithms = [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]



print ("Attempting: ", str(algorithms), '\n\n\n')



for algorithm in algorithms:

    print("Starting: " ,str(algorithm))

    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)

    

    tmp = pd.DataFrame.from_dict(results).mean(axis=0)

    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))

    benchmark.append(tmp)

    print("Done: " ,str(algorithm), "\n\n")



print ('\n\tDONE\n')
surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
surprise_results
# df['organization'].head()
# for column in df['menu_item'].columns:

#     print(column)
# for column in df['order'].columns:

#     print(column)
# for k,v in df.items():

#     print('*'*10)

#     print("Table Name:",k)

#     print("Shape: ", v.shape)

#     print("Columns: ", v.columns.to_list())

#     print('*'*10)
# for k,v in df.items():

#     print('*'*10)

#     print("Table Name:",k)

#     print("Dataframe infos:")

#     print(v.info())

#     print('*'*10)

    
# for k,v in df.items():

#     print('*'*10)

#     print("Table Name:",k)

#     print("Dataframe infos:")

#     print(v.describe())

#     print('*'*10)
# for k,v in df.items():

#     print('*'*10)

#     print("Table Name:",k)

#     column_missing_state(v)

#     print('*'*10)