# importing all needed libraries 

import time
import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# display the output of plotting commands inline within frontends, directly below the code cell that produced it.
%matplotlib inline
plt.style.use('ggplot')


from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

from sklearn.model_selection import cross_val_score
from sklearn.externals.six import StringIO  
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report
import pydot
from IPython.display import Image

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import GridSearchCV
import sklearn.linear_model as skl_lm
from sklearn.preprocessing import scale 


# print all files available in the data folder
import os
print(os.listdir("../input/elo-merchant-category-recommendation/"))

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Set matplotlib figure sizes to 10 and 6, font size to 12.
from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 6)
rcParams['font.size'] = 12
# it takes more than 50s to read, because there are 29 million lines in historical_transactions
train = pd.read_csv('../input/elo-merchant-category-recommendation/train.csv', parse_dates=['first_active_month'])
test = pd.read_csv('../input/elo-merchant-category-recommendation/test.csv', parse_dates=['first_active_month'])
historical_transactions = pd.read_csv('../input/elo-merchant-category-recommendation/historical_transactions.csv', parse_dates=['purchase_date'])
new_merchant_transactions = pd.read_csv('../input/elo-merchant-category-recommendation/new_merchant_transactions.csv', parse_dates=['purchase_date'])
merchants = pd.read_csv('../input/elo-merchant-category-recommendation/merchants.csv')
historical_transactions = historical_transactions.sample(n=5000000, random_state=1111)  # random_state is the seed 
new_merchant_transactions = new_merchant_transactions.sample(n=500000, random_state=1111)  # random_state is the seed 
train_data = pd.read_excel('../input/elo-merchant-category-recommendation/Data_Dictionary.xlsx', sheet_name='train')
history_data = pd.read_excel('../input/elo-merchant-category-recommendation/Data_Dictionary.xlsx', sheet_name='history')
new_merchant_period = pd.read_excel('../input/elo-merchant-category-recommendation/Data_Dictionary.xlsx', sheet_name='new_merchant_period')
merchant = pd.read_excel('../input/elo-merchant-category-recommendation/Data_Dictionary.xlsx', sheet_name='merchant')
# description of the train data
train_data
history_data
new_merchant_period
merchant
train.shape
historical_transactions.shape
new_merchant_transactions.shape
merchants.shape
'''The function prints out the number and persentage of null values a dataframe column has.'''
def print_null(df):
    for col in df:
        if df[col].isnull().any():
            print('%s has %.0f null values: %.3f%%'%(col, df[col].isnull().sum(), df[col].isnull().sum()/df[col].count()*100))
# Checking the types of the column values
print(merchants.dtypes)
# Checking for missing data
print_null(merchants)
#Now, let's look at column histograms:

cat_cols = ['active_months_lag6','active_months_lag3','most_recent_sales_range', 'most_recent_purchases_range','category_1','active_months_lag12','category_4', 'category_2']
num_cols = ['numerical_1', 'numerical_2','merchant_group_id','merchant_category_id','avg_sales_lag3', 'avg_purchases_lag3', 'subsector_id', 'avg_sales_lag6', 'avg_purchases_lag6', 'avg_sales_lag12', 'avg_purchases_lag12']

# Removing infinite values and replacing them with NAN
merchants.replace([-np.inf, np.inf], np.nan, inplace=True)

plt.figure(figsize=[15, 15])
plt.suptitle('Merchants table histograms', y=1.02, fontsize=20)
ncols = 4
nrows = int(np.ceil((len(cat_cols) + len(num_cols))/4))
last_ind = 0
for col in sorted(list(merchants.columns)):
    #print('processing column ' + col)
    if col in cat_cols:
        last_ind += 1
        plt.subplot(nrows, ncols, last_ind)
        vc = merchants[col].value_counts()
        x = np.array(vc.index)
        y = vc.values
        inds = np.argsort(x)
        x = x[inds].astype(str)
        y = y[inds]
        plt.bar(x, y, color=(0, 0, 0, 0.7))
        plt.title(col, fontsize=15)
    if col in num_cols:
        last_ind += 1
        plt.subplot(nrows, ncols, last_ind)
        merchants[col].hist(bins = 50, color=(0, 0, 0, 0.7))
        plt.title(col, fontsize=15)
    plt.tight_layout()
#Now, let's look at correlations between columns in merchants.csv:

corrs = np.abs(merchants.corr())
ordered_cols = (corrs).sum().sort_values().index
np.fill_diagonal(corrs.values, 0)
plt.figure(figsize=[10,10])
plt.imshow(corrs.loc[ordered_cols, ordered_cols], cmap='plasma', vmin=0, vmax=1)
plt.colorbar(shrink=0.7)
plt.xticks(range(corrs.shape[0]), list(ordered_cols), rotation=90)
plt.yticks(range(corrs.shape[0]), list(ordered_cols))
plt.title('Heat map of coefficients of correlation between merchant\'s features', fontsize=17)
plt.show()
x = np.array([12, 6, 3]).astype(str)
sales_rates = merchants[['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12']].mean().values
purchase_rates = merchants[['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']].mean().values
plt.bar(x, sales_rates, width=0.3, align='edge', label='average sales', edgecolor=[0.2]*3)
plt.bar(x, purchase_rates, width=-0.3, align='edge', label='average purchases', edgecolor=[0.2]*3)
plt.legend()
plt.title('Avergage sales and number of purchases\nover the last 12, 6, and 3 months', fontsize=17)
plt.show()
# Target distribution in the train dataframe
plt.hist(train['target'], bins= 50)
plt.title('Loyalty score')
plt.xlabel('Loyalty score')
plt.show()
((train['target']<-30).sum() / train['target'].count()) * 100 # percentage of outliers
print(max(train['first_active_month']))
print(max(test['first_active_month']))
d1 = train['first_active_month'].value_counts().sort_index()
d2 = test['first_active_month'].value_counts().sort_index()
data = [go.Scatter(x=d1.index, y=d1.values, name='train'), go.Scatter(x=d2.index, y=d2.values, name='test')]
layout = go.Layout(dict(title = "Counts of first active",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Count'),
                  ),legend=dict(
                orientation="v"))
py.iplot(dict(data=data, layout=layout))
# binarize authorized_flag, replace Y with 1 and N with 0
historical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].map({'Y':1, 'N':0})
# authorized_flag distribution in historical transactions
("At average " + str(historical_transactions['authorized_flag'].mean() * 100) + "% transactions are authorized")
historical_transactions['authorized_flag'].value_counts().plot(kind='bar', title='authorized_flag value counts');
historical_transactions['installments'].value_counts()
historical_transactions.groupby(['installments'])['authorized_flag'].mean()
# We know from the Dictionary that Purchase Amount is normalized
for i in [-1, 0]:
    n = historical_transactions.loc[historical_transactions['purchase_amount'] < i].shape[0]
    print("There are " + str(n) + " transactions with purchase_amount less than " + str(i) + ".")
for i in [0, 10, 100]:
    n = historical_transactions.loc[historical_transactions['purchase_amount'] > i].shape[0]
    print("There are " + str(n) + " transactions with purchase_amount more than " + str(i) + ".")
max(historical_transactions['purchase_amount'])
# Unique values in historical transactions
for col in ['city_id', 'merchant_category_id', 'merchant_id', 'state_id', 'subsector_id']:
    print("There are " + str(historical_transactions[col].nunique()) + " unique values in " + str(col) + ".")
# binarize authorized_flag, replace Y with 1 and N with 0
new_merchant_transactions['authorized_flag'] = new_merchant_transactions['authorized_flag'].map({'Y':1, 'N':0})
# authorized_flag distribution in new merchant transactions
print("At average " + str(new_merchant_transactions['authorized_flag'].mean() * 100) + "% transactions are authorized")
new_merchant_transactions['authorized_flag'].value_counts().plot(kind='bar', title='authorized_flag value counts');
new_merchant_transactions['installments'].value_counts()
# We know from the Dictionary that Purchase Amount is normalized
for i in [-1, 0]:
    n = new_merchant_transactions.loc[new_merchant_transactions['purchase_amount'] < i].shape[0]
    print("There are " + str(n) + " transactions with purchase_amount less than " + str(i) + ".")
for i in [0, 10, 100]:
    n = new_merchant_transactions.loc[new_merchant_transactions['purchase_amount'] > i].shape[0]
    print("There are " + str(n) + " transactions with purchase_amount more than " + str(i) + ".")
# Unique values in new merchant transactions
for col in ['city_id', 'merchant_category_id', 'merchant_id', 'state_id', 'subsector_id']:
    print("There are " + str(new_merchant_transactions[col].nunique()) + " unique values in " + str(col) + ".")
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
def impute_na(X_train, df, variable):
    # make temporary df copy
    temp = df.copy()
    
    # extract random from train set to fill the na
    # temp[variable].isnull().sum() is the size of our sample
    random_sample = X_train[variable].dropna().sample(temp[variable].isnull().sum(), random_state=1111, replace=True)
    
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = temp[temp[variable].isnull()].index
    temp.loc[temp[variable].isnull(), variable] = random_sample
    return temp[variable]
# It was noticed that clipping the outliers does not improve the model. 
# Maybe because the tree based models that were used are robust to outliers anyway.
'''Function to clip outliers
def clipping_outliers(X_train, df, var):
    # Calculate the IQR
    IQR = X_train[var].quantile(0.75) - X_train[var].quantile(0.25)
    # Get the data that is located in the lower bound
    lower_bound = X_train[var].quantile(0.25) - 6 * IQR
    # Get the data that is located in the upper bound
    upper_bound = X_train[var].quantile(0.75) + 6 * IQR
    # Extract the data out of the dataframe that is located between the bounds
    no_outliers = len(df[df[var]>upper_bound]) + len(df[df[var]<lower_bound])
    print('There are %i outliers in %s: %.3f%%' %(no_outliers, var, no_outliers/len(df)))
    df[var] = df[var].clip(lower_bound, upper_bound)
    return df
'''
'''# WE'RE NOT USING MERCHANTS ANYMORE
# Merchants null
merchants = merchants.replace([np.inf,-np.inf], np.nan)  # How does this change the values?
print('Merchants null')
print_null(merchants)

# We fill null values in the merchants data with the mean value of the column.
null_cols = ['avg_purchases_lag3','avg_sales_lag3', 'avg_purchases_lag6','avg_sales_lag6','avg_purchases_lag12','avg_sales_lag12']
for col in null_cols:
    merchants[col] = merchants[col].fillna(merchants[col].mean())

# Fill category_2 with random sampling from available data
merchants['category_2'] = impute_na(merchants, merchants, 'category_2')
'''
'''# WE'RE NOT USING MERCHANTS ANYMORE
merchants['category_1'] = merchants['category_1'].map({'Y':1, 'N':0})
merchants['category_4'] = merchants['category_4'].map({'Y':1, 'N':0})

map_cols = ['most_recent_purchases_range', 'most_recent_sales_range']
for col in map_cols:
    merchants[col] = merchants[col].map({'A':5,'B':4,'C':3,'D':2,'E':1})

numeric_cols = ['numerical_1','numerical_2'] + null_cols + map_cols

colormap = plt.cm.RdBu
plt.figure(figsize=(12,12))
sns.heatmap(merchants[numeric_cols].astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pair-wise correlation')

merchants.head()
'''
max(new_merchant_transactions['purchase_date']) # when did the last transaction happen?
# The last date to calculate time lags from 
REF_DATE = datetime.datetime.strptime('2018-12-31', '%Y-%m-%d')
# Create columns that calculate the number of days from the transaction day to the reference day (2018-12-31)
historical_transactions['days_to_date'] = ((REF_DATE - historical_transactions['purchase_date']).dt.days) 
#historical_transactions['days_to_date'] = historical_transactions['days_to_date'] #+ df_hist_trans['month_lag']*30
new_merchant_transactions['days_to_date'] = ((REF_DATE - new_merchant_transactions['purchase_date']).dt.days)#//30

### Here we're concatinatig historical transactions with new transactions, since they both have the same columns and form
### and therefore do not need to be joined together. ### 
transactions = pd.concat([historical_transactions, new_merchant_transactions])  

# Create column months_ro_date: this is the number of months from transaction date to reference date (2018-12-31)
transactions['months_to_date'] = transactions['days_to_date']//30
transactions = transactions.drop(columns=['days_to_date'])

# Reduce memory usage
transactions = reduce_mem_usage(transactions)

transactions.head()
# We do not need the 2 dataframes anymore, beccause we have all the data needed in transactions.
del historical_transactions
del new_merchant_transactions
'''# WE'RE NOT USING MERCHANTS ANYMORE
# Merge trasactions with merchant data
transactions = pd.merge(transactions, merchants, how='left', left_on='merchant_id', right_on='merchant_id')
transactions.head()
'''
'''# WE'RE NOT USING MERCHANTS ANYMORE
# Take the 2 last characters out from the column names of the transactions data frame.
t = list(transactions)
trans_cols = []
for e in t:
    trans_cols.append(e[:-2])
'''
'''# WE'RE NOT USING MERCHANTS ANYMORE
seen = {}
dupes = []

for x in trans_cols:
    if x not in seen:
        seen[x] = 1
    else:
        if seen[x] == 1:
            dupes.append(x)
        seen[x] += 1
dupes  # there are duplicate columns in transactions, which end in _x and _y
'''
'''# WE'RE NOT USING MERCHANTS ANYMORE
transactions = transactions.drop(columns=['category_1_y', 'category_2_y', 'city_id_y', 'state_id_y', 'merchant_category_id_y',
                                        'merchant_category_id_y', 'subsector_id_y'])

transactions.rename(columns={'category_1_x': 'category_1', 
                            'category_2_x': 'category_2',
                            'city_id_x': 'city_id',
                            'state_id_x': 'state_id',
                            'merchant_category_id_x': 'merchant_category_id',
                            'merchant_category_id_x': 'merchant_category_id',
                            'subsector_id_x': 'subsector_id'}, inplace=True)
'''
# Null ratio
print('Null ratio')
print_null(transactions)
# The function prints out the most common values of one column
def most_frequent(x):
    return x.value_counts().index[0]
print("merchant_id", most_frequent(transactions['merchant_id'])) ##A:'M_ID_00a6ca8a8a'
# print("category_4", most_frequent(merchants['category_4'])) ##A:'0.0'
# print("most_recent_sales_range", most_frequent(merchants['most_recent_sales_range'])) ##A:'1.0'
# print("most_recent_purchases_range: ", most_frequent(merchants['most_recent_purchases_range'])) ##A:'1.0'
print("category_2: ", most_frequent(transactions['category_2']))
print("category_3: ", most_frequent(transactions['category_3']))
# Fill null by most frequent data
transactions['category_2'].fillna(1.0,inplace=True)
transactions['category_3'].fillna('A',inplace=True)
transactions['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
'''# WE'RE NOT USING MERCHANTS ANYMORE
# Fill the merchant columns that have null values with random values.

# nan_cols = transacations.columns[transactions.isna().any()].tolist()
nan_cols = ['active_months_lag3','active_months_lag6','active_months_lag12','avg_purchases_lag3','avg_sales_lag3', 'avg_purchases_lag6','avg_sales_lag6','avg_purchases_lag12','avg_sales_lag12']
for col in nan_cols:
    transactions[col] = impute_na(transactions, transactions, col)

# merchants['category_4'].fillna(0.0,inplace=True)
# merchants['most_recent_sales_range'].fillna(1.0,inplace=True)
'''
print('Null ratio')
print_null(transactions) # There are no more null values in the transactions dataframe for the moment
# Encoding (Mapping/ Dummy vars)
# Binarizing Y to 1 and N to 0
# transactions['authorized_flag'] = transactions['authorized_flag'].map({'Y':1,'N':0})  # already done for authoried_flag
# Category 1 has only 2 distinct values
transactions['category_1'] = transactions['category_1'].map({'Y':1,'N':0})


# pd.get_dummies when applied to a column of categories where we have one category per observation 
# will produce a new column (variable) for each unique categorical value. 
# It will place a one in the column corresponding to the categorical value present for that observation.
dummies = pd.get_dummies(transactions[['category_2', 'category_3']], prefix = ['cat_2','cat_3'], columns=['category_2','category_3'])
transactions = pd.concat([transactions, dummies], axis=1) # axis=1 joins all the columns
 
transactions.head()
transactions = reduce_mem_usage(transactions)
transactions['weekend'] = (transactions['purchase_date'].dt.weekday >=5).astype(int)
transactions['hour'] = transactions['purchase_date'].dt.hour
transactions['day'] = transactions['purchase_date'].dt.day

# Calculate the weeks left till Christmas (2017-12-25)
transactions['weeks_to_Xmas_2017'] = ((pd.to_datetime('2017-12-25') - transactions['purchase_date']).dt.days//7).apply(lambda x: x if x>=0 and x<=60 else 0)
# Calculate the weeks left till Black Friday (2017-11-25)
transactions['weeks_to_BFriday'] = ((pd.to_datetime('2017-11-25') - transactions['purchase_date']).dt.days//7).apply(lambda x: x if x>=0 and x<=60 else 0)
#Mothers Day: May 14 2017 and 2018
transactions['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 60 else 0)
transactions['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 60 else 0)
#fathers day: August 13 2017
transactions['Fathers_day_2017']=(pd.to_datetime('2017-08-13')-transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 60 else 0)
#Childrens day: October 12 2017
transactions['Children_day_2017']=(pd.to_datetime('2017-10-12')-transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 60 else 0)
#Valentine's Day : 12th June, 2017
transactions['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 60 else 0)
# Carnival in Brasil 27.02.2017 - 28.02.2017
transactions['Carnival_2017']=(pd.to_datetime('2017-02-27')-transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 60 else 0)
# Carnival in Brasil 09.02.2018
transactions['Carnival_2018']=(pd.to_datetime('2018-02-09')-transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 60 else 0)
# Easter: 14.04.2017 - Restaurants
transactions['Easter_2017']=(pd.to_datetime('2017-04-14')-transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 60 else 0)
# Tiradentes: 21.04.2017 - Restaurants
transactions['Tiradentes_2017']=(pd.to_datetime('2017-04-21')-transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 60 else 0)
# Labour day: 01.05.2017 - Restaurants
transactions['Labour_day_2017']=(pd.to_datetime('2017-05-01')-transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 60 else 0)
# Independence day 01.09.2017 - Restaurants
transactions['Independence_day_2017']=(pd.to_datetime('2017-09-01')-transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 60 else 0)
# Categorize time in 4 categories: (0) from 5 to 11, (1) from 12 to 16, (2) from 17 to 20 and (3) from 21 to 4
#Hypothesis: when do people go out to restaurants?
def get_session(hour):
    hour = int(hour)
    if hour > 4 and hour < 12:
        return 0
    elif hour >= 12 and hour < 17:
        return 1
    elif hour >= 17 and hour < 21:
        return 2
    else:
        return 3
    
transactions['hour'] = transactions['hour'].apply(lambda x: get_session(x))
# Categorize day in 3 categories: (0) 0 to 10, (1) 11 to 20  and (2) over 20
# Hypothesis: People have more money or less money in the beginning or end of the month.
def get_day(day):
    if day <= 10:
        return 0
    elif day <=20:
        return 1
    else:
        return 2

transactions['day'] = transactions['day'].apply(lambda x: get_day(x))
transactions.head()
def aggregate_trans(df):
    agg_func = {
        'authorized_flag': ['mean', 'std'],
        'category_1': ['mean'],
        'cat_2_1.0': ['mean'],
        'cat_2_2.0': ['mean'],
        'cat_2_3.0': ['mean'],
        'cat_2_4.0': ['mean'],
        'cat_2_5.0': ['mean'],
        'cat_3_A': ['mean'],
        'cat_3_B': ['mean'],
        'cat_3_C': ['mean'],
        ###'numerical_1':['nunique','mean','std'], # merchants
        #'most_recent_sales_range': ['mean','std'], # merchants
        #'most_recent_purchases_range': ['mean','std'], # merchants
        ###'avg_sales_lag12':['mean','std'], # merchants
        ###'avg_purchases_lag12':['mean','std'], # merchants
        ###'active_months_lag12':['nunique'], # merchants
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],  # counts unique values of id for the rows that were groupped by card_id
        ###'state_id': ['nunique'], #
        'city_id': ['nunique'],
        ###'subsector_id': ['nunique'], # merchants
        ###'merchant_group_id': ['nunique'], # merchants
        'installments': ['sum','mean', 'max', 'min', 'std'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'weekend': ['mean', 'std'],
        'hour': ['mean', 'std'],
        'day': ['mean', 'std'],
        'weeks_to_Xmas_2017': ['mean', 'sum'],
        'weeks_to_BFriday': ['mean', 'sum'],
        'purchase_date': ['count'],
        'months_to_date': ['mean', 'max', 'min', 'std'],
        'Mothers_Day_2017': ['mean', 'sum'],
        'Fathers_day_2017': ['mean', 'sum'],
        'Children_day_2017': ['mean', 'sum'],
        'Valentine_Day_2017': ['mean', 'sum'],
        'Mothers_Day_2018': ['mean', 'sum'],
        'Labour_day_2017': ['mean', 'sum'],
        'Independence_day_2017': ['mean', 'sum'],
        'Easter_2017': ['mean', 'sum'],
        'Tiradentes_2017': ['mean', 'sum'],
        'Carnival_2017': ['mean', 'sum'],
        'Carnival_2018': ['mean', 'sum']
    }
    #'mer_category_4': ['mean'],
    #'mer_avg_sales_lag6':['nunique', 'mean','std'],
    #'mer_avg_purchases_lag6':['nunique', 'mean','std'],
    #'months_to_date': ['mean', 'max', 'min', 'std'],
    agg_df = df.groupby(['card_id']).agg(agg_func)
    agg_df.columns = ['_'.join(col)for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    return agg_df
def aggregate_per_month(history):
    
    # Group the dataframe by card_id and month_lag
    grouped = history.groupby(['card_id', 'month_lag'])
    # Convert the data type of the column installments to integer
    history['installments'] = history['installments'].astype(int)
    # Add aggregate functions count, sum, mean, min, max, std to the dataframe
    # agg_func is a dictionary that assigns the aggregate functions to the columns they will be applied on
    agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }

    #Aggregate using The above mentioned functions over the dictionary keys (purchase_amount, installments).
    intermediate_group = grouped.agg(agg_func)
    # Rename the columns add '-' between column name and aggregate function name
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    # Reset the index of the dataframe after aggregating
    intermediate_group.reset_index(inplace=True)

    # Group by card_id and add functions mean and std
    end_df = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    end_df.columns = ['_'.join(col).strip() for col in end_df.columns.values]
    end_df.reset_index(inplace=True)
    
    return end_df
# Aggregate transactions
agg_transactions = aggregate_trans(transactions)
agg_transactions_permonth = aggregate_per_month(transactions)
# Merge aggregated transactions
agg_trans = pd.merge(agg_transactions, agg_transactions_permonth, how='left', on='card_id')
agg_trans = reduce_mem_usage(agg_trans)
agg_trans.head()
# Delete agg_transactions and agg_transactions_per_month because they were already merged together
del agg_transactions
del agg_transactions_permonth
# Columns that have na values in them
nan_cols = agg_trans.columns[agg_trans.isna().any()].tolist()
nan_cols
## Replace infinite values with their
agg_trans = agg_trans.replace([np.inf,-np.inf], np.nan)
#agg_trans = agg_trans.fillna(value=0)  # Take the mean?

# agg_trans.mean(): calculates the mean of every column of the dataframe
#agg_trans = agg_trans.fillna(value=agg_trans.mean())
# It works faster when definined the columns containing nan values
agg_trans[nan_cols] = agg_trans[nan_cols].fillna(value=agg_trans[nan_cols].mean())
# Droping outliers worsened the performance
# What if we drop outliers from target variable
#train['outliers'] = 0
#train.loc[train['target'] < -30, 'outliers'] = 1
#train['outliers'].value_counts()
#train = train[train.outliers != 1]
#train = train.drop(columns=['outliers'])
#train.shape
# Add random values to the null values of the column
test['first_active_month'] = impute_na(test, train, 'first_active_month')
# Merge the train and test dataframes with the agg_trans dataframe
train = pd.merge(train, agg_trans, on='card_id', how='left')
test = pd.merge(test, agg_trans, on='card_id', how='left')
# Create columns year and month out of the column first_active_month 
train["year"] = train["first_active_month"].dt.year
test["year"] = test["first_active_month"].dt.year
train["month"] = train["first_active_month"].dt.month
test["month"] = test["first_active_month"].dt.month
# Get numerical features
numerical = [var for var in train.columns if train[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))

# Get discrete features
discrete = []
for var in numerical:
    if len(train[var].unique())<8:
        discrete.append(var)
        
print('There are {} discrete variables'.format(len(discrete)))

# Get continuous features
continuous = [var for var in numerical if var not in discrete and var not in ['card_id', 'first_active_month','target']]
print('There are {} continuous variables'.format(len(continuous)))
# Detect all null columns
train_null = train.columns[train.isnull().any()].tolist()
test_null = test.columns[test.isnull().any()].tolist()

# Get a set out of the null columns. The set only contains unique values.
in_first = set(train_null)
in_second = set(test_null)

# Get columns that are in the test dataframe but not in the train dataframe
in_second_but_not_in_first = in_second - in_first

# Create list of null columns
null_cols = train_null + list(in_second_but_not_in_first)
# Filling null
for col in null_cols:
    if col in continuous:
        # if it is a continuous column, fill with 0-s
        train[col] = train[col].fillna(0)#df_train[col].astype(float).mean())
        test[col] = test[col].fillna(0)#df_train[col].astype(float).mean())
    if col in discrete:
        # if it is a descrete columns fill with random values
        train[col] = impute_na(train, train, col)
        test[col] = impute_na(test, train, col)
print('Final null')
# There are no more null values in the dataframes
print_null(train)
print_null(test)
# Take card_id, first_active_month and target out of the list of the columns to use for training models, so that no errors are caused.
cols_to_use = list(train)
cols_to_use.remove('card_id')
cols_to_use.remove('first_active_month')
cols_to_use.remove('target')
# Take card_id, first_active_month and target out of the list of the columns to use for training models, so that no errors are caused.
names = list(train)
names.remove('card_id')
names.remove('first_active_month')
names.remove('target')
# Define a function that returns the cross-validation rmse error 
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train[names], train['target'], scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
#cv_ridge.min()
# Fit the training data to RidgeCV model
ridgeCV = RidgeCV(alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]).fit(train[names], train['target'])
rmse_cv(ridgeCV).mean()
# Predict the loyalty score of the test data
ridgeCV_pred = ridgeCV.predict(test[names])
#Submitting the prediction of the ridgecv regression.
submit = pd.DataFrame({"card_id":test["card_id"].values})
submit["target"] = ridgeCV_pred
submit.to_csv("elo_submission_ridgeCV.csv", index=False)
# Fit the training data to LassoCV model
lassoCV = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train[names], train['target'])
rmse_cv(lassoCV).mean()
# Predict the loyalty score of the test data
lassoCV_pred = lassoCV.predict(test[names])
# Submitting the prediction of the lasso regression.
submit = pd.DataFrame({"card_id":test["card_id"].values})
submit["target"] = lassoCV_pred
submit.to_csv("elo_submission_lasso.csv", index=False)
# Take a look at the coefficients
coef = pd.Series(lassoCV.coef_, index = train[names].columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# We are taking the first (highest) and last (lowest) 10 values of the coeffiecients and then we plot them.
imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])

plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
# Choosing max depth 3
# Fit the training data to the Regression Tree model
regr2 = DecisionTreeRegressor(max_depth=3)
regr2.fit(train[names], train['target'])

# Predict the loyalty score of the test data
pred_tree = regr2.predict(test[names])
# Submitting the prediction of a regression decision tree.
submit = pd.DataFrame({"card_id":test["card_id"].values})
submit["target"] = pred_tree
submit.to_csv("elo_submission_regression_tree.csv", index=False)
# This function creates images of tree models using pydot
def print_tree(estimator, features, class_names=None, filled=True):
    tree = estimator
    names = features
    color = filled
    classn = class_names
    
    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, feature_names=features, class_names=classn, filled=filled)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    return(graph)
# Print tree
graph, = print_tree(regr2, features=cols_to_use)
Image(graph.create_png())
# Fit the training data to Random Forest model
regr1 = RandomForestRegressor(max_features=13, random_state=1)
regr1.fit(train[names], train['target'])
# Predict the loyalty score of the test data
pred_forest = regr1.predict(test[names])

# Submitting the prediction of a random forest now.
submit = pd.DataFrame({"card_id":test["card_id"].values})
submit["target"] = pred_forest
submit.to_csv("elo_submission_random_forest_13.csv", index=False)
# Print feature importance
fig, ax = plt.subplots(figsize=(12,27))
Importance = pd.DataFrame({'Importance':regr1.feature_importances_*100}, index=train[names].columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='g', ax=ax )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
# Fit the training data to Boosting model
# n_estimators: The number of boosting stages to perform.
regr_b = GradientBoostingRegressor(n_estimators=600, learning_rate=0.005, random_state=1)
regr_b.fit(train[names], train['target'])
# Print feature importance
fig, ax = plt.subplots(figsize=(12,27))
feature_importance = regr_b.feature_importances_*100
rel_imp = pd.Series(feature_importance, index=train[names].columns).sort_values(inplace=False)
print(rel_imp)
rel_imp.T.plot(kind='barh', color='r', ax=ax)
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
# Predict the loyalty score of the test data
pred_b = regr_b.predict(test[names])
# Submitting the prediction of a boosting method.
submit = pd.DataFrame({"card_id":test["card_id"].values})
submit["target"] = pred_b
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    
    # Define the model parameters    
    params = {'num_leaves': 111,
             'min_data_in_leaf': 150,  # was 149 
             'objective':'regression',
             'max_depth': 9,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.75,
             "bagging_freq": 1,
             "bagging_fraction": 0.70,
             "bagging_seed": 11,
             "metric": 'rmse',
             "lambda_l1": 0.25,  # was 0.26
             "random_state": 1111,
             "verbosity": -1}

    # Convert train dataframe to a dataset
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    # Fit the training data to the lgb model
    model = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)
    # Predict the loyalty score of the test data
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

train_X = train[names]
test_X = test[names]
train_y = train.target.values

pred_test = 0  # Initialize pred_test

# Running a k-fold cross validation
kf = model_selection.KFold(n_splits=5, random_state=1111, shuffle=True)
for dev_index, val_index in kf.split(train):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test += pred_test_tmp
pred_test /= 5. 
# Plotting the first 50 features with highest importance
fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()
#Submitting the prediction of the lighgbm model.
submit = pd.DataFrame({"card_id":test["card_id"].values})
submit["target"] = pred_test
submit.to_csv("elo_submission_lightgbm.csv", index=False)
model_results = pd.read_excel('../input/model-results/submissions_table.xlsx')
model_results