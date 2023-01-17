import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import matplotlib

matplotlib.rcParams['figure.figsize'] = (20,10)
data = pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')

data.head()
data.shape
data.groupby('area_type')['area_type'].agg('count')
data2 = data.drop(['area_type','availability','society','balcony'],axis = 1)

data2.head()
data2.isnull().sum()
data3 = data2.dropna()
data3.shape
data3.head()
data3['size'].unique()
data3['BHK'] = data3['size'].apply(lambda x: int(x.split()[0]))
data3.head()
data3.drop('size', axis = 1)
data3.head()
data3['total_sqft'].unique()
def is_float(x):

    try:

        float(x)

    except:

        return False

    return True
def convert_sqft_to_num(x):

    token = x.split('-')

    if len(token) == 2:

        return (float(token[0]) + float(token[1])) / 2

    try:

        return float(x)

    except:

        return None
data3[~data3['total_sqft'].apply(is_float)]
data4 = data3.copy()

data4['total_sqft'] = data4['total_sqft'].apply(convert_sqft_to_num)
data4.head()
data4.loc[30]
data5 = data4.copy()

data5['price_per_sqft'] = data5['price'] * 100000 / data5['total_sqft']

data5.head()
data5['location'].nunique()
data5['location'] = data5['location'].apply(lambda x: x.strip())
location_stats = data5.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats
len(location_stats[location_stats <= 10])
location_stats_less_than_10 = location_stats[location_stats <= 10]

location_stats_less_than_10
data5.location = data5.location.apply(lambda x:'other' if x in location_stats_less_than_10 else x)
data5.head()
data5['location'].nunique()
data6 = data5[~(data5['total_sqft'] / data5['BHK'] < 300)]
data6.describe()
data6.price_per_sqft.describe()
def remove_pps_outliers(df):

    df_out = pd.DataFrame()

    for key, val in df.groupby('location'):

        m = np.mean(val.price_per_sqft)

        st = np.std(val.price_per_sqft)

        reduced_df = val[(val.price_per_sqft > (m - st)) & (val.price_per_sqft <= (m + st))]

        df_out = pd.concat([df_out, reduced_df], ignore_index = True)

    return df_out
data7 = remove_pps_outliers(data6)

data7
def plot_scatter_chart(df, location):

    bhk2 = df[(df.location == location) & (df.BHK == 2)]

    bhk3 = df[(df.location == location) & (df.BHK == 3)]

    matplotlib.rcParams['figure.figsize'] = (15,10)

    plt.scatter(bhk2.total_sqft,bhk2.price,color = 'blue',label = '2 BHK')

    plt.scatter(bhk3.total_sqft,bhk3.price,color = 'green',label = '3 BHK')

    plt.legend()

    plt.xlabel('Total Square Feet')

    plt.ylabel('Price per square feet')

    plt.title(location)
plot_scatter_chart(data7, 'Rajaji Nagar')
def remove_bhk_outliers(df):

    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):

        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby('BHK'):

            bhk_stats[bhk] = {

                'mean': np.mean(bhk_df.price_per_sqft),

                'std': np.std(bhk_df.price_per_sqft),

                'count': bhk_df.shape[0]

            }

        for bhk, bhk_df in location_df.groupby('BHK'):

            stats = bhk_stats.get(bhk-1)

            if stats and stats['count']>5:

                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)

    return df.drop(exclude_indices,axis='index')

data8 = remove_bhk_outliers(data7)

# df8 = df7.copy()

data8.shape
plot_scatter_chart(data8, 'Hebbal')
matplotlib.rcParams["figure.figsize"] = (20,10)

plt.hist(data8.price_per_sqft,rwidth=0.8)

plt.xlabel("Price Per Square Feet")

plt.ylabel("Count")
data8.head()
data8['bath'].nunique()
data8[data8['bath'] > 10]
plt.hist(data8.bath,rwidth=0.8)

plt.xlabel("Number of bathrooms")

plt.ylabel("Count")
data9 = data8[data8.bath <= data8.BHK + 2]
data9.tail()
data10 = data9.drop(['size','price_per_sqft'],axis = 1)

data10.head()
dummies = pd.get_dummies(data10['location'])
data11 = pd.concat([data10,dummies.drop('other',axis = 1)],axis = 1)
data11.head()
data11.drop(['location'],axis = 1,inplace = True)
data11.head()
data11.shape
x = data11.drop('price', axis = 1)

y = data11['price']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2)
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(xtrain, ytrain)
lr.score(xtest,ytest)
from sklearn.model_selection import ShuffleSplit, cross_val_score
cv = ShuffleSplit(n_splits = 10,test_size = 0.2, random_state = 0)
cross_val_score(LinearRegression(),x,y,cv = cv)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor



def find_best_model_using_gridsearchcv(X,y):

    algos = {

        'linear_regression' : {

            'model': LinearRegression(),

            'params': {

                'normalize': [True, False]

            }

        },

        'lasso': {

            'model': Lasso(),

            'params': {

                'alpha': [1,2],

                'selection': ['random', 'cyclic']

            }

        },

        'decision_tree': {

            'model': DecisionTreeRegressor(),

            'params': {

                'criterion' : ['mse','friedman_mse'],

                'splitter': ['best','random']

            }

        }

    }

    scores = []

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():

        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)

        gs.fit(X,y)

        scores.append({

            'model': algo_name,

            'best_score': gs.best_score_,

            'best_params': gs.best_params_

        })



    return pd.DataFrame(scores,columns=['model','best_score','best_params'])



find_best_model_using_gridsearchcv(x,y)
def predict_price(location,sqft,bath,bhk):    

    loc_index = np.where(x.columns==location)[0][0]



    x1 = np.zeros(len(x.columns))

    x1[0] = sqft

    x1[1] = bath

    x1[2] = bhk

    if loc_index >= 0:

        x1[loc_index] = 1



    return lr.predict([x1])[0]
predict_price('Rajaji Nagar', 1800, 2, 3)