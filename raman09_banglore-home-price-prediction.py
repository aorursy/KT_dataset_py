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
# import the requires libraries

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

import matplotlib 

matplotlib.rcParams["figure.figsize"] = (20,10)
# Load the data

df = pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")

df.head()
df.shape
df.describe()
df.columns
df.info()
# count of flats with different areas type

df['area_type'].value_counts()
df.groupby('area_type')['area_type'].agg('count')
# drop features that are not required and save it into df2

df2 = df.drop(['area_type','society','balcony','availability'],axis='columns')

df2.shape
# cheaking null values

df2.isnull().sum()
# find the num values in each columns

df2.isnull().sum()
# drop the null values

df3 = df2.dropna()
# there will be no null values now

df3.isnull().sum()
# check the total records of df2(with null values) and df3(removed null values)

print(df2.shape)

print(df3.shape)
# find all the distinct size of falts

df3['size'].unique()
# to make the flat size uniform remove the strings literals and save it into new 'bhk' column

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

df3.bhk.unique()
# these are the dummy data or corrupt data as for a 27bhk total_sqft is 8000 which is incorrect.

df3[df3.bhk>20]
df3.total_sqft.unique()
# this function will change the total_sqft i.e flat area into float from string.

def is_float(x):

    try:

        float(x)

    except:

        return False

    return True
df3[~df3['total_sqft'].apply(is_float)].head(10)
# 34.46Sq. Meter is incorrect values so, we will remove these incorrect data and then take the avg of

# flat size

def convert_sqft_to_num(x):

    tokens = x.split('-')

    if len(tokens) ==2:

        return (float(tokens[0]) + float(tokens[1]))/2

    try:

        return float(x)

    except:

        return None
df4 = df3.copy()

df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)

df4 = df4[df4.total_sqft.notnull()]

df4.head(2)
df5 = df4.copy()

df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']

df5.head()
df5_stats = df5['price_per_sqft'].describe()

df5_stats
df5['location'].head(10)
df5.location = df5.location.apply(lambda x: x.strip())

location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)

location_stats
location_stats_less_than_10 = location_stats[location_stats<=10]

location_stats_less_than_10
# total unique records for location

len(df5.location.unique())
# for all the locations having flats less than 10 make it's location as other 

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

len(df5.location.unique())
df5[df5.location == 'other'].head()
df5[df5.total_sqft/df5.bhk<300].head()
df6 = df5[~(df5.total_sqft/df5.bhk<300)]

df6.shape
df6.price_per_sqft.describe()
def remove_pps_outliers(df):

    df_out = pd.DataFrame()

    for key, subdf in df.groupby('location'):

        m = np.mean(subdf.price_per_sqft)

        st = np.std(subdf.price_per_sqft)

        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]

        df_out = pd.concat([df_out,reduced_df],ignore_index=True)

    return df_out

df7 = remove_pps_outliers(df6)

df7.shape
# Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like

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

    

plot_scatter_chart(df7,"Rajaji Nagar")
plot_scatter_chart(df7,"Hebbal")
# Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft 

# of 1 BHK apartment

def remove_bhk_outliers(df):

    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):

        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby('bhk'):

            bhk_stats[bhk] = {

                'mean': np.mean(bhk_df.price_per_sqft),

                'std': np.std(bhk_df.price_per_sqft),

                'count': bhk_df.shape[0]

            }

        for bhk, bhk_df in location_df.groupby('bhk'):

            stats = bhk_stats.get(bhk-1)

            if stats and stats['count']>5:

                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)

    return df.drop(exclude_indices,axis='index')

df8 = remove_bhk_outliers(df7)

# df8 = df7.copy()

df8.shape
plot_scatter_chart(df8,"Rajaji Nagar")
plot_scatter_chart(df8,"Hebbal")
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20,10)

plt.hist(df8.price_per_sqft,rwidth=0.8)

plt.xlabel("Price Per Square Feet")

plt.ylabel("Count")
df8[df8.bath>10]
# It is unusual to have 2 more bathrooms than number of bedrooms in a home

df8[df8.bath>df8.bhk+2]
df9 = df8[df8.bath<df8.bhk+2]

df9.shape
df10 = df9.drop(['size','price_per_sqft'],axis='columns')

df10.head(3)
dummies = pd.get_dummies(df10.location)

dummies.head(3)
df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')

df11.head()
df12 = df11.drop('location',axis='columns')

df12.head(2)
X = df12.drop(['price'],axis='columns')

y = df12.price

print("X_shape", X.shape)

print("y_shape", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)



from sklearn.linear_model import LinearRegression

lr_clf = LinearRegression()

lr_clf.fit(X_train,y_train)

lr_clf.score(X_test,y_test)
#Use K Fold cross validation to measure accuracy of our LinearRegression model

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)



cross_val_score(LinearRegression(), X, y, cv=cv)
# Find best model using GridSearchCV

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

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():

        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)

        gs.fit(X,y)

        scores.append({

            'model': algo_name,

            'best_score': gs.best_score_,

            'best_params': gs.best_params_

        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])



find_best_model_using_gridsearchcv(X,y)    
#Test the model for few properties

def predict_price(location,sqft,bath,bhk):    

    loc_index = np.where(X.columns==location)[0][0]



    x = np.zeros(len(X.columns))

    x[0] = sqft

    x[1] = bath

    x[2] = bhk

    if loc_index >= 0:

        x[loc_index] = 1



    return lr_clf.predict([x])[0]
predict_price('1st Phase JP Nagar',1000, 2, 2)