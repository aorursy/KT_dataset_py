# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline

import matplotlib 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_data=pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')

raw_data.head()
raw_data.shape
list(raw_data.groupby('area_type'))
list(raw_data.groupby('area_type')['area_type'])
list(raw_data.groupby('area_type')['location'])
raw_data.describe()
raw_data.groupby('area_type')['location'].agg('count')
featured_data=raw_data.drop(['area_type','availability','society'],axis='columns')

featured_data.head()
featured_data.isnull().sum()
featured_data.dropna(inplace=True)
featured_data.isnull().sum()
featured_data.shape
featured_data['size'].unique()
featured_data['balcony'].unique()
featured_data['bhk']=featured_data['size'].apply(lambda x: int(x.split(' ')[0]))

featured_data.head()
featured_data.drop(['size'],axis='columns',inplace=True)

featured_data.head()
featured_data.loc[featured_data['bhk']>11] #featured_data[featured_data['bhk']>15] #featured_data[featured_data.bhk>15]
featured_data.shape
featured_data.groupby('bhk')['bhk'].agg('count')
featured_data.bhk.describe()
featured_data.groupby('total_sqft')['total_sqft'].agg('count')
featured_data=featured_data[featured_data['bhk']<=12]
featured_data.shape
featured_data.groupby('balcony')['total_sqft'].agg('count')
featured_data=featured_data[featured_data['bath']<=8]

featured_data.shape
featured_data.groupby('bath')['bath'].agg('count')
featured_data.head()
featured_data['total_sqft'].unique()
def is_float(x):

    try:

        float(x)

    except:

        return False

    return True
featured_data[featured_data['total_sqft'].apply(is_float)]
featured_data[~featured_data['total_sqft'].apply(is_float)].head()
def clean_data(x):

    tokens=x.split('-')

    if len(tokens) == 2:

        return (float(tokens[0])+float(tokens[1]))/2

    

    try:

        return float(x)

    except:

        return None        
clean_data('1000-1000')
featured_data['total_sqft']=featured_data['total_sqft'].apply(clean_data)

featured_data['total_sqft'].unique()
featured_data.shape
featured_data['total_sqft']
featured_data.isnull().sum()
featured_data.dropna(inplace=True)
featured_data.isnull().sum()
featured_data.shape
len(featured_data['location'].unique())
featured_data['location']=featured_data['location'].apply(lambda x : x.strip())
featured_data.groupby('location')['location'].agg('count')
location_stats=featured_data.groupby('location')['location'].agg('count').sort_values()

location_stats.head(100)
less_locations=location_stats[location_stats<10]

less_locations
featured_data['location']=featured_data['location'].apply(lambda x: 'others' if x in less_locations else x)

len(featured_data['location'].unique())
featured_data['price_per_sqft']=(featured_data['price']*100000)/featured_data['total_sqft']

featured_data.head()
featured_data=featured_data[~(featured_data['total_sqft']/featured_data['bhk']<250)]

featured_data.shape
featured_data['price_per_sqft'].describe()
def remove_outlier_ppsqft(df):

    new_df=pd.DataFrame()

    for key,sub_data in df.groupby('location'):

        mn=np.mean(sub_data['price_per_sqft'])

        stdv=np.std(sub_data['price_per_sqft'])

        accepted_data=sub_data[(sub_data['price_per_sqft']>(mn-stdv)) & (sub_data['price_per_sqft']<=(mn+stdv)) ]

        new_df=pd.concat([new_df,accepted_data])

    return new_df    
new_data=remove_outlier_ppsqft(featured_data)
new_data.shape
new_data.head()
def plot_area(df,location):

    bhk2=df[(df['location']==location) & (df['bhk']==2)]

    bhk3=df[(df['location']==location) & (df['bhk']==3)]

    matplotlib.rcParams['figure.figsize']=(15,10)

    plt.scatter(bhk2.total_sqft,bhk2.price,color="blue",label="2 bhk",s=70)

    plt.scatter(bhk3.total_sqft,bhk3.price,marker="+",color="green",label="3 bhk",s=80)

    plt.legend()

    plt.title(location,color='red')

    plt.xlabel('Total Square Feet')

    plt.ylabel('Total Price')

    

plot_area(new_data,"Rajaji Nagar")    
new_data.groupby('location')['location'].agg('count')
new_data.groupby('location')['location'].agg('count').sort_values(ascending=False).head(10)
plot_area(new_data,"Whitefield")
plot_area(new_data,"Marathahalli")
def remove_outliers(df):

    arr_del=np.array([])

    for location_name,location_data in df.groupby('location'):

        bhk_stats=dict()

        for bhk_number,bhk_data in location_data.groupby('bhk'):

            bhk_stats[bhk_number]={

                'mean':np.mean(bhk_data.price_per_sqft),

                'stdv':np.std(bhk_data.price_per_sqft),

                'count':bhk_data.shape[0]

            }

        for bhk_number,bhk_data in location_data.groupby('bhk'):

            prev_stats=bhk_stats.get(bhk_number-1)

            if prev_stats:

                arr_del=np.append(arr_del,bhk_data[bhk_data.price_per_sqft<(prev_stats['mean'])].index.values)

            #exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)

    return df.drop(arr_del,axis='index')



final_data=remove_outliers(new_data)
final_data.shape
plot_area(final_data,"Whitefield")
plot_area(final_data,"Rajaji Nagar")   
dummies=pd.get_dummies(final_data['location'])

dummies.head()
final_data=pd.concat([final_data,dummies.drop('others',axis='columns')],axis='columns')

final_data.head()
final_data
final_data.drop(['location'],axis='columns',inplace=True)

final_data.head()
#out=final_data['price']

out.head(60)
out.shape
final_data.drop(['price'],axis='columns',inplace=True)

final_data.head()
from sklearn.model_selection import ShuffleSplit

from  sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score
cv=ShuffleSplit(n_splits=6 ,test_size=0.3,random_state=10)

a=list(cross_val_score(LinearRegression(),final_data,out,cv=cv))

for x in a:

    print(x)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor



def find_the_best(X,y):

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

    

find_the_best(final_data,out)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

x_train,x_test,y_train,y_test=train_test_split(final_data,out,test_size=0.3,random_state=82)



Dt_model=DecisionTreeRegressor(criterion='friedman_mse',splitter= 'best')

Dt_model.fit(x_train,y_train)

Dt_model.score(x_test,y_test)
a=np.where(final_data.columns=="bath")

a[0][0]
def predict_price(location,sqft,bath,balcony,bhk):

    location_index=np.where(final_data.columns==location)[0][0]

    x=np.zeros(len(final_data.columns))

    x[0]=sqft

    x[1]=bath

    x[2]=balcony

    x[3]=bhk

    if location_index>=0:

        x[location_index]=1

    return Dt_model.predict([x])[0]

predict_price('Indira Nagar',1000, 3, 1,3)
final_data.loc[210][1]
import pickle

with open('house_price.pickle','wb') as f:

    pickle.dump(Dt_model,f)
import json

columns={

    'data_cols':[col.lower() for col in final_data.columns]

}

with open('column_names.json','w') as f:

    f.write(json.dumps(columns))