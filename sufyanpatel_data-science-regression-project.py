import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)
df = pd.read_csv('../input/flats-for-rent-in-mumbai/prop_data_clean.csv')
df.head()
df['city'].unique()
df.columns
df.count().sum()
df['floor_count'].isnull().sum()
df1 = df.drop(['city','desc','dev_name','floor_count', 'floor_num', 'id', 'id_string','post_date', 'poster_name','project',
       'title', 'trans', 'type', 'url', 'latitude','longitude'],axis='columns')
df1.head()
df1['user_type'].unique()
df1.shape
df1.isnull().sum()
df2 = df1.dropna()
df2.isnull().sum()
df2.shape
df2.head()
location = df2['locality'].value_counts()
location
location.values.sum()
len(location[location>10])
len(location)
len(location[location<=10])
location_stats_less_than_10 = location[location<=10]
location_stats_less_than_10
len(df2.locality.unique())
df2['locality'] = df2.locality.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df2.locality.unique())
df2.head(10)
df3 = df2.copy()
df3.head()
df3['furnishing'].unique()
df3.head()
df3['price_per_sqft'] = df3['price']/df3['area']
df3.head()
df3_stats = df3['price_per_sqft'].describe()
df3_stats
df3.to_csv("stage1_sufyan_project_mumbai_home_price.csv",index=False)
df3.head()
df3['bedroom_num'].value_counts()
bathroom_stat = df3['bathroom_num'].value_counts()
bathroom_stat
bathroom_stat_greater_six = bathroom_stat[bathroom_stat<=4]
bathroom_stat_greater_six
df3[df3.area/df3.bedroom_num<300].head(20)
df3.price_per_sqft.describe()
df3.shape
df4 = df3.copy()
df4.head()
temp = df4[df4.area/df4.bedroom_num<200]
len(temp)
df4.shape
df5 = df4[~(df4.area/df4.bedroom_num<200)]
df5.shape
df5['price_per_sqft'].describe()
df6 = df5.copy()
df6.head()
df6.shape
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('locality'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape
df7.head()
def plot_scatter_chart(df,locality):
    bhk2 = df[(df.locality==locality) & (df.bedroom_num==1)]
    bhk3 = df[(df.locality==locality) & (df.bedroom_num==2)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.area,bhk2.price,color='blue',label='1 BHK', s=50)
    plt.scatter(bhk3.area,bhk3.price,marker='+', color='green',label='2 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(locality)
    plt.legend()
    
plot_scatter_chart(df7,"Andheri East")
def plot_scatter_chart(df,locality):
    bhk2 = df[(df.locality==locality) & (df.bedroom_num==1)]
    bhk3 = df[(df.locality==locality) & (df.bedroom_num==2)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.area,bhk2.price,color='blue',label='1 BHK', s=50)
    plt.scatter(bhk3.area,bhk3.price,marker='+', color='green',label='2 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(locality)
    plt.legend()
    
plot_scatter_chart(df7,"Lokhandwala Complex")
df7.shape
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('locality'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bedroom_num'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bedroom_num'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape
def plot_scatter_chart(df,locality):
    bhk2 = df[(df.locality==locality) & (df.bedroom_num==1)]
    bhk3 = df[(df.locality==locality) & (df.bedroom_num==2)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.area,bhk2.price,color='blue',label='1 BHK', s=50)
    plt.scatter(bhk3.area,bhk3.price,marker='+', color='green',label='2 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(locality)
    plt.legend()
    
plot_scatter_chart(df8,"Andheri East")
def plot_scatter_chart(df,locality):
    bhk2 = df[(df.locality==locality) & (df.bedroom_num==1)]
    bhk3 = df[(df.locality==locality) & (df.bedroom_num==2)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.area,bhk2.price,color='blue',label='1 BHK', s=50)
    plt.scatter(bhk3.area,bhk3.price,marker='+', color='green',label='2 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(locality)
    plt.legend()
    
plot_scatter_chart(df8,"Lokhandwala Complex")
df9 = df8.copy()
df9.head()
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df9.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
df9['bathroom_num'].unique()
df9[df9['bathroom_num']>df9['bedroom_num']+2]
df9 = df9[df9['bathroom_num']<df9['bedroom_num']+2]
df9.shape
df10 = df9.drop(['price_per_sqft','user_type'],axis='columns')
df10.head()
df10['furnishing'].unique()
df11 = df10.copy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df11['furnishing']= le.fit_transform(df10['furnishing']) 
df11['furnishing'].unique()
df11.head()
dummies = pd.get_dummies(df11.locality)
dummies.head(3)
df11 = pd.concat([df11,dummies.drop('other',axis='columns')],axis='columns')
df11.head()
df12 = df11.drop('locality',axis='columns')
df12.head(2)
df12.shape
X = df12.drop(['price'],axis='columns')
X.head(3)
X.shape
y = df12.price
y.head(3)
len(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)
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
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]
predict_price('4 Bunglows',700, 2, 2)
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
