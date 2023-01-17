import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
data= pd.read_csv('../input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
data.head()
#Shape of dataset
data.shape
data.groupby('area_type')['area_type'].agg('count')
data2 = data.drop(['area_type','availability','society','balcony'],axis='columns')
data2.head()
# number of rows where every particular column's value is null
data2.isnull().sum()
#Drop null value
data3 = data2.dropna()
data3.isnull().sum()
# Size column unique values 
data3['size'].unique()
# Keep only numbers of bedrooms in size column
data3['bhk'] = data3['size'].apply(lambda x : int(x.split(' ')[0]))
data3['bhk'].unique()
# Total sqft values 
data3['total_sqft'].unique() 
# some total sqft values are range '1133-1384'
# first find float values
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
data3[~data3['total_sqft'].apply(is_float)]['total_sqft'].head(10)
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2 
    try:
        return float(x)
    except:
        return None
data4 = data3.copy()
data4['total_sqft'] = data4['total_sqft'].apply(convert_sqft_to_num)
data4['total_sqft'].head()
data5 = data4.copy()
data5['price_per_sqft'] = data5['price']*100000/data5['total_sqft']
data5.head()
# Number of Location
len(data5.location.unique())
#Reduce dimension
# remove spaces before and after the location
data5.location = data5.location.apply(lambda x: x.strip())
# how many point exists for every location
location_stats = data5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats
# locations with less than 10 datapoints
len(location_stats[location_stats<=10])

# there are 1052 location with less than 10 datapoints (1052 out of 1293)
location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10
# Convert location with less than 10 datapoints to 'other'
data5.location = data5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(data5.location.unique())
data5.head(10)
# Outliers detection : we admit that area of a given room is >= 300sqft
data5[data5.total_sqft/data5.bhk <300].head()

#remove houses with room's areas <300 sqft
data6 = data5[~(data5.total_sqft/data5.bhk <300)]
data6.shape
data6.price_per_sqft.describe()
# min 267 max 176470 let's remove extreme values based on standard deviation
# we  filter all datapoints that stand beyone 1 std
# Since the price depends on the location we filter with std of price_per_sqft per location
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>m-st)&(subdf.price_per_sqft<=m+st)]
        df_out = pd.concat([df_out,reduced_df], ignore_index=True)
    return df_out
data7 = remove_pps_outliers(data6)
# We notice that in some location house with 2 rooms are more expensive than ones with 3 rooms
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk==2)]
    bhk3 = df[(df.location == location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,20)
    plt.scatter(bhk2.total_sqft,bhk2.price, color= 'blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, color='green', label='3 BHK', s=50)
    plt.xlabel('Total square feet area')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(data7, "Rajaji Nagar")
# same with 'Hebbal' location
# for a given location we will build a dictionary of stats per bhk, 
# we then remove those 2 BHK apartement whose price per sqft is more than mean of price per sqft of 1 BHK apartement
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'std'  : np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
                
    return df.drop(exclude_indices, axis= "index")
    
    
data8 = remove_bhk_outliers(data7)
plot_scatter_chart(data8, "Rajaji Nagar")
# Historgram of price_per_sqft
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(data8.price_per_sqft, rwidth=0.8)
plt.xlabel = 'price_per_sqft'
plt.ylabel = 'count'
# Number of bathrooms Outliers
data8.bath.unique()
# datapoints with more than 10 bathrooms
data8[data8.bath>=10]
# historgram of bathrooms
plt.hist(data8.bath, rwidth=0.8)
plt.xlabel = 'Number of bathrooms'
plt.ylabel = 'count'
# Outliers = apartements that have more bathrooms than bhk+2
data9 = data8[data8.bath<data8.bhk+2]
data10 = data9.drop(['size', 'price_per_sqft'], axis='columns')
# turn location into dummies variable
dummies = pd.get_dummies(data10.location)
data11 = pd.concat([data10.drop('location', axis ='columns'), dummies.drop('other', axis = 'columns')], axis = 'columns')
data11.head()
# Dependent variables
X = data11.drop('price', axis ='columns')
#Independent variable
y = data11.price
# Split to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 10)
# Train linear regression model
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# Model performance
lr_clf.score(X_test, y_test)
# Cross validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits= 5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv= cv) 
# Run our model on different regressors : Lasso and Decision Tree
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression': {
            'model' : LinearRegression(),
            'params': {
                'normalize':[True, False]
            }
        },
        'lasso':{
            'model': Lasso(),
            'params':{
                'alpha':[1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree':{
            'model': DecisionTreeRegressor(),
            'params':{
                'criterion': ['mse','friedman_mse'],
                'splitter' : ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits= 5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        #print(config['params'])
        gs = GridSearchCV(config['model'], config['params'],cv=cv, return_train_score = False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_,
        })
    return pd.DataFrame(scores, columns= ['model', 'best_score', 'best_params'])
    
find_best_model_using_gridsearchcv(X,y)
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0 :
        x[loc_index] = 1
      
    return lr_clf.predict([x])[0]
predict_price('1st Phase JP Nagar',1000,2,2)
