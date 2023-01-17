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
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)
df1 = pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")
df1.head()
df1.shape
# get the information of data
df1.info()
df1.columns
df1['area_type'].unique()
df1['area_type'].value_counts()
import seaborn as sns
sns.scatterplot(df1['balcony'], df1['price'])
sns.countplot(df1['area_type'], hue='balcony', data=df1)
sns.jointplot(x=df1['bath'], y=df1['price'], data=df1)
df1.describe()
# We have only 3 neumerical features - bath, balcony and price
# 6 categorical features - area type, availability, size, society, and total_srft
# Target Feature =======>>>>>> price >>>>>>
# Price in lakh
 
#observe 75% and max value it shows huge diff
sns.pairplot(df1)

# bath and price have slightly linear correlation with some outliers
# value count of each feature
def value_count(df1):
  for var in df1.columns:
    print(df1[var].value_counts())
    print("--------------------------------")
value_count(df1)
# correlation heatmap
num_vars = ["bath", "balcony", "price"]
sns.heatmap(df1[num_vars].corr(),cmap="coolwarm", annot=True)
 
# correlation of bath is greater than a balcony with price
df1.isnull().sum()
df1.shape
df1.isnull().mean()*100 # % of measing value

#society has 41.3% missing value (need to drop)
# visualize missing value using heatmap to get idea where is the value missing
 
plt.figure(figsize=(16,9))
sns.heatmap(df1.isnull())
del_col = ['area_type','availability','society','balcony']
df2 = df1.drop(del_col, axis=1)
# drop na value rows from df2
# because there is very less % value missing
df3 = df2.dropna()
df3.shape
df3.isnull().sum()
df3.head()
df3['size'].unique()
df3['bhk'] = df3['size'].apply(lambda x : int(x.split(' ')[0]))
df3.head()
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
df3[~df3['total_sqft'].apply(is_float)].head(10)
# here we observe that 'total_sqft' contain string value in diff format
#float, int like value 1689.28,817 
# range value: 540 - 740 
# number and string: 142.84Sq. Meter, 117Sq. Yards, 1Grounds
 
# best strategy is to convert it into number by spliting it

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()
df4.isna().sum()
df5 = df4.copy()
df5['price_per_sqft'] = df5['price']* 100000 / df5['total_sqft']
df5.head()
df5.dtypes
# function to create histogram, Q-Q plot and boxplot
 
# for Q-Q plots
import scipy.stats as stats

def diagnostic_plots(df, variable):
    # function takes a dataframe (df) and
    # the variable of interest as arguments
 
    # define figure size
    plt.figure(figsize=(16, 4))
 
    # histogram
    plt.subplot(1, 3, 1)
    sns.distplot(df[variable], bins=30)
    plt.title('Histogram')
 
    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')
 
    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')
 
    plt.show()
    
num_var = ["bath","total_sqft","bhk","price"]
for var in num_var:
    print("******* {} *******".format(var))
    diagnostic_plots(df5, var)
 
  # here we observe outlier using histogram,, qq plot and boxplot
df5['location'] = df5['location'].apply(lambda x : x.strip())

loc_status = df4.groupby('location')['location'].agg('count')
loc_status.sort_values(ascending = False)
len(loc_status[loc_status <=10])
loc_status_less_10 = loc_status[loc_status <=10]
df5['location'] = df5['location'].apply(lambda x : 'other' if x in loc_status_less_10 else x)
df5.head()
df5.shape
df5[df5['total_sqft']/ df5['bhk'] <300 ].head()  #remove these rows
df6 = df5[~(df5['total_sqft']/ df5['bhk'] <300) ]
print(df6.head())
df6.shape
df6['price_per_sqft'].describe()
# Removing outliers using help of 'price per sqrt'  taking std and mean per location
def remove_pps_outliers(df):
  df_out = pd.DataFrame()
  for key, subdf in df.groupby('location'):
    m=np.mean(subdf.price_per_sqft)
    st=np.std(subdf.price_per_sqft)
    reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
    df_out = pd.concat([df_out, reduced_df], ignore_index = True)
  return df_out
df7 = remove_pps_outliers(df6)
df7.shape
def scatter_chart(df, location):
    bhk2 = df[(df['location'] == location) & (df['bhk'] == 2)]
    bhk3 = df[(df['location'] == location) & (df['bhk'] == 3)]
    
    matplotlib.rcParams['figure.figsize'] = (15,10)
    
    plt.scatter(bhk2['total_sqft'], bhk2['price_per_sqft'], label='2 BHK', s=50)
    plt.scatter(bhk3['total_sqft'], bhk3['price_per_sqft'], marker='+',label= '3 BHK', s= 50, color='green')
    plt.xlabel("Total Square Feat Area")
    plt.ylabel("Price per Sqft")
    plt.title(location)
    plt.legend()
scatter_chart(df7, 'Rajaji Nagar')

# in below scatterplot we observe that at same location price of
# 2 bhk house is greater than 3 bhk so it is outlier
scatter_chart(df7, "Hebbal")

# in below scatterplot we observe that at same location price of
# 3 bhk house is less than 2 bhk so it is outlier
def rm_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df['price_per_sqft']),
                'std'  : np.std(bhk_df['price_per_sqft']),
                'count' : bhk_df.shape[0]
            }
            
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats  = bhk_stats.get(bhk -1)
            
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df['price_per_sqft'] < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')
df8 = rm_bhk_outliers(df7)
df8.shape
scatter_chart(df8, 'Rajaji Nagar')
scatter_chart(df8, "Hebbal")
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
df8.bath.unique()
plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
df8[df8.bath>10]

#It is unusual to have 2 more bathrooms than number of bedrooms in a home
df8[df8.bath>df8.bhk+2]
#if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max.

df9 = df8[df8.bath<df8.bhk+2]
df9.shape
df9.head(2)
df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)
dummies = pd.get_dummies(df10.location)
dummies.head(3)
df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()
df12 = df11.drop('location',axis='columns')
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

cv = ShuffleSplit(n_splits=6, test_size=0.2, random_state=10)

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
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)
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
predict_price('1st Phase JP Nagar',1000, 2, 2)
predict_price('Indira Nagar', 1000, 2, 2 )
predict_price('1st Phase JP Nagar', 1000, 3, 3)