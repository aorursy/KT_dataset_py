import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline

import matplotlib 

matplotlib.rcParams["figure.figsize"] = (20,10)

import seaborn as sns

from sklearn import preprocessing

from sklearn import model_selection

import sklearn

import xgboost



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
home = pd.read_csv("/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv")

home.head()
home.info()
## finding null values in %form



round(100*(home.isnull().sum()/len(home.index)),2)
#removing NaN values from the dataset

home.dropna(inplace =True)
home = home.drop(columns='society')
home.reset_index(drop= True, inplace =True)
home['bhk'] = home['size'].str.split().str[0]

home['bhk'].dropna(inplace = True)

home['bhk'] = home['bhk'].astype('int')
print(home['total_sqft'].iloc[[17]])



## fucntion to remove 2100 - 2850 by taking there average

def convert_sqft_to_num(x):

    tokens = x.split('-')

    if len(tokens) == 2:

        return (float(tokens[0])+float(tokens[1]))/2

    try:

        return float(x)

    except:

        return None
## applying the fucntion to the column: - 'total_sqft'

home.total_sqft = home.total_sqft.apply(convert_sqft_to_num)

# Taking only the Numeric values from the data and storing it in 'home'

home = home[home.total_sqft.notnull()]

# display the first 2 columns from the dataset

home.head(2)
##removing invalid data entry

## Example: The total sqft divided by the number of bhk should always be more than 300



home = home[~(home.total_sqft/home.bhk<200)]

home.shape
## dividing the dataset into Continous and Categorical variables:

cont_ = home.select_dtypes(exclude = 'object')

cat_ = home.select_dtypes(include  = 'object')
## displaying only the continous variables from the dataset

## to determine the variables which have outliers and those which needs to be removed

fig = plt.figure(figsize = (10,8))

for index,col in enumerate(cont_):

    plt.subplot(3,2,index+1)

    sns.boxplot(y = cont_.loc[:,col])

fig.tight_layout(pad = 1.0)
home = home.drop(home[home['bath']>6].index)

home = home.drop(home[home['bhk']>7.0].index)
## Feature Engineering step

home['price_per_sqft'] = home['price']*100000/home['total_sqft']

home.head()
home['price_per_sqft'].describe()
## taking only the values with 1st Standard devaition values.

## as per Normal Distribution, 95% of our data lies within 1st Standard Deviation as per the location



def remove_pps_outliers(df):

    df_out = pd.DataFrame()

    for key, subdf in df.groupby('location'):

        m = np.mean(subdf.price_per_sqft)

        st = np.std(subdf.price_per_sqft)

        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]

        df_out = pd.concat([df_out,reduced_df],ignore_index=True)

    return df_out

home = remove_pps_outliers(home)

home.shape
## finding correlation values within the dataset

## we remove features which are highly related to each other as they do not provide

## any significance value to our Model



corr = home.corr()

plt.figure(figsize = (10,8))

sns.heatmap(corr,mask = corr<0.8 ,annot= True,cmap = 'Blues')
home.drop(columns=['availability','size','area_type'],inplace = True)
## checking the dataset with highest location data provided

## because havind values for a location less than 10 wont give us good information on the dataset



home.location = home.location.str.strip()

location_stats = home['location'].value_counts(ascending=False)

location_stats
## cretaing a Series of all the location having less than 10 entries against its  

location_stats_less_than_10 = location_stats[location_stats<=10]

location_stats_less_than_10
## using lambda function to naming 'location_stats_less_than_10' as 'other' and then removing it



home.location = home.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)



home = home[home.location != 'other']
## Keeping in mind that the number of Bathroom shouldn't be more than BHK+2

## Example for a 3 bhk, the number of bathrooms shouldn't be more than 5



home = home[home.bath<home.bhk+2]
## representing Numerical Data and Visualizing the same usin Distplot to gain further info



num_ = home.select_dtypes(exclude = 'object')

fig = plt.figure(figsize =(10,8))

for index, col in enumerate(num_):

    plt.subplot(3,2,index+1)

    sns.distplot(num_.loc[:,col],kde = False)

fig.tight_layout(pad = 1.0)
## performing One hot encoding on the Categorical values

## 1st step. create dummies

dummies = pd.get_dummies(home.location)

dummies.head(3)
## adding the dummies dataframe to our main DataFrame



home = pd.concat([home,dummies],axis='columns')



## removing 'location' as we have already created the dummies

home1 = home.drop('location',axis = 1)



## removing columns which will not be required by our model

home1 = home1.drop(columns=['balcony','price_per_sqft'])

home1
home1.reset_index(drop = True)
## Dividing our dataset to Independent and Dependent Variables



X = home1.drop('price',axis = 1).values ## Independent Variables

y = home1.price.values ## Dependent Variables
## adding a new axis

y = y[:,np.newaxis]
## preprocessing the data values to StandardScaler

sc = preprocessing.StandardScaler()

X1 = sc.fit_transform(X)

## Standardize a dataset along any axis



## Center to the mean and component wise scale to unit variance.



Std_x1 = preprocessing.scale(X)
## importing the required libraries for Machine Learning



from sklearn.model_selection import cross_val_score,cross_val_predict

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

from sklearn.model_selection import cross_validate as CV
## using Cross Validation of 5 andscoring of Negative mean sqaured error



cross1 = cross_val_score(lr,Std_x1,y,cv=5,scoring='neg_mean_squared_error')

print(cross1.mean())
sklearn.metrics.SCORERS.keys()
# from the model selection module import train_test_split for the ML training and testing.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1,y,test_size=0.3,random_state=10)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

acc = mean_squared_error(y_pred,y_test)

rscore = r2_score(y_pred,y_test)

print(rscore)
