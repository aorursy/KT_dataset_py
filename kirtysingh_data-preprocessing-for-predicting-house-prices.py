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
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("../input/housedata/data.csv")
data.head(10)
#lets see the descriptive statistics
data.describe()
data.info()
data['price'].describe()
#we can see there are few prices with value 0. Lets have a brief look
min(data['price'])
#lets remove the rows which have 0 values
data=data[data['price']!=0]
data['price'].describe()
data.info()
#lets visualise the whether there are outliers in our target variable or not
plt.scatter(range(len(data)),data['price'],color='red')
plt.xlabel("ID")
plt.ylabel("Price")
plt.title("PRICE SCATTER PLOT")
plt.show()
#lets fix these outliers using interquartile range
q1=data['price'].quantile(0.25)
q3=data['price'].quantile(0.75)
q1,q3
IQR=q3-q1
upper_limit=q3+1.5*IQR
lower_limit=q1-1.5*IQR
upper_limit,lower_limit
#lets create a function to remove all the outliers from our dataset as it would affect the efficiency of our model
def impute_outliers(data):
    if data>upper_limit:
        return upper_limit
    if data<lower_limit:
        return lower_limit
    else:
        return data
data['price']=data['price'].apply(impute_outliers)
#now again lets visualise our price column to make sure we have no outliers
plt.scatter(range(len(data)),data['price'],color='red')
plt.xlabel("ID")
plt.ylabel("Price")
plt.title("PRICE SCATTER PLOT")
plt.show()
data['price'].describe()
sns.boxplot(data['price'])
#Lets plot a histogram to visualise how data is distributed over a range
plt.hist(data['price'],bins=25,color='green')
plt.xlabel('intervals')
plt.ylabel('price')
plt.title("Price Distribution")
plt.show()
feature=[feature for feature in data.columns if min(data[feature])==0.0]
feature
data[feature].tail(10)
#impute bedrooms and bathrooms with 0 value with their median
numeric_feature=['bathrooms']
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=0.0,strategy='median')
data[numeric_feature]=imputer.fit_transform(data[numeric_feature])

#we will replace bedrooms with 0 values as mostfrequent value while doing we will observe an unexpected error lets have a look
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=0.0,strategy='most_frequent')
data['bedrooms']=imputer.fit_transform(data['bedrooms'])
#lets check the shape of our data
data['bedrooms'].shape
#now we will reshape the bedroom column
column=data['bedrooms'].values.reshape(-1,1)
column.shape
imputer=SimpleImputer(missing_values=0.0,strategy='most_frequent')
data['bedrooms']=imputer.fit_transform(column)
#convert zipcode to categorical varaible
data['statezip']=data['statezip'].astype('object')
data.info()
data.head(10)
#lets create 2 new columns "ever_renovated" and "year_since_renovation" from the column yr_renovated to keep track of the renovation year of the house
data['ever_renovated']=np.where(data['yr_renovated']==0,'No','Yes')
data['purchase_yr']=pd.DatetimeIndex(data['date']).year
data['yr_since_renovated']=np.where(data['ever_renovated']=='Yes',abs(data['purchase_yr']-data['yr_renovated']),0)
#now we can remove date,yr_renovated,purchase_yr column as we have stored its information in yr_since_renovated
data.drop(columns=['purchase_yr','yr_renovated','date'],inplace=True)
data.head()
data['waterfront']=np.where(data['waterfront']==1,'Yes','No')
data.head()
data['view'].value_counts()
data['view']=data['view'].astype('object')
data['condition']=data['condition'].astype('object')
#lets view the correlation of independent variables
data.corr()
#lets remove uncorrelated columns
data.drop(columns=['sqft_above','yr_built','yr_since_renovated'],inplace=True)
data.info()
#lets view the unique values in each categorical variable
data['waterfront'].unique()
len(data['street'].unique())
len(data['city'].unique())
len(data['country'].unique())
#lets view the distribution of categorical variables over target variable
data.groupby('waterfront')['price'].mean().plot(kind='bar')
data.groupby('street')['price'].mean().plot(kind='bar')
data.groupby('city')['price'].mean().sort_values().plot(kind='bar')
data.groupby('statezip')['price'].mean().sort_values().plot(kind='bar')
data.groupby('country')['price'].mean().plot(kind='bar')
data.groupby('ever_renovated')['price'].mean().plot(kind='bar')
data.groupby('condition')['price'].mean().sort_values().plot(kind='bar')
data.groupby('view')['price'].mean().sort_values().plot(kind='bar')
from statsmodels.formula.api import ols
import statsmodels.api as sm
mod=ols('price~waterfront',data=data).fit()
sm.stats.anova_lm(mod,typ=2)
mod=ols('price~street',data=data).fit()
sm.stats.anova_lm(mod,typ=2)
mod=ols('price~city',data=data).fit()
sm.stats.anova_lm(mod,typ=2)
mod=ols('price~statezip',data=data).fit()
sm.stats.anova_lm(mod,typ=2)
mod=ols('price~ever_renovated',data=data).fit()
sm.stats.anova_lm(mod,typ=2)
mod=ols('price~condition',data=data).fit()
sm.stats.anova_lm(mod,typ=2)
mod=ols('price~view',data=data).fit()
sm.stats.anova_lm(mod,typ=2)
#lets drop all those varaibles which are least correlated
data.drop(columns=['street','city','statezip','country'],inplace=True)
data.head(10)
data.info()

#Creating dummy variables
data=pd.get_dummies(data=data,columns=['waterfront','view','condition','ever_renovated'],drop_first=True)
data.head()
data.to_csv("Transformed_Data.csv",index=False)
