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
unique_category = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.csv')

unique_category_sort = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv')

summer_products = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
pd.set_option('display.max_rows',None)

pd.set_option('display.max_columns',None)
summer_products.head()
summer_products.shape
summer_products.isnull().sum()
summer_products.drop(['title','title_orig','shipping_option_name','urgency_text','merchant_info_subtitle','merchant_id','merchant_has_profile_picture','merchant_profile_picture','product_url','product_picture','product_id'],axis=1,inplace=True)
summer_products.shape
summer_products.isnull().sum()
summer_products.head()
summer_products['crawl_month'].unique()
summer_products['currency_buyer'].value_counts()
summer_products.drop(['crawl_month','currency_buyer'],axis=1,inplace=True)
max_rating_count = max(summer_products['rating_count'])
summer_products['rating/rating_count'] = (summer_products['rating']/summer_products['rating_count'])*100
summer_products.drop(['rating','rating_count'],axis=1,inplace=True)
summer_products.drop('tags',axis=1,inplace=True)
summer_products['product_color'].value_counts()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()
from sklearn.pipeline import Pipeline



class MultiColumnLabelEncoder:

    def __init__(self,columns = None):

        self.columns = columns # array of column names to encode



    def fit(self,X,y=None):

        return self # not relevant here



    def transform(self,X):

        '''

        Transforms columns of X specified in self.columns using

        LabelEncoder(). If no columns specified, transforms all

        columns in X.

        '''

        output = X.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col] = LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname] = LabelEncoder().fit_transform(col)

        return output



    def fit_transform(self,X,y=None):

        return self.fit(X,y).transform(X)
summer_products['product_color'] = summer_products['product_color'].astype('str')
summer_products['product_variation_size_id'] = summer_products['product_variation_size_id'].astype('str')
summer_products.drop('theme',axis=1,inplace=True)
summer_products.drop('merchant_name',axis=1,inplace=True)
summer_products.isnull().sum()
summer_products['has_urgency_banner'] = summer_products['has_urgency_banner'].fillna(0)
summer_products['origin_country'] = summer_products['origin_country'].fillna(summer_products['origin_country'].mode()[0])
summer_products.shape
def replace_with_median(data,column_list):

    for col in column_list:

        data[col] = data[col].fillna(data[col].median())
column_list = [i for i in summer_products if summer_products[i].isnull().sum() != 0]
column_list
replace_with_median(summer_products,column_list)
summer_products.head()
!nvidia-smi
summer_products_corr = summer_products.corr()
summer_products_corr['price']
import matplotlib.pyplot as plt

import seaborn as sns
sns.jointplot(x='price',y='retail_price',data=summer_products,kind='scatter')
summer_products.drop(summer_products[summer_products['retail_price'] > 200].index,inplace=True)
summer_products.drop(summer_products[summer_products['price'] > 30].index,inplace=True)
summer_products['rating/rating_count_of_merchant'] = (summer_products['merchant_rating']/summer_products['merchant_rating_count'])*100
summer_products.drop(['merchant_rating','merchant_rating_count'],axis=1,inplace=True)
sns.jointplot(x='price',y='units_sold',data=summer_products,kind='scatter')
from scipy import stats

sns.distplot(summer_products['units_sold'],fit=stats.norm)
summer_products['units_sold'].skew()
sns.boxplot(summer_products['units_sold'])
sns.violinplot(summer_products['units_sold'])
summer_products['units_sold'].unique()
bins = [0, 100, 1000, 5000, 10000, 50000, 100000,np.inf]

labels = [1,2,3,4,5,6,7]

summer_products['units_sold_binned'] = pd.cut(summer_products['units_sold'], bins=bins, labels=labels)
summer_products['units_sold_binned'] = summer_products['units_sold_binned'].astype('int')
summer_products.drop(['units_sold'],axis=1,inplace=True)
summer_products.drop(column_list,axis=1,inplace=True)
summer_products['product_color'] = le.fit_transform(summer_products['product_color'])
summer_products['product_variation_size_id'] = le.fit_transform(summer_products['product_variation_size_id'])
summer_products.head()
summer_products_corr = summer_products.corr()

summer_products_corr['price']
sns.jointplot(data=summer_products,x='price',y='badges_count')
sns.jointplot(data=summer_products,x='price',y='badge_local_product')
sns.jointplot(data=summer_products,x='price',y='badge_product_quality')
sns.jointplot(data=summer_products,x='price',y='badge_fast_shipping')
sns.jointplot(data=summer_products,x='price',y='shipping_is_express')
summer_products.drop(['badge_local_product','shipping_is_express'],axis=1,inplace=True)
summer_products['countries_shipped_to'].unique()
expensive_country = summer_products.groupby('countries_shipped_to').agg({'price':'mean'})
expensive_country.sort_values(by='price',ascending=False,inplace=True)

expensive_country.head(10)
expensive_country.tail(10)
summer_products['inventory_total'].unique()
bins = [0, 10, 20, 30, 40, 50,np.inf]

labels = [1,2,3,4,5,6]

summer_products['inventory_total_binned'] = pd.cut(summer_products['inventory_total'], bins=bins, labels=labels)
summer_products.drop('inventory_total',axis=1,inplace=True)

summer_products['inventory_total_binned'] = summer_products['inventory_total_binned'].astype('int')
summer_products['has_urgency_banner'].value_counts()
sns.jointplot(data=summer_products,x='price',y='has_urgency_banner')
summer_products['origin_country'].unique()
summer_products['origin_country'] = le.fit_transform(summer_products['origin_country'])
summer_products.drop(['merchant_title'],axis=1,inplace=True)
summer_products.shape
summer_products.head()
features = summer_products.drop('price',axis=1)

labels = summer_products['price']
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
features.isnull().sum()
from sklearn.ensemble import RandomForestRegressor



rfr = RandomForestRegressor(n_estimators=100)
features.drop(['rating/rating_count','rating/rating_count_of_merchant'],axis=1,inplace=True)
rfr.fit(features,labels)
ranked_features=pd.Series(rfr.feature_importances_,index=features.columns)

ranked_features.nlargest(7).plot(kind='barh')

plt.show()
ranked_features.sort_values(ascending=False)[:8]
features = features[['shipping_option_price','retail_price','countries_shipped_to','product_color','product_variation_size_id','units_sold_binned','product_variation_inventory']]
rfr_ = RandomForestRegressor()
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score

from sklearn import metrics
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=42)
X_train.shape,X_test.shape
cross_score = cross_val_score(X=features,y=labels,estimator=rfr_,cv=3,verbose=10,n_jobs=-1,scoring='neg_root_mean_squared_error')
cross_score.mean()
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression,Lasso,Ridge
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
sns.distplot(summer_products['price'],fit=stats.norm)
summer_products['price'].skew()
lr = LinearRegression(n_jobs=-1)
lr.fit(X_train_scaled,y_train)
ypred_lr = lr.predict(X_test_scaled)
metrics.mean_absolute_error(y_test,ypred_lr)
lasso = Lasso()
lasso.fit(X_train_scaled,y_train)
ypred_lasso = lasso.predict(X_test_scaled)
metrics.mean_absolute_error(y_test,ypred_lasso)
metrics.r2_score(y_test,ypred_lr)
rfr_.fit(X_train,y_train)
ypred_rfr_ = rfr_.predict(X_test)
metrics.r2_score(y_test,ypred_rfr_) #best model
from xgboost import XGBRegressor

reagressor=XGBRegressor()

reagressor.fit(X_train,y_train)

Y_pred_xgb=reagressor.predict(X_test)

from sklearn.metrics import r2_score

r2_score(Y_pred_xgb,y_test)