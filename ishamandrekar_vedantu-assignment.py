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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

pd.set_option('display.max_columns', 100)
customer_features_df = pd.read_csv("/kaggle/input/customer_features.csv")

last_month_assortment_df = pd.read_csv("/kaggle/input/last_month_assortment.csv")

next_month_assortment_df = pd.read_csv("/kaggle/input/next_month_assortment.csv")

next_purchase_order_df = pd.read_csv("/kaggle/input/next_purchase_order.csv")

original_purchase_order_df = pd.read_csv("/kaggle/input/original_purchase_order.csv")

product_features_df = pd.read_csv("/kaggle/input/product_features.csv")
customer_features_df.head()
customer_features_df.info()
# Checking the percentage of missing values in this file

col_list = customer_features_df.columns



for col_name in customer_features_df.columns:

    missing_percent = round(100* ((customer_features_df[col_name].isnull()) | (customer_features_df[col_name].astype(str) == 'Select')).sum() /len(customer_features_df.index) , 2)

    print(col_name + " - " + str(missing_percent))
customer_features_df['Likes_Self_Help'] = 0

customer_features_df['Likes_Biography'] = 0

customer_features_df['Likes_History'] = 0

customer_features_df['Likes_Thriller'] = 0

customer_features_df['Likes_Sci_Fi'] = 0

customer_features_df['Likes_Romance'] = 0

customer_features_df['Likes_Pop_Psychology'] = 0

customer_features_df['Likes_Beach_Read'] = 0

customer_features_df['Likes_Drama'] = 0

customer_features_df['Likes_Classic'] = 0

customer_features_df['Likes_Pop_Sci'] = 0
customer_features_df['Likes_Self_Help'] = customer_features_df.apply(lambda row: 1 if 'Self-Help' in str(row.favorite_genres) else 0,axis=1)

customer_features_df['Likes_Biography'] = customer_features_df.apply(lambda row: 1 if 'Biography' in str(row.favorite_genres) else 0,axis=1)

customer_features_df['Likes_History'] = customer_features_df.apply(lambda row: 1 if 'History' in str(row.favorite_genres) else 0,axis=1)

customer_features_df['Likes_Thriller'] = customer_features_df.apply(lambda row: 1 if 'Thriller' in str(row.favorite_genres) else 0,axis=1)

customer_features_df['Likes_Sci_Fi'] = customer_features_df.apply(lambda row: 1 if 'Sci-Fi' in str(row.favorite_genres) else 0,axis=1)

customer_features_df['Likes_Romance'] = customer_features_df.apply(lambda row: 1 if 'Romance' in str(row.favorite_genres) else 0,axis=1)

customer_features_df['Likes_Pop_Psychology'] = customer_features_df.apply(lambda row: 1 if 'Pop-Psychology' in str(row.favorite_genres) else 0,axis=1)

customer_features_df['Likes_Beach_Read'] = customer_features_df.apply(lambda row: 1 if 'Beach-Read' in str(row.favorite_genres) else 0,axis=1)

customer_features_df['Likes_Drama'] = customer_features_df.apply(lambda row: 1 if 'Drama' in str(row.favorite_genres) else 0,axis=1)

customer_features_df['Likes_Classic'] = customer_features_df.apply(lambda row: 1 if 'Classic' in str(row.favorite_genres) else 0,axis=1)

customer_features_df['Likes_Pop_Sci'] = customer_features_df.apply(lambda row: 1 if 'Pop-Sci' in str(row.favorite_genres) else 0,axis=1)
customer_features_df.head()
customer_features_df.loc[customer_features_df.Likes_Self_Help==1].age_bucket.value_counts()
customer_features_df.loc[customer_features_df.Likes_Biography==1].age_bucket.value_counts()
customer_features_df.loc[customer_features_df.Likes_History==1].age_bucket.value_counts()
customer_features_df.loc[customer_features_df.Likes_Thriller==1].age_bucket.value_counts()
customer_features_df.loc[customer_features_df.Likes_Sci_Fi==1].age_bucket.value_counts()
customer_features_df.loc[customer_features_df.Likes_Romance==1].age_bucket.value_counts()
customer_features_df.loc[customer_features_df.Likes_Pop_Psychology==1].age_bucket.value_counts()
customer_features_df.loc[customer_features_df.Likes_Beach_Read==1].age_bucket.value_counts()
customer_features_df.loc[customer_features_df.Likes_Drama==1].age_bucket.value_counts()

customer_features_df.loc[customer_features_df.Likes_Classic==1].age_bucket.value_counts()

customer_features_df.loc[customer_features_df.Likes_Pop_Sci==1].age_bucket.value_counts()

original_purchase_order_df.head()
original_purchase_order_df.info()
original_purchase_order_df['total_cost_to_buy'] = original_purchase_order_df.cost_to_buy * original_purchase_order_df.quantity_purchased
original_purchase_order_df.head()
next_purchase_order_df.head()
next_purchase_order_df.info()
next_purchase_order_df['total_cost_to_buy'] = next_purchase_order_df.quantity_purchased * next_purchase_order_df.cost_to_buy
product_features_df.head()
product_features_df.info()
last_month_assortment_df.head()
last_month_assortment_df.info()
next_month_assortment_df.head()
next_month_assortment_df.info()
#Total Loan Amount

initial_loan_amount = round(original_purchase_order_df.total_cost_to_buy.sum(),2)

initial_loan_amount
total_spending_reqd_next_mnth_purchase = round(next_purchase_order_df.total_cost_to_buy.sum(),2)

total_spending_reqd_next_mnth_purchase
df_temp = pd.merge(last_month_assortment_df, original_purchase_order_df, how='inner', on='product_id')
df_temp['purchased_price'] = df_temp.apply(lambda row: row.retail_value if row.purchased else 0, axis=1)
df_temp['send_shipping_cost'] = 0.60
df_temp['return_shipping_cost'] = df_temp.purchased.apply(lambda x: 0.60 if not x else 0)
total_revenue_last_mnth_assortment = df_temp.purchased_price.sum() - (df_temp.send_shipping_cost + df_temp.return_shipping_cost).sum()

total_revenue_last_mnth_assortment
profit_earned_last_month = round( 116024.76 - 135546.42, 2)

profit_earned_last_month
loan_pending = round(135546.42 - 116024.76 , 2)

loan_pending

# Defining the map function

def binary_map(x):

    return x.map({True: 1, False: 0})
last_month_assortment_df.head()
customer_features_df.head()
last_month_assortment_df.customer_id.nunique()
next_month_assortment_df.customer_id.nunique()
last_month_assortment_df.product_id.nunique()
next_month_assortment_df.product_id.nunique()
len(product_features_df)
last_month_assortment_df.head()
next_month_assortment_df1 = next_month_assortment_df

next_month_assortment_df1['purchased'] = False

df_temp3 = pd.concat([last_month_assortment_df, next_month_assortment_df1], axis = 0)

df_temp3.purchased = df_temp3.purchased.apply(lambda x: 1 if x else 0)

df_temp3 = df_temp3.pivot_table(

    index='customer_id',

    columns='product_id',

    values='purchased'

).fillna(0)
len(df_temp3)
len(customer_features_df)
df_temp3 = pd.merge( df_temp3, customer_features_df,how='inner',on='customer_id')

df_temp3.drop('favorite_genres',axis=1,inplace=True)

df_temp3.is_returning_customer = df_temp3.is_returning_customer.apply(lambda x: 1 if x else 0)

dummy1 = pd.get_dummies(df_temp3[['age_bucket']], drop_first=True)

df_temp3 = pd.concat([df_temp3, dummy1], axis=1)

df_temp3.drop('age_bucket',axis=1,inplace=True)

df_temp3.head()
from sklearn.metrics.pairwise import pairwise_distances



# User Similarity Matrix

user_correlation = pairwise_distances(df_temp3, metric='euclidean')

user_correlation_df = pd.DataFrame(user_correlation)
user_correlation_df.head()
product_features_df.head()
product_features_df.product_id.nunique()
original_purchase_order_df.head()
original_purchase_order_df.product_id.nunique()
df_temp4 = pd.merge(product_features_df,original_purchase_order_df,how='inner',on='product_id')

df_temp4.drop(['quantity_purchased','cost_to_buy','total_cost_to_buy'],axis=1,inplace=True)

df_temp4.fiction = df_temp4.fiction.apply(lambda x: 1 if x else 0)

dummy1 = pd.get_dummies(df_temp4[['genre']], drop_first=True)

df_temp4 = pd.concat([df_temp4, dummy1], axis=1)

df_temp4.drop('genre',axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_temp4[['retail_value','length','difficulty']] = scaler.fit_transform(df_temp4[['retail_value','length','difficulty']])
df_temp4.head()
# User Similarity Matrix

item_correlation = pairwise_distances(df_temp4, metric='euclidean')
item_correlation_df = pd.DataFrame(item_correlation)
item_correlation_df.info()
item_correlation_df.head()
last_month_assortment_df.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

col_name = user_correlation_df.columns

user_correlation_df[col_name] = scaler.fit_transform(user_correlation_df[col_name])

user_correlation_df = user_correlation_df.apply(lambda x : 1 - x)
scaler = MinMaxScaler()

col_name = item_correlation_df.columns

item_correlation_df[col_name] = scaler.fit_transform(item_correlation_df[col_name])

item_correlation_df = item_correlation_df.apply(lambda x : 1 - x)
user_correlation_df.head()
item_correlation_df.head()
last_month_assortment_df.head()
#purchase_score_product - purchase score calculated based on other products purchased by same user, how similar are they to this product.

#purchase_score_user - purchase score calculated based on this product purchased other users, how similar are they to this user.

# The more score closer to 0, more probability of buying the product

last_month_assortment_df['purchase_score_product'] = 0

last_month_assortment_df['purchase_score_user'] = 0
user_correlation_df.head()
user_correlation_df.info()
def func_purchase_score(row,type):

    if type == 'product':

        product_lst = list(last_month_assortment_df.loc[(last_month_assortment_df.customer_id == row.customer_id) & (last_month_assortment_df.product_id!= row.product_id),'product_id'])

        index_lst = product_features_df.loc[product_features_df.product_id.isin(product_lst)].index

        idx = product_features_df.loc[product_features_df.product_id == row.product_id].index

        arr1 = np.array(item_correlation_df.iloc[idx,index_lst])

        arr2 = np.array(last_month_assortment_df.loc[(last_month_assortment_df.customer_id == row.customer_id) & (last_month_assortment_df.product_id!= row.product_id),'purchased'])

        #row.purchase_score_product = 

        return np.dot(arr1,arr2)[0]

    if type == 'user':

        user_lst = list(last_month_assortment_df.loc[(last_month_assortment_df.product_id == row.product_id) & (last_month_assortment_df.customer_id!= row.customer_id),'customer_id'])

        index_lst = customer_features_df.loc[customer_features_df.customer_id.isin(user_lst)].index

        idx = customer_features_df.loc[customer_features_df.customer_id == row.customer_id].index

        arr1 = np.array(user_correlation_df.iloc[idx,index_lst])

        arr2 = np.array(last_month_assortment_df.loc[(last_month_assortment_df.product_id == row.product_id) & (last_month_assortment_df.customer_id!= row.customer_id),'purchased'])

        #row.purchase_score_user = 

        return np.dot(arr1,arr2)[0]

purchase_score_product = last_month_assortment_df.apply(lambda row: func_purchase_score(row,'product'),axis=1)

#purchase_score_user = last_month_assortment_df.apply(lambda row: func_purchase_score(row,'user'),axis=1)

purchase_score_product.head()
purchase_score_user = last_month_assortment_df.apply(lambda row: func_purchase_score(row,'user'),axis=1)

purchase_score_user.head()
last_month_assortment_df['purchase_score_product'] = purchase_score_product

last_month_assortment_df['purchase_score_user'] = purchase_score_user
last_month_assortment_df['purchase_score_total'] = last_month_assortment_df.purchase_score_product + last_month_assortment_df.purchase_score_user

last_month_assortment_df.head()
# Plot the histogram of the Purchase Score

fig = plt.figure()

sns.distplot((last_month_assortment_df.purchase_score_total), bins = 20)

fig.suptitle('Purchase Score', fontsize = 20)                  # Plot heading 

plt.xlabel('Purchase Score', fontsize = 18)                         # X-label
# Plot the histogram of the Purchase Score Product

fig = plt.figure()

sns.distplot((last_month_assortment_df.purchase_score_product), bins = 20)

fig.suptitle('Purchase Score Product', fontsize = 20)                  # Plot heading 

plt.xlabel('Purchase Score Product', fontsize = 18)                         # X-label
# Plot the histogram of the Purchase Score User

fig = plt.figure()

sns.distplot((last_month_assortment_df.purchase_score_user), bins = 20)

fig.suptitle('Purchase Score User', fontsize = 20)                  # Plot heading 

plt.xlabel('Purchase Score User', fontsize = 18)     
sum(last_month_assortment_df.purchase_score_user.isnull())
sum(last_month_assortment_df.purchase_score_user==0)
sns.boxplot(x='purchased', y='purchase_score_total',data=last_month_assortment_df)
sns.boxplot(x='purchased', y='purchase_score_product',data=last_month_assortment_df)
sns.boxplot(x='purchased', y='purchase_score_user',data=last_month_assortment_df)
df_temp = pd.merge(df_temp, customer_features_df, how='inner', on='customer_id')

df_temp = pd.merge(df_temp, product_features_df, how='inner', on='product_id')

df_temp.head()
df_temp.drop(['quantity_purchased','cost_to_buy','total_cost_to_buy','send_shipping_cost','return_shipping_cost','purchased_price','favorite_genres'],axis=1, inplace=True)
len(df_temp)
df_temp.customer_id.nunique()
#### Each of the 7200 customers have been sent 5 books thus

7200 * 5
df_temp['purchase_score_user'] = last_month_assortment_df.purchase_score_user
df_temp.info()
#Dropping Customer rows with null age brackets below % data

(36000 - 34435)*100/36000
df_temp = df_temp[~df_temp.age_bucket.isnull()]
df_temp.drop(['customer_id','product_id'], inplace=True,axis=1)
df_temp.head()
# Defining the map function

def binary_map(x):

    return x.map({True: 1, False: 0})
varlist =  ['purchased', 'is_returning_customer', 'fiction']



df_temp[varlist] = df_temp[varlist].apply(binary_map)
dummy1 = pd.get_dummies(df_temp[['age_bucket', 'genre']], drop_first=True)



# Adding the results to the master dataframe

df_temp = pd.concat([df_temp, dummy1], axis=1)
df_temp.drop(['age_bucket', 'genre'],axis=1,inplace=True)
# Let's see the correlation matrix 

plt.figure(figsize = (100,100))        # Size of the figure

sns.set(font_scale=4)

corrmat = df_temp.corr()

sns.heatmap(df_temp.corr())

plt.show()
# Putting feature variable to X

X = df_temp.drop(['purchased'], axis=1)

y = df_temp['purchased']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#X_train[['retail_value','length','difficulty','purchase_score_user']] = scaler.fit_transform(X_train[['retail_value','length','difficulty','purchase_score_user']])

X[['retail_value','length','difficulty','purchase_score_user']] = scaler.fit_transform(X[['retail_value','length','difficulty','purchase_score_user']])



#X_train.head()

X.head()
# Importing random forest classifier from sklearn library

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(bootstrap=True,

                             max_depth=12,

                             n_estimators=195,

                             max_features=8,

                             min_samples_leaf=22,

                             min_samples_split=10)
# fit

rfc.fit(X,y)
len(next_month_assortment_df)
next_month_assortment_df.head()
X.head()
purchase_score_user1 = next_month_assortment_df.apply(lambda row: func_purchase_score(row,'user'),axis=1)
sum(purchase_score_user1.isnull())
sum(purchase_score_user1==0)
# Plot the histogram of the Purchase Score

fig = plt.figure()

sns.distplot((purchase_score_user1), bins = 20)

fig.suptitle('Purchase Score', fontsize = 20)                  # Plot heading 

plt.xlabel('Purchase Score', fontsize = 18)                         # X-label
next_month_assortment_df.purchased = next_month_assortment_df.purchased.apply(lambda x: 1 if x else 0)

#next_month_assortment_df['purchase_score_user'] = purchase_score_user1

next_month_assortment_df.head()

df_test_temp = pd.merge(next_month_assortment_df, original_purchase_order_df, how='inner', on='product_id')

df_test_temp = pd.merge(df_test_temp, customer_features_df, how='inner', on='customer_id')

df_test_temp = pd.merge(df_test_temp, product_features_df, how='inner', on='product_id')

df_test_temp.drop(['quantity_purchased','cost_to_buy','total_cost_to_buy','favorite_genres'],axis=1, inplace=True)

df_test_temp['purchase_score_user'] = purchase_score_user1

varlist =  ['purchased', 'is_returning_customer', 'fiction']

df_test_temp[varlist] = df_test_temp[varlist].apply(binary_map)

dummy1 = pd.get_dummies(df_test_temp[['age_bucket', 'genre']], drop_first=True)

# Adding the results to the master dataframe

df_test_temp = pd.concat([df_test_temp, dummy1], axis=1)

df_test_temp.drop(['age_bucket', 'genre','customer_id','product_id'],axis=1,inplace=True)

# Putting feature variable to X

X_test = df_test_temp.drop(['purchased'], axis=1)

y_test = df_test_temp['purchased']

X_test[['retail_value','length','difficulty','purchase_score_user']] = scaler.transform(X_test[['retail_value','length','difficulty','purchase_score_user']])

X_test.head()
X.head()
y_test = rfc.predict(X_test)
next_month_assortment_df['purchased_predicted'] = y_test
next_month_assortment_df
df_pred = pd.merge(next_month_assortment_df, original_purchase_order_df, how='inner', on='product_id')

df_pred['purchased_price'] = df_pred.apply(lambda row: row.retail_value if row.purchased else 0, axis=1)

df_pred['send_shipping_cost'] = 0.60

df_pred['return_shipping_cost'] = df_pred.purchased.apply(lambda x: 0.60 if not x else 0)
predicted_revenue_next_assortment = df_pred.purchased_price.sum() - (df_pred.send_shipping_cost + df_pred.return_shipping_cost).sum()

predicted_revenue_next_assortment
if predicted_revenue_next_assortment - loan_pending >= total_spending_reqd_next_mnth_purchase:

    print('Yes we will be able to pay back our loan and afford our next book purchase order')

else:

    print('No we will not be able to pay back our loan and afford our next book purchase order')

    

print('Initial Loan Amount: '+ str(initial_loan_amount))

print('Total revenue Last Month Assortment: '+ str(total_revenue_last_mnth_assortment))

print('Profit Earned Last Month Assortment(): '+ str(profit_earned_last_month))

print('Loan Pending: '+ str(loan_pending))

print('Predicted revenue Next Month Assortment: '+str(predicted_revenue_next_assortment))

print('Total Spending required for Next Month Purchase: '+str(total_spending_reqd_next_mnth_purchase))
import sys

sys.stdout.write('***************Final Results************* \n')

if predicted_revenue_next_assortment - loan_pending >= total_spending_reqd_next_mnth_purchase:

    os.system('echo Yes we will be able to pay back our loan and afford our next book purchase order \n')

else:

    os.system('echo No we will not be able to pay back our loan and afford our next book purchase order \n')

    

os.system('echo Initial Loan Amount: '+ str(initial_loan_amount))

os.system('echo Total revenue Last Month Assortment: '+ str(total_revenue_last_mnth_assortment))

os.system('echo Profit Earned Last Month Assortment(): '+ str(profit_earned_last_month))

os.system('echo Loan Pending: '+ str(loan_pending))

os.system('echo Predicted revenue Next Month Assortment: '+str(predicted_revenue_next_assortment))

os.system('echo Total Spending required for Next Month Purchase: '+str(total_spending_reqd_next_mnth_purchase))