import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_rows', 150)

pd.set_option('display.max_columns', 150)
product = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')

uni_cat = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.csv')

uni_cat_sort = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv')
product.head()
product.shape
product.describe()
product.info()
product.isnull().sum()
product.head(2)
col_drop = ['title','title_orig','currency_buyer','rating_five_count', 'rating_four_count','rating_three_count',

            'rating_two_count', 'rating_one_count','has_urgency_banner', 'urgency_text', 'merchant_id',

            'merchant_has_profile_picture','merchant_profile_picture', 'product_url','product_picture','product_id',

            'theme', 'crawl_month']



product.drop(columns = col_drop, axis = 1, inplace = True)



product.isnull().sum()
100*product.isnull().sum()/product.shape[0]
product.product_color.fillna(product.product_color.mode()[0], inplace = True)
product.origin_country.fillna(product.origin_country.mode()[0], inplace = True)
product.head()
plt.figure(figsize=(20,6))

plt.subplot(1,2,1)

sns.boxplot(data=product, x='price')

plt.subplot(1,2,2)

sns.distplot(product.price, bins=15)

plt.show()
plt.figure(figsize=(20,6))

plt.subplot(1,2,1)

sns.boxplot(data=product, x='retail_price')

plt.subplot(1,2,2)

sns.distplot(product.retail_price, bins=15)

plt.show()
product['Profit_Loss'] = product.retail_price - product.price
product['revenue'] = product.Profit_Loss*product.units_sold
maxrevenue = product.sort_values('revenue', ascending = False).head(10)

maxrevenue
bottom10loss = product.sort_values('revenue').head(10)

bottom10loss
top10rating = product.sort_values('rating', ascending = False).head(10)

top10rating
bottom10rating = product.sort_values('rating').head(10)

bottom10rating
df_profit = product[product.revenue>=0].sort_values('revenue', ascending = False)

df_loss = product[product.revenue<0].sort_values('revenue', ascending = False)
df_profit.head()
df_loss.sort_values('Profit_Loss')
print(df_profit.shape)

print(df_loss.shape)
loss_rating = df_loss.sort_values('rating', ascending = False).head(20)
loss_rating.sort_values('revenue').head()
df_profit.revenue.describe()
df_profit.revenue
bins = [-1,100.00,1000,10000,5000000]

label = ['Low', 'Medium', 'High', 'Very_High']



df_profit['revenue_cat'] = pd.cut(df_profit.revenue, bins = bins, labels = label)
sns.boxplot(data=df_profit, x='revenue_cat', y='rating')

plt.show()
sns.boxplot(data=df_profit, x='revenue_cat', y='units_sold')

plt.show()
sns.boxplot(data=product, x='uses_ad_boosts', y='units_sold')

plt.show()
df1 = product[['units_sold', 'uses_ad_boosts']][product.units_sold<20000]
df1.uses_ad_boosts.value_counts()
sns.boxplot(data=df1, x='uses_ad_boosts', y='units_sold')

plt.show()
sns.distplot(product['rating'])

plt.show()
product.columns
corr_col = product[['price', 'retail_price', 'units_sold', 'uses_ad_boosts', 'rating',

       'rating_count','badges_count','badge_local_product', 'badge_product_quality',

       'badge_fast_shipping','product_variation_inventory',

       'shipping_option_price', 'shipping_is_express', 'countries_shipped_to',

       'inventory_total', 'merchant_rating_count',

       'merchant_rating']]
var = product[['price', 'retail_price', 'units_sold', 'rating', 'merchant_rating']]
sns.pairplot(var)

plt.show()
sns.scatterplot(data=product, x='rating',y='rating_count')

plt.show()
sns.scatterplot(data=product, x='rating',y='merchant_rating')

plt.show()
plt.figure(figsize=(20,10))

sns.heatmap(corr_col.corr(), annot= True, cmap='YlGnBu')

plt.show()
corr_col.drop(columns=['rating_count', 'badge_product_quality'], axis=1, inplace = True)
plt.figure(figsize=(20,10))

sns.heatmap(corr_col.corr(), annot= True, cmap='YlGnBu')

plt.show()
final_vars = product[['price','shipping_option_price','retail_price', 'product_variation_inventory']]
X = final_vars[['shipping_option_price','retail_price', 'product_variation_inventory']]

y = final_vars['price']
from sklearn.model_selection import train_test_split



X_train , X_test , y_train, y_test = train_test_split(X , y , train_size = 0.7, test_size = 0.3, random_state = 100)
import statsmodels.api as sm



X_train_sm = sm.add_constant(X_train)



lr = sm.OLS(y_train, X_train_sm).fit()



lr.params
print(lr.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_test_sm = sm.add_constant(X_test)



y_test_pred = lr.predict(X_test_sm)
y_train_price = lr.predict(X_train_sm)
fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)  
print(lr.summary())