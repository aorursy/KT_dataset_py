import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
sales=pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv',header=0)

sales.head()
sales['price'].plot(kind='hist',figsize=(10,10))

plt.xlabel('Value of price')

plt.ylabel('Frequency')
sales['retail_price'].plot(kind='hist',figsize=(10,10))

plt.xlabel('Retail price')

plt.ylabel('Frequency')
sales['units_sold'].plot(kind='hist',figsize=(10,10))

plt.xlabel('Units Sold')

plt.ylabel('Frequency')
sales['rating'].plot(kind='hist',figsize=(10,10))

plt.xlabel('Rating')

plt.ylabel('Frequency')
sales['rating_count'].plot(kind='hist',figsize=(10,10))

plt.xlabel('Rating Count')

plt.ylabel('Frequency')
sales['badges_count'].plot(kind='hist',figsize=(10,10))

plt.xlabel('Number of Badges')

plt.ylabel('Frequency')
sales['shipping_option_price'].plot(kind='hist',figsize=(10,10))

plt.xlabel('Shipping Cost')

plt.ylabel('Frequency')
sales.groupby('origin_country')['product_id'].count().sort_values(ascending=False).plot(kind='bar',figsize=(10,10),title='Country wise products count')
sales.info()
to_be_removed=['merchant_title','merchant_has_profile_picture','merchant_profile_picture','product_url','product_picture','crawl_month','has_urgency_banner','urgency_text']
sales.drop(to_be_removed,axis=1,inplace=True)

sales.info()
sales.describe()
sales['price_difference']=sales['retail_price']-sales['price']
sales.columns
sales['currency_buyer'].value_counts()
to_be_used=['price','retail_price','uses_ad_boosts','rating','rating_count','rating_five_count','rating_four_count','rating_three_count','rating_two_count','rating_one_count','badges_count','badge_local_product','badge_product_quality','badge_fast_shipping','shipping_option_price','shipping_is_express','countries_shipped_to','merchant_rating_count','units_sold']
f=plt.figure(figsize=(25,25))

plt.matshow(sales[to_be_used].corr(),fignum=f.number)

plt.xticks(range(sales[to_be_used].shape[1]), sales[to_be_used].columns, fontsize=14, rotation=45)

plt.yticks(range(sales[to_be_used].shape[1]), sales[to_be_used].columns, fontsize=14)

cb=plt.colorbar()

cb.ax.tick_params(labelsize=14)
l=[]



for i in to_be_used:

    if abs(sales['units_sold'].corr(sales[i])>=0.05):

        l.append(i)

l.remove('units_sold')

l
sales.describe(include='object')
sales['shipping_option_name'].value_counts()
sales['origin_country'].value_counts()
a=pd.get_dummies(sales['origin_country'])

a.info()
sales1=sales.merge(a,how='outer',left_index=True, right_index=True)

sales1.info()
to_be_used1=['AT','CN','GB','SG','US','VE','units_sold']

f1=plt.figure(figsize=(10,15))

plt.matshow(sales1[to_be_used1].corr(),fignum=f1.number)

plt.xticks(range(sales1[to_be_used1].shape[1]), sales1[to_be_used1].columns, fontsize=14, rotation=45)

plt.yticks(range(sales1[to_be_used1].shape[1]), sales1[to_be_used1].columns, fontsize=14)

cb=plt.colorbar()

cb.ax.tick_params(labelsize=14)
b=pd.get_dummies(sales['shipping_option_name'])

b
sales2=sales.merge(b,how='outer',left_index=True,right_index=True)

sales2.info()
to_be_used2=list(sales2.columns[36:51])
f2=plt.figure(figsize=(10,10))

plt.matshow(sales2[to_be_used2].corr(),fignum=f2.number)

plt.xticks(range(sales2[to_be_used2].shape[1]), sales2[to_be_used2].columns, fontsize=14, rotation=45)

plt.yticks(range(sales2[to_be_used2].shape[1]), sales2[to_be_used2].columns, fontsize=14)

cb=plt.colorbar()

cb.ax.tick_params(labelsize=14)
#to_be_removed2.remove('units_sold')

for i in to_be_used2:

    if abs(sales2['units_sold'].corr(sales2[i]))>=0.5:

        print(i)
sales['units_sold'].describe()
category=[]

for i in sales['units_sold']:

    if i>=0 and i<=100:

        category.append('very_cold')

    elif i>=101 and i<=1000:

        category.append('cold')

    elif i>=1001 and i<=5000:

        category.append('warm')

    elif i>=5001:

        category.append('hot')

len(category)        
sales['class']=category
sales.head()
l.append('units_sold')

l.append('class')
l=np.array(l)

l
sales_df=sales[l]

sales_df.head()
sales_df.info()
sales_df.dropna(inplace=True)
sales_df.info()
X=sales_df[['rating_count', 'rating_five_count', 'rating_four_count','rating_three_count', 'rating_two_count', 'rating_one_count','badge_product_quality', 'merchant_rating_count', 'units_sold']].values

X[0:5]
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X[0:5]
y = sales_df['class'].values

y[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
k = 4

#Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

neigh
yhat = neigh.predict(X_test)

yhat[0:5]
from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
Ks = 10

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1,Ks):

    

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    yhat=neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)



    

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])



mean_acc
plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Nabors (K)')

plt.tight_layout()

plt.show()
pwd
