import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
raw_data = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')#

raw_data.head()
data=raw_data.copy()
drop_col=['merchant_title','merchant_name','merchant_info_subtitle','merchant_rating_count','merchant_rating','merchant_has_profile_picture',

         'merchant_profile_picture','product_url','product_picture','crawl_month','title','title_orig']
def graphs(column,data,column_to_exclude):

    if column not in column_to_exclude:

        if data[column].dtype not in['int64','float64']:

            f,ax=plt.subplots(1,1,figsize=(7,7))

            sns.countplot(x=column,data = data)

            plt.xticks(rotation=90)

            plt.suptitle(column)

            plt.show()

        else:

            g=sns.FacetGrid(data,margin_titles=True,aspect=4,height=4)

            g.map(plt.hist,column,bins=100)

            plt.show()

        plt.show()

        

#for column in raw_data.columns:

    graphs(column,raw_data,drop_col);
data=data.drop(data[drop_col],axis=1)
data.head()
data=data.drop('urgency_text',axis=1)

data=data.drop(['currency_buyer','merchant_id','theme'],axis=1)
data=data.drop(['shipping_option_name','shipping_is_express'],axis=1)

data['origin_country']=data['origin_country'].fillna('CN')

data.isna().sum()
data['has_urgency_banner']=data['has_urgency_banner'].fillna(data['has_urgency_banner'].mean())
for col in data.columns:

    print(col,':',len(data[col].unique()),'label')
data['product_variation_size_id'].value_counts()
data['product_variation_size_id']=data['product_variation_size_id'].fillna('S')

data['product_color'].value_counts()
data['product_color']=data['product_color'].fillna('black')
plt.figure(figsize=(6,6))

sns.boxenplot(data['rating_five_count']);
data[data['rating_five_count'].isna()==True][['rating','rating_count','rating_five_count','rating_four_count',

                                              'rating_three_count','rating_two_count','rating_one_count'

                                             ]].head(12)
def is_successful(col):

    if col>1000:

        return 1

    else:

        return 0

data['success']=data['units_sold'].apply(is_successful)

print('percent of successfull product is:',data['success'].value_counts()[1]/len(data['success'])*100)

sns.countplot(data=data,x='success');

plt.show()
print('///////////////////////////////////////////')

print('overall status:')

print('For price:',data['price'].mean())

print('Retail_price:',data['retail_price'].mean())

print('---------------------------------------')

print('Status for Successful product:')

print('For price:',data[data['success']==1]['price'].mean())

print('For retail_price:',data[data['success']==1]['retail_price'].mean())

print('////////////////////////////////////////////////////')

print('NO CONCLUSION')
data['rating_five_count']=data['rating_five_count'].fillna(0)

data['rating_four_count']=data['rating_four_count'].fillna(0)

data['rating_three_count']=data['rating_three_count'].fillna(0)

data['rating_two_count']=data['rating_two_count'].fillna(0)

data['rating_one_count']=data['rating_one_count'].fillna(0)
sns.boxenplot(data['countries_shipped_to'],data['origin_country']);
print('Mean of Success by Ratings:-')

print('for five=',data[data['success']==1]['rating_five_count'].mean())

print('for four=',data[data['success']==1]['rating_four_count'].mean())

print('for three=',data[data['success']==1]['rating_three_count'].mean())

print('for two=',data[data['success']==1]['rating_two_count'].mean())

print('for one=',data[data['success']==1]['rating_one_count'].mean())
print('Mean of unsuccess by ratings:-')

print('for five=',data[data['success']==0]['rating_five_count'].mean())

print('for four=',data[data['success']==0]['rating_four_count'].mean())

print('for three=',data[data['success']==0]['rating_three_count'].mean())

print('for two=',data[data['success']==0]['rating_two_count'].mean())

print('for one=',data[data['success']==0]['rating_one_count'].mean())
fig,((ax1,ax2))=plt.subplots(1,2,figsize=(6,6),sharey=True)

ax1=sns.boxplot(data['success'],data['rating_five_count'])





ax2=sns.barplot(data['success'],data['units_sold'])

fig.suptitle('BEST SUCCESS AND FALIURE GRAPH(UNITS_SOLD &RATING)');

#Creating a column of Top Products:-

data['top_product']=data[data['success']==1]['product_id']

x=pd.DataFrame(raw_data['merchant_rating'],data['success'])

print('the Effect of merchant_rating v/s success is:-')



x.groupby(['success']).mean()

sns.distplot(x);
y=pd.DataFrame(data['tags'],data['success'])

z=y.groupby(['success'])

z.describe(include='all')