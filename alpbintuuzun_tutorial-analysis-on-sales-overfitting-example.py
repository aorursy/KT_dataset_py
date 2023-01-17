# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

import matplotlib

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")

data.head()
categoryData = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv")

categoryData['count'].head(100).sum()/categoryData['count'].sum()
importantTags = categoryData.keyword.head(100).str.lower().tolist()
print("There are ",len(data.columns), " columns in this dataset.\nColumns:\n",data.columns)
importantTags = categoryData.keyword.head(20).str.lower().tolist()
selectedFeatures = data.drop(columns=['title','title_orig'])
pid_dt = selectedFeatures['product_id']

mid_dt = selectedFeatures['merchant_id']

print(len(pid_dt), len(mid_dt))

selectedFeatures=selectedFeatures.drop(columns=['merchant_id','merchant_title',

                                                'merchant_name','merchant_profile_picture',

                                                'merchant_info_subtitle','product_id','product_url',

                                                'product_picture','shipping_option_name','urgency_text'])

selectedFeatures.head()
rptp = selectedFeatures[['price','retail_price']]

selectedFeatures['ret_to_price_ratio'] = rptp['retail_price']/rptp['price']

selectedFeatures[['ret_to_price_ratio','units_sold']].sort_values(by=['units_sold'],ascending = False)
rtrc_columns = selectedFeatures[['rating','rating_count','units_sold']]

rtrc = rtrc_columns['rating']/(rtrc_columns['rating_count']+1) # +1 is in order to deflect infinity

#These two lines will cause overfitting!!!!!

selectedFeatures['sales_to_rating'] = rtrc_columns['units_sold']/rtrc_columns['rating'] #Overfitting 1

selectedFeatures['sales_to_rating_count'] = rtrc_columns['units_sold']/(rtrc_columns['rating_count']+1) #Overfitting 2

#Change the two lines marked as "Overfitting"!!!!!

selectedFeatures['rating_to_rating_count'] = rtrc

matplotlib.pyplot.scatter(rtrc_columns['units_sold'],rtrc)
print(selectedFeatures['currency_buyer'].unique())

selectedFeatures = selectedFeatures.drop(columns=['currency_buyer'])
ad_boost_success = selectedFeatures[['uses_ad_boosts','units_sold']]

ad_boost_success = ad_boost_success.groupby(['uses_ad_boosts']).mean().sort_values(by=['units_sold'], ascending = False)

ad_boost_success
conditions = [(selectedFeatures['rating']<2),

              ((selectedFeatures['rating']>=2) & (selectedFeatures['rating']<3)),

              ((selectedFeatures['rating']>=3) & (selectedFeatures['rating']<4)),

              ((selectedFeatures['rating']>=4) & (selectedFeatures['rating']<=5))]

tags = ['tag_1','tag_2','tag_3','tag_4']

selectedFeatures = selectedFeatures.assign(categorical_rating = np.select(conditions,tags))

#selectedFeatures['categorical_rating'] = round(selectedFeatures['rating'],1)



#These three lanes are for generating the plot and will be repeated frequently

rating_to_sales = selectedFeatures[['units_sold','categorical_rating']]

rating_to_sales = rating_to_sales.groupby(['categorical_rating']).mean().sort_values(by=['categorical_rating'])

rating_to_sales.plot()
print("Min: ", min(selectedFeatures['rating_count']),"\nMax: ", max(selectedFeatures['rating_count']),"\nMean: ",selectedFeatures['rating_count'].mean())
low = selectedFeatures['rating_count'].quantile(0.3)

mid = selectedFeatures['rating_count'].quantile(0.6)

high = selectedFeatures['rating_count'].quantile(0.9)



conditions = [(selectedFeatures['rating_count']<low),

              ((selectedFeatures['rating_count']>=low) & (selectedFeatures['rating_count']<mid)),

              ((selectedFeatures['rating_count']>=mid) & (selectedFeatures['rating_count']<high)),

              (selectedFeatures['rating_count']>=high)

             ]

tags = ['tag0_low','tag2_mid','tag4_high','tag5_extreme']

selectedFeatures = selectedFeatures.assign(categorical_rating_count = np.select(conditions,tags))

rating_count_to_sales = selectedFeatures[['units_sold','categorical_rating_count']]

rating_count_to_sales = rating_count_to_sales.groupby(['categorical_rating_count']).mean().sort_values(by=['categorical_rating_count'])

rating_count_to_sales.plot()
low =selectedFeatures['rating_five_count'].quantile(0.3)

mid = selectedFeatures['rating_five_count'].quantile(0.6)

high = selectedFeatures['rating_five_count'].quantile(0.9)



conditions = [(selectedFeatures['rating_five_count']<low),

              ((selectedFeatures['rating_five_count']>=low) & (selectedFeatures['rating_five_count']<mid)),

              ((selectedFeatures['rating_five_count']>=mid) & (selectedFeatures['rating_five_count']<high)),

              (selectedFeatures['rating_five_count']>=high)

             ]

tags = ['tag0_low','tag2_mid','tag4_high','tag5_extreme']

selectedFeatures = selectedFeatures.assign(categorical_rating_five_count = np.select(conditions,tags))

rating5_count_to_sales = selectedFeatures[['units_sold','categorical_rating_five_count']]

rating5_count_to_sales = rating5_count_to_sales.groupby(['categorical_rating_five_count']).mean().sort_values(by=['categorical_rating_five_count'])

rating5_count_to_sales.plot()
low =selectedFeatures['rating_four_count'].quantile(0.3)

mid = selectedFeatures['rating_four_count'].quantile(0.6)

high = selectedFeatures['rating_four_count'].quantile(0.9)



conditions = [(selectedFeatures['rating_four_count']<low),

              ((selectedFeatures['rating_four_count']>=low) & (selectedFeatures['rating_four_count']<mid)),

              ((selectedFeatures['rating_four_count']>=mid) & (selectedFeatures['rating_four_count']<high)),

              (selectedFeatures['rating_four_count']>=high)

             ]

tags = ['tag0_low','tag2_mid','tag4_high','tag5_extreme']

selectedFeatures = selectedFeatures.assign(categorical_rating_four_count = np.select(conditions,tags))



rating4_count_to_sales = selectedFeatures[['units_sold','categorical_rating_four_count']]

rating4_count_to_sales = rating4_count_to_sales.groupby(['categorical_rating_four_count']).mean().sort_values(by=['categorical_rating_four_count'])

rating4_count_to_sales.plot()
low =selectedFeatures['rating_three_count'].quantile(0.3)

mid = selectedFeatures['rating_three_count'].quantile(0.6)

high = selectedFeatures['rating_three_count'].quantile(0.9)



conditions = [(selectedFeatures['rating_three_count']<low),

              ((selectedFeatures['rating_three_count']>=low) & (selectedFeatures['rating_three_count']<mid)),

              ((selectedFeatures['rating_three_count']>=mid) & (selectedFeatures['rating_three_count']<high)),

              (selectedFeatures['rating_three_count']>=high)

             ]

tags = ['tag0_low','tag2_mid','tag4_high','tag5_extreme']

selectedFeatures = selectedFeatures.assign(categorical_rating_three_count = np.select(conditions,tags))

rating3_count_to_sales = selectedFeatures[['units_sold','categorical_rating_three_count']]

rating3_count_to_sales = rating3_count_to_sales.groupby(['categorical_rating_three_count']).mean().sort_values(by=['categorical_rating_three_count'])

rating3_count_to_sales.plot()
low =selectedFeatures['rating_two_count'].quantile(0.3)

mid = selectedFeatures['rating_two_count'].quantile(0.6)

high = selectedFeatures['rating_two_count'].quantile(0.9)



conditions = [(selectedFeatures['rating_two_count']<low),

              ((selectedFeatures['rating_two_count']>=low) & (selectedFeatures['rating_two_count']<mid)),

              ((selectedFeatures['rating_two_count']>=mid) & (selectedFeatures['rating_two_count']<high)),

              (selectedFeatures['rating_two_count']>=high)

             ]

tags = ['tag0_low','tag2_mid','tag4_high','tag5_extreme']

selectedFeatures = selectedFeatures.assign(categorical_rating_two_count = np.select(conditions,tags))

rating2_count_to_sales = selectedFeatures[['units_sold','categorical_rating_two_count']]

rating2_count_to_sales = rating2_count_to_sales.groupby(['categorical_rating_two_count']).mean().sort_values(by=['categorical_rating_two_count'])

rating2_count_to_sales.plot()
low =selectedFeatures['rating_one_count'].quantile(0.3)

mid = selectedFeatures['rating_one_count'].quantile(0.6)

high = selectedFeatures['rating_one_count'].quantile(0.9)



conditions = [(selectedFeatures['rating_one_count']<low),

              ((selectedFeatures['rating_one_count']>=low) & (selectedFeatures['rating_one_count']<mid)),

              ((selectedFeatures['rating_one_count']>=mid) & (selectedFeatures['rating_one_count']<high)),

              (selectedFeatures['rating_one_count']>=high)

             ]

tags = ['tag0_low','tag2_mid','tag4_high','tag5_extreme']

selectedFeatures = selectedFeatures.assign(categorical_rating_one_count = np.select(conditions,tags))

rating1_count_to_sales = selectedFeatures[['units_sold','categorical_rating_one_count']]

rating1_count_to_sales = rating1_count_to_sales.groupby(['categorical_rating_one_count']).mean().sort_values(by=['categorical_rating_one_count'])

rating1_count_to_sales.plot()
selectedFeatures = selectedFeatures.drop(columns=['rating','rating_count','rating_five_count','rating_four_count',

                                                  'rating_three_count','rating_two_count','rating_one_count'])
conditions = [(selectedFeatures['merchant_rating']<2),

              ((selectedFeatures['merchant_rating']>=2) & (selectedFeatures['merchant_rating']<3)),

              ((selectedFeatures['merchant_rating']>=3) & (selectedFeatures['merchant_rating']<4)),

              ((selectedFeatures['merchant_rating']>=4) & (selectedFeatures['merchant_rating']<=5))]

tags = ['tag_1','tag_2','tag_3','tag_4']

#selectedFeatures['categorical_merchant_rating'] = round(selectedFeatures['merchant_rating'],1)

selectedFeatures = selectedFeatures.assign(categorical_merchant_rating = np.select(conditions,tags))





m_rating_to_sales = selectedFeatures[['units_sold','categorical_merchant_rating']]

m_rating_to_sales = m_rating_to_sales.groupby(['categorical_merchant_rating']).mean().sort_values(by=['categorical_merchant_rating'])

m_rating_to_sales.plot()
low = selectedFeatures['merchant_rating_count'].quantile(0.3)

mid = selectedFeatures['merchant_rating_count'].quantile(0.6)

high = selectedFeatures['merchant_rating_count'].quantile(0.9)



conditions = [(selectedFeatures['merchant_rating_count']<low),

              ((selectedFeatures['merchant_rating_count']>=low) & (selectedFeatures['merchant_rating_count']<mid)),

              ((selectedFeatures['merchant_rating_count']>=mid) & (selectedFeatures['merchant_rating_count']<high)),

              (selectedFeatures['merchant_rating_count']>=high)

             ]

selectedFeatures = selectedFeatures.assign(categorical_merchant_rating_count = np.select(conditions,tags))



m_rating_count_to_sales = selectedFeatures[['units_sold','categorical_merchant_rating_count']]

m_rating_count_to_sales = m_rating_count_to_sales.groupby(['categorical_merchant_rating_count']).mean().sort_values(by=['categorical_merchant_rating_count'])

m_rating_count_to_sales.plot()
selectedFeatures = selectedFeatures.drop(columns=['merchant_rating_count','merchant_rating'])

selectedFeatures
origin_country_success = selectedFeatures[['origin_country','units_sold']]

origin_country_success = origin_country_success.groupby(['origin_country']).mean().sort_values(by=['units_sold'], ascending = False)

selectedFeatures = selectedFeatures.drop(columns=['origin_country'])

origin_country_success.plot.pie(subplots=True, figsize=(10,10))
themes = selectedFeatures['theme']

print(themes.unique())

selectedFeatures = selectedFeatures.drop(columns=['theme'])
print(selectedFeatures.duplicated().sum())

selectedFeatures.drop_duplicates(inplace=True)

selectedFeatures = selectedFeatures.reset_index(drop=True)

selectedFeatures.info()
selectedFeatures = selectedFeatures.drop(columns=['product_variation_size_id','product_color'])



encoder = LabelEncoder() #From sklearn.preprocessing library



selectedFeatures['categorical_merchant_rating_count'] = encoder.fit_transform(selectedFeatures['categorical_merchant_rating_count'])

selectedFeatures['categorical_merchant_rating'] = encoder.fit_transform(selectedFeatures['categorical_merchant_rating'])



selectedFeatures['categorical_rating_five_count'] = encoder.fit_transform(selectedFeatures['categorical_rating_five_count'])

selectedFeatures['categorical_rating_four_count'] = encoder.fit_transform(selectedFeatures['categorical_rating_four_count'])

selectedFeatures['categorical_rating_three_count'] = encoder.fit_transform(selectedFeatures['categorical_rating_three_count'])

selectedFeatures['categorical_rating_two_count'] = encoder.fit_transform(selectedFeatures['categorical_rating_two_count'])

selectedFeatures['categorical_rating_one_count'] = encoder.fit_transform(selectedFeatures['categorical_rating_one_count'])



selectedFeatures['categorical_rating_count'] = encoder.fit_transform(selectedFeatures['categorical_rating_count'])

selectedFeatures['categorical_rating'] = encoder.fit_transform(selectedFeatures['categorical_rating'])

selectedFeatures['crawl_month'] = encoder.fit_transform(selectedFeatures['crawl_month'])
selectedFeatures['ret_to_price_ratio'] = np.float32(selectedFeatures['ret_to_price_ratio'])

selectedFeatures['sales_to_rating'] = np.float32(selectedFeatures['sales_to_rating'])

selectedFeatures['price'] = np.float32(selectedFeatures['price'])
selectedFeatures.isnull().sum()


selectedFeatures = selectedFeatures.assign(has_urgency = (data['has_urgency_banner']==1.0).astype(int))

selectedFeatures = selectedFeatures.drop(columns=['has_urgency_banner','crawl_month'])
selectedFeatures.T
weightedTags = []

for i in range(len(selectedFeatures['tags'])):

    count = 0

    weight = 100

    for j in range(len(importantTags)):

        if importantTags[j] in selectedFeatures['tags'][i]:

            count+=weight

        weight-=5

    weightedTags.append(count)

    

df = pd.DataFrame({'weightedTags':weightedTags})

selectedFeatures['weightedTags'] = df['weightedTags']

selectedFeatures = selectedFeatures.drop(columns=['tags'])



weighted_tags_to_sales = selectedFeatures[['units_sold','weightedTags']]

weighted_tags_to_sales = weighted_tags_to_sales.groupby(['weightedTags']).mean().sort_values(by=['weightedTags'])

weighted_tags_to_sales.plot()
selectedFeatures.info()
features = selectedFeatures.drop(columns=['units_sold'])

sales = selectedFeatures['units_sold']



feature_train,feature_test,sale_train,sale_test=train_test_split(features,sales,test_size=0.2,random_state=0)
regressor=RandomForestRegressor(n_estimators=10000)

regressor.fit(feature_train,sale_train)
sale_pred=regressor.predict(feature_test)

print("Prediction score: ", r2_score(sale_pred,sale_test))