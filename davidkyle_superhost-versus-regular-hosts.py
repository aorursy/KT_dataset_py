# A quick Analysis of Airbnb Data from Seattle 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


'''See https://medium.com/@dkylemiller/the-number-one-thing-that-you-should-know-if-you-want-to-make-more-money-on-airbnb-6605070a4dca'''
'''for more commentary'''
data = pd.read_csv('../input/listings.csv')
data.info()
data.head()
data.columns
# data modifications by column

data['host_is_superhost'].replace('t','SuperHost',inplace = True) 
data['host_is_superhost'].replace('f','Regular Host',inplace =True)

data['price'] = data['price'].str.replace('$','')
data['price'] = data['price'].str.replace(',','').astype('float64')

data['weekly_price'] = data['weekly_price'].str.replace('$','')
data['weekly_price'] = data['weekly_price'].str.replace(',','').astype('float64')

data['monthly_price'] = data['monthly_price'].str.replace('$','')
data['monthly_price'] = data['monthly_price'].str.replace(',','').astype('float64')
data['cleaning_fee'] = data['cleaning_fee'].str.replace('$','')
data['cleaning_fee'] = data['cleaning_fee'].str.replace(',','').astype('float64')

data['host_response_rate'] = data['host_response_rate'].str.replace('%','').astype('float64')/100
data['host_acceptance_rate'] = data['host_acceptance_rate'].str.replace('%','').astype('float64')/100
data.info()
data.columns
super_data1 = data[['host_is_superhost', 'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'price', 'cleaning_fee', 'guests_included',
       'minimum_nights', 'maximum_nights','weekly_price', 'monthly_price', 'availability_365']]
sns.heatmap(super_data1.corr(method='kendall'))
super_data1.groupby('host_is_superhost',axis =0).mean()
host_table = pd.DataFrame(data= super_data1.groupby('host_is_superhost',axis =0).mean())

host_table.loc['Delta'] = host_table.loc['Regular Host'] - host_table.loc['SuperHost']

host_table.drop(['accommodates', 'guests_included', 'minimum_nights', 'maximum_nights'], axis = 1, inplace = True)
host_table.head()
host_table['host_total_listings_count'].plot(kind ='bar')
host_table[['price', 'cleaning_fee']].plot(kind ='bar')
super_data1.groupby(['bedrooms','host_is_superhost'],axis =0).mean()
super_data1.groupby(['bedrooms','host_is_superhost'],axis =0).count()['host_total_listings_count']
super_data2 = data[[ 'host_is_superhost','number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value',
       'reviews_per_month']]
super_data2.dropna().info()
super_data2.groupby('host_is_superhost',axis =0).mean()
review_table = pd.DataFrame(data = super_data2.groupby('host_is_superhost',axis =0).mean())

#review_table.drop(['review_scores_checkin', 'review_scores_value','review_scores_location' ,'review_scores_checkin', 'review_scores_accuracy'], axis =1)

review_table.head()

super_data2.groupby(['host_is_superhost','review_scores_rating'],axis =0).mean()
review_table = pd.DataFrame(data=super_data2.groupby(['host_is_superhost','review_scores_rating'],axis =0).mean())

reg_host_ratings = pd.DataFrame(data =list(review_table.index.get_level_values(1)[:42]), columns =['Regular Host Ratings'])
super_host_ratings =pd.DataFrame(data =list(review_table.index.get_level_values(1)[42:]), columns =['SuperHost Ratings'])
                        
#print(reg_host_ratings)
#print(super_host_ratings)

reg_host_number = pd.DataFrame(data =list(review_table['number_of_reviews'][:42]), columns =['Number of Reviews'])
super_host_number = pd.DataFrame(data= list(review_table['number_of_reviews'][42:]),columns =['Number of Reviews'])


review_table.index
reg_table = pd.concat([reg_host_ratings, reg_host_number], axis =1)

super_table = pd.concat([super_host_ratings, super_host_number], axis =1)

print(super_table)
#print(reg_table)

reg_host_ratings.hist()
super_host_ratings.hist()



super_data3 = data[['host_response_time', 'host_response_rate', 
                    'host_is_superhost','cancellation_policy','instant_bookable', 'host_acceptance_rate']]
super_data3 = pd.DataFrame(data=super_data3)
super_data3.head()
super_data3.info()
super_data3.dropna(subset=['host_acceptance_rate'], inplace=True)
super_data3.info()
# Number of Super Host 
super_data3['host_is_superhost'].value_counts()
# Approx percentage of Superhost 

728/(728+3045)
super_data3['cancellation_policy'].value_counts()
response_speed = pd.DataFrame(data = super_data3.groupby(['host_response_time', 'host_is_superhost'],axis =0).count())
response_speed['host_response_rate']
response_speed['host_response_rate'].plot(kind='bar')
cancel_policy = pd.DataFrame(data = super_data3.groupby(['cancellation_policy', 'host_is_superhost'],axis =0).count())
# Cancel policy counts

cancel_policy['host_response_rate']
cancel_policy['host_response_rate'].plot(kind ='bar')
instant_book = pd.DataFrame(data = super_data3.groupby(['instant_bookable', 'host_is_superhost'],axis =0).count())
instant_book['host_response_rate']
super_data3[:5]
super_data3[['host_is_superhost','host_response_rate','host_acceptance_rate']].groupby(['host_is_superhost'], axis =0).mean()
drop = ['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary',
       'space', 'description', 'experiences_offered', 'neighborhood_overview',
       'notes', 'transit', 'thumbnail_url', 'medium_url', 'picture_url',
       'xl_picture_url', 'host_id', 'host_url', 'host_name', 'host_since',
       'host_location', 'host_about','host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
       'street', 'neighbourhood', 'neighbourhood_cleansed',
       'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
       'smart_location', 'country_code', 'country', 'latitude', 'longitude',
       'amenities', 'square_feet', 'calendar_last_scraped', 'number_of_reviews',
       'first_review', 'last_review', 'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value', 'requires_license',
       'license', 'jurisdiction_names', 'reviews_per_month']
model_data = data.drop(labels=drop, axis =1)
model_data.info()
model_data['security_deposit'] = model_data['security_deposit'].str.replace('$', '')
model_data['security_deposit'] = model_data['security_deposit'].str.replace(',', '').astype('float64')

model_data['extra_people'] = model_data['extra_people'].str.replace('$', '')
model_data['extra_people'] = model_data['extra_people'].str.replace(',', '').astype('float64')
model_data.info()
model_data.dropna(subset =['bedrooms'], axis =0, inplace = True)

model_data.fillna(value =0, inplace=True)

model_data.drop(1297, inplace =True)
model_data.drop(1419, inplace =True)

model_data.info()
Y = model_data['host_is_superhost']

Y = Y.str.replace('Regular Host','0')
Y = Y.str.replace('SuperHost','1')
Y = Y.astype('float64')


X = pd.get_dummies(model_data.drop(labels=['host_is_superhost'], axis =1))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.15, random_state=1)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
model.score(X_test, y_test)
a = model.coef_.ravel()
b = X_train.columns

c = pd.DataFrame({'feature': b, 'coef': a})

c.sort_values(by = 'coef', ascending =False)[:25].plot(x='feature', y='coef', kind='bar')
features = c.sort_values(by = 'coef', ascending =False)[:12]

from IPython.display import HTML

HTML(features.to_html(index=False))
from sklearn.ensemble import RandomForestRegressor

model_forest = RandomForestRegressor(max_depth=20, random_state=0)
model_forest.fit(X_train, y_train)

print(model_forest.score(X_test, y_test))

model_forest_importances = model_forest.feature_importances_
model_forest_result = pd.DataFrame({'feature': X_train.columns, 'importance': model_forest_importances})
model_forest_result.sort_values(by='importance',ascending=False)[:10].plot(x='feature', y='importance', kind='bar')