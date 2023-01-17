# import relevant packages

import os

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as plticker

import seaborn as sns #unused

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error



#set up data path

#this is currently the path to access the data on kaggle, modify as needed

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#read in data

list_df=pd.read_csv('/kaggle/input/seattle/listings.csv')

review_df=pd.read_csv('/kaggle/input/seattle/reviews.csv')

date_df=pd.read_csv('/kaggle/input/seattle/calendar.csv')

print(list_df.shape)

list_df.head()
#what variables can we play with

list_df.columns
#quick summary of data

list_df.describe()
#get an understanding of categorical and numerical data columns

list_df.select_dtypes('number').columns
list_df.describe(include='object')
print(review_df.shape)

review_df.head()
review_df.describe(include='object')
print(date_df.shape)

date_df.head()
date_df.describe(include='object')
print('first date in the dataset is '+date_df['date'].min()+' and last date is '+date_df['date'].max())
#which columns are not missing data

len(list_df.columns[list_df.isnull().sum()==0])
#how much data is missing per column

list_df[list_df.columns[list_df.isnull().sum()>0]].isnull().mean().sort_values()
list_df['neighbourhood_cleansed'].value_counts().hist(bins=40)
list_df['neighbourhood_group_cleansed'].value_counts()
#some columns have strange formatting and hey need to be fixed

def pricing_reformat(df_col):

    return df_col.str.strip('$').str.replace(',','').astype(float)



list_df['price']=pricing_reformat(list_df['price'])

list_df['extra_people']=pricing_reformat(list_df['extra_people'])
review_df.isnull().sum()
# rows missing prices are not useful since I'm interested in pricing, so drop them

date_df2=date_df.dropna(subset=['price'])

date_df2.describe(include='object')
# as expected, all rows with prices are for when the listing is available, so the avalability column isn't very useful

date_df2=date_df2.drop(['available'],axis=1)

date_df2.head()
# I want to analyze pricing by month and by date, create relevant columns

y_md=date_df2['date'].str.split('-',expand=True,n=1)

date_df2=date_df2.assign(monthday=y_md[1], month=y_md[1].str.split('-',expand=True)[0].astype(int))

date_df2['price']=pricing_reformat(date_df2['price'])



date_df2.head()
#need to check there are enough data for each month and each date

date_df2['monthday'].hist(bins=365,xrot=90)
list_df=list_df.rename(index=str, columns={'id':'listing_id'})

model_df = pd.merge(date_df2, list_df, on = 'listing_id')

model_df.shape
### drop columns



#these cols have way too many missing numbers to be predictive

drop_cols=['cleaning_fee','neighborhood_overview','notes','weekly_price','security_deposit','monthly_price','square_feet','license']

model_df=model_df.drop(drop_cols, axis=1)



#to reduce dimensionality of the problem, I decided to drop the following columns

drop_cols = ['listing_id', 'monthday','date', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary',

       'space', 'description', 'experiences_offered', 'thumbnail_url', 'medium_url', 'picture_url',

       'xl_picture_url', 'host_id', 'host_url', 'host_name','host_since', 'host_response_rate',

       'host_location', 'host_about', 'host_response_time','city','state',

        'host_acceptance_rate', 'host_is_superhost',

       'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',

       'host_listings_count', 'host_total_listings_count',

       'host_verifications', 'host_has_profile_pic', 'host_identity_verified',

       'street', 'neighbourhood', 'neighbourhood_cleansed',

        'zipcode', 'market', 'transit',

       'smart_location', 'country_code', 'country', 'latitude', 'longitude',

       'is_location_exact',  'room_type', 'bed_type', 'amenities', 'minimum_nights',

       'maximum_nights', 'calendar_updated', 'has_availability',

       'availability_30', 'availability_60', 'availability_90',

       'availability_365', 'calendar_last_scraped',

       'first_review', 'last_review',  'requires_license',

        'jurisdiction_names', 'instant_bookable',

       'cancellation_policy', 'require_guest_profile_picture',

       'require_guest_phone_verification', 'calculated_host_listings_count',

       'reviews_per_month','price_y']

model_df=model_df.drop(drop_cols, axis=1)
#drop any row with missing price

model_df.dropna(subset=['price_x'])



#split response column

y=model_df['price_x']

model_df=model_df.drop(['price_x'],axis=1)



#fill missing vals for numerical variables

num_cols=model_df.select_dtypes(include=['float','int']).columns

for col in num_cols:

    model_df[col].fillna(model_df[col].median(), inplace=True)



#dummy variables for categorical variables

cat_cols=model_df.select_dtypes(include=['object']).columns

for col in cat_cols:

    model_df=pd.concat([model_df.drop(col, axis=1), pd.get_dummies(model_df[col], prefix=col, prefix_sep='_', drop_first=True)], axis=1)



model_df.head()
#calculate monthly and daily price averages

monthly_prices=date_df2.groupby(['month'], as_index=False, group_keys=False)['price']

daily_prices=date_df2.groupby(['monthday'], as_index=False, group_keys=False)['price']

monthly_avg=monthly_prices.mean()

daily_avg=daily_prices.mean()

daily_avg
%matplotlib inline

plt.plot(monthly_avg['month'],monthly_avg['price'])

plt.xlabel('month')

plt.ylabel('prices per day')

plt.title('average price by month')



print(f'the cheapest month is January with average price of ${monthly_avg.price.min():.2f}')

print(f'the most expensive month is July with average price of ${monthly_avg.price.max():.2f}')


fig, ax = plt.subplots()

ax.plot(daily_avg['monthday'],daily_avg['price'])

plt.xlabel('date')

plt.xticks(rotation=90, fontsize=8)

loc = plticker.MultipleLocator(base=14.0) 

ax.xaxis.set_major_locator(loc)

plt.ylabel('prices per day')

plt.title('average price by date')
max_val=daily_avg['price'].max()

max_date=daily_avg.iloc[daily_avg['price'].argmax()].monthday

min_val=daily_avg['price'].min()

min_date=daily_avg.iloc[daily_avg['price'].argmin()].monthday

print(f'maximum daily average prices is ${max_val:.2f} on {max_date}')

print(f'minimum daily average prices is ${min_val:.2f} on {min_date}')
daily_avg_smooth=daily_avg.rolling(7, center=True).mean()  

daily_avg_smooth.head()
fig, ax = plt.subplots()

ax.plot(daily_avg['monthday'],daily_avg_smooth['price'])

plt.xlabel('date')

plt.xticks(rotation=90, fontsize=8)

loc = plticker.MultipleLocator(base=14.0) 

ax.xaxis.set_major_locator(loc)

plt.ylabel('prices per day')

plt.title('average price by date')

max_val=daily_avg_smooth['price'].max()

max_date=daily_avg.iloc[daily_avg_smooth['price'].argmax()].monthday

min_val=daily_avg_smooth['price'].min()

min_date=daily_avg.iloc[daily_avg_smooth['price'].argmin()].monthday

print(f'maximum daily average prices is ${max_val:.2f} in the week +/-3 days around {max_date}')

print(f'minimum daily average prices is ${min_val:.2f} in the week +/-3 days around {min_date}')
area_prices=list_df.groupby(['neighbourhood_group_cleansed'], as_index=False, group_keys=False)['price']

area_avg=area_prices.mean().sort_values(['price'])

area_avg
plt.bar(area_avg['neighbourhood_group_cleansed'],area_avg['price'])

plt.xticks(rotation=90)

plt.xlabel('Neighbourhood')

plt.ylabel('prices per day')

plt.title('Average price by area')


## modelling

X_train, X_test, y_train, y_test = train_test_split(model_df, y, test_size = 0.2, random_state=7)

#lin_model = LinearRegression(normalize=True) # Instantiate

rf_model = RandomForestRegressor(n_estimators=150, 

                               criterion='mse', random_state=7, n_jobs=-1)



rf_model.fit(X_train, y_train) #Fit



result_train = rf_model.predict(X_train)

result_test = rf_model.predict(X_test)

    

test_score = r2_score(y_test, result_test)

train_score= r2_score(y_train, result_train)



print('R2 test score: '+str(test_score))

print('R2 train score: '+str(train_score))
test_score = mean_squared_error(y_test, result_test)

train_score= mean_squared_error(y_train, result_train)



print('MSE test score: '+str(test_score))

print('MSE train score: '+str(train_score))
headers = ["name", "score"]

values = sorted(zip(X_train.columns, rf_model.feature_importances_), key=lambda x: x[1] * -1)

forest_feature_importances = pd.DataFrame(values, columns = headers)

forest_feature_importances = forest_feature_importances.sort_values(by = ['score'], ascending = False)



features = forest_feature_importances['name'][:20]

y_pos = np.arange(len(features))

scores = forest_feature_importances['score'][:20]



#plot feature importances

plt.figure()

plt.bar(y_pos, scores, align='center', alpha=0.5)

plt.xticks(y_pos, features, rotation='vertical')

plt.ylabel('Score')

plt.xlabel('Features')

plt.title('Feature importances')

 

plt.show()
list_df2 = list_df[list_df['number_of_reviews']>5] #get rid of extremes that might skew data 

plt.figure()

plt.plot(list_df2['number_of_reviews'],list_df2['review_scores_rating'],'o',label='data')

lin_fit_param=np.polyfit(list_df2['number_of_reviews'], list_df2['review_scores_rating'],1)

plt.plot(list_df2['number_of_reviews'], lin_fit_param[0]*list_df2['number_of_reviews']+lin_fit_param[1],0, label='fit line')

plt.xlabel('number of reviews')

plt.ylabel('average score of reviews')

plt.title('Does more reviews mean better reviews?')

plt.legend(loc='lower right')

print(f'slope of linear fit is {lin_fit_param[0]:.5f}')
list_df3 = list_df[list_df['number_of_reviews']>=20]  #number of rentals with more than 20 reviews

list_df4 = list_df[list_df['number_of_reviews']<20] 

print(list_df3.shape) #check that the number of data points is roughly the same

print(list_df4.shape)

x=['less than 20','more than or equal 20']

y=[list_df3['review_scores_rating'].mean(),list_df4['review_scores_rating'].mean()]

plt.figure()

plt.bar(x,y)

plt.xlabel('number of reviews')

plt.ylabel('average score')

plt.title('Average score compared with number of reviews')