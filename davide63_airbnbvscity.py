%%javascript

IPython.OutputArea.auto_scroll_threshold = 300;
from IPython.display import Image

Image(filename='../input/airbnbbasermam/florencia-potter-s2q1_cxLHSE-unsplash.jpg')
%%javascript

IPython.OutputArea.auto_scroll_threshold = 100;
from IPython.display import Image

Image(filename='../input/airbnbbasermam/feature importances XGB - Seattle.png')
Image(filename='../input/airbnbbasermam/feature importances XGB - Barcelona.png') 
Image(filename='../input/airbnbbasermam/feature importances RF - Seattle.png')
Image(filename='../input/airbnbbasermam/feature importances RF - Seattle.png')
Image(filename='../input/airbnbbasermam/neighbourhood_ba.png')
Image(filename='../input/airbnbbasermam/neighbourhood_se.png')
Image(filename='../input/airbnbbasermam/NeighbourhoodGroup_se.png')
#Import linear algebra and data manipulation

import numpy as np

import pandas as pd



#Import plotting packages

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



#Import machine learning

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import xgboost



from sklearn.model_selection import train_test_split #split

from sklearn.metrics import r2_score, mean_squared_error #metrics





import matplotlib.image as mpimg

import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score



from sklearn.tree import DecisionTreeRegressor



from wordcloud import WordCloud



import folium

from folium.plugins import HeatMap
# read listing data and print the related shape



listings_am_df = pd.read_csv('../input/airbnbbasermam/listings_am.csv')

listings_ba_df = pd.read_csv('../input/airbnbbasermam/listings_ba.csv')

listings_rm_df = pd.read_csv('../input/airbnbbasermam/listings_rm.csv')

listings_se_df = pd.read_csv('../input/airbnbbasermam/listings_se.csv')



print('Shape of AM: %s' % (str(listings_am_df.shape)))

print('Shape of BA: %s' % (str(listings_ba_df.shape)))

print('Shape of RM: %s' % (str(listings_rm_df.shape)))

print('Shape of SE: %s' % (str(listings_se_df.shape)))



# check if AM and BA have the same columns

check = listings_am_df.columns.tolist() == listings_ba_df.columns.tolist()

print('AM and BA file have the same columns: %s' % (check))
listings_am_df.columns.tolist()
listings_am_df.info
#Cleaning data

listings_am_df.duplicated().sum()

listings_am_df.drop_duplicates(inplace=True)

print('Amsterdam Data')

print(listings_am_df.isnull().sum())



print("")

listings_ba_df.duplicated().sum()

listings_ba_df.drop_duplicates(inplace=True)

print('Barcelona Data')

print(listings_ba_df.isnull().sum())



print("")

listings_rm_df.duplicated().sum()

listings_rm_df.drop_duplicates(inplace=True)

print('Rome Data')

print(listings_rm_df.isnull().sum())



print("")

listings_se_df.duplicated().sum()

listings_se_df.drop_duplicates(inplace=True)

print('Seattle Data')

print(listings_se_df.isnull().sum())
columns_to_drop = ['id',

 'listing_url',

 'scrape_id',

 'last_scraped',

 'name',

 'summary',

 'space',

 'description',

 'experiences_offered',

 'neighborhood_overview',

 'notes',

 'transit',

 'access',

 'interaction',

 'house_rules',

 'thumbnail_url',

 'medium_url',

 'picture_url',

 'xl_picture_url',

 'host_url',

 'host_name',

 'host_since',

 'host_location',

 'host_about',

 'host_response_time',

 'host_response_rate',

 'host_acceptance_rate',

 'host_is_superhost',

 'host_thumbnail_url',

 'host_picture_url',

 'host_neighbourhood',

 'host_listings_count',

 'host_total_listings_count',

 'host_verifications',

 'host_has_profile_pic',

 'host_identity_verified',

 'street',

 'neighbourhood',

 'city',

 'state',

 'zipcode',

 'market',

 'smart_location',

 'country_code',

 'country',

 'is_location_exact',

 'property_type',

 'accommodates',

 'bathrooms',

 'bedrooms',

 'beds',

 'bed_type',

 'amenities',

 'square_feet',

 'weekly_price',

 'monthly_price',

 'security_deposit',

 'cleaning_fee',

 'guests_included',

 'extra_people',

 'maximum_nights',

 'minimum_minimum_nights',

 'maximum_minimum_nights',

 'minimum_maximum_nights',

 'maximum_maximum_nights',

 'minimum_nights_avg_ntm',

 'maximum_nights_avg_ntm',

 'calendar_updated',

 'has_availability',

 'availability_30',

 'availability_60',

 'availability_90',

 'calendar_last_scraped',

 'number_of_reviews_ltm',

 'first_review',

 'last_review',

 'review_scores_rating',

 'review_scores_accuracy',

 'review_scores_cleanliness',

 'review_scores_checkin',

 'review_scores_communication',

 'review_scores_location',

 'review_scores_value',

 'requires_license',

 'license',

 'jurisdiction_names',

 'instant_bookable',

 'is_business_travel_ready',

 'cancellation_policy',

 'require_guest_profile_picture',

 'require_guest_phone_verification',

 'calculated_host_listings_count_entire_homes',

 'calculated_host_listings_count_private_rooms',

 'calculated_host_listings_count_shared_rooms',

 ]

listings_am_df.drop(columns_to_drop, axis=1, inplace=True)

listings_ba_df.drop(columns_to_drop, axis=1, inplace=True)

listings_rm_df.drop(columns_to_drop, axis=1, inplace=True)

listings_se_df.drop(columns_to_drop, axis=1, inplace=True)
listings_am_df.head(5)
listings_ba_df.head(5)
listings_rm_df.head(5)
listings_se_df.head(5)
print('NaN reviews per month ante filling: %d'% listings_am_df.reviews_per_month.isnull().sum())

listings_am_df.fillna({'reviews_per_month':0}, inplace=True)

#examing changes

print('NaN reviews per month post filling with 0: %d'% listings_am_df.reviews_per_month.isnull().sum())



print('NaN reviews per month ante filling: %d'% listings_ba_df.reviews_per_month.isnull().sum())

listings_ba_df.fillna({'reviews_per_month':0}, inplace=True)

#examing changes

print('NaN reviews per month post filling with 0: %d'% listings_ba_df.reviews_per_month.isnull().sum())



print('NaN reviews per month ante filling: %d'% listings_rm_df.reviews_per_month.isnull().sum())

listings_rm_df.fillna({'reviews_per_month':0}, inplace=True)

#examing changes

print('NaN reviews per month post filling with 0: %d'% listings_rm_df.reviews_per_month.isnull().sum())



print('NaN reviews per month ante filling: %d'% listings_se_df.reviews_per_month.isnull().sum())

listings_se_df.fillna({'reviews_per_month':0}, inplace=True)

#examing changes

print('NaN reviews per month post filling with 0: %d'% listings_se_df.reviews_per_month.isnull().sum())



listings_am_df.isnull().sum()

listings_am_df.dropna(how='any',inplace=True)

listings_am_df.info()
listings_ba_df.isnull().sum()

listings_ba_df.dropna(how='any',inplace=True)

listings_ba_df.info()
listings_rm_df.isnull().sum()

listings_rm_df.dropna(how='any',inplace=True)

listings_rm_df.info()
listings_se_df.isnull().sum()

listings_se_df.dropna(how='any',inplace=True)

listings_se_df.info()
listings_ba_df.describe()
listings_se_df.describe()
corr = listings_ba_df.corr(method='kendall')

plt.figure(figsize=(15,8))

sns.heatmap(corr, annot=True)

listings_ba_df.columns
corr = listings_se_df.corr(method='kendall')

plt.figure(figsize=(15,8))

sns.heatmap(corr, annot=True)

listings_se_df.columns
listings_ba_df['neighbourhood_group_cleansed'].unique()
listings_se_df['neighbourhood_group_cleansed'].unique()
sns.countplot(listings_ba_df['neighbourhood_group_cleansed'], palette="plasma")

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Neighbourhood Group')
sns.countplot(listings_se_df['neighbourhood_group_cleansed'], palette="plasma")

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Neighbourhood Group - Seattle')

plt.savefig('NeighbourhoodGroup_se.png')
sns.countplot(listings_ba_df['neighbourhood_cleansed'], palette="plasma")

fig = plt.gcf()

fig.set_size_inches(25,6)

plt.title('Neighbourhood')
sns.countplot(listings_se_df['neighbourhood_cleansed'], palette="plasma")

fig = plt.gcf()

fig.set_size_inches(25,6)

plt.title('Neighbourhood')
sns.countplot(listings_ba_df['room_type'], palette="plasma")

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Restaurants delivering online or Not')
sns.countplot(listings_se_df['room_type'], palette="plasma")

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Restaurants delivering online or Not')
plt.figure(figsize=(10,10))

ax = sns.boxplot(data=listings_ba_df, x='neighbourhood_group_cleansed',y='availability_365',palette='plasma')
plt.figure(figsize=(10,10))

ax = sns.boxplot(data=listings_se_df, x='neighbourhood_group_cleansed',y='availability_365',palette='plasma')
plt.figure(figsize=(10,6))

sns.scatterplot(listings_ba_df.longitude,listings_ba_df.latitude,hue=listings_ba_df.neighbourhood_group_cleansed)

plt.ioff()
plt.figure(figsize=(10,6))

sns.scatterplot(listings_se_df.longitude,listings_se_df.latitude,hue=listings_se_df.neighbourhood_group_cleansed)

plt.ioff()
plt.figure(figsize=(10,6))

sns.scatterplot(listings_ba_df.longitude,listings_ba_df.latitude,hue=listings_ba_df.neighbourhood_cleansed)

plt.ioff()
plt.figure(figsize=(10,6))

sns.scatterplot(listings_se_df.longitude,listings_se_df.latitude,hue=listings_se_df.neighbourhood_cleansed)

plt.ioff()
sns.scatterplot(listings_ba_df.longitude,listings_ba_df.latitude,hue=listings_ba_df.room_type)

plt.ioff()
sns.scatterplot(listings_se_df.longitude,listings_se_df.latitude,hue=listings_se_df.room_type)

plt.ioff()
plt.figure(figsize=(10,6))

sns.scatterplot(listings_ba_df.longitude,listings_ba_df.latitude,hue=listings_ba_df.availability_365)

plt.ioff()
plt.figure(figsize=(10,6))

sns.scatterplot(listings_se_df.longitude,listings_se_df.latitude,hue=listings_se_df.availability_365)

plt.ioff()
wordcloud_ba = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(listings_ba_df.neighbourhood_cleansed))

plt.imshow(wordcloud_ba)

plt.axis('off')

plt.savefig('neighbourhood_ba.png')

plt.show()
wordcloud_se = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(listings_se_df.neighbourhood_cleansed))

plt.imshow(wordcloud_se)

plt.axis('off')

plt.savefig('neighbourhood_se.png')

plt.show()
m=folium.Map([41.3887901,-2.1589899],zoom_start=11)

HeatMap(listings_ba_df[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
m=folium.Map([47.6062,-122.3321],zoom_start=11)

HeatMap(listings_se_df[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
listings_ba_df.drop(['host_id','latitude','longitude','neighbourhood_cleansed','number_of_reviews','reviews_per_month'], axis=1, inplace=True)

#examing the changes

listings_ba_df.head(5)
listings_se_df.drop(['host_id','latitude','longitude','neighbourhood_cleansed','number_of_reviews','reviews_per_month'], axis=1, inplace=True)

#examing the changes

listings_se_df.head(5)
def Encode(listings_df):

    for column in listings_ba_df.columns[listings_ba_df.columns.isin(['neighbourhood_group_cleansed', 'room_type'])]:

        listings_ba_df[column] = listings_ba_df[column].factorize()[0]

    return listings_ba_df



listings_ba_df_en = Encode(listings_ba_df.copy())

listings_se_df_en = Encode(listings_se_df.copy())
listings_ba_df_en.head(15)
listings_se_df_en.head(15)
corr = listings_ba_df_en.corr(method='kendall')

plt.figure(figsize=(18,12))

sns.heatmap(corr, annot=True)

listings_ba_df_en.columns
corr = listings_se_df_en.corr(method='kendall')

plt.figure(figsize=(18,12))

sns.heatmap(corr, annot=True)

listings_se_df_en.columns
x = listings_ba_df_en.iloc[:,[0,1,3,4,5]]

y = listings_ba_df_en[['price']]

y = y.replace({'\$': '', ',': ''}, regex=True).astype(float)

#Getting Test and Training Set

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)

x_train.head()

y_train.head()
y_train.head()
x_train.shape
reg=LinearRegression()

reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

r2_score(y_test,y_pred)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)

DTree=DecisionTreeRegressor(min_samples_leaf=.0001)

DTree.fit(x_train,y_train)

y_predict=DTree.predict(x_test)

r2_score(y_test,y_predict)

#2nd run

DTree.fit(x_train,y_train)

y_predict=DTree.predict(x_test)

r2_score(y_test,y_predict)
print(x_test[:5])

print()

print(y_pred[:5])
TEST_SIZE = 0.3

RAND_STATE = 42

forest = RandomForestRegressor(n_estimators=100, 

                                   criterion='mse', 

                                   random_state=RAND_STATE, 

                                   n_jobs=-1)

forest.fit(x_train, y_train.squeeze())



#calculate scores for the model

y_train_preds = forest.predict(x_train)

y_test_preds = forest.predict(x_test)

print('Random Forest')

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_preds), mean_squared_error(y_test, y_test_preds)))

print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_preds), r2_score(y_test, y_test_preds)))



#get feature importances from the model

headers = ["name", "score"]

values = sorted(zip(x_train.columns, forest.feature_importances_), key=lambda x: x[1] * -1)

forest_feature_importances = pd.DataFrame(values, columns = headers)

forest_feature_importances = forest_feature_importances.sort_values(by = ['score'], ascending = False)

features = forest_feature_importances['name'][:15]

y_pos = np.arange(len(features))

scores = forest_feature_importances['score'][:15]

#plot feature importances

plt.figure(figsize=(10,5))

plt.bar(y_pos, scores, align='center', alpha=0.5)

plt.xticks(y_pos, features, rotation='vertical')

plt.ylabel('Score')

plt.xlabel('Features')

plt.title('Feature importances (Random Forest) - Barcelona')

plt.savefig('feature importances RF - Barcelona.png')

plt.show()



#train XGBoost model

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

xgb.fit(x_train,y_train)

#calculate and print scores for the model for top 15 features

y_train_preds = xgb.predict(x_train)

y_test_preds = xgb.predict(x_test)

print('XGBoost')

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_preds), mean_squared_error(y_test, y_test_preds)))

print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_preds), r2_score(y_test, y_test_preds)))



#get feature importances from the model

headers = ["name", "score"]

values = sorted(zip(x_train.columns, xgb.feature_importances_), key=lambda x: x[1] * -1)

xgb_feature_importances = pd.DataFrame(values, columns = headers)



#plot feature importances for top 15 features

features = xgb_feature_importances['name'][:15]

y_pos = np.arange(len(features))

scores = xgb_feature_importances['score'][:15]

plt.figure(figsize=(10,5))

plt.bar(y_pos, scores, align='center', alpha=0.5)

plt.xticks(y_pos, features, rotation='vertical')

plt.ylabel('Score')

plt.xlabel('Features')

plt.title('Feature importances (XGBoost) - Barcelona')

plt.savefig('feature importances XGB - Barcelona.png')

plt.show()
x = listings_se_df_en.iloc[:,[0,1,3,4,5]]

y = listings_se_df_en[['price']]

y = y.replace({'\$': '', ',': ''}, regex=True).astype(float)

#Getting Test and Training Set

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)

x_train.head()
y_train.head()
reg=LinearRegression()

reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

print('R^2 score for linear regression: ' + str(r2_score(y_test,y_pred)))

      

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)

DTree=DecisionTreeRegressor(min_samples_leaf=.0001)

DTree.fit(x_train,y_train)

y_predict=DTree.predict(x_test)

print('R^2 score for decision tree regressor (1st run): ' + str(r2_score(y_test,y_pred)))

DTree.fit(x_train,y_train)

y_predict=DTree.predict(x_test)

print('R^2 score for decision tree regressor (2nd run): ' + str(r2_score(y_test,y_pred)))
print(x_test[:5])

print()

print(y_pred[:5])
TEST_SIZE = 0.3

RAND_STATE = 42

forest = RandomForestRegressor(n_estimators=100, 

                                   criterion='mse', 

                                   random_state=RAND_STATE, 

                                   n_jobs=-1)

forest.fit(x_train, y_train.squeeze())



#calculate scores for the model

y_train_preds = forest.predict(x_train)

y_test_preds = forest.predict(x_test)

print('Random Forest')

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_preds), mean_squared_error(y_test, y_test_preds)))

print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_preds), r2_score(y_test, y_test_preds)))



#get feature importances from the model

headers = ["name", "score"]

values = sorted(zip(x_train.columns, forest.feature_importances_), key=lambda x: x[1] * -1)

forest_feature_importances = pd.DataFrame(values, columns = headers)

forest_feature_importances = forest_feature_importances.sort_values(by = ['score'], ascending = False)

features = forest_feature_importances['name'][:15]

y_pos = np.arange(len(features))

scores = forest_feature_importances['score'][:15]

#plot feature importances

plt.figure(figsize=(10,5))

plt.bar(y_pos, scores, align='center', alpha=0.5)

plt.xticks(y_pos, features, rotation='vertical')

plt.ylabel('Score')

plt.xlabel('Features')

plt.title('Feature importances (Random Forest) - Seattle')

plt.savefig('feature importances RF - Seattle.png')

plt.show()



#train XGBoost model

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

xgb.fit(x_train,y_train)

#calculate and print scores for the model for top 15 features

y_train_preds = xgb.predict(x_train)

y_test_preds = xgb.predict(x_test)

print('XGBoost')

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_preds), mean_squared_error(y_test, y_test_preds)))

print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_preds), r2_score(y_test, y_test_preds)))



#get feature importances from the model

headers = ["name", "score"]

values = sorted(zip(x_train.columns, xgb.feature_importances_), key=lambda x: x[1] * -1)

xgb_feature_importances = pd.DataFrame(values, columns = headers)



#plot feature importances for top 15 features

features = xgb_feature_importances['name'][:15]

y_pos = np.arange(len(features))

scores = xgb_feature_importances['score'][:15]

plt.figure(figsize=(10,5))

plt.bar(y_pos, scores, align='center', alpha=0.5)

plt.xticks(y_pos, features, rotation='vertical')

plt.ylabel('Score')

plt.xlabel('Features')

plt.title('Feature importances (XGBoost) - Seattle')

plt.savefig('feature importances XGB - Seattle.png')

plt.show()