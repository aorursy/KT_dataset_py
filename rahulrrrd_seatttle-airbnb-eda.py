

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

style.use('ggplot')



import plotly.offline as pyoff

import plotly.graph_objs as go



pyoff.init_notebook_mode()





from datetime import datetime

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from datetime import datetime

from sklearn.cluster import KMeans

from sklearn.preprocessing import minmax_scale

from sklearn.metrics import mean_squared_error





import os 





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

Files required:



reviews.csv containing reviews info in Seattle city,

calender.csv  containing dates and price info in Seattle city,

listings.csv containing major features and information for analysis.





'''



#Reading the .csv file into pandas dataframe



c_data = pd.read_csv("/kaggle/input/seattle/calendar.csv")

r_data = pd.read_csv("/kaggle/input/seattle/reviews.csv")

l_data = pd.read_csv("/kaggle/input/seattle/listings.csv")



#checking and handling rows and columns with missing values



r_data.isnull().sum().plot(kind='bar')

plt.xticks(rotation=60);
#listings data



l_data.info()
#Get the column names with null values



l_data.isnull().sum()[l_data.isnull().sum().to_numpy().nonzero()[0]].index
plt.figure(figsize=(12,6));

plt.xticks(rotation=90);

plt.yticks(np.arange(0,4000,200));

sns.barplot(x=l_data.isnull().sum()[l_data.isnull().sum().to_numpy().nonzero()[0]].index,y=l_data.isnull().sum()[l_data.isnull().sum().to_numpy().nonzero()[0]].values);
# columns with more than 75 missing values

set(l_data.columns[l_data.isnull().mean()>= 0.75])
# columns with more than 100% missing values

set(l_data.columns[l_data.isnull().mean()>= 1])
#Iisnull columns contains more tha n100% missig value and sqaure feet contains more than 75%

#it wise to drop them

l_data.drop(['license','square_feet'],axis=1,inplace=True)
# missing values in Calender data



plt.figure(figsize=(10,6));

null_price = c_data.isnull().sum()

(null_price/c_data.shape[0]).plot(kind='bar');

#function to clean and convert str columns to numeric



def str_to_num(df,column):

    df[column] = pd.to_numeric(df[column].apply(lambda x : str(x).replace('$','').replace(",",'')),errors='coerce')

    return df



columns = ['price','monthly_price','weekly_price','security_deposit','cleaning_fee']



for col in columns:

    l_data = str_to_num(l_data,col)



l_data[columns][:5]
c_data.dropna(axis=0,subset=['price'],inplace=True)

r_data.dropna(axis=0,subset=['comments'],inplace=True)
#Explore different property types and room types



prop_type = l_data.property_type.value_counts()



plt.figure(figsize=(12,6));

(prop_type/l_data.shape[0]).plot(kind='bar');

plt.xticks(rotation=60);

plt.title('Percentange of different property types in Seattle city ');

plt.xlabel('Property Type');

plt.ylabel('% Availability');
room_type = l_data.room_type.value_counts()



plt.figure(figsize=(10,6));

(room_type/l_data.shape[0]).plot(kind='bar');

plt.xticks(rotation=60);

plt.title('Percentange of room types in Seattle city ');


plt.figure(figsize=(14,6))

l_data.groupby(['property_type','room_type'])['price'].mean().sort_values(ascending=False).plot(kind='bar');

plt.title('Price of room types as per different property types');

plt.ylabel('$ Price');
plt.figure(figsize=(14,6));

l_data.groupby(['property_type'])['price'].mean().sort_values(ascending=False).plot(kind='bar');

plt.title('Property price as per Property Type');

plt.xlabel('Property Type');

plt.ylabel('$ Price');
#Exploring neighbourhood and reviews data





# neighbourhood distribution

neighbourhood = l_data.neighbourhood.value_counts()[:40]



plt.figure(figsize=(14,8))

(neighbourhood/l_data.shape[0]).plot(kind="bar");

plt.title("Neighbourhood Listings distribution");

plt.xlabel('Neighbourhoods');


# num of reviews distribution 



fig = plt.figure(figsize=(10,6))

l_data.number_of_reviews.hist();

plt.title('Number of reviews distribution');

plt.xlabel('Number of Reviews');

plt.ylabel('Count');


# price distribution

plt.figure(figsize=(10,6))

l_data.price.hist();

plt.title('Price distribution');

plt.xlabel('Price nights');

plt.ylabel('Count');
#get summary of required columns



l_data[['reviews_per_month','number_of_reviews','calculated_host_listings_count','price','minimum_nights','availability_365']].describe()
#Calender dataset





# change the date column to a datetime 

c_data['date'] = pd.to_datetime(c_data.date)

c_data.info(verbose=True, null_counts=True)


# drop the missing values in price and drop 

c_data = c_data.dropna(subset=['price'], axis = 0)

c_data.info(verbose=True, null_counts=True)


# to clean the price and convert it into a float 

c_data = str_to_num(c_data,'price')

c_data.info(verbose=True, null_counts=True)
c_data.describe()
# add month and year column to the calender dataset

c_data['month'], c_data['year'] = c_data.date.dt.month, c_data.date.dt.year

c_data.info(verbose=True, null_counts=True)
c_data.available.value_counts()


price = pd.DataFrame(c_data.groupby(['month','available']).mean()['price'].reset_index())



data = [

    go.Scatter(

        x = price['month'],

        y = price.price,

        name = 'Price'

    )

]



layout = go.Layout(

    title = 'Booking prices as per months',

    xaxis = dict(title='Months'),

    yaxis = dict(title= '$ Price'),

    showlegend=True,

    

)

fig = go.Figure(data=data,layout=layout)



pyoff.iplot(fig)
#as per the dataset all listings are available



available_count_daily = c_data.groupby('date').count()[['price']]

available_count_daily = available_count_daily.rename({"price":"total_available_houses"},axis='columns')



average_price_daily = c_data.groupby('date').mean()[['price']]

# change column name

average_price_daily = average_price_daily.rename({"price":"average_prices"},axis='columns')
# plot total available houses and average prices in one figure

f, ax = plt.subplots(figsize=(15, 6))

plt1 = sns.lineplot(x = available_count_daily.index,y = 'total_available_houses', 

                  data = available_count_daily,color="b",legend=False,label='No. of houses available')



ax2 = ax.twinx()

plt2 = sns.lineplot(x = average_price_daily.index,y = 'average_prices',

             data=average_price_daily,ax=ax2,linestyle=':', legend=False,label='Daily prices')

ax.set_title('Comparing the daily availability of airbnb listing with the daily listing prices');

ax2.legend();

ax.legend();
# group the listings by neighbourood and get the average price

l_data.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)

(l_data.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)).plot(kind="bar", figsize=(16,8));

plt.title("Average Listing price across Seattle neighbourhoods");

plt.xlabel('Neighbourhood');

plt.ylabel('Average Price');
#top 10 expensive neighbourhoods



(l_data.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)[:10]).plot(kind="bar", figsize=(16,8));

plt.title("Average Listing price across Seattle neighbourhoods");

plt.xlabel('Neighbourhood');

plt.ylabel('Average Price $');

plt.xticks(rotation=60);
(l_data.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)[-5:]).plot(kind="bar", figsize=(16,8));

plt.title("Average Listing price across Seattle neighbourhoods");

plt.xlabel('Neighbourhood');

plt.ylabel('Average Price $');

plt.xticks(rotation=60);


# drop redundent columns

l_data.drop(columns = 'neighbourhood' , inplace = True, axis = 1)

l_data.info()
# columns with more than 75% missing values

cols_with_missing_val = list(set(l_data.columns[l_data.isnull().mean()>75]))

cols_with_missing_val
l_data[['review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','review_scores_rating']].describe()
# create new columns

#Fix columns that have only 2 possible with binary values 1 and 0



l_data['has_transit_info'] = l_data.transit.apply(lambda x: 0 if pd.isnull(x) else 1)

l_data['has_description_info'] = l_data.description.apply(lambda x: 0 if pd.isnull(x) else 1)

l_data['has_summary_info'] = l_data.summary.apply(lambda x: 0 if pd.isnull(x) else 1)

l_data['has_neighborhood_info'] = l_data.neighborhood_overview.apply(lambda x: 0 if pd.isnull(x) else 1)

l_data['has_host_info'] = l_data.host_about.apply(lambda x: 0 if pd.isnull(x) else 1)

l_data.host_is_superhost = l_data.host_is_superhost.apply(lambda x: 0 if pd.isnull(x) else 1)



# fix t and f columns

l_data['instant_bookable'] =  l_data['instant_bookable'].apply(lambda x: 1 if x=='t' else 0)

l_data['require_guest_phone_verification'] =  l_data['require_guest_phone_verification'].apply(lambda x: 1 if x=='t' else 0)

l_data['require_guest_profile_picture'] =  l_data['require_guest_profile_picture'].apply(lambda x: 1 if x=='t' else 0)

l_data['host_identity_verified'] =  l_data['host_identity_verified'].apply(lambda x: 1 if x=='t' else 0)

l_data.host_has_profile_pic = l_data.host_has_profile_pic.apply(lambda x: 1 if x=='t' else 0)
l_data.info()

# Drop irrelevent object columns



l_data.select_dtypes(include=['object']).columns

columns_to_drop =['listing_url', 'scrape_id', 'last_scraped', 'country_code', 'country', 'picture_url', 'host_url', 'host_thumbnail_url','notes', 

'host_picture_url','transit', 'space', 'description', 'summary', 'neighborhood_overview', 'name', 'latitude', 'longitude'

, 'is_location_exact', 'host_about','jurisdiction_names','experiences_offered','amenities','calendar_last_scraped','requires_license', 'smart_location', 'host_verifications', 'street',

'host_verifications', 'calendar_updated','has_availability', 'city', 'state','host_id','zipcode','market','host_location','host_name','host_neighbourhood']



l_data.drop(columns=columns_to_drop, inplace=True)

l_data.info()
l_data.drop(['xl_picture_url','thumbnail_url','medium_url'],axis=1,inplace=True)
l_data.select_dtypes(include=['object']).columns



numerical_columns = l_data.select_dtypes(include=['float', 'int']).columns

numerical_columns
def days_between(date):

    """

    return the number of days from today to the given date

    """

    try:

        date_format = "%Y-%m-%d"

        NOW = datetime.now()

        d = datetime.strptime(date, date_format)

        return abs((NOW - d).days)

    except:

        return date
l_data.first_review.apply(lambda x : days_between(x)).head()

l_data['first_review'] = l_data.first_review.apply(lambda x : days_between(x))

l_data['last_review'] = l_data['last_review'].apply(lambda x : days_between(x))

l_data['host_since'] = l_data['host_since'].apply(lambda x : days_between(x))

l_data.info()
l_data = str_to_num(l_data,'extra_people')
def clean_response_rate(x):

    """

    clean the response rate column by eliminating the % and

    from the string and return a float to be stored in the column

    """

    try:

        return float(str( x[:-1]))/100

    except:

        return x
l_data.host_response_rate = l_data.host_response_rate.apply(lambda x: clean_response_rate(x))
l_data.info()
l_data.select_dtypes(include=['object']).columns
l_data.drop(['host_acceptance_rate','neighbourhood_group_cleansed'],axis=1,inplace=True)
#Handling categorcial data



def create_dummy_df(df, cat_cols, dummy_na):

    '''

    INPUT:

    df - pandas dataframe with categorical variables you want to dummy

    cat_cols - list of strings that are associated with names of the categorical columns

    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not

    

    OUTPUT:

    df - a new dataframe that has the following characteristics:

            1. contains all columns that were not specified as categorical

            2. removes all the original columns in cat_cols

            3. dummy columns for each of the categorical columns in cat_cols

            4. if dummy_na is True - it also contains dummy columns for the NaN values

            5. Use a prefix of the column name with an underscore (_) for separating 

    '''

    for col in cat_cols:

        try:

            df = pd.concat([df.drop(col, axis =1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)

        except:

            continue

    return df
cat_cols_lst = l_data.select_dtypes(include=['object'])

cat_cols_lst
# create new columns



# print the shape of the dataframe before handeling the categorical variables

print(l_data.shape)



l_data_categorized = create_dummy_df(l_data, cat_cols_lst, dummy_na=False) 



print(l_data_categorized.shape)
#Feature Scaling and transformation



l_data_categorized.dropna(axis=1,inplace=True)



l_data_categorized.isnull().sum(axis=1).describe()



numerical_columns = l_data_categorized.select_dtypes(include=['float', 'int']).columns



listings_df_scaled = pd.DataFrame(data = l_data_categorized)



# define the scaler and scale numerical values

scaler = MinMaxScaler()

listings_df_scaled[numerical_columns] = scaler.fit_transform(l_data_categorized[numerical_columns])



# Shows scaled data

listings_df_scaled.head()
#Checking kmeans score to find the optimal cluster number



listings_df_scaled.dropna(axis=1,inplace=True)



sse = {}

for i in range(1,30):

    Kmeans = KMeans(n_clusters=i).fit(listings_df_scaled)

    sse[i] = Kmeans.inertia_

    

plt.figure(figsize=(8,6))

plt.plot(list(sse.keys()),list(sse.values()))

plt.xlabel("Number of Clusters")

plt.show()
#Refit fitting with optimal cluster number

kmeans = KMeans(n_clusters=10)

model_kmeans = kmeans.fit(listings_df_scaled)

labels_listings = model_kmeans.transform(listings_df_scaled)



listings_kmeans = model_kmeans.predict(listings_df_scaled)

listings_kmeans
# add the cluster column to the dataset 

listings_df_scaled['Cluster'] = pd.Series(listings_kmeans, index=listings_df_scaled.index)

listings_df_scaled.head()
# group the data by different clusters 

listings_grouped_by_clusters = listings_df_scaled.groupby('Cluster').mean()

listings_grouped_by_clusters.head()
avg_price_per_cluster = listings_grouped_by_clusters.price.sort_values(ascending=False)

avg_price_per_cluster.plot.bar(figsize=(10,4));

plt.title('Average Listings Price per Cluster');

plt.ylabel('Average Price');
# clusters column mean value for 

clusters_mean = listings_grouped_by_clusters.mean()

clusters_mean
# in order to compare the different clusters i will subtract the mean

# and the value in each cluster



# display the percentage for each cluster and the average mean 

clusters_differences_in_mean = ((listings_grouped_by_clusters- clusters_mean)*100/clusters_mean)

clusters_differences_in_mean
clusters_differences_in_mean_transposed = clusters_differences_in_mean.T

clusters_differences_in_mean_transposed
clusters_differences_in_mean['price']

clusters_differences_in_mean[['price']].sort_values(ascending=False, by='price').plot(kind='barh', figsize= [10,5], fontsize =12, legend=False);

plt.title('Percentage of Price Difference Across Clusters');

plt.xlabel('Difference');

plt.ylabel('Clusters');
# investigating the most cluster with the highest and lowest price 

# cluster 5 highest difference score compared to the other clusters

# cluster 1 had the lowest price



cluster5 = clusters_differences_in_mean_transposed[5].sort_values(ascending=False).head(20)

cluster5.plot.bar(figsize=(10,4))

plt.title('Attributes Differeniating Cluster 5');

plt.ylabel('Difference Percentage');

plt.xlabel('Attribute');
cluster5



# Cluster 5



# the highest difference is for property_type_dorm

# property type: dorm,ced_couch,Tent,Condonium

# neighbourhoods: South park,Laurwlhuest,Queeen Anne

# it had the highest price difference
cluster1 = clusters_differences_in_mean_transposed[1].sort_values(ascending=False).head(20)

cluster1.plot.bar(figsize=(10,4))

plt.title('Attributes Differeniating Cluster 1');

plt.ylabel('Difference Percentage');

plt.xlabel('Attribute');
# Cluster 1:



# property Bunglow

# strict cancellation policy moderate

# neighbourhoods: Pinehusrt,Highland_park,View_ridge
# Medium range price cluster 8

cluster8 = clusters_differences_in_mean_transposed[8].sort_values(ascending=False).head(20)

cluster8.plot.bar(figsize=(10,4))



plt.title('Attributes Differeniating Cluster 8');

plt.ylabel('Difference Percentage');

plt.xlabel('Attribute');


# Cluster 8:



# property type mostly Treehouse,Cabin,Townhouse

# the highest difference is for montlake neighbourhood

# Strict Canecllation policy

# neighbourhoods: SunsetHill,Arbor_heights,Ravenna



cluster8
# comparing the 3 clusters 

comparsion_columns = list(set(cluster5.index[:10].tolist() + cluster1.index[:10].tolist() + cluster8.index[:10].tolist()))

clusters_differences_in_mean_transposed[[5,8,1]].loc[comparsion_columns].sort_values(ascending=False, by=5)
l_data.last_review.mean()

l_data_categorized.accommodates.value_counts()

l_data_categorized.rename(columns={'property_type_Bed_&_Breakfast':'property_type_Bed/Breakfast'},inplace=True)

#implementing a Logistic model



all_columns = []

for column in l_data_categorized.columns:

    column = column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_").replace("/","_")

    all_columns.append(column)



l_data_categorized.columns = all_columns
glm_columns = 'host_is_superhost'



for column in l_data_categorized.columns:

    if column not in ['price','id']:

        glm_columns = glm_columns + ' + ' + column
import statsmodels.api as sm

import statsmodels.formula.api as smf

 



glm_model = smf.glm(formula='price ~ {}'.format(glm_columns), data=l_data_categorized, family=sm.families.Binomial())

res = glm_model.fit()

print(res.summary())
print(np.exp(res.params));
#create feature set and labels

X = l_data_categorized.drop(['price','id'],axis=1)

y = l_data_categorized.price
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



#train and test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)
import xgboost as xgb



#building the model

xgb_model = xgb.XGBClassifier(learning_rate=0.01, objective= 'binary:logistic',n_jobs=-1).fit(X_train, y_train)



pred = xgb_model.predict(X_test)

results = pd.DataFrame(pred,y_test)

results = results.reset_index()

results.columns = ['price','pred']

mse = mean_squared_error(y_pred=results['pred'],y_true=results['price'])

print(f"Mean Squared error of model on test set : {mse}")
from xgboost import plot_importance



fig, ax = plt.subplots(figsize=(10,8))

plot_importance(xgb_model, ax=ax)