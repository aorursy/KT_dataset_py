"""

want to work on this locally?





1. make conda environment, 

                                    conda create --name workshop

                                    conda activate workshop







2. install dependencies wiithin conda-env

                                    pip install kaggle

                                    conda install folium 

                                    conda install seaborn

                                    pip install descartes

                                    kaggle datasets download -d bbissey/barcelonaairbnbgeojson

                                    or just download the dataset on kaggle's website





^^ maybe some more package installs 

"""



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline





# scraped on sept19, 2019 from insideairbnb.com

df_0 = pd.read_csv('../input/barcelonaairbnbgeojson/listings.csv')

num_rows = len(df_0['id'])

df_0.head(5)



# trouble with only displaying some columns or rows?

# pd.set_option('display.max_columns', None)

# pd.set_option('display.max_rows', None)





display(df_0.describe())
display(df_0.isnull().sum())
#Remove Majority Null Columns



colsToDrop = []

for col in df_0.columns:

    if df_0[col].isnull().sum() > (.6 * num_rows):

        colsToDrop.append(col)  

print(f'Number of columns to be dropped: {len(colsToDrop)}')

for col in colsToDrop:

    print(col)

df_0.shape
df_1 = df_0.drop(colsToDrop, axis=1)



df_1.shape
# Remove Columns with One Unique Value

# Why? For example, country. If all data is in the same country, we don't need 20404 rows that say Spain

# Yes, there are faster and more efficient ways to do this.



colsToDrop = []



for col in df_1.columns:

    if df_1[col].nunique() == 1:

        colsToDrop.append(col)

        

for col in colsToDrop:

    print(col)

df_1.shape

# Reassign to df_2 variable



df_2 = df_1.drop(colsToDrop, axis = 1)

df_2.shape
# what data types make up our dataframe?

# object == string



df_2.dtypes
# As previously seen, pandas' nunique function counts the number of unique values in a column

display(df_2.nunique())
# Bulk Removal of Redundant Columns and String Columns



# Some of these string columns may be helpful for categorical analysis and language processing, 

# but for the purpose of this workshop we will leave them out.

# 





df_3 = df_2.drop(['listing_url', 'last_scraped', 'name', 'summary', 'space', 'description',

                 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'picture_url',

                 'host_url', 'host_name', 'host_since', 'host_location', 'host_about', 'host_thumbnail_url', 

                  'host_neighbourhood', 'host_listings_count', 'host_total_listings_count', 'host_response_time', 

                  'host_response_rate', 'street', 'host_verifications', 'host_picture_url', 'amenities', 'calendar_updated', 

                  'calendar_last_scraped', 'availability_30', 'availability_90', 'availability_60','neighbourhood', 

                  'smart_location', 'is_location_exact', 'first_review', 'last_review', 'license','minimum_minimum_nights',

                  'maximum_minimum_nights','minimum_maximum_nights','maximum_maximum_nights','minimum_nights_avg_ntm',

                  'maximum_nights_avg_ntm','calculated_host_listings_count_entire_homes',

                  'calculated_host_listings_count_shared_rooms','calculated_host_listings_count_private_rooms'], axis =1)

print(f'Before removing redundant values: {df_2.shape}')

print(f'After removing redundant values:  {df_3.shape}')
# Look at what data remains, along with how many unique values there are

df_3.nunique()
df_3 = df_3.set_index('id')
df_3.head(20)
# Why? To visualize the proportions of data relative to each other

# Normalization of value counts helps with seeing proportions easier



df_3.room_type.value_counts(normalize=True)
df_3.cancellation_policy.value_counts(normalize=True)



df_3.neighbourhood_group_cleansed.value_counts(normalize=True)

        
df_3.bed_type.value_counts(normalize=True)
# Almost everybody has a real bed, let's remove this variable

df_4 = df_3.drop(['bed_type', 'review_scores_accuracy',

'review_scores_cleanliness',

'review_scores_checkin',

'review_scores_communication',

'review_scores_location',

'review_scores_value',

'state',

'zipcode',

'market',

], axis = 1)
# Check nulls again



df_4.isnull().sum()



# Returns every row where the 'availability_365' column equals 0.

df_4.loc[df_4['availability_365'] == 0].head()
# Removal of Inactive Listings



df_4 = df_4[df_4['availability_365'] != 0] 

#Parsing Floats from Price Columns



df_5 = df_4

df_6 = df_5

df_6['price'] = df_6['price'].str.replace('$', '').str.replace(',', '').astype(float)

df_6['security_deposit'] = df_6['security_deposit'].str.replace('$', '').str.replace(',', '').astype(float)

df_6['cleaning_fee'] = df_6['cleaning_fee'].str.replace('$', '').str.replace(',', '').astype(float)

df_6['extra_people'] = df_6['extra_people'].str.replace('$', '').str.replace(',', '').astype(float)
df_6['security_deposit'].fillna(0, inplace = True)

df_6['cleaning_fee'].fillna(0, inplace = True)

df_6['extra_people'].fillna(0, inplace = True)

df_6['review_scores_rating'].fillna(0, inplace = True)

df_6['reviews_per_month'].fillna(0, inplace = True)
# Review the remaining data

df_6.head()
df_7 = df_6.drop(['review_scores_rating', 'number_of_reviews', 'number_of_reviews_ltm'], axis=1)

df_7.shape
#PRICE LIMIT SETTING AND PRICE DISTRIBUTION

df_7.price.describe()






df_7[df_7.price > 1000].head()
df_8 = df_7.drop(df_7[df_7.price > 650].index, axis=0)

df_8 = df_8.drop('maximum_nights', axis=1)

df_8.price.describe()



#df_7 has prices above 500, df_8 does not. there are about 500 listings priced above 500
plt.figure(figsize=(10,10))

sns.violinplot(x=df_8.price, palette = "Set3").set_title("Price Distribution (removed listings $650 and above)", size=16)

sns.despine
df_8.shape
df_8.isnull().sum()

#because there are so few, we will investigate the cases where there is null
null_cases = df_8[df_8.isnull().any(axis=1)]

null_cases.head(25)
null_cases.info()
df_9 = df_8.dropna()

df_9.shape
df_9.head()
df_9.isnull().sum()
df_9.nunique()
# Explain the parts of the violin chart:

# Curve = Normal Disribution

# Bottom of veritcal line = Smallest value

# Top of vertical line = Highest value

# Bottom of the middle box = First quartile

# Top of the middle box = Third quartile

# White dot = mean





plt.figure(figsize=(20,10))

sorted_room = df_9.groupby(['room_type'])['price'].median().sort_values()

sns.violinplot(x=df_9.room_type, y=df_9.price, order=list(sorted_room.index)).set_title("Price by Room Type", size=16)

#sns.despine


plt.figure(figsize=(20,10))



sorted_room = df_9.groupby(['room_type'])['reviews_per_month'].median().sort_values()

sns.violinplot(x=df_9.room_type, y=df_9.reviews_per_month, order=list(sorted_room.index)).set_title("Reviews Per Month by Room Type", size=16)

sns.despine
# Explain skew

df_9.skew()
df_9['minimum_nights'][df_9['minimum_nights'] > 365] = 366
#Predictor variables and Skewed Distribution





f, axes = plt.subplots(2, 4, figsize=(15, 15))

sns.despine(left=True)



sns.distplot(color = sns.color_palette("Set3")[0], a=df_9.price, ax = axes[0,0])

sns.distplot(color = sns.color_palette("Set3")[0], a=df_9.minimum_nights, ax = axes[1,0])

sns.distplot(color = sns.color_palette("Set3")[0], a=df_9.calculated_host_listings_count, ax = axes[0,1])

sns.distplot(color = sns.color_palette("Set3")[0], a=df_9.accommodates, ax = axes[1,1])

sns.distplot(color = sns.color_palette("Set3")[0], a=df_9.reviews_per_month, ax = axes[0,2])

sns.distplot(color = sns.color_palette("Set3")[0], a=df_9.bedrooms, ax = axes[0,3])

sns.distplot(color = sns.color_palette("Set3")[0], a=df_9.bathrooms, ax = axes[1,3])

sns.distplot(color = sns.color_palette("Set3")[0], a=df_9.extra_people, ax = axes[1,2])





df_9.skew(axis=0)
df_9.nunique()

df_9.describe()
df_9.minimum_nights.describe()
df_9.calculated_host_listings_count.describe()
# BINNING VARIABLES (drop beds because we have bedrooms)



df_10 = df_9.drop(['beds', 'accommodates'], axis=1)

df_10['binned_min_nights'] = pd.cut(df_9['minimum_nights'], bins=[0, 1, 2, 10, 1900],

                                                labels = ['oneNight', '2to3','4to10', '11+']

                                                )

df_10.binned_min_nights.value_counts()

df_10.skew()




df_10.calculated_host_listings_count.describe()
#log transforming price, sec deposit, cleaning fee, extra people, reviews_per_month variables



df_10['log_price'] = np.log(df_10['price']*10 + 1)

df_10['log_security'] = np.log(df_10['security_deposit'] + 1)

df_10['log_cleaning'] = np.log(df_10['cleaning_fee']*10 + 1)



df_10['log_extra_people'] = np.log(df_10['extra_people']*10 + 1)

df_10['log_reviews_pm'] = np.log((df_10['reviews_per_month']+ 1) * 10)

df_10['log_bedrooms'] = np.log(df_10['bedrooms'] + 1)

df_10['log_bathrooms'] = np.log(df_10['bathrooms']*10 + 1)

df_10['log_guests_included'] = np.log(df_10['guests_included']*100 + 1)

df_10['log_listings_count'] = np.log(df_10['calculated_host_listings_count']*10 + 1)



df_10.skew()

# Predictor variables and Skewed Distribution



sns.set({'xtick.labelsize': 10, 'ytick.labelsize': 10})



f, axes = plt.subplots(2,1, figsize=(5, 10))

sns.despine(left=True)



sns.distplot(color = sns.color_palette("Set3")[0], a=df_10.reviews_per_month, ax = axes[0]).set_title("Log transformed Reviews Distribution")

sns.distplot(color = sns.color_palette("Set3")[0], a=df_10.log_reviews_pm, ax = axes[1])







df_10.skew(axis=0)
# Predictor variables and Skewed Distribution





f, axes = plt.subplots(2,1, figsize=(18, 9))

sns.despine(left=True)



sns.distplot(color = sns.color_palette("Set3")[0], a=df_10.price, ax = axes[0]).set_title("Log transformed Price Distribution")

sns.distplot(color = sns.color_palette("Set3")[0], a=df_10.log_price, ax = axes[1])







df_10.skew(axis=0)
df_10.room_type.value_counts()
plt.figure(figsize=(15,20))

sns.set({'xtick.labelsize': 20, 'ytick.labelsize': 20})

sorted_hood = df_10.groupby(['neighbourhood_group_cleansed'])['log_price'].median().sort_values()

sns.boxplot(palette = 'Set3', y=df_10.neighbourhood_group_cleansed, x=df_10.log_price, order=list(sorted_hood.index)).set_title("Log(Price) by Neighbourhood", size=20)

sns.despine
plt.figure(figsize=(20,20))

sns.heatmap(df_10.corr(), annot=True, cmap='coolwarm', linewidth=.2)
f, axes = plt.subplots(2, 2, figsize=(10, 10))

sns.set()



sns.distplot(df_10.bedrooms, ax = axes[0,0])

sns.distplot(df_10.log_bedrooms, ax = axes[1,0])

sns.distplot(df_10.log_reviews_pm, ax = axes[1,1])

sns.distplot(df_10.reviews_per_month, ax = axes[0,1])

import descartes

import geopandas as gpd



gjsonFile = "../input/barcelonaairbnbgeojson/neighbourhoods.geojson"

barc_hoods = gpd.read_file(gjsonFile)



barc_hoods.plot(figsize=(10,10), column="neighbourhood_group", cmap = "tab10")
barc_hoods['neighbourhood_group'].value_counts()
barc_hoods.plot(figsize=(20,20), column="neighbourhood_group", cmap='tab10', alpha = .5)

#plt.figure(figsize=(20,20))



sns.scatterplot(x='longitude', 

                y = 'latitude', 

                hue='price', 

                size = 'price', 

                sizes= (20, 600),

                alpha = .8,

                marker=".",

                data = df_10,

                )
barc_hoods.plot(figsize=(10,10), alpha = .5)

#plt.figure(figsize=(20,20))

#here we add lat and longitude lines to plot for context





sns.set({'font.size' : 10})

sns.set_style("whitegrid")

#plot by reviews per month

sns.scatterplot(x='longitude', 

                y = 'latitude', 

                hue='neighbourhood_group_cleansed', 

                alpha = .5,

                marker="o",

                data = df_10,

                cmap='Set3')

                #size = 15)
import folium

from folium.plugins import HeatMap, MarkerCluster



mapp = folium.Map(location=[41.40,2.15], zoom_start=12, figsize=(20,20))

cluster = MarkerCluster().add_to(mapp)

# add a marker for every record in the filtered data, use a clustered view

for each in df_10.iterrows():

    folium.Marker(

        location = [each[1]['latitude'],each[1]['longitude']], 

        clustered_marker = True).add_to(cluster)

  

mapp.save(outfile='map.html')

display(mapp)




max_price_map = df_10['price'].max() #this should be 650

barc_map = folium.Map(location=[41.40, 2.15], zoom_start=12, )



heatmap = HeatMap( list(zip(df_10.latitude, df_10.longitude, df_10.price)),

                 min_opacity = .3,

                 max_val = max_price_map, 

                 radius = 3,

                 blur = 2,

                 max_zoom=1)



folium.GeoJson(barc_hoods).add_to(barc_map)

barc_map.add_child(heatmap)



barc_map.save(outfile="mapp.html")


import folium

from folium.plugins import HeatMap





max_reviews_pm = df_10['reviews_per_month'].max() 

barc_map = folium.Map(location=[41.40, 2.15], zoom_start=12, )



heatmap = HeatMap( list(zip(df_10.latitude, df_10.longitude, df_10.reviews_per_month)),

                 min_opacity = .3,

                 max_val = max_reviews_pm, 

                 radius = 3,

                 blur = 2,

                 max_zoom=1)



folium.GeoJson(barc_hoods).add_to(barc_map)

barc_map.add_child(heatmap)



sns.set()

plt.figure(figsize=(20,10))

sorted_hood_by_price = df_10.groupby(['neighbourhood_group_cleansed'])['log_price'].median().sort_values()



sns.boxplot(x="neighbourhood_group_cleansed",

            y="log_price",

            data=df_10, 

            palette="tab10", 

            order = list(sorted_hood_by_price.index)

           ).set_title("Log(Price) by neighbourhood", size=14)



plt.figure(figsize=(20,10))

sorted_hood_by_price = df_10.groupby(['neighbourhood_group_cleansed'])['reviews_per_month'].median().sort_values()



sns.boxplot(x="neighbourhood_group_cleansed",y="log_reviews_pm",

            data=df_10, palette="tab10", order = list(sorted_hood_by_price.index)

           ).set_title("Log(ReviewsPerMonth) by neighbourhood", size=14)



plt.figure(figsize=(20,10))

sorted_hood_by_price = df_10.groupby(['room_type'])['log_price'].median().sort_values()



sns.boxplot(x="room_type",

            y="log_price",

            data=df_10, 

            palette="tab10", 

            order = list(sorted_hood_by_price.index)

           ).set_title("Log(Price) by Room Type", size=15)

plt.figure(figsize=(20,10))



sorted_hood_by_activity = df_10.groupby(['neighbourhood_group_cleansed'])['log_price'].median().sort_values()

sns.violinplot(x=df_10.neighbourhood_group_cleansed, y=df_10.log_price, order=list(sorted_hood_by_activity.index))

sns.despine
plt.figure(figsize=(20,10))



sorted_hood_by_activity = df_10.groupby(['room_type'])['log_reviews_pm'].median().sort_values()

sns.violinplot(x=df_10.room_type, y=df_10.log_reviews_pm, order=list(sorted_hood_by_activity.index))

sns.despine

#private rooms getting reviewed the most
# UGLY, but potentially helpful plot



# why may we want to do a scatterplot for categorical data??





f, axes = plt.subplots(1, 1, figsize=(10, 15))

sns.despine



sns.scatterplot(x='log_guests_included', 

                y = 'log_price', 

                hue='room_type',

                size = 'log_reviews_pm', 

                sizes= (5, 400),

                alpha = .8,

                palette = "tab10",

                data = df_10).set_title("Price by Bedrooms (weighted by Reviews)")
#appears that reviews_per_month is an important activity heuristic, 

#we can see there is a potential relationship between price and bedrooms, especially in active airbnbs
df_10.info()
df_10.nunique()
#Removing Log Transformed Variables



df_lg = df_10.drop(['price', 

                    'security_deposit', 

                    'cleaning_fee', 

                    'extra_people', 

                    'reviews_per_month', 

                    'bedrooms', 

                    'bathrooms', 

                    'guests_included', 

                    'calculated_host_listings_count'], axis = 1)
df_lg.nunique()
df_lg.head()
df_lg2 = df_lg.drop(['host_id', 'neighbourhood_cleansed', 'city', 'property_type'], axis=1)
df_lg2.nunique()
df_lg2.skew()
df_lg2.dtypes
#MODELING STARTS

X = df_lg2

X = pd.get_dummies(data=X, drop_first = True)



Y = X['log_price']

X2 = X.drop('log_price', axis = 1)

X = X2



X.nunique()
                          
#save cleaned X and Y for further analysis in R or elsewhere



X.to_csv("./Xcleaned.csv")

Y.to_csv("./Ycleaned.csv")
#BASELINE MULTIVARIATE REGRESSION MODEL



from sklearn import linear_model

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor





X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state = 40)



est = sm.OLS(Y_train, X_train).fit()

display(est.summary())



# explore Variance inflation factor of each variable in regression model



# big VIF == bad



# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]



vif["features"] = X_train.columns



vif.round(1)
from sklearn.metrics import mean_squared_error, r2_score



#EVALUATION OF MODEL



predicted = est.predict(X_test)



print("MSE of model when comparing Y_test and predicted is %lf" %mean_squared_error(Y_test, predicted))



    

fig, ax = plt.subplots(figsize=(7,7))

ax.scatter(Y_test, predicted)

ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4, alpha=.5)

ax.set_xlabel('measured')

ax.set_ylabel('predicted')

ax.set_title("Baseline Train Predictions")

plt.show()
plot_lm_1 = plt.figure(1)

plot_lm_1.set_figheight(5)

plot_lm_1.set_figwidth(14)



plot_lm_1.axes[0] = sns.residplot(est.fittedvalues, y=Y_train, data=X_train, 

                          lowess=True, 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



plot_lm_1.axes[0].set_title('Residuals vs Fitted')

plot_lm_1.axes[0].set_xlabel('Fitted values')

plot_lm_1.axes[0].set_ylabel('Residuals')


model_fitted_y = est.fittedvalues

# model residuals

model_residuals = est.resid

# normalized residuals

model_norm_residuals = est.get_influence().resid_studentized_internal

# absolute squared normalized residuals

model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# absolute residuals

model_abs_resid = np.abs(model_residuals)

# leverage, from statsmodels internals

model_leverage = est.get_influence().hat_matrix_diag



QQ = sm.ProbPlot(model_norm_residuals)

plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)



plot_lm_2.set_figheight(6)

plot_lm_2.set_figwidth(6)



plot_lm_2.axes[0].set_title('Normal Q-Q')

plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')

plot_lm_2.axes[0].set_ylabel('Standardized Residuals');



# annotations

abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)

abs_norm_resid_top_3 = abs_norm_resid[:3]



for r, i in enumerate(abs_norm_resid_top_3):

    plot_lm_2.axes[0].annotate(i, 

                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],

                                   model_norm_residuals[i]));
coef = pd.Series(est.params, index = X_train.columns)





imp_coef = pd.concat([coef.sort_values().head(18),

                     coef.sort_values().tail(18)])

plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Baseline Model")