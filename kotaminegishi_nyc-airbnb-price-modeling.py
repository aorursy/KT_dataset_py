import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.metrics import r2_score



%matplotlib inline
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

df.shape
df.head() 
df.isnull().sum()
df.name = df.name.fillna('no_name')

df.reviews_per_month = df.reviews_per_month.fillna(0)
# Set date for relevant variables

df["last_review"] = pd.to_datetime(df.last_review)



# Derive a year-month truncated variable 

df['last_review_YrMo'] = df.last_review.dt.strftime('%Y-%m')



YrMo_count = df.last_review_YrMo.value_counts().sort_index()

YrMo_count
sns.relplot(data = YrMo_count[YrMo_count.index>"2018-01"],

            aspect = 3, kind='line');

plt.xticks(rotation = 45);
sns.distplot(df.price[df.price<1000]).set_title("Distribution of Price");
(sns.distplot(df.minimum_nights[df.minimum_nights<30],

              bins=30, kde=False)

 .set_title("Distribution of Minimum Nights"));
(sns.distplot(

    df.reviews_per_month[df.reviews_per_month<=30],  

    kde=False)

 .set_title("Distribution of Reviews per Month"));
# Subset data

df = df[(df.price < 500) & (df.minimum_nights <= 5) &

        (df.reviews_per_month <= 30) &

        ((df.last_review >= "2019-01-01") |

         (df.last_review.isnull()))]

len(df)
import matplotlib.image as mpimg

plt.figure(figsize=(14,10))



# Read image 

nyc_img = mpimg.imread('../input/new-york-city-airbnb-open-data/New_York_City_.png', 0) 



# Scale the image based on the latitude and longitude max and mins

plt.imshow(nyc_img, zorder=0,

           extent=[-74.258, -73.7, 40.49, 40.92])



# Add a data layer

sns.scatterplot(x = 'longitude', y='latitude', 

                hue='neighbourhood_group', 

                data = df);

plt.legend(loc='upper left',fontsize=16);
# Show top 12 neighborhoods by data count  

df.neighbourhood.value_counts()[:12]
top12_nbhd = df.neighbourhood.value_counts()[:12].index

top12_data = (df.set_index('neighbourhood')

              .loc[top12_nbhd]

              .reset_index())

len(top12_data)
plt.figure(figsize=(14,10))



# Scale the image based on the latitude and longitude max and mins

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49, 40.92])



# Add a data layer

filled_markers = ('o', 'v', '^', '<', '>', 's', '*', 'H', 'P', 'X', 'P', 'D')

sns.scatterplot(x = 'longitude', y='latitude', 

                hue='neighbourhood', 

                style='neighbourhood', 

                markers=filled_markers,

                data = top12_data)

plt.legend(loc='upper left',fontsize=16,

           title='Top 12 Neighbourhoods', title_fontsize=20);
# Check neighbourhood_group for the top 12 neighbourhood

pd.crosstab(top12_data.neighbourhood, 

            top12_data.neighbourhood_group)
# Barplot of listings by neighborhood

sns.barplot(x = 'id', y = 'neighbourhood', orient='h',

            data = (top12_data[['neighbourhood','id']]

                    .groupby('neighbourhood')

                    .count()

                    .sort_values('id', ascending=False)

                    .reset_index())

           )

plt.xlabel('Listings')

plt.title('Number of Listings in the Top 12 Neighborhoods');
#  Barplot of listings by neighborhood and room_type 

(pd.crosstab(top12_data.neighbourhood, top12_data.room_type)

 .reindex(top12_nbhd) # order by the total counts

 .iloc[::-1] # reverse the order 

).plot.barh(stacked=True)

plt.xlabel('Listings')

plt.title('Number of Listings in the Top 12 Neighborhoods');
sns.boxplot(x = 'room_type', y = 'price', hue = 'room_type', 

            data= df[df.neighbourhood == top12_nbhd[0]])

plt.title(top12_nbhd[0] + ' Listings');
from statsmodels.distributions.empirical_distribution import ECDF



# Plot a neighborhood cumulative distribution of listing prices by room type 

def map_nbhd_listings(data, nbhd, ax=None):

    # pass axis 

    ax = ax or plt.gca()

    



    

    # Make a subset for the provided neighborhood

    loc_df = data[data.neighbourhood == nbhd] 

    nbhd_group = loc_df.neighbourhood_group.iloc[0]

    

    # Find room types 

    room_types = loc_df.room_type.value_counts().sort_index().index

    room_types_len = loc_df.room_type.value_counts().sort_index().values



    ecdf1 = ECDF(loc_df[loc_df.room_type == room_types[0]].price)

    ecdf2 = ECDF(loc_df[loc_df.room_type == room_types[1]].price)

    ecdf3 = ECDF(loc_df[loc_df.room_type == room_types[2]].price)



    # Plot the cdf 

    #  where y-axis is the price

    #        x-axis is the prob * the number of observation 

    ax.plot(ecdf1.y*room_types_len[0], ecdf1.x, color='blue', 

            linewidth=3,

            label = room_types[0])

    ax.plot(ecdf2.y*room_types_len[1], ecdf2.x, color='darkorange', 

             linestyle='dashed', linewidth=3,

            label = room_types[1])

    ax.plot(ecdf3.y*room_types_len[2], ecdf3.x, color='green',

             linestyle='dashed', linewidth=3, 

            label = room_types[2])

    [ax.axhline(y=y,color='gray',linestyle='--') for y in range(50,550,50)]

    ax.set_ylabel('Price, $')

    ax.set_xlabel('Listings')

    ax.set_title(nbhd + ' (' + nbhd_group + ')' + ' Supply')

    ax.legend(loc='upper left')

    return(ax) 

    

map_nbhd_listings(df, nbhd = top12_nbhd[0]);
# Generate supply curves for the top 12 neighborhoods 

fig, axs = plt.subplots(6, 2)

fig.set_size_inches(6*2, 3*6) 

fig.tight_layout(pad=2.0)



for n in range(0,12):

    # Create axis references  

    pos_1 = n//2

    pos_2 = np.mod(n,2)

    

    # Add a plot into subplots figure

    map_nbhd_listings(df, nbhd = top12_nbhd[n], 

                      ax = axs[pos_1, pos_2])

    

    # Remove excess axis labels 

    if pos_1 < 5:

        axs[pos_1, pos_2].set(xlabel='')

    if pos_2 ==1:

        axs[pos_1, pos_2].set(ylabel='')
# Tabulate `calculated_host_listings_count`

df.calculated_host_listings_count.value_counts().sort_index()
# Calulate current host listings count 

current_listings = (df.host_id.value_counts()

                    .reset_index()

                    .rename(columns = 

                            {'index':'host_id',

                             'host_id':'current_listings'}))



df = df.set_index('host_id').join(current_listings.set_index('host_id')).reset_index()
# Tabulate `current_listings`

df.current_listings.value_counts().sort_index()
# Add number of hosts column to the previous output

tbl_current_listings = (df.current_listings

                        .value_counts().sort_index().reset_index())

tbl_current_listings['num_hosts'] = (

    (tbl_current_listings.current_listings

    /tbl_current_listings['index']).astype(int)

)

tbl_current_listings
# Define `host_class` variable

df['host_class'] = pd.cut(df.current_listings,

                          bins = np.array([0,1, 5, 12, 200]), 

                          labels = ['single (1)','multi (2-5)', 

                                   'serial (6-12)', 'mega (13+)']

                         )



# Tabulate `host_class`

df.host_class.value_counts()
# Define `mega_host_id` by the last 3 digits of host_id + num of listings 

df['mega_host_id'] = (df.host_id.astype(str).str[-3:] 

                      + '_#list_' 

                      + df.current_listings.astype(str)) 

df['mega_host_id'].loc[df.host_class != 'mega (13+)'] = 'NA'



# Mega host id, listings, mega_host_id

mega_hosts = (df[df.host_class == 'mega (13+)']

              [['host_id', 'current_listings', 'mega_host_id']]

              .drop_duplicates().sort_index()

              .sort_values(by=['current_listings'], 

                           ascending = False)

             )

mega_hosts_order = mega_hosts.mega_host_id

mega_hosts
# Define a 4-digit (decimal places) longitude and latitude

# 4-digit: 0.0001 degree = 11.132 meters accuracy 

# 5-digit: 0.00001 degree = 1.1132 meters accuracy 

df['longitude4'] = df.longitude.round(4)

df['latitude4'] = df.latitude.round(4)
# At host_listings <= 5, how many "identical" listing locations by the host exist? 

(df[df.current_listings ==5]

 [['id','host_id','latitude4','longitude4']]

 .pivot_table(index=['host_id','latitude4','longitude4'],

              aggfunc='size')

 .value_counts().sort_index()

)
# Example 1: longitudes and latitudes of a host with 5 listings  

(df[['id','host_id','longitude4', 'latitude4']]

 [df.host_id == 51038])
# Example 2: longitudes and latitudes of a host with 5 listings  

(df[['id','host_id','longitude4', 'latitude4']]

 [df.host_id == 116382])
#  Barplot of listings by host_class and room_type 

(pd.crosstab(df.host_class, df.room_type)

).plot.bar(stacked=True)

plt.xlabel('Host Class')

plt.ylabel('Listings')

plt.xticks(rotation=45);

plt.title('Number of Listings by Host Class and Room Type');
#  Barplot of listings by host_class and room_type, in percent 

(pd.crosstab(df.host_class, df.room_type, 

            normalize= 'index')

).plot.bar(stacked=True)

plt.xlabel('Host Class')

plt.ylabel('Share of listings')

plt.xticks(rotation=45)

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.title('Share of Room Types by Host Class');
#  Barplot of listings by top hosts and room_type, in percent 

(pd.crosstab(df[df.host_class=="mega (13+)"].mega_host_id, 

             df[df.host_class=="mega (13+)"].room_type, 

            normalize= 'index')

 .reindex(mega_hosts_order).iloc[::-1] 

 # sort by listings, high to low

).plot.barh(stacked=True)

plt.ylabel('Mega Host ID')

plt.xlabel('Share of listings')

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.title('Share of Room Types by Mega Host');
#  Barplot of listings by top hosts and neighborhood groups, in percent 

(pd.crosstab(df[df.host_class=="mega (13+)"].mega_host_id, 

             df[df.host_class=="mega (13+)"].neighbourhood_group, 

            normalize= 'index')

  .reindex(mega_hosts_order).iloc[::-1] # sort by listings, high to low

).plot.barh(stacked=True)

plt.ylabel('Mega Host ID')

plt.xlabel('Share of listings')

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.title('Share of Neighborhood Groups of Listings by Mega Host');
#  Barplot of listings by top hosts and neighborhood, in percent 

(pd.crosstab(df[df.host_class=="mega (13+)"].mega_host_id, 

             df[df.host_class=="mega (13+)"].neighbourhood, 

            normalize= 'index')

  .reindex(mega_hosts_order).iloc[::-1] # sort by listings, high to low

).plot.barh(stacked=True, cmap='rainbow')

plt.ylabel('Mega Host ID')

plt.xlabel('Share of listings')

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.title('Share of Neighborhoods of Listings by Mega Host');
# Count mega hosts for neighborhoods where they have their majority of listings 

(pd.crosstab(df[df.host_class=="mega (13+)"].mega_host_id, 

             df[df.host_class=="mega (13+)"].neighbourhood, 

            normalize= 'index')

).apply(lambda x: x>.5).sum().sort_values(ascending = False) 
# Violin plot of price by host class 

ax = sns.violinplot(x = 'host_class', y = 'price', data = df)

[ax.axhline(y=y,color='gray',linestyle='--') for y in range(0,600,50)]

plt.title('Violin plot of price by host class');
# Boxplot of price by mega host 

ax = sns.boxplot(y = 'mega_host_id', x = 'price', 

                 orient='h',

                 order= mega_hosts_order,

                 data = df[df.host_class=="mega (13+)"])

[ax.axvline(x=x,color='gray',linestyle='--') for x in range(0,600,50)]

plt.title('Boxplot of price by mega host');
# Violin plot of price by host class 

ax = sns.violinplot(x = 'host_class', y = 'reviews_per_month', 

                    data = df[df.reviews_per_month<=20])

[ax.axhline(y=y,color='gray',linestyle='--') for y in range(0,25,5)]

plt.title('Violin plot of reviews per month by host class');
# Boxplot of reviews_per_month by mega host 

ax = sns.boxplot(y = 'mega_host_id', x = 'reviews_per_month',

                 orient='h',

                 order= mega_hosts_order,

                 data = df[df.host_class=="mega (13+)"])

[ax.axvline(x=x,color='gray',linestyle='--') for x in range(0,10, 2)]

plt.title('Boxplot of reviews per month by mega host');
# Violin plot of availability by host class 

ax = sns.violinplot(x = 'host_class', y = 'availability_365', 

                    data = df)

[ax.axhline(y=y,color='gray',linestyle='--') for y in range(0,400,50)]

plt.title('Violin plot of availability by host class');
# Boxplot of availability_365 by mega host 

ax = sns.boxplot(y = 'mega_host_id', x = 'availability_365', orient='h',

                 order= mega_hosts_order,

                 data = df[df.host_class=="mega (13+)"])

[ax.axvline(x=x,color='gray',linestyle='--') for x in range(0,400,50)]

plt.title('Boxplot of availability by mega host');
# Violin plot of availability by host class 

ax = sns.violinplot(x = 'host_class', y = 'minimum_nights', 

                    data = df)

plt.title('Violin plot of minimum nights by host class');
# Reviews per Month vs Price

sns.lmplot(x = 'reviews_per_month', y = 'price', lowess = True, 

           scatter=False,

           hue = 'room_type', data = df)

plt.title('Reviews per Month vs Price by Room Type');
# Minimum Nights vs Price

sns.lmplot(x = 'minimum_nights', y = 'price', lowess = True, 

           scatter=False,

           hue = 'room_type', data = df)

plt.title('Minimum Nights vs Price by Room Type');
# Availability vs Price

sns.lmplot(x = 'availability_365', y = 'price', lowess = True, 

           scatter=False,

           hue = 'room_type', data = df)

plt.title('Availability vs Price by Room Type');
# To use discrete categories of minimum nights, define a string variable

df['minimum_nights_str'] = df.minimum_nights.astype(str)
# Last Review Year-Month vs Price

sns.relplot(x = 'last_review_YrMo', y = 'price',

            hue = 'room_type', kind ='line',

            palette=['green','blue','red'],

            data = ((df.pivot_table(index ='last_review_YrMo',

                                   values='price', aggfunc='mean',

                                   columns = ['room_type'])

                    ).unstack()

                    .reset_index()

                    .rename(columns={0:'price'})

                   )

)

plt.xticks(rotation=45)

plt.title('Last Review Year-Month vs Price by Room Type');
# Add a string version of last_review_YrMo

df['last_review_YrMo_str'] = df.last_review_YrMo.astype('str')
# word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.name)



# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show();
# Define a list of keywords 

keywords = ['amazing','luxury','huge','large','cozy',

            'central park','spacious','comfy','house',

            'bright','comfortable','chic','beautiful',

           'private','NYC','modern','gorgeous','quiet',

           'clean','suite','loft','times square','oasis',

           'brand new','soho','sunny','manhattan',

           'lovely','studio','duplex',"charming"]



# Generate an indicator/dummy variable for each keyword

for kw in keywords:

    df['kw_'+ str.replace(kw, ' ','_')] = (df.name

                                           .str.contains(kw, case=False)*1)

    

# Generate a string variable containing keywords to be used in smf-formula

kw_names = ['kw_' + str.replace(kw, ' ','_') for kw in keywords] 

fmla_kw = str('')

for kw in kw_names:

 fmla_kw = fmla_kw + ' + ' + kw 
# Calculate a neighborhood average price for each room type  

df['nbhd_price'] = (df.groupby(['neighbourhood', 'room_type'])

                    .price.transform(np.mean))
nbhd_counts = df.neighbourhood.value_counts() 

nbhd_use = nbhd_counts[nbhd_counts>=30].index



# Print neibhorhoods to include 

[print(x, end=', ') for x in nbhd_use]; 
# Print neibhorhoods to exclude 

[print(x, end=', ') for x in df.neighbourhood.unique() if x not in nbhd_use];
# Define a subset that excludes neibhorhoods with less than 30 listings 

print(len(df))



df = df.set_index('neighbourhood').loc[nbhd_use].reset_index()

print(len(df))
# price distribution

sns.distplot(df.price);
# log(price) distribution

sns.distplot(np.log(df.price + 1)); # add 1 for avoiding log(0)
# Define log-transformed prices  

df['ln_nbhd_price'] = np.log(df.nbhd_price + 1) 

df['ln_price'] = np.log(df.price + 1)
# a simple OLS regression for price 

rlt1 = smf.ols('price ~ nbhd_price', data=df)

rlt1.fit().summary2()
print('Total Mean Squared Error (MSE):', rlt1.fit().mse_total.round())

print('Residual MSE:', rlt1.fit().mse_resid.round())
# data subset for Bedford-Stuyvesant and Williamsburg

rlt1a = smf.ols('price ~ nbhd_price', 

                data=df[

                    df.neighbourhood.isin(

                        ["Bedford-Stuyvesant", "Williamsburg"])])

rlt1a.fit().summary2()
print('Total MSE:', rlt1a.fit().mse_total.round())

print('Residual MSE:', rlt1a.fit().mse_resid.round())
# data subset for Hell's Kitchen and Midtow

rlt1b = smf.ols('price ~ nbhd_price', 

                data=df[

                    df.neighbourhood.isin(

                        ["Hell's Kitchen", "Midtow"])])

rlt1b.fit().summary2()
print('Total MSE:', rlt1b.fit().mse_total.round())

print('Residual MSE:', rlt1b.fit().mse_resid.round())
# a simple OLS regression for log(price) 

rlt2 = smf.ols('ln_price ~ ln_nbhd_price', data=df)

rlt2.fit().summary2()
print('Total MSE:', rlt2.fit().mse_total.round(3))

print('Residual MSE:', rlt2.fit().mse_resid.round(3))
# Define a dummy variable for the mega-host class

df['host_class_mega'] = (df.host_class == "mega (13+)") * 1
rlt3 = smf.ols('ln_price ~ ln_nbhd_price +'

               + 'room_type:minimum_nights_str +' 

               + 'room_type:availability_365 + '

               + 'room_type:last_review_YrMo_str +'

               + 'host_class + mega_host_id', 

               data=df)

rlt3.fit().summary2()
print('Residual MSE:', rlt3.fit().mse_resid.round(3))
rlt4 = smf.ols('ln_price ~ ln_nbhd_price +'

               + 'room_type:minimum_nights_str +' 

               + 'room_type:availability_365 +'

               + 'room_type:last_review_YrMo_str +'

               + 'host_class + mega_host_id' + fmla_kw, 

               data=df)

rlt4.fit().summary2()
print('Residual MSE:', rlt4.fit().mse_resid.round(3))
rlt5 = smf.ols('ln_price ~ ln_nbhd_price +' 

               + 'room_type:minimum_nights_str +' 

               + 'room_type:availability_365 +'

               + 'room_type:last_review_YrMo_str +'

               + 'host_class + mega_host_id' + fmla_kw

               + '+ neighbourhood +'

               + 'neighbourhood:longitude4 +'

               + 'neighbourhood:latitude4', 

               data=df)

rlt5.fit().summary().tables[0] # show only the stats table
print('Residual MSE:', rlt5.fit().mse_resid.round(3))
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.neighbors import KNeighborsRegressor
# Prepare model matrix via smf.ols-formula  

knn1_model = smf.ols('ln_price ~ 0 + longitude4 + latitude4 + ' 

                     + 'room_type + host_class', 

               data=df)

knn1_model_matrix = pd.DataFrame(knn1_model.fit().model.exog)



# Show variables 

knn1_model.exog_names
# Set up data split into training and test sets

np.random.seed(2020)



n_sample = len(df)

random_indices = np.random.permutation(n_sample)

num_training_samples = int(n_sample*0.75)



X_vars = knn1_model_matrix

Y_var = df.ln_price



x_train = X_vars.iloc[random_indices[:num_training_samples]]

x_test = X_vars.iloc[random_indices[num_training_samples:]]



y_train = Y_var.iloc[random_indices[:num_training_samples]]

y_test = Y_var.iloc[random_indices[num_training_samples:]]
# Function to show main test results

def test_stats(y_test, pred, digits=3): 

    print("Test MSE: ", 

          round(mean_squared_error(y_test, pred), digits))

    print("Test R2 score: ", 

          round(r2_score(y_test, pred), digits))
# Run KNN 

knn1 = KNeighborsRegressor(n_neighbors=20, weights='uniform')

knn1.fit(x_train, y_train)

knn1_pred = knn1.predict(x_test)

test_stats(y_test, knn1_pred)
# Try n_neighbors= 4, 5, ... , 30 

knn1_results_MSE, knn1_results_R2  = [], []

num_neighbors = range(4,30)

for n in num_neighbors:

    knn0 = KNeighborsRegressor(n_neighbors = n)

    knn0.fit(x_train, y_train)

    knn0_pred = knn0.predict(x_test)

    knn1_results_MSE.append(mean_squared_error(y_test, knn0_pred))

    knn1_results_R2.append(r2_score(y_test, knn0_pred))
# Plot the performance (MSE) along n_neighbors = 4, 5, ... , 30 

plt.plot(num_neighbors, knn1_results_MSE)

plt.xlabel('Number of neighbors')

plt.ylabel('Mean Square Error')

plt.title("Nearest K Neighbors Regression Performance, MSE")

plt.show()
# Plot the performance (R2) along n_neighbors = 4, 5, ... , 30 

plt.plot(num_neighbors, knn1_results_R2)

plt.xlabel('Number of  neighbors')

plt.ylabel('R squared')

plt.title("Nearest K Neighbors Regression Performance, R2")

plt.show()
# Use the OLS model (rlt5) above

X_vars2 = pd.DataFrame(rlt5.fit().model.exog)



x_train2 = X_vars2.iloc[random_indices[:num_training_samples]]

x_test2 = X_vars2.iloc[random_indices[num_training_samples:]]
# Assessing the OLS performance with the same data split 

ols2 = LinearRegression()

ols2.fit(x_train2, y_train)

ols2_pred = ols2.predict(x_test2)

test_stats(y_test, ols2_pred)

print("number of features used: ", np.sum(ols2.coef_!=0))
# Add the knn1 prediction to the data

df['pred_knn1'] = knn1.predict(X_vars)



# Run KNN + OLS model with 'pred_knn1' in place of 'ln_nbhd_price'

rlt6 = smf.ols('ln_price ~ pred_knn1 +'

               + 'room_type:minimum_nights_str +' 

               + 'room_type:availability_365 +'

               + ' room_type:last_review_YrMo_str +'

               + 'host_class + mega_host_id' + fmla_kw

               + '+ neighbourhood +'

               + 'neighbourhood:longitude4 +'

               + 'neighbourhood:latitude4', 

               data=df)

rlt6.fit().summary().tables[0] # show only the stats table
print('Residual MSE:', rlt6.fit().mse_resid.round(3))
# Use the KNN + OLS model (rlt6) above

X_vars3 = pd.DataFrame(rlt6.fit().model.exog)



x_train3 = X_vars3.iloc[random_indices[:num_training_samples]]

x_test3 = X_vars3.iloc[random_indices[num_training_samples:]]
ols3 = LinearRegression()

ols3.fit(x_train3, y_train)

ols3_pred = ols3.predict(x_test3)

test_stats(y_test, ols3_pred)

print("number of features used: ", np.sum(ols3.coef_!=0))
# LASSO with high penality parameter, alpha = 0.1

lasso1 = Lasso(alpha = 0.1)

lasso1.fit(x_train3, y_train)

lasso1_pred = lasso1.predict(x_test3)

test_stats(y_test, lasso1_pred)

print("number of features used: ", np.sum(lasso1.coef_!=0))
# LASSO with medium penality parameter, alpha = 0.01

lasso2 = Lasso(alpha = 0.01)

lasso2.fit(x_train3, y_train)

lasso2_pred = lasso2.predict(x_test3)

test_stats(y_test, lasso2_pred)

print("number of features used: ", np.sum(lasso2.coef_!=0))
# LASSO with low penality parameter, alpha = 0.001

lasso3 = Lasso(alpha = 0.001)

lasso3.fit(x_train3, y_train)

lasso3_pred = lasso3.predict(x_test3)

test_stats(y_test, lasso3_pred)

print("number of features used: ", np.sum(lasso3.coef_!=0))
# LASSO with very low penality parameter, alpha = 0.0001

lasso4 = Lasso(alpha = 0.00025)

lasso4.fit(x_train3, y_train)

lasso4_pred = lasso4.predict(x_test3)

test_stats(y_test, lasso4_pred)

print("number of features used: ", np.sum(lasso4.coef_!=0))
# Extract the coefficients from the LASSO estimate above

lasso3_coef = pd.DataFrame(

    data = {'variable': rlt6.exog_names,

            'coef': lasso3.coef_})

lasso3_coef
# Remove zero-coefficients for keywords

lasso3_coef_kw = lasso3_coef[

    (lasso3_coef.variable.str.startswith("kw_")) &

   ((lasso3_coef.coef > 0) | (lasso3_coef.coef < 0))

]

lasso3_coef_kw.variable = (lasso3_coef_kw.variable

                           .str.replace('kw_', ''))

lasso3_coef_kw
# Plot the relevant keywords

ax = sns.barplot(y = 'variable', x='coef', orient='h', 

            data = lasso3_coef_kw)

[ax.axvline(x=x,color='gray',linestyle='--') for x in np.arange(-0.15,0.25,0.05)]

plt.xlabel("Keyword in the listing's name")

plt.xlabel('Predicted effect on price, percentage change')

plt.title("Estimated Associatons between Keywords and Price");
# Extract non-zero mega_host_id

lasso3_coef[

    (lasso3_coef.variable.str.startswith("mega_host_id")) &

   ((lasso3_coef.coef > 0) | (lasso3_coef.coef < 0))

]
# Check all coefficient estimates for mega_host_id 

lasso3_coef[(lasso3_coef.variable.str.startswith("mega_host_id"))]
lasso3_coef[(lasso3_coef.variable.str.startswith("host_class"))]
rlt8 = smf.ols('ln_price ~ ln_nbhd_price + host_class', data=df)

rlt8.fit().summary2()
rlt8 = smf.ols('ln_price ~ ln_nbhd_price + host_class +' 

               + 'room_type:availability_365', data=df)

rlt8.fit().summary2()
# Extract availability coefficients 

lasso3_coef[

    (lasso3_coef.variable.str.contains("availability_365"))]