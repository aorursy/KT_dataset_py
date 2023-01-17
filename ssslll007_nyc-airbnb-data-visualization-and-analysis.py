import pandas as pd

import numpy as np

import matplotlib

#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno
# Pycharm configuration to display the whole dataframe printed in the "run" output.

pd.set_option('display.width', 400)

pd.set_option('display.max_columns', 10)
airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
print('number of samples: ',airbnb.shape[0])

print('number of columns: ',airbnb.shape[1])
print('Dataframe details: \n')

print(airbnb.info(verbose=True))

print('\nNull values in dataset:\n')

print(airbnb.isnull().sum().sort_values(ascending=False))

print('\nPercentage of null values in last_review column: ',round(airbnb['last_review'].isnull().sum()/len(airbnb)*100,2),'%')
fig, ax = plt.subplots(figsize=(17,6))

plt.title('Null values in last_review and reviews_per_month', fontsize=15)

sns.heatmap(airbnb[['last_review','reviews_per_month']].isnull(), cmap="Blues", yticklabels=False, ax=ax, cbar_kws={'ticks': [0, 1]})

plt.show()
# Let's visualize the columns with missing values to see how they are distributed. 

msno.matrix(airbnb)

plt.show()
msno.heatmap(airbnb)

plt.show()
# Also it seems column number_of_reviews usually has a value of 0 when last_review is null. So column last_review may be dropped

airbnb.drop('last_review',axis=1,inplace=True)
# And we can fillna the other columns that have null values

airbnb['reviews_per_month'].fillna(value=0,inplace=True)

airbnb['name'].fillna(value='$',inplace=True)

airbnb['host_name'].fillna(value='#',inplace=True)
# Before we move on, there are a few listings with zero listing price. What's the purpose of listing a place asking for 0 dollars? Are those legit data entries? Let's take a look.

zero_price = airbnb[airbnb['price']==0]

print(zero_price.head(3))
# There are 11 listings with 0 prices, yet nothing looks special about those listings. So let's get rid of these 0 priced listings as they can be misleading.

airbnb = airbnb[airbnb['price']>0]
# Check if all null values are handled

print('Null values in dataset:\n')

print(airbnb.isnull().sum().sort_values(ascending=False))

# Learn about dtypes of each feature in the dataframe

print('\nData types: \n')

print(airbnb.info(verbose=True))

# Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution

print('\nData description: \n')

print(airbnb.describe())
# A quick overview of correlations between features using heatmap

plt.figure(figsize=(12,8))

plt.xticks(rotation=45)

plt.yticks(rotation=45)

sns.heatmap(airbnb.corr(),annot=True,linewidths=0.1,cmap='Reds')
# Number of listings per room type

print(airbnb['room_type'].value_counts())

sns.countplot(x='room_type',data=airbnb,palette='viridis')

plt.title('No of Listings per Room Type')
# Let's see how many listings a host usually has

def listing_count(count):

    if count == 1:

        return 'only 1'

    elif count<=3:

        return '2-3'

    elif count<=5:

        return '4-5'

    elif count<=10:

        return '6-10'

    elif count<=20:

        return '11-20'

    elif count<=30:

        return '21-30'

    elif count <= 50:

        return '31-50'

    elif count<=80:

        return '51-80'

    else:

        return '81+'

airbnb['host_listings_count_group'] = airbnb['calculated_host_listings_count'].apply(listing_count)

g = sns.countplot(x='host_listings_count_group',data=airbnb,order=['only 1','2-3','4-5','6-10','11-20','21-30','31-50','51-80','81+'],palette='viridis')

plt.title('Listing Ownership Distribution')

total = float(len(airbnb))

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x() + p.get_width() / 2.,

            height + 3,

            '{:1.1f}%'.format(height / total*100),

            ha="center")
# We can change the above visualization to a pie chart just for practice purpose

listing_counts = airbnb.groupby('host_listings_count_group').agg('count')

label = listing_counts['id'].sort_values(ascending=False).index

size = listing_counts['id'].sort_values(ascending=False)

exp = (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8)

fig, ax = plt.subplots()

wedges, texts, autotexts = ax.pie(size, explode=exp, labels=label, autopct='%1.1f%%',shadow=False, startangle=0)

ax.set(title='Listing Ownership Distribution')

plt.setp(autotexts,size=8,weight="bold")

ax.axis('equal')
# number of listings per neighborhood groups

print(airbnb['neighbourhood_group'].value_counts(ascending=False))

airbnb['neighbourhood_group'].value_counts(ascending=False).plot.bar()

plt.title('No of Listings Per Neighborhood Group')
# top 3 neighborhoods with most number of listings in each neighborhood groups

for group_name in airbnb['neighbourhood_group'].unique():

    neighbor = airbnb[airbnb['neighbourhood_group']==group_name]['neighbourhood'].value_counts(ascending=False)[:3]

    print('{} top 3 neighborhoods with corresponding count of listings:\n{} \n'.format(group_name,neighbor))
# number of listings per neighborhood group, categorized by room type

sns.countplot(x='neighbourhood_group',data=airbnb,hue='room_type',palette='viridis')

plt.title('Number of listings in each neighborhood per room types')
# number of listings per neighborhood, categorized by room type, sort in DESC order by count of listings

listing_per_neighbor = airbnb.groupby(['room_type','neighbourhood'],sort=False)['id'].agg([('count','count')]).reset_index().sort_values(by=['room_type','count'],ascending=[True,False])

print(listing_per_neighbor)
# top 10 neighborhoods with most number of listings per each room type

top10 = listing_per_neighbor.groupby(['room_type']).apply(lambda x: x.nlargest(10,'count'))

print(top10)
# visualize top 10 neighborhoods per each room type

fig,axes = plt.subplots(nrows=1,ncols=3,sharey=True,figsize=(18,6))

fig.suptitle('Top 10 Neighborhoods per Room Type', fontsize=16)

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

sns.barplot(x='neighbourhood',y='count',data=top10[top10['room_type']=='Entire home/apt'],ax=axes[0],palette='viridis')

axes[0].set_title('Entire home/apt')

sns.barplot(x='neighbourhood',y='count',data=top10[top10['room_type']=='Private room'],ax=axes[1],palette='viridis')

axes[1].set_title('Private room')

sns.barplot(x='neighbourhood',y='count',data=top10[top10['room_type']=='Shared room'],ax=axes[2],palette='viridis')

axes[2].set_title('Shared room')
# price distribution per neighborhood group, a quick overview

sns.violinplot(x='neighbourhood_group',y='price',data=airbnb,palette='viridis')

plt.title('Price Distribution per Neighborhood Group')
# let's get rid of the long tails and take a closer look

g = sns.boxplot(x='neighbourhood_group',y='price',data=airbnb,palette='viridis')

g.set(yscale='log')

plt.title('Price Distribution per Neighborhood Group')
# detailed price statistics in each neighborhood group, for example min/max, 25/50/75 percentile

price_stat = pd.DataFrame()

for group in airbnb['neighbourhood_group'].unique():

    prices = airbnb[airbnb['neighbourhood_group']==group][['price']]

    stats = prices.describe(percentiles=[.25,.5,.75])

    stats = stats.iloc[1:]

    stats.reset_index(inplace=True)

    stats.rename(columns={'index': 'Stats','price': group}, inplace=True)

    #print(stats)

    price_stat = pd.concat([price_stat,stats],axis=1)

price_stat = price_stat.loc[:,~price_stat.columns.duplicated()].set_index('Stats')

print(price_stat)
# average price of listings in each neighborhood

price_by_nei = airbnb.groupby(['neighbourhood_group','neighbourhood'],sort=False)['price'].agg([('price_avg','mean')]).reset_index().sort_values(by=['neighbourhood_group','price_avg'],ascending=[True,False])

highest10 = price_by_nei.groupby(['neighbourhood_group']).apply(lambda x: x.nlargest(10,'price_avg'))

print(highest10)
# visualize top 10 neighborhoods with highest average price within each neighborhood group

fig,axes = plt.subplots(nrows=1,ncols=5,sharey=True,figsize=(25,6))

fig.suptitle('Top 10 Neighborhoods with Highest Average Listing Price', fontsize=16)

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

    ax.yaxis.label.set_visible(False)

sns.pointplot(x='neighbourhood',y='price_avg',data=highest10[highest10['neighbourhood_group']=='Manhattan'],ax=axes[0],palette='viridis')

axes[0].set_title('Manhattan')

sns.pointplot(x='neighbourhood',y='price_avg',data=highest10[highest10['neighbourhood_group']=='Brooklyn'],ax=axes[1],palette='viridis')

axes[1].set_title('Brooklyn')

sns.pointplot(x='neighbourhood',y='price_avg',data=highest10[highest10['neighbourhood_group']=='Staten Island'],ax=axes[2],palette='viridis')

axes[2].set_title('Staten Island')

sns.pointplot(x='neighbourhood',y='price_avg',data=highest10[highest10['neighbourhood_group']=='Queens'],ax=axes[3],palette='viridis')

axes[3].set_title('Queens')

sns.pointplot(x='neighbourhood',y='price_avg',data=highest10[highest10['neighbourhood_group']=='Bronx'],ax=axes[4],palette='viridis')

axes[4].set_title('Bronx')
# check the distribution of listing availability

sns.distplot(airbnb['availability_365'],kde=False)

plt.title('Availability in a year')

plt.xlabel('Availability')

plt.ylabel('Frequency')
# Is there a pattern between price and availability? let's explore with jointplot

sns.jointplot(x='availability_365',y='price',data=airbnb)

plt.title('Price v.s. Availability in a year')
# let's zoom out and take a closer look

sns.jointplot(x='availability_365',y='price',data=airbnb[airbnb['price']<1000])

plt.title('Price v.s. Availability in a year')

# the answer to the above question is: no
# next let's do a trend analysis of availability between neighborhood groups

airbnb['availability_365_group_int'] = airbnb['availability_365']//30 * 30

airbnb['availability_365_group'] = airbnb['availability_365_group_int'].apply(lambda x: '0-30 d' if x == 0 else (str(x) + ' d+'))



plt.figure(figsize=(10,8))

g = sns.boxplot(x='neighbourhood_group',y='price',data=airbnb,hue='availability_365_group',dodge=True,palette='plasma',fliersize=1,linewidth=1,

              hue_order=['0-30 d','30 d+','60 d+','90 d+','120 d+','150 d+','180 d+','210 d+','240 d+','270 d+','300 d+','330 d+','360 d+'])

g.set(yscale='log')

plt.title('Trend analysis of availability with price in each neighborhood group')

plt.xlabel('Neighborhood')

plt.ylabel('Price')
# check the distribution of review numbers

plt.figure(figsize=(10,8))

map_img = plt.imread('../input/new-york-city-airbnb-open-data/New_York_City_.png',0)

plt.imshow(map_img,zorder=1,aspect='auto',extent=[-74.258, -73.7, 40.49, 40.92])

g = plt.scatter(x=airbnb['longitude'],y=airbnb['latitude'],c=airbnb['number_of_reviews'],cmap=plt.get_cmap('plasma'),alpha=0.5,s=10,zorder=2)

plt.title('Map of Price Distribution')

plt.colorbar(g).set_label('Number of Reviews')

plt.grid(True)
# top 500 most reviewed listings and their locations

most_reviewed_500 = airbnb.sort_values(by='number_of_reviews',ascending=False)[:500][['host_id','number_of_reviews','latitude','longitude','price']]

plt.figure(figsize=(10,8))

map_img = plt.imread('../input/new-york-city-airbnb-open-data/New_York_City_.png', 0)

plt.imshow(map_img, zorder=1, aspect='auto', extent=[-74.258, -73.7, 40.49, 40.92])

g = plt.scatter(x=most_reviewed_500['longitude'],y=most_reviewed_500['latitude'],c=most_reviewed_500['price'],cmap=plt.get_cmap('plasma'),alpha=0.5,s=10,zorder=2)

plt.title('Map of Top 500 Most Reviewed Listings')

plt.xlabel('Listing Longitude')

plt.ylabel('Listing Latitude')

plt.colorbar(g).set_label('Price of Listings')

plt.grid(True)
# check how number of reviews variate along with different availability groups

import matplotlib.ticker as mtick



plt.figure(figsize=(10,8))

# sns.swarmplot(x='neighbourhood_group',y='number_of_reviews',data=airbnb[airbnb['price']<600],hue='availability_365_group',dodge=True,palette='plasma',size=2,hue_order=['0-30 d','30 d+','60 d+','90 d+','120 d+','150 d+','180 d+','210 d+','240 d+','270 d+','300 d+','330 d+','360 d+'])

# swarmplot doesn't give a clear picture of how review numbers variate between different groups of availability

g = sns.boxplot(x='neighbourhood_group',y='number_of_reviews',data=airbnb[airbnb['number_of_reviews']>0],hue='availability_365_group',dodge=True,palette='plasma',fliersize=1,linewidth=1,

              hue_order=['0-30 d','30 d+','60 d+','90 d+','120 d+','150 d+','180 d+','210 d+','240 d+','270 d+','300 d+','330 d+','360 d+'])

g.set_yscale('log',basey=np.e)

g.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

plt.title('Trend analysis of review numbers with availability in each neighborhood group')

plt.xlabel('Neighborhood')

plt.ylabel('Number of Reviews')

plt.legend(loc="lower left",ncol=airbnb['availability_365_group'].nunique()//2+1,mode='expand',fancybox = True, shadow = True,bbox_to_anchor=(0,-0.2,1,0.2))
# for convenience we ignore outliers from price and minimum_nights columns

sns.jointplot(x='minimum_nights',y='price',data=airbnb[(airbnb['price']<600) & (airbnb['minimum_nights']<30)])
# now let's exam the location of listings, and display the locations on the NYC map

plt.figure(figsize=(10,8))

g = sns.scatterplot(x='longitude',y='latitude',data=airbnb,hue='neighbourhood_group',s=8,palette='viridis',zorder=2)

map_img = plt.imread('../input/new-york-city-airbnb-open-data/New_York_City_.png',0)

g.imshow(map_img,zorder=1,extent=g.get_xlim() + g.get_ylim(),aspect=g.get_aspect())

plt.title('Map of listings')

plt.legend(title='Neighbourhood Groups')

plt.xlabel('Listing Longitude')

plt.ylabel('Listing Latitude')

plt.legend(frameon=False)
# price distribution on the map

plt.figure(figsize=(10,8))

map_img = plt.imread('../input/new-york-city-airbnb-open-data/New_York_City_.png',0)

plt.imshow(map_img,zorder=1,aspect='auto',extent=[-74.258, -73.7, 40.49, 40.92])

g = plt.scatter(x=airbnb['longitude'],y=airbnb['latitude'],c=airbnb['price'],cmap=plt.get_cmap('coolwarm'),alpha=0.5,s=10,zorder=2,norm=matplotlib.colors.LogNorm())

plt.title('Map of Price Distribution')

plt.colorbar(g).set_label('Log Price of Listings')

plt.grid(True)
# let's exam the average price a host posts on all his/her listings

listing_mean_price = airbnb[airbnb['calculated_host_listings_count']<50].groupby(['host_id','calculated_host_listings_count'],sort=False)['price'].agg([('price_avg','mean')]).reset_index().sort_values(by=['calculated_host_listings_count','price_avg'])

print(listing_mean_price.head())
# let's first do a stripplot to see how host listings count distribute

g = sns.stripplot(x='calculated_host_listings_count',y='price_avg',data=listing_mean_price,palette='viridis')

g.set(yscale='log')
# from above we can see most hosts own less than 10 listings. Let's take a closer look at those hosts with 10 or less listings

g = sns.boxplot(x='calculated_host_listings_count',y='price_avg',data=listing_mean_price[listing_mean_price['calculated_host_listings_count']<11],palette='viridis')

g.set(yscale='log')
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn import metrics
airbnb_prep = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

airbnb_prep.drop(['id','name','host_name','last_review'],axis=1,inplace=True) # since we drop name, host_name and last_review, we no longer need to care about missing values from these columns

airbnb_prep['reviews_per_month'] = airbnb_prep['reviews_per_month'].fillna(value=0,inplace=False)

airbnb_prep = airbnb_prep[airbnb_prep['price']>0]
le = preprocessing.LabelEncoder()

le.fit(airbnb_prep['neighbourhood_group'])

airbnb_prep['neighbourhood_group'] = le.transform(airbnb_prep['neighbourhood_group'])

le.fit(airbnb_prep['neighbourhood'])

airbnb_prep['neighbourhood'] = le.transform(airbnb_prep['neighbourhood'])

le.fit(airbnb_prep['room_type'])

airbnb_prep['room_type'] = le.transform(airbnb_prep['room_type'])
airbnb_prep.sort_values(by='price',ascending=True,inplace=True)

print(airbnb_prep.head())
X = airbnb_prep.drop(['price'],axis=1,inplace=False)

y = airbnb_prep['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
lm = LinearRegression()

lm.fit(X_train,y_train)

# predict y using trained model

y_pred = lm.predict(X_test)

# get evaluation matrix

print("""

        Mean Squared Error: {}

        R2 Score: {}

        Mean Absolute Error: {}

      """.format(

    np.sqrt(metrics.mean_squared_error(y_test,y_pred)),

    metrics.r2_score(y_test,y_pred)*100,

    metrics.mean_absolute_error(y_test,y_pred)

))
linear_error = pd.DataFrame({'Actual Values':np.array(y_test).flatten(),

                             'Predicted Values':y_pred.flatten()})

print(linear_error.head(10))

sns.regplot(x=y_pred,y=y_test)

plt.title('Evaluated predictions - Linear')

plt.xlabel('Predictions')

plt.ylabel('Test')
GBoost = GradientBoostingRegressor()

GBoost.fit(X_train,y_train)

# predict y using trained model

y_pred2 = GBoost.predict(X_test)

# get evaluation matrix

print("""

        Mean Squared Error: {}

        R2 Score: {}

        Mean Absolute Error: {}

      """.format(

    np.sqrt(metrics.mean_squared_error(y_test,y_pred2)),

    metrics.r2_score(y_test,y_pred2)*100,

    metrics.mean_absolute_error(y_test,y_pred2)

))
gboost_error = pd.DataFrame({'Actual Values':np.array(y_test).flatten(),

                             'Predicted Values':y_pred2.flatten()})

print(gboost_error.head(10))

sns.regplot(x=y_pred2,y=y_test)

plt.title('Evaluated predictions - GBoost')

plt.xlabel('Predictions')

plt.ylabel('Test')