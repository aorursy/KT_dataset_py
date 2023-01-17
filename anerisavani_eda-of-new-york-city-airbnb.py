import pandas as pd
from pandas import DataFrame
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import n_colors
import numpy as np
import seaborn as sns
import pandas_profiling
%matplotlib inline
from matplotlib import rc
import scipy.stats
newyork = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv") #reading the file
newyork.head(50)
newyork.shape
newyork.columns
newyork.info()
newyork.nunique(axis=0)
newyork.describe()
newyork.neighbourhood_group.unique()
newyork.room_type.unique()
newyork.neighbourhood.unique()
newyork.isnull().sum()
newyork.fillna({'name':"NoName"}, inplace=True)
newyork.fillna({'host_name':"NoName"}, inplace=True)
last_review = newyork.last_review
percent_last_review = (last_review.isnull().sum()/(len(newyork)*1.0))*100
percent_last_review
newyork.drop(['last_review','reviews_per_month'], axis=1, inplace=True)
newyork.isnull().sum()
sns.distplot(newyork['price'])
np.mean(newyork['price'])
np.std(newyork['price'])
np.median(newyork['price'])
z_score_price = np.abs(scipy.stats.zscore(newyork['price']))
price_outliers = newyork.iloc[np.where(z_score_price>3)]
price_outliers.sort_values(['price'])
def get_lower_upper_bound(my_data):
    # Get first and third quartile
    q1 = np.percentile(my_data, 25)
    q3 = np.percentile(my_data, 75)
    
    # Calculate Interquartile range
    iqr = q3 - q1
    
    # Compute lower and upper bound
    lower_bound = q1 - (iqr * 6)
    upper_bound = q3 + (iqr * 6)
    
    return lower_bound, upper_bound
def get_outliers_iqr(my_data):
    lower_bound, upper_bound = get_lower_upper_bound(my_data)
    # Filter data less than lower bound and more than upper bound
    return my_data[np.where((my_data > upper_bound) |
                            (my_data < lower_bound))]
outliers_price = get_outliers_iqr(newyork['price'].values)
outliers_price
newyork = newyork[newyork['price'] < 815]
newyork = newyork[newyork['price']>0]
sns.distplot(newyork['price'])
sns.distplot(newyork['minimum_nights'])
z_score_nights = np.abs(scipy.stats.zscore(newyork['minimum_nights']))
nights_outliers = newyork.iloc[np.where(z_score_nights>3)]
nights_outliers.sort_values(['minimum_nights'])
newyork = newyork[newyork['minimum_nights'] < 999]
sns.distplot(newyork['number_of_reviews'])
newyork.loc[newyork['availability_365']==0,:]
newyork.shape
newyork.describe()
sub_set = newyork[['price','availability_365','minimum_nights','number_of_reviews','longitude','latitude']]
sns.pairplot(sub_set)
sns.set(rc={'figure.figsize':(8,8)})
hosts = newyork.host_id.value_counts().head(10)
hosts
fig = hosts.plot(kind='bar')
fig.set_title("Busiest top 10 host")
fig.set_xlabel("Host Ids")
fig.set_ylabel("Counts")

print(newyork['neighbourhood_group'].value_counts())
sns.countplot(x='neighbourhood_group',data=newyork,palette='viridis')
plt.title('neightbourhood groups')
newyork.loc[newyork['host_id'] == 219517861,'neighbourhood_group'].values #Manhattan
newyork.loc[newyork['host_id'] == 107434423 ,'neighbourhood_group'].values #Manhattan/Brooklyn
newyork.loc[newyork['host_id'] == 30283594  ,'neighbourhood_group'].values #Manhattan
newyork.loc[newyork['host_id'] == 12243051   ,'neighbourhood_group'].values #Manhattan
newyork.loc[newyork['host_id'] == 137358866   ,'neighbourhood_group'].values #Manhattan/Queens/Brooklyn
newyork.loc[newyork['host_id'] == 16098958   ,'neighbourhood_group'].values ##Manhattan
newyork.loc[newyork['host_id'] == 61391963   ,'neighbourhood_group'].values #Manhattan
newyork.loc[newyork['host_id'] == 22541573   ,'neighbourhood_group'].values #Manhattan/Brooklyn
newyork.loc[newyork['host_id'] == 200380610   ,'neighbourhood_group'].values #Manhattan
newyork.loc[newyork['host_id'] == 1475015   ,'neighbourhood_group'].values  #Manhattan
# top 10 neighborhoods with most number of listings per each neighbourhood group
listing_per_neighborhoodgroup = newyork.groupby(['neighbourhood_group','neighbourhood'],sort=False)['id'].agg([('count','count')]).reset_index().sort_values(by=['neighbourhood_group','count'],ascending=[True,False])
top_10 = listing_per_neighborhoodgroup.groupby(['neighbourhood_group']).apply(lambda x: x.nlargest(10,'count'))
fig,axes = plt.subplots(nrows=1,ncols=3,sharey=True,figsize=(18,6))
fig.suptitle('Top 10 neighborhoods per following three neighbourhood group', fontsize=16)
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
sns.barplot(x='neighbourhood',y='count',data=top_10[top_10['neighbourhood_group']=='Manhattan'],ax=axes[0],palette='viridis')
axes[0].set_title('Manhattan')
sns.barplot(x='neighbourhood',y='count',data=top_10[top_10['neighbourhood_group']=='Brooklyn'],ax=axes[1],palette='viridis')
axes[1].set_title('Brooklyn')
sns.barplot(x='neighbourhood',y='count',data=top_10[top_10['neighbourhood_group']=='Queens'],ax=axes[2],palette='viridis')
axes[2].set_title('Queens')
print(newyork['room_type'].value_counts())
sns.countplot(x='room_type',data=newyork,palette='viridis')
plt.title('Different room types')
neighbourhood_roomtype = newyork.groupby(by=['neighbourhood','room_type'],sort=False)['id'].agg([('count','count')]).reset_index().sort_values(by='count',ascending=False)

list = ['Bedford-Stuyvesant','Williamsburg','Bushwick','Crown Heights','Greenpoint','East Flatbush','East Village','Flatbush','Prospect-Lefferts Gardens','Clinton Hill','Park Slope','Harlem','Upper East Side','Upper West Side',
        'Midtown','East Village','East Harlem','Chelsea','Financial District','Washington Heights','Astoria','Flushing','Long Island City','Ridgewood','Sunnyside','Ditmars Steinway','Elmhurst','Woodside','East Elmhurst','Hell\'s Kitchen']
neightbourhood_roomtype_top_30 = neighbourhood_roomtype[neighbourhood_roomtype.neighbourhood.str.contains('|'.join(list))]

pivot_df = neightbourhood_roomtype_top_30.pivot(index='neighbourhood', columns='room_type', values='count')
pivot_df
colors = ["#8B0A50", "#EE1289","#1E90FF"]

pivot_df.loc[:,['Entire home/apt','Private room', 'Shared room']].plot.barh(stacked=True, color=colors, figsize=(10,7))




neighbourhood_price = newyork.groupby(by=['neighbourhood'],sort=False)['price'].mean().reset_index().sort_values(by='price',ascending=False)
neighbourhood_price
neightbourhood_price_top_30 = neighbourhood_price[neighbourhood_price.neighbourhood.str.contains('|'.join(list))]
neightbourhood_price_top_30
fig = px.bar(neightbourhood_price_top_30,  x='neighbourhood', y='price',text='price')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
price = newyork[newyork.price < 1000]
plt.figure(figsize=(20,10))

# The plot
sns.boxplot(x = 'neighbourhood_group',
            y = 'price', data = price, palette = "viridis", saturation = 1, width = 0.9, fliersize=4, linewidth=2)

# Make pretty
plt.title('Price distribution of neighbourhood_group', fontsize = 20)
plt.xlabel('Neighbourhood_group', fontsize = 15)
plt.ylabel('Price', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
Manhattan = price.loc[price['neighbourhood_group']=='Manhattan','price']
list_manhattan = Manhattan.values.tolist()
list_manhattan.sort()
np.mean(list_manhattan)
Brooklyn = price.loc[price['neighbourhood_group']=='Brooklyn','price']
list_brooklyn = Brooklyn.values.tolist()
list_brooklyn.sort()
np.mean(list_brooklyn)
Queens = price.loc[price['neighbourhood_group']=='Queens','price']
list_queens = Queens.values.tolist()
list_queens.sort()
np.mean(list_queens)
Staten_Island = price.loc[price['neighbourhood_group']=='Staten Island','price']
list_staten_island = Staten_Island.values.tolist()
list_staten_island.sort()
np.mean(list_staten_island)
Bronx = price.loc[price['neighbourhood_group']=='Bronx','price']
list_bronx = Bronx.values.tolist()
list_bronx.sort()
np.mean(list_bronx)
plt.figure(figsize=(10,6))
sns.scatterplot(newyork.longitude,newyork.latitude,hue=newyork.availability_365)
plt.ioff()
top_reviewed_listings=newyork.nlargest(10,'number_of_reviews')
top_reviewed_listings
reviews = newyork[newyork.number_of_reviews < 630]
sns.scatterplot(reviews.longitude,reviews.latitude,hue=reviews.number_of_reviews)
plt.ioff()
price_avrg=top_reviewed_listings.price.mean()
print('Average price per night: {}'.format(price_avrg))