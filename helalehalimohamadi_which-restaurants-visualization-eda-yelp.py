# Let's import the library we needed before we get started:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from wordcloud import WordCloud, STOPWORDS     
# import the data (chunksize returns jsonReader for iteration)
businesses = pd.read_json("/kaggle/input/yelp-dataset/yelp_academic_dataset_business.json", lines=True, orient='columns', chunksize=100000)
reviews = pd.read_json("/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json", lines=True, orient='columns', chunksize=100000)
checkins = pd.read_json("/kaggle/input/yelp-dataset/yelp_academic_dataset_checkin.json", lines=True, orient='columns', chunksize=100000)
users = pd.read_json("/kaggle/input/yelp-dataset/yelp_academic_dataset_user.json", lines=True, orient='columns', chunksize=100000)
# read the data 
for business in businesses:
    subset_business = business
    break
    
for review in reviews:
    subset_review = review
    break
      
for checkin in checkins:
    subset_checkin = checkin
    break
    
for user in users:
    subset_user = user
    break
# Let's give a look at the data:
display(subset_business.head(2))
display(subset_review.head(2))
display(subset_user.head(2))
display(subset_checkin.head(2))
# Create dataframe having only Restaurant business.
all_cities = subset_business[subset_business['categories'].str.contains('Rest.*')==True]

# Creating dummies dataframe from series for 'categories' 
df_rest = pd.Series(all_cities['categories']).str.get_dummies(',')

# Dropping Restaurants and Food columns as this analysis is for Resturants and these words are common to all entries
df_rest.drop(["Restaurants", " Restaurants", "Food", " Food"], axis=1, inplace=True)

# Removing the whitespaces from the column names
df_rest.columns = df_rest.columns.str.lstrip()

# Adding up all the rows to get the sum of columns and merging the columns with same names
all_rest = df_rest.groupby(by=df_rest.columns, axis=1).sum()
# Extracting name of all the cities
from_business = all_cities[['city']]
# Getting count of Vegetarian restaurants in each city
all_rest.join(from_business).groupby('city').sum()['Vegetarian'].sort_values(ascending=False)
plt.figure(figsize=(15,6))
all_rest.join(from_business).groupby('city').sum()['Vegetarian'].sort_values(ascending=True).tail(10)\
                .plot(kind='barh',color='Darkcyan')
plt.title('Top Cities for Vegetarian Restaurants',fontsize=18, pad=25.0) 
plt.xlabel('Counts', fontsize=15)
plt.ylabel('Cities', fontsize=15)
plt.show()
# Extracting data for the Vegetarian Restaurants in Toronto city which are open along with there location coordinates.
Toronto = all_rest.join(subset_business)[all_rest.join(subset_business)['city'] == 'Toronto']
tor_1 = Toronto[['Vegetarian','name','address','latitude','longitude','stars','is_open','hours']]

# Sorting data as per their star ratings.
tor_1[ (tor_1['Vegetarian']==1) & (Toronto['is_open']==1) ].sort_values(by='stars',ascending=False).head(10)
# Creating a geographical map for the location of top vegetarian restaurants in Toronto
import folium
import pandas as pd
from folium.plugins import MarkerCluster
 
# make a data frame with dots to show on the map
data_veg = tor_1[ (tor_1['Vegetarian']==1) & (Toronto['is_open']==1) ].sort_values(by='stars',ascending=False)\
            [['longitude','latitude','name','address','stars','hours']].head(20)
 
# create an empty map
toronto_veg_map = folium.Map(location=[43.651070,-79.347015], tiles='Stamen Terrain') #, default_zoom_start=20)
 
# add marker one by one on the map
for i in range(0, len(data_veg)):
    text = folium.Html('<b>Name: </b>'+ data_veg.iloc[i]['name'] + "<hr style='margin:10px;'>" + 
                       "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>" +
                       "<li>Star: " + str(data_veg.iloc[i]['stars']) + "</li>" +
                       "<li>Address: " + str(data_veg.iloc[i]['address']) + "</li>" +
                       "<li>Hours: " + str(data_veg.iloc[i]['hours']) + "</li>", script=True)

    popup = folium.Popup(text, max_width=2650)
    folium.Marker([ data_veg.iloc[i]['latitude'], data_veg.iloc[i]['longitude'] ], popup=popup).add_to(toronto_veg_map)
toronto_veg_map
#Creating dataset with the data for different cities for further analysis
cities=['Toronto','Las Vegas','Calgary','Montréal']
df_cities=[]

for c in cities:
    city = subset_business[subset_business['city']==c]
    rest = city[city['categories'].str.contains('Rest.*')==True]
    df = pd.Series(rest['categories']).str.get_dummies(',')
    df.drop(["Restaurants", " Restaurants", "Food", " Food"], axis=1, inplace=True)
    df.columns = df.columns.str.lstrip()
    result = df.groupby(by=df.columns, axis=1).sum()
    df_cities.append(result)
 
 # Creating separate dataframe for each city.
toronto, las_vegas, calgary, montreal = df_cities[0], df_cities[1], df_cities[2], df_cities[3]
# Getting 20 most popular Cuisine in Toronto
tor = pd.DataFrame(toronto.sum().sort_values(ascending=True).tail(20),columns=['counts'])
tor.reset_index(inplace=True)
tor.rename({'index':'name'}, axis=1, inplace=True)
tor
# Getting number of Vegetarian Restaurants in Toronto.
toronto[['Vegetarian']].sum()
import warnings
warnings.filterwarnings("ignore")

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12,24))

# Visualizing 20 most popular Categories in Toronto Restaurants
#plt.subplot(221)
#plt.barh(tor['name'], tor['counts'],color='Darkcyan')
toronto.sum().sort_values(ascending=True).tail(20).plot(kind='barh',color='Darkcyan',ax=axes[0])
axes[0].set_title('top categories for TORONTO restaurants',fontsize=25, pad=25.0) 
axes[0].xaxis.set_label_text("Category counts", fontsize=18)
axes[0].yaxis.set_label_text("Restaurants names", fontsize=18)
#axes[0].tick_params(width=1, labelsize=16)

# Visualizing 20 most popular Categories in Las Vegas Restaurants
#plt.subplot(222)
las_vegas.sum().sort_values(ascending=True).tail(20).plot(kind='barh',color='red',ax=axes[1])
axes[1].set_title('top categories for LAS VEGAS restaurants',fontsize=25, pad=25.0) 
axes[1].xaxis.set_label_text("Category counts", fontsize=18)
axes[1].yaxis.set_label_text("Restaurants names", fontsize=18)
#axes[1].tick_params(width=1, labelsize=16)

# Visualizing 20 most popular Categories in Calgary Restaurants
#plt.subplot(223)
calgary.sum().sort_values(ascending=True).tail(20).plot(kind='barh',color='blue',ax=axes[2])
axes[2].set_title('top categories for CALGARY restaurants',fontsize=25, pad=25.0) 
axes[2].xaxis.set_label_text("Category counts", fontsize=18)
axes[2].yaxis.set_label_text("Restaurants names", fontsize=18)
#axes[2].tick_params(width=1, labelsize=16)

# Visualizing 20 most popular Categories in Montreal Restaurants
#plt.subplot(224)
montreal.sum().sort_values(ascending=True).tail(20).plot(kind='barh',color='green',ax=axes[3])
axes[3].set_title('top categories for MONTREAL restaurants',fontsize=25, pad=25.0) 
axes[3].xaxis.set_label_text("Category counts", fontsize=18)
axes[3].yaxis.set_label_text("Restaurants names", fontsize=18)
#axes[3].tick_params(width=1, labelsize=16)

plt.tight_layout() # makes sure there is no overlap in plots 
plt.subplots_adjust(wspace=0.5, hspace=0.5)
"""
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
"""
plt.show()
# Getting the Users(Yelpers) based on review counts
review_total = subset_user.sort_values(by='review_count',ascending=True)

# Getting the Users based on number of fans
fans = subset_user.sort_values(by='fans',ascending=True)

# Getting the Users based on number of total friends on yelp
subset_user['total_friends'] = [len(subset_user.iloc[i,8]) for i in range(0,100000)]
friends = subset_user.sort_values(by='total_friends',ascending=True)

# Getting the Users based on number of useful reviews 
useful = subset_user.sort_values(by='useful',ascending=True)
# Create new column named 'ratio'
subset_user['ratio'] = (subset_user['useful'] * subset_user['review_count']) / sum(subset_user['review_count'])
# Creating new dataset with User data
Top_influencer = subset_user.sort_values(by='ratio',ascending=True)[['name','review_count','useful','ratio','yelping_since']]
import matplotlib.gridspec as gridspec

gridspec.GridSpec(3,2)
plt.figure(figsize=(14,18)) 

# Plotting the most popular reviewers based on the count of their fans
plt.subplot2grid((3,2), (0,0))
plt.barh(fans['name'][-10:],fans['fans'][-10:])
#plt.barh(fans['name'][:10],fans['fans'][:10])

plt.title('Top POPULAR',fontsize=16, pad=25.0) 
plt.xlabel('Reviewer names', fontsize=14, labelpad=15.0)
plt.ylabel('number of fans', fontsize=14)

# Plotting the 10 most popular reviewers based on the count of their useful reviews
#plt.subplot(322)
plt.subplot2grid((3,2), (0,1))
plt.barh(useful['name'][-10:],useful['useful'][-10:],color='green')

plt.title('Top USEFUL',fontsize=18, pad=25.0) 
plt.xlabel('Reviewer names', fontsize=14, labelpad=15.0)
plt.ylabel('number of reviews', fontsize=14)

# Plotting the 10 most popular reviewers based on the count of their friends on yelp
#plt.subplot(323)
plt.subplot2grid((3,2), (1,0))
plt.barh(friends['name'][-10:],friends['total_friends'][-10:],color='purple')

plt.title('Top SOCIAL',fontsize=18, pad=25.0) 
plt.xlabel('Reviewer names', fontsize=14, labelpad=15.0)
plt.ylabel('number of friends', fontsize=14)

# Plotting the 10 most popular reviewers based on the count of their reviews posted on yelp
#plt.subplot(324)
plt.subplot2grid((3,2), (1,1))
plt.barh(review_total['name'][-10:],review_total['review_count'][-10:],color='cyan')

plt.title('Top ACTIVE',fontsize=18, pad=25.0) 
plt.xlabel('Reviewer names', fontsize=14, labelpad=15.0)
plt.ylabel('number of reviews', fontsize=14)

# Plotting the 10 most popular reviewers based on the ratio of their useful reviews to total reviews
#plt.subplot(325)
plt.subplot2grid((3,2),(2,0),colspan=2, rowspan=2)
plt.barh(Top_influencer['name'][-10:],Top_influencer['ratio'][-10:],color='red')

plt.title('Top INFLUENCER',fontsize=18, pad=25.0) 
plt.xlabel('Reviewer names', fontsize=14, labelpad=15.0)
plt.ylabel('Ratio', fontsize=14)

# comment out the following line and run cell to see the difference it makes
plt.tight_layout() # makes sure there is no overlap in plots 
plt.show()
range(0,subset_checkin.shape[0])
# Creating a new column to see how many people cheacked in to the business over the years
subset_checkin['total'] = [len(subset_checkin.iloc[i,1]) for i in range(0,100000)]
subset_checkin.head(1)
# Sorting the businesses by total number of check-ins to get the top businesses with most check-ins
checkin_sort = subset_checkin.sort_values(by='total',ascending=False)

# Getiing the name of the businesses
business_check = pd.merge(checkin_sort,subset_business,on='business_id')
business_check.head(1)
# Creating a dataframe for getting the business in Toronto and their check-in history
Toronto = business_check[business_check['city']=='Toronto']

# getting just restaurants from Toronto business
rest = Toronto[Toronto['categories'].str.contains('Rest.*')==True]
rest.head(1)
# top popular rated restaurants in Toronto
top_pop = rest.sort_values(by=['total'],ascending=False)[['name','total','stars']]
plt.figure(figsize=(12,6))
ax = sns.barplot(top_pop['name'][:10], top_pop['total'][:10], alpha=0.8)
#plt.barh(top_pop['name'][:10],top_pop['review_count'][:10])
plt.title('Top popular restaurants in Toronto on Yelp (Check In)',fontsize=20, pad=30.0) 
plt.xlabel('Restaurants names', fontsize=16)
plt.ylabel('Checkin counts', fontsize=16)
plt.xticks(rotation='50', fontsize=12)
rects = ax.patches
labels = top_pop['stars']
for rect, label in zip(rects, labels):
    height = rect.get_height()
    #print(rect.get_x() + rect.get_width() / 2, height + 5, label)
    ax.text(rect.get_x() + rect.get_width() / 2, height, label, ha='center', va='bottom')
plt.show()
#Plotting Restaurants in Toronto having most check-ins
import folium
import pandas as pd
from folium.plugins import MarkerCluster
 
# make a data frame with dots to show on the map
data = rest[['longitude','latitude','name','total']].head(10)
 
# create an empty map
toronto_map = folium.Map(location=[43.651070,-79.347015], tiles='Stamen Terrain')#, default_zoom_start=20)
 
# add marker one by one on the map
for i in range(0,len(data)):
    test = folium.Html('<b>Name: </b>'+ data.iloc[i]['name'] + "<hr style='margin:10px;'>" + 
                       "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>" +
                       "<li>Total reviews: "+str(data.iloc[i]['total'])+"</li>", script=True)

    popup = folium.Popup(test, max_width=2650)
    folium.Marker([ data.iloc[i]['latitude'], data.iloc[i]['longitude'] ], popup=popup).add_to(toronto_map)
    
    #folium.Marker([ data.iloc[i]['latitude'], data.iloc[i]['longitude'] ], popup=data.iloc[i][['name','total']].values).add_to(toronto_map)
# plot the top 10 restuarant in Toronto (checkIn)
toronto_map
gridspec.GridSpec(2,2)

plt.figure(figsize=(18,10)) 

# Plotting daily activity of users on yelp
plt.subplot2grid((2,2), (0,0))
sns.distplot(subset_review['date'].dt.day, color='green')
plt.title('Yelp daily user activity',fontsize=16)
plt.xlabel('days', fontsize=14)
plt.ylabel('ratio', fontsize=14)

# Plotting activity of usres on yelp over a month
plt.subplot2grid((2,2), (0,1))
sns.distplot(subset_review['date'].dt.month, color='red')
plt.title('Yelp user activity over months',fontsize=16)
plt.xlabel('Months', fontsize=14)
plt.ylabel('ratio', fontsize=14)

# Plotting activity of usres on yelp over an year
plt.subplot2grid((2,2), (1,0),colspan=2, rowspan=2)
sns.distplot(subset_review['date'].dt.year, color='orange')
plt.title('Yelp user activity over years',fontsize=16)
plt.xlabel('Months', fontsize=14)
plt.ylabel('ratio', fontsize=14)

# Function that extract keys from the nested dictionary
def extract_keys(attr, key):
    if attr == None:
        return "{}"
    if key in attr:
        return attr.pop(key)

# convert string to dictionary
import ast
def str_to_dict(attr):
    if attr != None:
        return ast.literal_eval(attr)
    else:
        return ast.literal_eval("{}") 
# get dummies from nested attributes
rest['BusinessParking'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'BusinessParking')), axis=1)
rest['Ambience'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Ambience')), axis=1)
rest['GoodForMeal'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'GoodForMeal')), axis=1)
rest['Dietary'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Dietary')), axis=1)
rest['Music'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Music')), axis=1)
# create the features table with attribute dummies
df_attr = pd.concat([ r['attributes'].apply(pd.Series), r['BusinessParking'].apply(pd.Series),
                    r['Ambience'].apply(pd.Series), r['GoodForMeal'].apply(pd.Series), 
                    r['Dietary'].apply(pd.Series) ], axis=1)
df_attr_dummies = pd.get_dummies(df_attr,drop_first=True)
df_attr_dummies.head(3)
# get dummies from categories
df_categories_dummies = pd.Series(rest['categories']).str.get_dummies(',')

# Dropping Restaurants and Food columns as this analysis is for Resturants and these words are common to all entries
df_categories_dummies.drop(["Restaurants", " Restaurants", "Food", " Food"], axis=1, inplace=True)

# Removing the whitespaces from the column names
df_categories_dummies.columns = df_categories_dummies.columns.str.lstrip()

# Adding up all the rows to get the sum of columns and merging the columns with same names
df_cat = df_categories_dummies.groupby(by=df_categories_dummies.columns, axis=1).sum()
df_cat.head()
df_final = pd.concat([df_attr_dummies, df_cat,rest[['stars']]], axis=1)
df_final.head()
# map floating point stars to an integer
mapper = {1.0:1,1.5:2, 2.0:2, 2.5:3, 3.0:3, 3.5:4, 4.0:4, 4.5:5, 5.0:5}
df_final['stars'] = df_final['stars'].map(mapper)

# Create X (all the features) and y (target)
X = df_final.iloc[:,:-1]
y = df_final['stars']
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg_model = LogisticRegression()
logreg_model.fit(X, y)
logreg_model.score(X,y)
coefficients = logreg_model.coef_[0]
df = pd.DataFrame(data={'feature':X.columns, 'Coef':list(coefficients)})
df.head()
top_20 = df.sort_values(by="Coef",ascending=False).head(10)
bottom_20 = df.sort_values(by="Coef",ascending=False).tail(10)
merge = pd.concat([top_20,bottom_20], axis=0)
merge
plt.figure(figsize=(15,8))
plt.title('The importance of features for ratings',fontsize= 28)
bar_colors=np.where(merge['Coef'][0:-1] > 0.0,"blue","red")
plt.barh(merge['feature'][0:-1],merge['Coef'][0:-1],color= bar_colors, label= 'positively correlated')
plt.xlabel(' Coefficients',fontsize='20')
plt.ylabel(' Features',fontsize='20')
plt.yticks(fontsize='16')
plt.legend()
plt.show()
