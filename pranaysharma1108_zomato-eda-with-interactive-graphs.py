import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from subprocess import check_output
%matplotlib inline

df1 = pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1") #import the zomato restaurants data file
df2 = pd.read_excel('../input/Country-Code.xlsx') #import zomato country code file
zomato = pd.merge(df1,df2, on = 'Country Code')
zomato.head(3)
#Now we have both country code and name in same data set.
zomato.info()
#Top 15 Restro with maximum number of outlets
ax=zomato['Restaurant Name'].value_counts().head(15).plot.bar(figsize =(12,6))
ax.set_title("Top 15 Restarurents with maximum outlets")
for i in ax.patches:
    ax.annotate(i.get_height(), (i.get_x() * 1.005, i.get_height() * 1.005))
#-------------------------------------------------------------
stopwords = set(STOPWORDS)

wordcloud = (WordCloud(width=500, height=300, relative_scaling=0.5, stopwords=stopwords).generate_from_frequencies(zomato['Restaurant Name'].value_counts().head(35)))
fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#---------------------------------------------------------
#No. of unique countries & number of restro listed in data set
zomato.Country.value_counts()
#Number of restro on zomato in different cities in India
zomato.loc[zomato['Country']=='India'].City.value_counts().head(10)
#--------------------------------------------------------
#Top 10 Restro with highest no. of votes
max_votes =zomato.Votes.sort_values(ascending=False).head(10)
zomato.loc[zomato['Votes'].isin(max_votes)][['Restaurant Name','Votes']]
zomato.loc[zomato['Votes'].isin(max_votes)][['Restaurant Name','Votes']].plot.bar(x='Restaurant Name', y='Votes',
                                                                                  figsize = (10,6), color='purple')
#-------------------------------------------------------
zomato_india = zomato.loc[zomato['Country']=='India']
zomato_india.head(3)
#Is there any relation between average cost for two and aggregate rating of restaurants
zomato_india.plot.scatter(x='Average Cost for two',y='Aggregate rating',figsize=(10,6), color='orange', title="Cost vs Agg Rating")
#-------------------------------------------------------------
#Better view of relation between average cost for two and aggregate rating of restaurants
sns.jointplot(x='Average Cost for two',y='Aggregate rating',kind ='hex',gridsize=18,data =zomato_india,color='blue')
#--------------------------------------------------
#Top 10 Cuisines served by restaurants
zomato_india['Cuisines'].value_counts().sort_values(ascending=False).head(10)
zomato_india['Cuisines'].value_counts().sort_values(ascending=False).head(10).plot(kind='pie',figsize=(10,6), 
title="Most Popular Cuisines", autopct='%1.2f%%')
plt.axis('equal')
#-------------------------------------------------------
#Correlation among avg cost, price range, agg rating
zomato_corr = zomato[['Average Cost for two', 'Price range', 'Aggregate rating']]
sns.heatmap(zomato_corr.corr(),linewidth=1.0)
#cmap='PuOr' cmap='YlGnBu'
#------------------------------------------------------------
#More insight for correlation by using pair plot keeping top 10 cities with max restro
top5_indian_cities = ['New Delhi', 'Gurgaon', 'Noida','Faridabad', 'Ghaziabad']
zomato_p = zomato.loc[zomato['City'].isin(top5_indian_cities)]
zomato_pair = zomato_p[['Average Cost for two', 'Price range', 'Aggregate rating', 'City']]
sns.pairplot(zomato_pair, size=3, hue='City', palette="husl")
#-----------------------------------------------
#Correlation of cost, price range with top 10 cuisines
top5cuisines_list=['North Indian', 'North Indian, Chinese', 'Fast Food', 'North Indian, Mughlai', 'Cafe' ]

zomato_cuisines = zomato.loc[zomato['Cuisines'].isin(top5cuisines_list)]
zomato_cuisines_data = zomato_cuisines[['Average Cost for two', 'Price range', 'Aggregate rating', 'Cuisines']]
sns.pairplot(zomato_cuisines_data, size=3, hue='Cuisines')

#--------------------------------------------------------
#ANalysis of top 10 cuisines with price range and agg rating
top10cuisines_list=['North Indian', 'North Indian, Chinese', 'Fast Food', 'North Indian, Mughlai', 'Cafe', 'Bakery',
                   'North Indian, Mughlai, Chinese', 'Bakery, Desserts', 'Street Food' ]
zomato_cuisines = zomato.loc[zomato['Cuisines'].isin(top10cuisines_list)]
zomato_cuisines_data = zomato_cuisines[['Average Cost for two', 'Price range', 'Aggregate rating', 'Cuisines']]

fig, axx =plt.subplots(figsize=(16,8))
sns.barplot(x='Cuisines', y='Aggregate rating', hue='Price range', data=zomato_cuisines_data, palette="Set1")
axx.set_title("Analysis of Top10 Cuisines with price range and Agg. rating ")
#------------------------------------------------------
#Most common agg. rating for each type of cuisine
top10cuisines_list=['North Indian', 'North Indian, Chinese', 'Fast Food', 'North Indian, Mughlai', 'Cafe', 'Bakery',
                   'North Indian, Mughlai, Chinese', 'Bakery, Desserts', 'Street Food' ]
zomato_cuisines = zomato.loc[zomato['Cuisines'].isin(top10cuisines_list)]
zomato_cuisines_data = zomato_cuisines[['Average Cost for two', 'Price range', 'Aggregate rating', 'Cuisines']]
fig, axx =plt.subplots(figsize=(16,6))
sns.boxplot(x='Cuisines', y='Aggregate rating', data=zomato_cuisines_data)
#----------------------------------------------------------------------
#Restaurant Percentage wise rating in top 5 cities
top5_indian_cities = ['New Delhi', 'Gurgaon', 'Noida','Faridabad', 'Ghaziabad']
zomato_rate = zomato.loc[zomato['City'].isin(top5_indian_cities)]

#Find total number of restaurants
total_restro = zomato_rate.groupby(['City'], as_index=False).count()[['City','Restaurant ID']]
total_restro.columns=['City','Total Restaurants']

#Find total rating count of each type
top5rest = zomato_rate.groupby(['City','Rating text'], as_index=False)[['Restaurant Name']].count()
top5rest.columns=['City','Rating text', 'Total Ratings']

#Merge both the dataframes and calculate percentage
top5restro_rating_percent = pd.merge(total_restro, top5rest, on='City')
top5restro_rating_percent['Percentage']= (top5restro_rating_percent['Total Ratings']/
                                       top5restro_rating_percent['Total Restaurants'])*100

top5restro_rating_percent
#Plot Rating percentage of restaurants in top 5 cities
fig, axx =plt.subplots(figsize=(12,6))
axx.set_title("Percentage Rating of Restaurants in Top 5 Cities")
sns.barplot(x='City', y='Percentage',hue='Rating text', data=top5restro_rating_percent, palette='Set3')