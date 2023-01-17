#! /usr/bin/env python3
# coding: utf-8

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style='whitegrid'
#load dataset
dataset = pd.read_csv('../input/TA_restaurants_curated.csv', encoding='utf8', index_col=0)
print(dataset.head(), '\n')
print(dataset.tail(), '\n')
print(dataset.info())
#Turn city and price range columns into categorical type for less memory usage
#dataset_categorized = dataset
#dataset_categorized['City'] = dataset_categorized['City'].astype('category')
#dataset_categorized['Price Range'] = dataset_categorized['Price Range'].astype('category')
#print(dataset_categorized.info())
#Get sorting of cities by number of restaurants
global_number_rest = dataset['City'].value_counts(dropna=False)
print("\n Sorted by number of restaurants \n")
print(global_number_rest)
total_rest = global_number_rest.sum()
print("\n Total number of restaurants: {}".format(total_rest))

#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pie.html
explode = [0.1 for i in global_number_rest]
global_number_rest.plot(kind='pie', figsize=(25, 25), explode=explode, fontsize=20, autopct=lambda v: int(v*total_rest/100),
                        title="Repartition by cities of number of restaurants (TA)")
#plt.savefig('figures/restaurants repartition by city.png')
#plt.savefig('figures/restaurants repartition by city.svg')
plt.show()
#Wikipedia: city population
pop_dict = {'London': 8673713, 'Paris': 2220445, 'Madrid':  3141991, 'Barcelona': 1620809, 'Berlin': 3520031,
           'Milan': 1368590, 'Rome': 2877215, 'Prague': 1280508, 'Lisbon': 545245, 'Vienna': 1840573, 'Amsterdam':840486,
                'Munich': 1548319, 'Hamburg': 1787408, 'Brussels': 1202953, 'Stockholm': 1515017, 'Budapest': 1733685,
                'Warsaw': 1726581, 'Copenhagen':  602481, 'Dublin': 5276120, 'Lyon': 513275, 'Athens': 664046,
                'Edinburgh': 482640, 'Zurich': 402762, 'Oporto': 237591, 'Geneva': 201813, 'Krakow': 758334, 
                'Helsinki': 642045, 'Oslo': 623966, 'Bratislava': 455800, 'Luxembourg': 115227, 'Ljubljana':280278}

#Wikipedia: Area of the city in km²
area_dict = {'London': 1575, 'Paris': 105.4, 'Madrid':  608, 'Barcelona': 100.4, 'Berlin': 3891.9,
           'Milan': 182, 'Rome': 1285.3, 'Prague': 496, 'Lisbon': 83.84, 'Vienna': 414.9, 'Amsterdam':219.3,
                'Munich': 310.4, 'Hamburg': 755.3, 'Brussels': 161.38, 'Stockholm': 414, 'Budapest': 525.1,
                'Warsaw': 517.24, 'Copenhagen':  88.25, 'Dublin': 115, 'Lyon': 47.9, 'Athens': 38,
                'Edinburgh': 259, 'Zurich': 87.9, 'Oporto': 41.3, 'Geneva': 15.9, 'Krakow': 326.8, 
                'Helsinki': 213.75, 'Oslo': 454, 'Bratislava': 367.6, 'Luxembourg': 51.7, 'Ljubljana': 275}
#For Brussels, it's the whole aggloemration, not the city (too tiny)

#Dataframe creation
pop_rest_df = pd.DataFrame()
pop_rest_df['Restaurants'] = global_number_rest
pop_rest_df['Population'] = pd.DataFrame([pop_dict]).T
pop_rest_df['Resto/1000 Inhabitants'] = 1000 * pop_rest_df['Restaurants'] / pop_rest_df['Population']
pop_rest_df['Area'] = pd.DataFrame([area_dict]).T
pop_rest_df['Resto/km²'] = pop_rest_df['Restaurants'] / pop_rest_df['Area']
pop_rest_df = pop_rest_df.sort_values('Resto/1000 Inhabitants', ascending=False)

print(pop_rest_df.head(10), '\n')
print(pop_rest_df.describe())
#Visualization with bar plot
pop_rest_df.plot(kind='bar', subplots=True, y=['Resto/1000 Inhabitants', 'Resto/km²'], figsize=(20,10))
#plt.savefig('figures/Densities.svg')
#plt.savefig('figures/Densities.png')
plt.show()

#Visualization with boxplot:
pop_rest_df.plot(kind='box', subplots=True, y=['Resto/1000 Inhabitants', 'Resto/km²'], figsize=(20,5))
#plt.savefig('figures/Densities_box.svg')
#plt.savefig('figures/Densities_box.png')
plt.show()
#Replace NaN in the Price range column by 'Unknown' 
dataset['Price Range'] = dataset['Price Range'].fillna('Unknown')

#Get the count per price range globally
global_pricerange_count = dataset.groupby('Price Range')['Name'].count()

#Dataframe of count of global restaurants by price range
global_pricerange_count = pd.DataFrame({'Restaurants': global_pricerange_count})
global_pricerange_count['percent'] = global_pricerange_count['Restaurants'] / total_rest
global_pricerange_count.index.names = ['Price Range']
global_pricerange_count.index = ['low', 'mid', 'high', 'unknown']
print(global_pricerange_count)

#Get price range counts per city
pricerange_count_city = dataset.groupby(['Price Range', 'City'])['Name'].count() #Series
pricerange_count_city = pd.DataFrame(pricerange_count_city) #multi indexed df (price range, city)
pricerange_count_city.columns = ['Restaurants']
#Pivot table to get the price range as columns instead of rows index
pricerange_count_city = pricerange_count_city.pivot_table(index='City', columns='Price Range')
pricerange_count_city['Total'] = global_number_rest

#Add percentage columns
pricerange_count_city['% in low price range'] = pricerange_count_city['Restaurants']['$'] / pricerange_count_city['Total']
pricerange_count_city['% in mid price range'] = pricerange_count_city['Restaurants']['$$ - $$$'] / pricerange_count_city['Total']
pricerange_count_city['% in high price range'] = pricerange_count_city['Restaurants']['$$$$'] / pricerange_count_city['Total']
pricerange_count_city['% unknown price range'] = pricerange_count_city['Restaurants']['Unknown'] / pricerange_count_city['Total']
print(pricerange_count_city.head())

#Statistical information
print('\n', pricerange_count_city['Restaurants'].describe())
explode = [0.02 for i in range(4)]

#Global price range count with pie chart
global_pricerange_count.index = ['low range', 'mid range', 'high range', 'unknown'] #$$$$ label returns error
global_pricerange_count['Restaurants'].plot.pie(figsize=(5,5), legend=False, autopct='%.0f%%', explode=explode,
                                               title="Global restaurants by price range")
#plt.savefig("figures/Global price range rep.svg")
#plt.savefig("figures/Global price range rep.png")
plt.show()

#Global price range count with pie chart without unknown price range restaurants
global_pricerange_count.index = ['low range', 'mid range', 'high range', 'unknown'] #$$$$ label returns error
global_pricerange_count.iloc[0:3, 0].plot.pie(figsize=(5,5), legend=False, autopct='%.0f%%', explode=explode[:-1],
                                              title="Global restaurants by price range (unknown not included)")
#plt.savefig("figures/Global price range rep_unknow not included.svg")
#plt.savefig("figures/Global price range rep_unknown not included.png")
plt.show()
#Visualization with pie charts
height = int(np.ceil(len(global_number_rest)/3))
fig, axs = plt.subplots(height, 3, figsize=(20,40)) #organizes the pies according to a grid
r=0 ; h=0 #r is the position in the row, h is the position in the heigh
labels = [pricerange_count_city.iloc[:, 5:].columns[i][0] for i in range(4)]

for k in range(len(global_number_rest)): #over each line (city)
    city = pricerange_count_city.iloc[k, :].name
    table = pricerange_count_city.iloc[k, 5:].T

    axs[h, r].pie(table, labels=labels, autopct='%.0f%%', explode=explode)
    axs[h, r].axis('equal')
    axs[h, r].set(title=city)
    k += 1
    #Change line every 3rd pie
    if r < 2:
        r += 1       
    elif r == 2:
        r = 0
        h += 1
#plt.savefig("figures/price_range_pies.png")
#plt.savefig("figures/price_range_pies.svg")
plt.show()
percent = pricerange_count_city.iloc[:, 5:]
percent.columns = ['low range', 'mid range', 'high range', 'unknown range']

#Visualization with box plots
percent.plot(kind='box', subplots=True, figsize=(40,10), title="% in each price range", fontsize=20)
#plt.savefig('figures/price range rep boxplot.svg')
#plt.savefig('figures/price range rep boxplot.png')
plt.show()
cuisine_df = dataset[['Cuisine Style', 'City']]

#Counting function to parse the cuisine styles lists
def cuisine_count(liste):
    cuisine_dict = {'Unknown': 0}
    for styles in liste:
        if styles is not np.nan:
            styles = ast.literal_eval(styles)  #recognize items as lists instead of string objects
            for style in styles:  #iterates over each cuisine style in the list
                if style in cuisine_dict:
                    cuisine_dict[style] += 1
                else :
                    cuisine_dict[style] = 1
        else:
            cuisine_dict['Unknown'] +=1
    return(cuisine_dict)
            
#Global cuisine styles count
global_cuisine_count = cuisine_count(cuisine_df['Cuisine Style'])
print("Total number of different cuisine styles ('unknown' included) :", len(global_cuisine_count))
print(global_cuisine_count)

#Count for each city and dataframe building
cuisine_count_df = pd.DataFrame()
city_number_styles = pd.Series()  # Initialize the df and series to avoid unwanted concatenation
for city in global_number_rest.index:
    city_cuisine = cuisine_count(cuisine_df[cuisine_df['City'] == city]['Cuisine Style'])
    city_number_styles = pd.concat([city_number_styles, pd.Series(data=len(city_cuisine), index=[city])]) #number of styles for the city in a series
    city_cuisine = pd.DataFrame(city_cuisine, index=[city]) #row for the city
    cuisine_count_df = pd.concat([cuisine_count_df, city_cuisine])
cuisine_count_df = cuisine_count_df.fillna(0)  #Replace NaN by 0
city_number_styles = city_number_styles.sort_values(ascending=False) #sorting values for visualization
print(cuisine_count_df.head(5))
print(city_number_styles)

#Statistical exploration
#print(cuisine_count_df.describe())
#Visualization of global cuisine styles count
pd.DataFrame(global_cuisine_count, index=['Global']).T.sort_values('Global', ascending=False).plot(kind='bar', figsize=(30,10), 
                                                            title='Global cuisine styles', sort_columns=True)
#plt.savefig("figures/global cuisine count.svg")
#plt.savefig("figures/global cuisine count.png")
plt.show()

#Visualization of cuisine style diversity
city_number_styles.plot(kind='bar', figsize=(30,10), title='Cuisine styles diversity (number of different styles)')
#plt.savefig("figures/global cuisine diversity.svg")
#plt.savefig("figures/global cuisine diversity.png")
plt.show()

#Visualization for each city for the 15 main cuisine styles in the city
for city in cuisine_count_df.index:
    cuisine_count_df.loc[city].sort_values(ascending=False).iloc[0:16].plot(kind='bar', 
                                                                            figsize=(30,10), fontsize=20,
                                                                            title='20 main Cuisine styles for restaurants in {}'.format(city))
    #plt.savefig("figures/global cuisine count {}.svg".format(city))
    #plt.savefig("figures/global cuisine count {}.png".format(city))
    plt.show()
#Cities with the most of restaurants with special diets cuisine
special_count_df = cuisine_count_df[['Vegetarian Friendly', 'Gluten Free Options', 'Vegan Options', 'Halal', 'Kosher']]
special_count_df['Total Restaurants'] = global_number_rest
special_count_df.plot(kind='bar', subplots=True, figsize=(30,40), fontsize=20,
                      y=['Vegetarian Friendly', 'Gluten Free Options', 'Vegan Options', 'Halal', 'Kosher'])
#plt.savefig("figures/special diets rest per city.svg")
#plt.savefig("figures/special diets rest per city.png")
plt.show()

#With ratio out of total number of restaurants (normalized)
special_count_df['% Vegetarian Friendly'] = special_count_df['Vegetarian Friendly'] / global_number_rest
special_count_df['% Gluten Free Options'] = special_count_df['Gluten Free Options'] / global_number_rest
special_count_df['% Vegan Options'] = special_count_df['Vegan Options'] / global_number_rest
special_count_df['% Halal'] = special_count_df['Halal'] / global_number_rest
special_count_df['% Kosher'] = special_count_df['Kosher'] / global_number_rest
special_count_df.plot(kind='bar', subplots=True, figsize=(30,40), fontsize=20,
                      y=['% Vegetarian Friendly', '% Gluten Free Options', '% Vegan Options', '% Halal', '% Kosher'])
#plt.savefig("figures/special diets rest per city_ratio.svg")
#plt.savefig("figures/special diets rest per city_ratio.png")
plt.show()

print(special_count_df.head())
print(special_count_df.describe())
#Get least numerous styles per city
#print(cuisine_count_df.head())
#print(cuisine_count_df[cuisine_count_df <10])
rare_styles_df = pd.DataFrame()

#Iterrate over filtered dataset to get only rare cuisine styles in each city
for city, city_row in cuisine_count_df[(cuisine_count_df <5) & (cuisine_count_df >0)].iterrows():
    city_row = city_row.dropna()  #removes NaN values (filtered cuisine styles)
    city_rare_styles = city_row.index.tolist()
    #filter dataset to get corresponding city &cuisine style, no cuisine style NaN rows
    for restaurant, restaurant_row in dataset[dataset['City'] == city].dropna(subset=['Cuisine Style']).iterrows():
        styles = ast.literal_eval(restaurant_row.loc['Cuisine Style'])
        #Check if rare styles of the city is in the restaurant styles (les iterations than the other way)
        for rare_style in city_rare_styles:
            if rare_style in styles:
                rare_styles_df = pd.concat([rare_styles_df, dataset[dataset['ID_TA'] == restaurant_row.loc['ID_TA']]])

#The scripts adds several times a restaurant if it has more than one rare style
rare_styles_df = rare_styles_df.drop_duplicates()
print(rare_styles_df.head())
#Visualization of the properties of the rare styles restaurants with violin plots
print("Restaurants with rare cuisine style in their city: ", rare_styles_df['Name'].count())
print("Rank range: ", rare_styles_df['Ranking'].min(),'\t', rare_styles_df['Ranking'].max())
#Change values because of '$' not recognized
rare_styles_df['PriceRange_new'] = rare_styles_df['Price Range'].map({'$':'low', '$$ - $$$':'mid', '$$$$':'high', 'Unknown':'unknown'})

#Restaurants with rare cuisine styles that ranked in top20
print("Restaurants ranked in top 20 with rare cuisine styles in their city \n", rare_styles_df[rare_styles_df['Ranking'] <=20][['Name', 'City']])

#Swarm plot with rate, price range & ranking adjusted
sns.swarmplot(x=rare_styles_df['Rating'], y=rare_styles_df['Ranking'], hue=rare_styles_df['PriceRange_new'])
sns.ylabel="Rank of the restaurant"
plt.xlabel="Price Range of the restaurant"
plt.title="Repartition of restaurants with rare cuisine styles in their city"
plt.show()

#Swarm plot with rate, price range & number of reviews
sns.swarmplot(x=rare_styles_df['Rating'], y=rare_styles_df['Number of Reviews'], hue=rare_styles_df['PriceRange_new'])
plt.show()

#Swarm plot with rate, price range & ranking adjusted
sns.swarmplot(x=rare_styles_df['Rating'], y=rare_styles_df['Ranking'], hue=rare_styles_df['PriceRange_new'])
sns.ylabel="Rank of the restaurant"
plt.xlabel="Price Range of the restaurant"
plt.title="Repartition of restaurants with rare cuisine styles in their city"
plt.show()
#plt.savefig("figures/rare styles repartiton.png")
#plt.savefig("figures/rare styles repartiton.svg")
#Global Repartition of rates
dataset['Rating'] = dataset['Rating'].fillna('Unknown')
global_rate_df = dataset.groupby('Rating').count()['Name']

#Bar chart
global_rate_df.plot(kind='bar')
plt.title="Repartition of restaurants per rate"
#plt.savefig("figures/rate repartiton.svg")
#plt.savefig("figures/rate repartiton.png")
plt.show()

#Restaurants with rate = -1
#print(dataset[dataset['Rating'] == -1][['Name', 'City']])
#5 rated restaurants subset
five_restaurants = dataset[dataset['Rating'] == 5]
five_restaurants['PriceRange_new'] = five_restaurants['Price Range'].map({'$':'low', '$$ - $$$':'mid', '$$$$':'high', 'Unknown':'unknown'})
print(five_restaurants.head())

#Aggregated by city
five_restaurants_agg = five_restaurants.groupby('City').count()[['Name']]
five_restaurants_agg.columns = ['Number of 5 rated']
five_restaurants_agg['Total Restaurants'] = global_number_rest
five_restaurants_agg['% of 5 rated'] = five_restaurants_agg['Number of 5 rated'] / five_restaurants_agg['Total Restaurants']
print(five_restaurants_agg)
#Visualization of percentage of 5 rated restaurants per city
five_restaurants_agg['% of 5 rated'].sort_values(ascending=False).plot(kind='bar', figsize=(40,10))
plt.ylabel("% of 5 rated restaurants in the city")
#plt.savefig("figures/5 rated percentage city.svg")
#plt.savefig("figures/5 rated percentage city.png")
plt.show()

#Visualization of he rank of the 5 rated restaurants
sns.stripplot(x=five_restaurants['PriceRange_new'], y=five_restaurants['Ranking'],
              jitter=True, size=2)
#plt.savefig("figures/5 rated ranking.svg")
#plt.savefig("figures/5 rated ranking.png")
plt.show()

#Visulization of 5 rated restaurants properties
sns.stripplot(x=five_restaurants['PriceRange_new'], y=five_restaurants['Number of Reviews'],
             jitter=True, size=2)
#plt.savefig("figures/5 rated properties.svg")
#plt.savefig("figures/5 rated properties.png")
plt.show()
print(five_restaurants[five_restaurants['Ranking'] > 15000])
#Global number of reviews and per price range
total_reviews = dataset['Number of Reviews'].sum()
total_reviews_pr = dataset.groupby('Price Range')['Number of Reviews'].sum()
total_reviews_pr = pd.DataFrame({'Number of Reviews': total_reviews_pr})
total_reviews_pr.index = global_pricerange_count.index
#Add weighting by number of restaurants and reviewed restaurants
total_reviews_pr['Reviews/Restaurants'] = total_reviews_pr['Number of Reviews'] / global_pricerange_count['Restaurants']
reviewed_restaurants_df = dataset[dataset['Number of Reviews'] >= 1]  #Subset with reviewed restaurants only
reviewed_restaurants_series =  reviewed_restaurants_df.groupby('Price Range').count()['Name']
reviewed_restaurants_series.index = global_pricerange_count.index
total_reviews_pr['Reviews/Reviewed Restaurants'] = total_reviews_pr['Number of Reviews'] / reviewed_restaurants_series

print("Total number of reviews:", int(total_reviews))
print("Total number of reviews per price range: \n", total_reviews_pr, '\n')

#Total number of reviews per city
reviews_df = pd.DataFrame()
for city in global_number_rest.index:
    city_reviews = dataset[dataset['City'] == city]
    city_reviews = pd.DataFrame([city_reviews['Number of Reviews'].sum()], index=[city], columns=['Number of Reviews'])    
    reviews_df = pd.concat([reviews_df, city_reviews]) 
reviews_df = reviews_df.sort_values('Number of Reviews', ascending=False)
#Reviews per restaurants
reviews_df['Reviews/Restaurant'] = reviews_df['Number of Reviews'] / global_number_rest
#Reviews per reviewed restaurants
reviewed_restaurants_df = reviewed_restaurants_df.groupby('City').count()
reviews_df['Reviews/Reviewed Restaurant'] = reviews_df['Number of Reviews'] / reviewed_restaurants_df['Name']
#Reviews per price range
reviews_count_pr = dataset.groupby(['City', 'Price Range'])['Number of Reviews'].sum()
reviews_count_pr = reviews_count_pr.unstack(level='Price Range')
reviews_count_pr.columns = ['low price range', 'mid price range', 'high price range', 'unknown price range']
reviews_df = pd.concat([reviews_df, reviews_count_pr], axis=1)

print(reviews_df)
print('\n', reviews_df.describe())
#Total reviews per price range in pie chart
total_reviews_pr.iloc[:,0].T.plot.pie(figsize=(5,5), legend=False, autopct='%.0f%%', explode=[0.04 for i in range(4)],
                        title="Global number of reviews by price range" )
#plt.savefig("figures/reviews per pricerange.svg")
#plt.savefig("figures/reviews per pricerange.png")
plt.show()

#Total reviews per price range per restaurants and reviewed restaurants
total_reviews_pr.iloc[:, 1:].plot(kind='bar', title="Reviews per restaurant according to price range")
#plt.savefig("figures/reviews per restaurants per price range.svg")
#plt.savefig("figures/reviews per restaurants per price range.png")
plt.show()

#Visualization of the dataframe containing the reviews dat per city
reviews_df.iloc[:, :3].plot(kind='bar', figsize=(40,30), fontsize=20, subplots=True)
#plt.savefig("figures/reviews per city and restaurant.svg")
#plt.savefig("figures/reviews per city and restaurant.png")
plt.show()
#Dataframe with reviews and rank and rate with fixed ranks (no NaN)
reviews_rankrate_df = dataset.loc[:, ['Name', 'City', 'Number of Reviews' , 'Rating']]
reviews_rankrate_df['Ranking'] = reviews_rankrate_df.index +1
print(reviews_rankrate_df.tail(50))
#Line plot for correlation visualisation
reviews_rankrate_df.plot(kind='scatter', figsize=(60,40), y='Number of Reviews', x='Ranking',
                          title="Correlation between number of reviews and popularity")
plt.xlim((0, 100))
#plt.savefig("figures/correlation_number reviews_rate rank.svg")
#plt.savefig("figures/correlation_number reviews_rate rank.png")
plt.show()
#Subset the dataset with price range = $$$$
dataset_highpr = dataset[dataset['Price Range'] == '$$$$']
#print(dataset_highpr.head(50))
dataset_highpr.drop(['Ranking'])
dataset_highpr['Ranking'] = dataset_highpr.index +1
print(dataset_highpr.info())
print(dataset_highpr[['Number of Reviews', 'Rating']].describe(), "\n compared to the global dataset:\n")
print(dataset[['Number of Reviews', 'Rating']].describe())
#Ranking and number of reviews analysis
dataset_highpr.plot(kind='scatter', x='Ranking', y='Number of Reviews', alpha=0.3, figsize=(40,20),
                   title="Number of reviews by rank for the high price range restaurants")
plt.xlim((0, 500))
#plt.savefig('figures/high price range restaurants.png')
#plt.savefig('figures/high price range restaurants.svg')
plt.show()

#Swarmplots by rate and number of reviews
dataset_highpr.plot(kind='box', subplots=True, figsize=(40,20), title="High price range satistical repartition")
#plt.savefig('figures/high price range repartition.png')
#plt.savefig('figures/high price range repartition.svg')
plt.show()
#Create the dataframe with top 20 restaurants for each city
top_restaurants = dataset[dataset['Ranking'] <= 30]
top_restaurants['PriceRange_new'] = top_restaurants['Price Range'].map({"$": 'low', '$$ - $$$': 'mid', '$$$$':'high'})
top_restaurants['PriceRange_new'] = top_restaurants['PriceRange_new'].fillna('unknown')
print(top_restaurants.info())

#Verification that there are only 30 restaurants per city
top_restaurants.groupby('City').count()['Name'].plot(kind='bar', figsize=(20,2))
plt.show()
print(top_restaurants[(top_restaurants['Price Range'].isnull()) | (top_restaurants['Cuisine Style'].isnull())])
#Common points in the top 30 restaurants of each city
explode_r = [0.04 for k in top_restaurants['Rating'].unique().tolist()]
top_restaurants.groupby('Rating').count().iloc[:, 0].plot.pie(legend=False, autopct='%.0f%%', explode=explode_r,
                                                             title="Repartion of top20 restaurants by rate")
plt.axis('equal')
plt.show()
#plt.savefig('figures/top20 repartition rate.png')
#plt.savefig('figures/top20 repartition rate.svg')

explode_pr = [0.04 for k in top_restaurants['PriceRange_new'].unique().tolist()]
top_restaurants.groupby('PriceRange_new').count().iloc[:, 0].plot.pie(legend=False, autopct='%.0f%%', explode=explode_pr,
                                                             title="Repartion of top20 restaurants by price range")
plt.axis('equal')
#plt.savefig('figures/top20 repartition pricerange.png')
#plt.savefig('figures/top20 repartition pricerange.svg')
plt.show()

top_restaurants.plot(kind='scatter', x='Ranking', y='Number of Reviews', figsize=(20,5), xticks=range(31),
                     alpha=0.5, title='Number of reviews of top20 restaurants')
#plt.savefig('figures/top20 number of reviews.png')
#plt.savefig('figures/top20 number of reviews.svg')
plt.show()
#Cuisine style analysis fo the top restaurants
count = {}
for i,styles in top_restaurants['Cuisine Style'].iteritems():
    if styles is not np.nan:
        styles = ast.literal_eval(styles)
        for style in styles:
            if style in count:
                count[style] += 1
            else:
                count[style]=1
print(count)
count_styles = pd.Series(count)
count_styles.sort_values(ascending=False).plot(kind='bar', figsize=(20,10), title='Cuisine styles among top 20 restaurants')
#plt.savefig('figures/top20 cuisine styles.png')
#plt.savefig('figures/top20 cuisine styles.svg')
plt.show()
