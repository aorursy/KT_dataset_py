#importing the most relevant libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams.update({'font.size': 22})



# wish_df = pd.read_csv('summer-products-with-rating-and-performance_2020-08.csv')

wish_df = pd.read_csv( '../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
print('Dimensions of the df', wish_df.shape)

print(wish_df.isnull().sum()) #checking how many values are not filled in each column

wish_df = wish_df.drop_duplicates() # removing the duplicates if any

print('Dimensions of the df after dropping the duplicates', wish_df.shape)
print('{:<30} {:<15}'.format('column', 'unique values'))

for key in wish_df.keys():

    uniques = len(wish_df[str(key)].unique())

    print('{:<30} {:<15}'.format(str(key) , uniques)) #prints how many unique values are present in a column

    if uniques <= 20 : # If a column has less than or equal to 20 values, it shows the categories

         print('\t:', wish_df[str(key)].unique()) 
wish_df.head()
wish_df['units_sold'] = [10 if x < 10 else x for x in wish_df['units_sold']]

wish_df.iloc[0,:] #Here we take a look at the sample
import seaborn as sns

sns.set(font_scale=1.2)

plt.figure(figsize=(12,6))

sns.barplot(y='merchant_rating_count', x='units_sold', data=wish_df) 
plt.figure(figsize=(12,6))

merch_pic = (wish_df.groupby(['units_sold'])['merchant_has_profile_picture'].value_counts(normalize=True).rename('Percentage').mul(100)

            .reset_index())

# print(merch_pic)

p = sns.barplot(x='units_sold', y='Percentage', hue='merchant_has_profile_picture', data=merch_pic, alpha=0.8)
plt.figure(figsize=(12,6))

sns.scatterplot(x=wish_df.merchant_rating, y=wish_df.units_sold, s=80)
wish_df['origin_country'] = wish_df.origin_country.fillna('Nil') #Marking the gaps as unknown

origin_countries = wish_df.origin_country.unique()

print(origin_countries)

volume_sold = np.zeros(len(origin_countries)) #to calculate the total volume of prodcut sold from a country

for i, country in enumerate(origin_countries):

#     print(i, country)

    volume_sold[i] = wish_df[wish_df['origin_country'] == str(country)]['units_sold'].sum()



# print(volume_sold)

fig, ax = plt.subplots(figsize=(8,4))

ax.set_yscale('log')

sns.scatterplot(x=origin_countries,y=volume_sold, size=volume_sold, sizes=(100,1000))
plt.figure(figsize=(14,6))

geo_sales = (wish_df.groupby(['units_sold'])['origin_country'].value_counts().rename('item_count')

            .reset_index())

# print(merch_pic)

sns.barplot(x='units_sold', y='item_count', hue='origin_country', data=geo_sales, alpha=0.8)

plt.legend(title='origin_country', loc='upper right')

plt.ylim(0,80)
target_countries = wish_df.countries_shipped_to.unique()

# print(target_countries)

fig, ax = plt.subplots(figsize=(15,6))

sns.scatterplot(x=wish_df.countries_shipped_to,y=wish_df.units_sold, s=100, alpha=0.8)
#If no urgency text is present, then filled none.

wish_df['urgency_text'] = wish_df['urgency_text'].fillna('None')

#If has_urgency_banner is not present, then filled with zero.

wish_df['has_urgency_banner'] = wish_df['has_urgency_banner'].fillna(0)



plt.figure(figsize=(10,6))

# sns.countplot(y='rating', hue='units_sold', data=wish_df)

sns.scatterplot(wish_df['rating'], wish_df['units_sold'], size=wish_df.rating_count, sizes=(10,750), alpha=0.75)

plt.legend(loc='upper left')
rating = ['rating_count', 'rating_five_count','rating_four_count', 'rating_three_count', 'rating_two_count', 'rating_one_count']

markers = ['p', 'd', 's', 'v', 'h', 'X']



plt.figure(figsize=(15,6))

plt.ylim(-500, 25000)

plt.xlim(-150, 6000)

for i in range(len(rating)):

    sns.regplot(x=rating[i], y='units_sold', data=wish_df, label=rating[i], marker=markers[i])



plt.legend()

plt.xlabel('Number of ratings')

plt.ylabel('Units sold')

rating.append('units_sold')

rating_corr = wish_df[rating].corr().round(2)

plt.figure(figsize=(8,8))

ax = sns.heatmap(rating_corr.round(2), annot=True, linewidths=.5, cmap='hot')

ax.set_ylim(len(rating), 0) 
from wordcloud import WordCloud



good_sales = wish_df['units_sold'] >= 1000

# with this we can focus only on those items that are sold considerably

alltitle = ''

alltitle_total = ''

for i, j in enumerate(wish_df[good_sales].title_orig):

    sold_units = wish_df[good_sales]['units_sold'].iloc[i]/1000

    # title_orig only focuses on the keywords

    # it doesn't say how much it is sold

    # so in order to understand the extent of popularity of a keyword,

    # it needs to multiplied by the times it's sold

    # to make things more efficient we divide it by 1000

    alltitle += ' ' + j

    # alltitle makes a huge sentence composed of all the words in the title combined together

    alltitle_total += (' ' + j)*int(sold_units)

    # alltitle_total has each word multiplied by the number of products sold



plt.figure(figsize=(16,16))

cloud = WordCloud(max_words=100, background_color="gray").generate(alltitle)

plt.imshow(cloud, interpolation='bilinear')

plt.axis("off")

plt.show()
plt.figure(figsize=(16,16))

cloud = WordCloud(max_words=100, stopwords=['Women', 'Summer', 'Fashion'], background_color="black").generate(alltitle_total)

plt.imshow(cloud, interpolation='bilinear')

plt.axis("off")

plt.show()
from wordcloud import WordCloud

wish_df['product_variation_size_id'] = wish_df['product_variation_size_id'].fillna('X')

wish_df['product_variation_size_id'] = wish_df['product_variation_size_id'].str.replace('Size','')

allsize = ''

allsize_total = ''

for i, j in enumerate(wish_df[good_sales].product_variation_size_id):

    sold_units = wish_df[good_sales]['units_sold'].iloc[i]/1000

    allsize += ' ' + str(j)

    allsize_total += ('  ' + str(j))*int(sold_units)



plt.figure(figsize=(16,16))

cloud = WordCloud(max_words=100, background_color="gray").generate(allsize)

plt.imshow(cloud,interpolation='bilinear')

plt.axis("off")

plt.show()
plt.figure(figsize=(16,16))

cloud = WordCloud(max_words=100, background_color="black").generate(allsize_total)

plt.imshow(cloud, interpolation='bilinear')

plt.axis("off")

plt.show()
allcolors = ''

allcolors_total = ''

wish_df['product_color'] = wish_df['product_color'].fillna('colorful')



for i, j in enumerate(wish_df[good_sales].product_color):

    sold_units = wish_df[good_sales]['units_sold'].iloc[i]/1000

    allcolors += ' ' + str(j)

    allcolors_total += ('  ' + str(j))*int(sold_units)



plt.figure(figsize=(16,16))

cloud = WordCloud(max_words=100, background_color="gray").generate(allcolors)

plt.imshow(cloud, interpolation='bilinear')

plt.axis("off")

plt.show()
plt.figure(figsize=(16,16))

cloud = WordCloud(max_words=100, background_color="black").generate(allcolors_total)

plt.imshow(cloud)

plt.axis("off", interpolation='bilinear')

plt.show()
# all letters made lower case

wish_df['product_color'] = wish_df['product_color'].str.lower()



# to simplify the number of colours some approximations are taken

# all light/dark shades are treated as the same parent color

wish_df['product_color'] = wish_df['product_color'].str.replace('light','')

wish_df['product_color'] = wish_df['product_color'].str.replace('dark','')

wish_df['product_color'] = wish_df['product_color'].str.replace('white & black','black & white')

# now some crude approximations

wish_df['product_color'] = wish_df['product_color'].str.replace('tan','camel')

wish_df['product_color'] = wish_df['product_color'].str.replace('rose','pink')

wish_df['product_color'] = wish_df['product_color'].str.replace('grey','gray')

wish_df['product_color'] = wish_df['product_color'].str.replace('coffee','brown')





# we take these colors are main colors and anything else derived from it as the main color

# except when there is & in the name of the color

colors = ['red', 'green', 'blue', 'black', 'pink', 'yellow', 'white', 'gold']

for subcolor in wish_df['product_color'].unique():

    if '&' not in subcolor:

        for maincolor in colors:

            if maincolor in subcolor:

                wish_df['product_color'] = wish_df['product_color'].str.replace(subcolor, maincolor)
print(wish_df['product_color'].unique())



unique_colors = wish_df['product_color'].unique()

color_sales = np.zeros(len(unique_colors))

for i, icolor in enumerate(unique_colors):

    color_sales[i] = wish_df[wish_df['product_color'] == icolor].units_sold.sum()

good_sales = wish_df[wish_df['units_sold'] >= 5000]





chart = sns.catplot(x='product_color', y='units_sold', kind="strip", data=wish_df[wish_df['units_sold'] > 5000], aspect=3, height=4, s=10);

chart.set_xticklabels(rotation=65, horizontalalignment='right')
plt.figure(figsize=(10,5))

sns.scatterplot(y=wish_df.units_sold, x=wish_df.inventory_total, s=100)
params = ['badges_count', 'badge_local_product', 'badge_product_quality', 'badge_fast_shipping', 'units_sold']

badge_correlation = wish_df[params].corr().round(2) 



# plt.figure(figsize=(6,6))

# ax = sns.heatmap(badge_correlation.round(2), annot=True, linewidths=.5, cmap='hot')

# ax.set_ylim(len(params), 0) 



plt.figure(figsize=(12,6))

badge_sales = (wish_df.groupby(['units_sold'])['badges_count'].value_counts(normalize=True).rename('Percentage').mul(100)

            .reset_index())

# print(merch_pic)

sns.barplot(x='units_sold', y='Percentage', hue='badges_count', data=badge_sales, palette="terrain_r")

plt.legend(title='badges_count', loc=(1,0.5))
plt.figure(figsize=(12,6))



sns.scatterplot(x=wish_df.shipping_option_price, y=wish_df.price, size=wish_df.units_sold, sizes=(100,1000), alpha=0.8)
plt.figure(figsize=(12,6))

banner_sales = (wish_df.groupby(['units_sold'])['has_urgency_banner'].value_counts(normalize=True).rename('Percentage').mul(100)

            .reset_index())

# print(merch_pic)

sns.barplot(x='units_sold', y='Percentage', hue='has_urgency_banner', data=banner_sales, palette="Reds", alpha=0.8)

plt.legend(title='has urgency banner', loc=(0.07,0.81))
plt.figure(figsize=(12,6))

ad_sales = (wish_df.groupby(['units_sold'])['uses_ad_boosts'].value_counts(normalize=True).rename('Percentage').mul(100)

            .reset_index())

# print(merch_pic)

sns.barplot(x='units_sold', y='Percentage', hue='uses_ad_boosts', data=ad_sales, palette="Blues", alpha=0.8)

plt.legend(title='uses_ad_boosts', loc=(0.07,0.81))
# discounted price vs original retail price

wish_df['price_diff'] = wish_df['retail_price']-wish_df['price']

#if price_diff is positive, then retail price is higher. then the customer sees a profit

#if price_diff is negative, then retail price is lower. then the customer doesn't see a profit

wish_df['price_diff_perc'] = (wish_df['retail_price']-wish_df['price'])*100/wish_df['retail_price']



params = ['price', 'retail_price', 'price_diff', 'price_diff_perc', 'units_sold']

params1 = ['price', 'retail_price', 'price_diff_perc', 'units_sold']

price_correlation = wish_df[params].corr().round(2) 



plt.figure(figsize=(10,10))

sns.pairplot(data=wish_df, vars=params, hue='has_urgency_banner',  plot_kws=dict(alpha=0.6) )
pdneg = wish_df[wish_df['price_diff'] < 0 ]

pd0 = wish_df[wish_df['price_diff'] == 0 ]

pd100 = wish_df[(wish_df['price_diff'] <= 100) & (wish_df['price_diff'] > 0)]

pd1000 = wish_df[wish_df['price_diff'] > 100]

plt.figure(figsize=(15,10))

sns.scatterplot(x=pdneg.price_diff, y=pdneg.units_sold, s=100, label='Price diff < 0 EUR' )

sns.scatterplot(x=pd0.price_diff, y=pd0.units_sold, s=100, label='Price diff = 0 EUR' )

sns.scatterplot(x=pd100.price_diff, y=pd100.units_sold, s=100, label='Price diff < 100 EUR' )

sns.scatterplot(x=pd1000.price_diff, y=pd1000.units_sold, s=100, label='Price diff > 100 EUR'  )

plt.legend()

net_sold = wish_df.units_sold.sum()

pdnegsum = pdneg.units_sold.sum()

pd0sum = pd0.units_sold.sum()

pd100sum = pd100.units_sold.sum()

pd1000sum = pd1000.units_sold.sum()

print('If the price_diff is negative, then the retail price is lower. The customer then doesn\'t see an opportunity to save.\n')

print('%2.1f %% products are sold at negative price_diff.'%(pdnegsum/net_sold*100))

print('%2.1f %% products are sold at zero price_diff.'%(pd0sum/net_sold*100))

print('%2.1f %% products are sold at less than 100 EUR price_diff.'%(pd100sum/net_sold*100))

print('%2.1f %% products are sold at more than 100 EUR price_diff.'%(pd1000sum/net_sold*100))
pcneg = wish_df[wish_df['price_diff_perc'] < 0]

pc0 = wish_df[wish_df['price_diff_perc'] == 0]

pc50 = wish_df[(wish_df['price_diff_perc'] <= 50) & (wish_df['price_diff_perc'] > 0)]

pc100 = wish_df[wish_df['price_diff_perc'] > 50]



plt.figure(figsize=(15,10))

sns.scatterplot(x=pcneg.price_diff_perc, y=pcneg.units_sold, label='Price diff % < 0 %' )

sns.scatterplot(x=pc0.price_diff_perc, y=pc0.units_sold, label='Price diff % = 0 %' )

sns.scatterplot(x=pc50.price_diff_perc, y=pc50.units_sold, label='Price diff % = 0-50 %' )

sns.scatterplot(x=pc100.price_diff_perc, y=pc100.units_sold, label='Price diff % >50 %' )

plt.legend()

pcnegsum = pcneg.units_sold.sum()

pc0sum = pc0.units_sold.sum()

pc50sum = pc50.units_sold.sum()

pc100sum = pc100.units_sold.sum()

# print(pcnegsum/net_sold, pc0sum/net_sold, pc50sum/net_sold, pc100sum/net_sold)

print('If the price_diff is negative, then the retail price is lower. The customer then doesn\'t see an opportunity to save. \n')

print('%2.1f %% products are sold at negative price_diff: '%(pcnegsum/net_sold*100))

print('%2.1f %% products are sold at the same price: '%(pc0sum/net_sold*100))

print('%2.1f %% products are sold at less than 0-50 %% price difference: '%(pc50sum/net_sold*100))

print('%2.1f %% products are sold at more than 50-100 %% price difference: '%(pc100sum/net_sold*100))