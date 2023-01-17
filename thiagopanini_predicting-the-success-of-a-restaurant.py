# Standard libraries

import pandas as pd

import numpy as np

from warnings import filterwarnings

filterwarnings('ignore')

pd.set_option('display.max_columns', 500)

from collections import Counter

from PIL import Image



# Viz libs

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from matplotlib.gridspec import GridSpec

from mpl_toolkits.axes_grid.inset_locator import InsetPosition

import folium

from folium.plugins import HeatMap, FastMarkerCluster

from wordcloud import WordCloud



# Geolocation libs

from geopy.geocoders import Nominatim



# Utils modules

from custom_transformers import *

from viz_utils import *

from ml_utils import *



# ML libs

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

import shap
# Reading restaurants data

data_path = r'../input/zomato-bangalore-restaurants/zomato.csv'

df_restaurants = import_data(path=data_path, n_lines=5000)



# Results

print(f'Dataset shape: {df_restaurants.shape}')

df_restaurants.head()
# An overview from the data

df_overview = data_overview(df_restaurants)

df_overview
# Changing the data type from approx_cost columns

df_restaurants['approx_cost'] = df_restaurants['approx_cost(for two people)'].astype(str).apply(lambda x: x.replace(',', ''))

df_restaurants['approx_cost'] = df_restaurants['approx_cost'].astype(float)



# Extracting the rate in a float column

df_restaurants['rate_num'] = df_restaurants['rate'].astype(str).apply(lambda x: x.split('/')[0])

while True:

    try:

        df_restaurants['rate_num'] = df_restaurants['rate_num'].astype(float)

        break

    except ValueError as e1:

        noise_entry = str(e1).split(":")[-1].strip().replace("'", "")

        print(f'Threating noisy entrance on rate: {noise_entry}')

        df_restaurants['rate_num'] = df_restaurants['rate_num'].apply(lambda x: x.replace(noise_entry, str(np.nan)))



# Dropping old columns

df_restaurants.drop(['approx_cost(for two people)', 'rate'], axis=1, inplace=True)

df_restaurants.head()
# Building a figure

fig = plt.figure(constrained_layout=True, figsize=(15, 9))



# Axis definition with GridSpec

gs = GridSpec(1, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])

ax2 = fig.add_subplot(gs[0, 1:3])



# Axis 1 - Big Number for total restaurants and total types in the data

total_restaurants = len(df_restaurants)

total_types = len(df_restaurants['rest_type'].value_counts())

ax1.text(0.00, 0.75, 'There are', fontsize=14, ha='center')

ax1.text(0.00, 0.63, f'{total_restaurants}', fontsize=64, color='orange', ha='center')

ax1.text(0, 0.59, 'restaurants in Bengaluru divided into', fontsize=14, ha='center')

ax1.text(0.00, 0.43, total_types, fontsize=44, ha='center', color='orange', style='italic', weight='bold',

         bbox=dict(facecolor='gold', alpha=0.5, pad=14, boxstyle='round, pad=.7'))

ax1.text(0, 0.39, 'different types', fontsize=14, ha='center')

ax1.axis('off')



# Axis 2 - Total number of restaurants per type (Top N)

top = 10

single_countplot(df_restaurants, ax2, x='rest_type', top=top)

ax2.set_title(f'Top {top} Restaurants Type in Bengaluru', color='dimgrey', size=18)

for tick in ax2.get_xticklabels():

    tick.set_rotation(25)

    

# Axis 3 - Representative of the top two restaurant type

df_restaurants['top_types'] = df_restaurants['rest_type'].apply(lambda x: 'Quick Bites + Casual Dining' if x in ('Quick Bites', 'Casual Dining') else 'Other')

ax3 = plt.axes([0, 0, 1, 1])

ip = InsetPosition(ax2, [0.57, 0.3, 0.6, 0.65])

ax3.set_axes_locator(ip)

donut_plot(df_restaurants, col='top_types', ax=ax3, colors=['darkslateblue', 'silver'], title='')
popular_franchises = df_restaurants.groupby(by='name', as_index=False).agg({'votes': 'sum',

                                                                            'url': 'count',

                                                                            'approx_cost': 'mean',

                                                                            'rate_num': 'mean'})

popular_franchises.columns = ['name', 'total_votes', 'total_unities', 'mean_approx_cost', 'mean_rate_num']

popular_franchises['votes_per_unity'] = popular_franchises['total_votes'] / popular_franchises['total_unities']

popular_franchises = popular_franchises.sort_values(by='total_unities', ascending=False)

popular_franchises = popular_franchises.loc[:, ['name', 'total_unities', 'total_votes', 'votes_per_unity',

                                                'mean_approx_cost', 'mean_rate_num']]



# Correcting a restaurant name

bug_name = 'SantÃ\x83Â\x83Ã\x82Â\x83Ã\x83Â\x82Ã\x82Â\x83Ã\x83Â\x83Ã\x82Â\x82Ã\x83Â\x82Ã\x82Â\x83Ã\x83Â\x83Ã\x82Â\x83Ã\x83Â\x82Ã\x82Â\x82Ã\x83Â\x83Ã\x82Â\x82Ã\x83Â\x82Ã\x82Â© Spa Cuisine'

popular_franchises['name'] = popular_franchises['name'].apply(lambda x: 'Santa Spa Cusisine' if x == bug_name else x)





popular_franchises.head(10)
# Creating a figure por restaurants overview analysis

fig, axs = plt.subplots(3, 3, figsize=(15, 15))



# Plot Pack 01 - Most popular restaurants (votes)

sns.barplot(x='total_votes', y='name', data=popular_franchises.sort_values(by='total_votes', ascending=False).head(),

            ax=axs[1, 0], palette='plasma')

axs[1, 0].set_title('Top 5 Most Voted Restaurants', size=12)

sns.barplot(x='total_votes', y='name', 

            data=popular_franchises.sort_values(by='total_votes', ascending=False).query('total_votes > 0').tail(),

            ax=axs[2, 0], palette='plasma_r')

axs[2, 0].set_title('Top 5 Less Voted Restaurants\n(with at least 1 vote)', size=12)

for ax in axs[1, 0], axs[2, 0]:

    ax.set_xlabel('Total Votes')

    ax.set_xlim(0, popular_franchises['total_votes'].max())

    format_spines(ax, right_border=False)

    ax.set_ylabel('')



# Annotations

axs[0, 0].text(0.50, 0.30, int(popular_franchises.total_votes.mean()), fontsize=45, ha='center')

axs[0, 0].text(0.50, 0.12, 'is the average of votes', fontsize=12, ha='center')

axs[0, 0].text(0.50, 0.00, 'received by restaurants', fontsize=12, ha='center')

axs[0, 0].axis('off')



# Plot Pack 02 - Cost analysis

sns.barplot(x='mean_approx_cost', y='name', data=popular_franchises.sort_values(by='mean_approx_cost', ascending=False).head(),

            ax=axs[1, 1], palette='plasma')

axs[1, 1].set_title('Top 5 Most Expensives Restaurants', size=12)

sns.barplot(x='mean_approx_cost', y='name', 

            data=popular_franchises.sort_values(by='mean_approx_cost', ascending=False).query('mean_approx_cost > 0').tail(),

            ax=axs[2, 1], palette='plasma_r')

axs[2, 1].set_title('Top 5 Less Expensive Restaurants', size=12)

for ax in axs[1, 1], axs[2, 1]:

    ax.set_xlabel('Avg Approx Cost')

    ax.set_xlim(0, popular_franchises['mean_approx_cost'].max())

    format_spines(ax, right_border=False)

    ax.set_ylabel('')



# Annotations

axs[0, 1].text(0.50, 0.30, round(popular_franchises.mean_approx_cost.mean(), 2), fontsize=45, ha='center')

axs[0, 1].text(0.50, 0.12, 'is mean approx cost', fontsize=12, ha='center')

axs[0, 1].text(0.50, 0.00, 'for Bengaluru restaurants', fontsize=12, ha='center')

axs[0, 1].axis('off')



# Plot Pack 03 - Rate analysis

sns.barplot(x='mean_rate_num', y='name', data=popular_franchises.sort_values(by='mean_rate_num', ascending=False).head(),

            ax=axs[1, 2], palette='plasma')

axs[1, 2].set_title('Top 5 Rstaurants with Highest Rates', size=12)

sns.barplot(x='mean_rate_num', y='name', 

            data=popular_franchises.sort_values(by='mean_rate_num', ascending=False).query('mean_rate_num > 0').tail(),

            ax=axs[2, 2], palette='plasma_r')

axs[2, 2].set_title('Top 5 Restaurants with Lowest Rate', size=12)

for ax in axs[1, 2], axs[2, 2]:

    ax.set_xlabel('Avg Rate')

    ax.set_xlim(0, popular_franchises['mean_rate_num'].max())

    format_spines(ax, right_border=False)

    ax.set_ylabel('')



# Annotations

axs[0, 2].text(0.50, 0.30, round(popular_franchises.mean_rate_num.mean(), 2), fontsize=45, ha='center')

axs[0, 2].text(0.50, 0.12, 'is mean rate given by customers', fontsize=12, ha='center')

axs[0, 2].text(0.50, 0.00, 'for Bengaluru restaurants', fontsize=12, ha='center')

axs[0, 2].axis('off')



plt.tight_layout()

plt.suptitle('The Best and the Worst Restaurants to Visit in Bengaluru', size=16)

plt.show()
fig, axs = plt.subplots(1, 2, figsize=(15, 10))

donut_plot(df_restaurants, col='book_table', colors=['crimson', 'mediumseagreen'], ax=axs[0], 

           title='Book Table Service in Bengaluru')

donut_plot(df_restaurants, col='online_order', colors=['darkslateblue', 'lightsalmon'], ax=axs[1], 

           title='Online Order Service in Bengaluru')
# Building a figure

fig = plt.figure(constrained_layout=True, figsize=(15, 12))



# Axis definition with GridSpec

gs = GridSpec(2, 5, figure=fig)

ax2 = fig.add_subplot(gs[0, :3])

ax3 = fig.add_subplot(gs[0, 3:])

ax4 = fig.add_subplot(gs[1, :3])

ax5 = fig.add_subplot(gs[1, 3:])



# First Line (01) - Rate

sns.kdeplot(df_restaurants.query('rate_num > 0 & book_table == "Yes"')['rate_num'], ax=ax2,

             color='mediumseagreen', shade=True, label='With Book Table Service')

sns.kdeplot(df_restaurants.query('rate_num > 0 & book_table == "No"')['rate_num'], ax=ax2,

             color='crimson', shade=True, label='Without Book Table Service')

ax2.set_title('Restaurants Rate Distribution by Book Table Service Offer', color='dimgrey', size=14)

sns.boxplot(x='book_table', y='rate_num', data=df_restaurants, palette=['mediumseagreen', 'crimson'], ax=ax3)

ax3.set_title('Box Plot for Rate and Book Table Service', color='dimgrey', size=14)



# First Line (01) - Cost

sns.kdeplot(df_restaurants.query('approx_cost > 0 & book_table == "Yes"')['approx_cost'], ax=ax4,

             color='mediumseagreen', shade=True, label='With Book Table Service')

sns.kdeplot(df_restaurants.query('approx_cost > 0 & book_table == "No"')['approx_cost'], ax=ax4,

             color='crimson', shade=True, label='Without Book Table Service')

ax4.set_title('Restaurants Approx Cost Distribution by Book Table Service Offer', color='dimgrey', size=14)

sns.boxplot(x='book_table', y='approx_cost', data=df_restaurants, palette=['mediumseagreen', 'crimson'], ax=ax5)

ax5.set_title('Box Plot for Cost and Book Table Service', color='dimgrey', size=14)





# Customizing plots

for ax in [ax2, ax3, ax4, ax5]:

    format_spines(ax, right_border=False)

    

plt.tight_layout()
# Building a figure

fig = plt.figure(constrained_layout=True, figsize=(15, 12))



# Axis definition with GridSpec

gs = GridSpec(2, 5, figure=fig)

ax2 = fig.add_subplot(gs[0, :3])

ax3 = fig.add_subplot(gs[0, 3:])

ax4 = fig.add_subplot(gs[1, :3])

ax5 = fig.add_subplot(gs[1, 3:])



# First Line (01) - Rate

sns.kdeplot(df_restaurants.query('rate_num > 0 & online_order == "Yes"')['rate_num'], ax=ax2,

             color='darkslateblue', shade=True, label='With Online Order Service')

sns.kdeplot(df_restaurants.query('rate_num > 0 & online_order == "No"')['rate_num'], ax=ax2,

             color='lightsalmon', shade=True, label='Without Online Order Service')

ax2.set_title('Restaurants Rate Distribution by Online Order Service Offer', color='dimgrey', size=14)

sns.boxplot(x='online_order', y='rate_num', data=df_restaurants, palette=['darkslateblue', 'lightsalmon'], ax=ax3)

ax3.set_title('Box Plot for Rate and Online Order Service', color='dimgrey', size=14)



# First Line (01) - Cost

sns.kdeplot(df_restaurants.query('approx_cost > 0 & online_order == "Yes"')['approx_cost'], ax=ax4,

             color='darkslateblue', shade=True, label='With Online Order Service')

sns.kdeplot(df_restaurants.query('approx_cost > 0 & book_table == "No"')['approx_cost'], ax=ax4,

             color='lightsalmon', shade=True, label='Without Online Order Service')

ax4.set_title('Restaurants Approx Cost Distribution by Online Order Service Offer', color='dimgrey', size=14)

sns.boxplot(x='online_order', y='approx_cost', data=df_restaurants, palette=['darkslateblue', 'lightsalmon'], ax=ax5)

ax5.set_title('Box Plot for Cost and Online Order Service', color='dimgrey', size=14)





# Customizing plots

for ax in [ax2, ax3, ax4, ax5]:

    format_spines(ax, right_border=False)

    

plt.tight_layout()
# Grouping data into location

good_ones = df_restaurants.groupby(by='location', as_index=False).agg({'votes': 'sum',

                                                                       'url': 'count',

                                                                       'approx_cost': 'mean',

                                                                       'rate_num': 'mean'})

good_ones.columns = ['location', 'total_votes', 'total_unities', 'mean_approx_cost', 'mean_rate_num']

good_ones['votes_per_unity'] = good_ones['total_votes'] / good_ones['total_unities']

good_ones = good_ones.sort_values(by='total_unities', ascending=False)

good_ones = good_ones.loc[:, ['location', 'total_unities', 'total_votes', 'votes_per_unity',

                                                'mean_approx_cost', 'mean_rate_num']]

good_ones.head(10)
# Creating a figure por restaurants overview analysis

fig, axs = plt.subplots(3, 3, figsize=(15, 15))

list_cols = ['total_votes', 'mean_approx_cost', 'mean_rate_num']



# PLotting best and worst by grouped data

answear_plot(grouped_data=good_ones, grouped_col='location', axs=axs, list_cols=list_cols, top=10, palette='magma')



# Finishing the chart

plt.suptitle('Where Are the Top Restaurants in Bengaluru?', size=16)

plt.tight_layout()

plt.show()
# Grouping data by city

city_group = df_restaurants.groupby(by='listed_in(city)', as_index=False).agg({'rate_num': 'mean',

                                                                               'approx_cost': 'mean'})

city_group.sort_values(by='rate_num', ascending=False, inplace=True)



# Ploting

fig, ax = plt.subplots(figsize=(15, 8))

sns.barplot(x='listed_in(city)', y='approx_cost', data=city_group, palette='cividis', 

            order=city_group['listed_in(city)'])

ax2 = ax.twinx()

sns.lineplot(x='listed_in(city)', y='rate_num', data=city_group, color='gray', ax=ax2, sort=False)



# Labeling line chart (rate)

xs = np.arange(0, len(city_group), 1)

ys = city_group['rate_num']

for x,y in zip(xs, ys):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text

                 (x,y), # this is the point to label

                 textcoords="offset points", # how to position the text

                 xytext=(0,10), # distance from text to points (x,y)

                 ha='center', # horizontal alignment can be left, right or center

                 color='black')

    

# Labeling bar chart (cost)

for p in ax.patches:

    x = p.get_bbox().get_points()[:, 0]

    y = p.get_bbox().get_points()[1, 1]

    ax.annotate('{}'.format(int(y)), (x.mean(), 15), va='bottom', rotation='vertical', color='white', 

                fontweight='bold')



# Customizing chart

format_spines(ax)

format_spines(ax2)

ax.tick_params(axis='x', labelrotation=90)

ax.set_title('Bengaluru Cities and all its Restaurants by Approx Cost (bars) and Rate (line)')

plt.show()
# Extracting lat and long from the restaurant city using an API service

geolocator = Nominatim(user_agent="Y_BzShFZceZ_rj_t-cI13w")



# Creating a auxiliar dataset with cities location (reducing the API calls and time consuming by consequence)

cities_aux = pd.DataFrame(df_restaurants['listed_in(city)'].value_counts())

cities_aux.reset_index(inplace=True)

cities_aux.columns = ['city', 'total_restaurants']



# Extracting cities lat and long features

cities_aux['lat'] = cities_aux['city'].apply(lambda x: geolocator.geocode(x)[1][0])

cities_aux['lng'] = cities_aux['city'].apply(lambda x: geolocator.geocode(x)[1][1])



# Adding more features do further analysis

city_group = df_restaurants.groupby(by='listed_in(city)', as_index=False).agg({'votes': 'sum',

                                                                               'approx_cost': 'mean',

                                                                               'rate_num': 'mean'})

city_group.columns = ['city', 'total_votes', 'avg_approx_cost', 'avg_rate_num']



# Creating an unique city data

cities_aux = cities_aux.merge(city_group, how='left', on='city')



# Merging the original data to the grouped cities lat and long

df_restaurants = df_restaurants.merge(cities_aux, how='left', left_on='listed_in(city)', right_on='city')

df_restaurants.drop(['city', 'total_restaurants'], axis=1, inplace=True)



# Results on cities grouped data

cities_aux
# Zipping locations for folium map

locations = list(zip(df_restaurants['lat'].values, df_restaurants['lng'].values))



# Creating a map using folium

map1 = folium.Map(

    location=[12.97, 77.63],

    zoom_start=11.5

)



# Plugin: FastMarkerCluster

FastMarkerCluster(data=locations).add_to(map1)



map1
map1 = folium.Map(

    location=[12.97, 77.63],

    zoom_start=11.0,

    tiles='cartodbdark_matter'

)



HeatMap(

    data=cities_aux.loc[:, ['lat', 'lng', 'avg_rate_num']],

    radius=35

).add_to(map1)



map1
rest_types = list(df_restaurants['listed_in(type)'].value_counts().index)

colors = ['darkslateblue', 'mediumseagreen', 'gray', 'salmon', 'cornflowerblue', 'cadetblue', 'gold']



fig, axs = plt.subplots(2, 1, figsize=(17, 15))

for r_type in rest_types:

    idx = rest_types.index(r_type)

    kde_data = df_restaurants[(df_restaurants['rate_num'] > 0) & (df_restaurants['listed_in(type)'] == r_type)]

    sns.kdeplot(kde_data['rate_num'], ax=axs[0], color=colors[idx], shade=True, label=r_type)

    sns.kdeplot(kde_data['approx_cost'], ax=axs[1], color=colors[idx], shade=True, label=r_type)



# Customizing charts

axs[0].set_title('Distribution of Rate by Restaurant Type', color='dimgrey', size=18)

axs[1].set_title('Distribution of Approx Cost by Restaurant Type', color='dimgrey', size=18)

for ax in axs:

    format_spines(ax, right_border=False)

plt.tight_layout()
# Creating a list with all options available

cuisines = list(df_restaurants['cuisines'].astype(str).values)

cuisines_word_list = []

for lista in [c.split(',') for c in cuisines]:

    for word in lista:

        cuisines_word_list.append(word.strip())

        

# Creating a Counter for unique options and generating the wordcloud

cuisines_wc_dict = Counter(cuisines_word_list)



wordcloud = WordCloud(width=1280, height=720, collocations=False, random_state=42, 

                      colormap='magma', background_color='white').generate_from_frequencies(cuisines_wc_dict)



# Visualizing the WC created and the total for each cuisine

fig, axs = plt.subplots(1, 2, figsize=(20, 12))

ax1 = axs[0]

ax2 = axs[1]

ax1.imshow(wordcloud)

ax1.axis('off')

ax1.set_title('WordCloud for Cuisines Available on Bengaluru Restaurants', size=18, pad=20)



# Total for each cuisine

df_cuisines = pd.DataFrame()

df_cuisines['cuisines'] = cuisines_wc_dict.keys()

df_cuisines['amount'] = cuisines_wc_dict.values()

df_cuisines.sort_values(by='amount', ascending=False, inplace=True)

sns.barplot(x='cuisines', y='amount', data=df_cuisines.head(10), palette='magma', ax=ax2)

format_spines(ax2, right_border=False)

ax2.set_title('Top 10 Cuisines in Bengaluru Restaurants', size=18)



# Customizing chart

ncount = df_cuisines['amount'].sum()

x_ticks = [item.get_text() for item in ax2.get_xticklabels()]

ax2.set_xticklabels(x_ticks, rotation=45, fontsize=14)

for p in ax2.patches:

    x = p.get_bbox().get_points()[:, 0]

    y = p.get_bbox().get_points()[1, 1]

    ax2.annotate('{}\n{:.1f}%'.format(int(y), 100. * y / ncount), (x.mean(), y), fontsize=14, ha='center', va='bottom')



plt.tight_layout()

plt.show()
# Creating a list with all options available

dishes = list(df_restaurants['dish_liked'].dropna().astype(str).values)

dishes_word_list = []

for lista in [c.split(',') for c in dishes]:

    for word in lista:

        dishes_word_list.append(word.strip())

        

# Creating a Counter for unique options and generating the wordcloud

dished_wc_dict = Counter(dishes_word_list)



# Reading and preparing a mask for serving as wordcloud background

food_mask = np.array(Image.open("../input/img-icons/delivery_icon.png"))

food_mask = food_mask[:, :, -1]

transf_mask = np.ndarray((food_mask.shape[0], food_mask.shape[1]), np.int32)

for i in range(len(food_mask)):

    transf_mask[i] = [255 if px == 0 else 0 for px in food_mask[i]]



# Generating the wordcloud    

wordcloud = WordCloud(width=1000, height=500, collocations=False, random_state=42, colormap='rainbow', 

                      background_color='black', mask=transf_mask).generate_from_frequencies(dished_wc_dict)



# Visualizing the WC created

plt.figure(figsize=(20, 17))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('WordCloud for Dish Liked on Bengaluru Restaurants', size=18, pad=20)

plt.show()
# Splitting restaurants

df_restaurants['rated'] = df_restaurants['rate_num'].apply(lambda x: 1 if x >= 0 else 0)

new_restaurants = df_restaurants.query('rated == 0')

train_val_restaurants = df_restaurants.query('rated == 1')



# PLotting a donut chart for seeing the distribution

fig, ax = plt.subplots(figsize=(10, 10))

donut_plot(df_restaurants, col='rated', ax=ax, label_names=['Rated', 'New or Not Rated'], 

           colors=['darkslateblue', 'silver'], title='Amount of Rated and Non Rated Restaurants',

           text=f'Total of Restaurants:\n{len(df_restaurants)}')
# Defining a custom threshold for splitting restaurants into good and bad

threshold = 3.75

train_val_restaurants['target'] = train_val_restaurants['rate_num'].apply(lambda x: 1 if x >= threshold else 0)



# Donut chart

fig, ax = plt.subplots(figsize=(10, 10))

label_names = ['Bad' if target == 0 else 'Good' for target in train_val_restaurants['target'].value_counts().index]

color_list = ['salmon' if label == 'Bad' else 'cadetblue' for label in label_names]

donut_plot(train_val_restaurants, col='target', ax=ax, label_names=label_names, 

           colors=color_list, title='Amount of Good and Bad Restaurants \n(given the selected threshold)',

           text=f'Total of Restaurants:\n{len(train_val_restaurants)}\n\nThreshold: \n{threshold}')
# Selecting initial features

initial_features = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines', 

                    'listed_in(type)', 'listed_in(city)', 'approx_cost', 'target']

train_val_restaurants = train_val_restaurants.loc[:, initial_features]



# Extracting new features

train_val_restaurants['multiple_types'] = train_val_restaurants['rest_type'].astype(str).apply(lambda x: len(x.split(',')))

train_val_restaurants['total_cuisines'] = train_val_restaurants['cuisines'].astype(str).apply(lambda x: len(x.split(',')))



# Dropping another ones

train_val_restaurants.drop('cuisines', axis=1, inplace=True)

train_val_restaurants.head()
# Splitting the data

X = train_val_restaurants.drop('target', axis=1)

y = train_val_restaurants['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
# Splitting features by data type

cat_features= [col for col, dtype in X_train.dtypes.items() if dtype == 'object']

num_features = [col for col, dtype in X_train.dtypes.items() if dtype != 'object']



# Apply encoding for categorical features

X_train_cat = X_train[cat_features]

for col in cat_features:

    col_encoded = pd.get_dummies(X_train_cat[col], prefix=col, dummy_na=True)

    X_train_cat = X_train_cat.merge(col_encoded, left_index=True, right_index=True)

    X_train_cat.drop(col, axis=1, inplace=True)

    

print(f'Total categorical features after encoding: {X_train_cat.shape[1]}')
# Class for applying initial prep on key columns

class PrepareCostAndRate(BaseEstimator, TransformerMixin):

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        # Extracting the approx cost feature

        X['approx_cost'] = X['approx_cost(for two people)'].astype(str).apply(lambda x: x.replace(',', '.'))

        X['approx_cost'] = X['approx_cost'].astype(float)

        

        # Extracting the rate feature

        X['rate_num'] = X['rate'].astype(str).apply(lambda x: x.split('/')[0])

        while True:

            try:

                X['rate_num'] = X['rate_num'].astype(float)

                break

            except ValueError as e1:

                noise_entry = str(e1).split(":")[-1].strip().replace("'", "")

                #print(f'Threating noisy entrance on rate feature: {noise_entry}')

                X['rate_num'] = X['rate_num'].apply(lambda x: x.replace(noise_entry, str(np.nan)))              

        

        return X



# Class for selection the initial features

class InitialFeatureSelection(BaseEstimator, TransformerMixin):

    

    def __init__(self, initial_features=['online_order', 'book_table', 'location', 'rest_type', 'cuisines', 

                                         'listed_in(type)', 'listed_in(city)', 'approx_cost', 'rate_num']):

        self.initial_features = initial_features

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        return X[self.initial_features]

                

# Class for creating some features

class RestaurantAdditionalFeatures(BaseEstimator, TransformerMixin):

    

    def __init__(self, multiples_types=True, total_cuisines=True, top_locations=10, top_cities=10, top_types=10):

        self.multiples_types = multiples_types

        self.total_cuisines = total_cuisines

        self.top_locations = top_locations

        self.top_cities = top_cities

        self.top_types = top_types

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        

        # Adding features based on counting of restaurant types and cuisines

        if self.multiples_types:

            X['multiple_types'] = X['rest_type'].astype(str).apply(lambda x: len(x.split(',')))

        if self.total_cuisines:

            X['total_cuisines'] = X['cuisines'].astype(str).apply(lambda x: len(x.split(',')))

            X.drop('cuisines', axis=1, inplace=True)

            

        # Creating for features for reducing granularity on location

        main_locations = list(X['location'].value_counts().index)[:self.top_locations]

        X['location_feature'] = X['location'].apply(lambda x: x if x in main_locations else 'Other')

        X.drop('location', axis=1, inplace=True)

        

        # Creating for features for reducing granularity on city

        main_cities = (X['listed_in(city)'].value_counts().index)[:self.top_cities]

        X['city_feature'] = X['listed_in(city)'].apply(lambda x: x if x in main_cities else 'Other')

        X.drop('listed_in(city)', axis=1, inplace=True)

        

        # Creating for features for reducing granularity on restaurant type

        main_rest_type = (X['rest_type'].value_counts().index)[:self.top_types]

        X['type_feature'] = X['rest_type'].apply(lambda x: x if x in main_rest_type else 'Other')

        X.drop('rest_type', axis=1, inplace=True)

        

        return X

            

# Class for creating a target based on a threshold (training only)

class CreateTarget(BaseEstimator, TransformerMixin):

    

    def __init__(self, threshold=3.75):

        self.threshold = threshold

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X['target'] = X['rate_num'].apply(lambda x: 1 if x >= self.threshold else 0)

        

        return X

    

# Class for splitting the data into new (not rated) and old (rated) restaurants

class SplitRestaurants(BaseEstimator, TransformerMixin):

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        # Splits the restaurants based on rate column (rated and non rated)

        rated = X[~X['rate_num'].isnull()]

        non_rated = X[X['rate_num'].isnull()]

        

        # Dropping the rate column

        rated.drop('rate_num', axis=1, inplace=True)

        non_rated.drop('rate_num', axis=1, inplace=True)

        

        return rated, non_rated
# Reading raw data

data_path = r'../input/zomato-bangalore-restaurants/zomato.csv'

raw_data = import_data(path=data_path, n_lines=5000)



# Defining a commoon pipeline to be applied after reading the raw data

common_pipeline = Pipeline([

    ('initial_preparator', PrepareCostAndRate()),

    ('selector', InitialFeatureSelection()),

    ('feature_adder', RestaurantAdditionalFeatures()),

    ('target_creator', CreateTarget()),

    ('new_splitter', SplitRestaurants())

])



# Applying the initial pipeline

train_restaurants, new_restaurants = common_pipeline.fit_transform(raw_data)

print(f'Total restaurants to be used on training: {len(train_restaurants)}')

print(f'Total restaurants to be used on prediction: {len(new_restaurants)}')
train_restaurants.head()
# Splitting into training and testing data

X = train_restaurants.drop('target', axis=1)

y = train_restaurants['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)



# Splitting into cat and num data

cat_features = [col for col, dtype in X_train.dtypes.items() if dtype == 'object']

num_features = [col for col, dtype in X_train.dtypes.items() if dtype != 'object']



# Building a numerical processing pipeline

num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median'))

])



# Building a categorical processing pipeline

cat_pipeline = Pipeline([

    ('encoder', DummiesEncoding(dummy_na=True))

])



# Building a complete Pipeline

full_pipeline = ColumnTransformer([

    ('num', num_pipeline, num_features),

    ('cat', cat_pipeline, cat_features)

])



# Applying the full pipeline into the data

X_train_prep = full_pipeline.fit_transform(X_train)

X_test_prep = full_pipeline.fit_transform(X_test)

print(f'Shape of X_train_prep: {X_train_prep.shape}')

print(f'Shape of X_test_prep: {X_test_prep.shape}')



# returning categorical features after encoding and creating a new set of features after the pipeline

encoded_features = full_pipeline.named_transformers_['cat']['encoder'].features_after_encoding

model_features = num_features + encoded_features

print(f'\nSanity check! Number of features after the pipeline (must be the same as shape[1]): {len(model_features)}')
# Logistic Regression hyperparameters

logreg_param_grid = {

    'C': np.linspace(0.1, 10, 20),

    'penalty': ['l1', 'l2'],

    'class_weight': ['balanced', None],

    'random_state': [42],

    'solver': ['liblinear']

}



# Decision Trees hyperparameters

tree_param_grid = {

    'criterion': ['entropy', 'gini'],

    'max_depth': [3, 5, 10, 20],

    'max_features': np.arange(1, X_train.shape[1]),

    'class_weight': ['balanced', None],

    'random_state': [42]

}



# Random Forest hyperparameters

forest_param_grid = {

    'bootstrap': [True, False],

    'max_depth': [3, 5, 10, 20, 50],

    'n_estimators': [50, 100, 200, 500],

    'random_state': [42],

    'max_features': ['auto', 'sqrt'],

    'class_weight': ['balanced', None]

}



# LightGBM hyperparameters

lgbm_param_grid = {

    'num_leaves': list(range(8, 92, 4)),

    'min_data_in_leaf': [10, 20, 40, 60, 100],

    'max_depth': [3, 4, 5, 6, 8, 12, 16],

    'learning_rate': [0.1, 0.05, 0.01, 0.005],

    'bagging_freq': [3, 4, 5, 6, 7],

    'bagging_fraction': np.linspace(0.6, 0.95, 10),

    'reg_alpha': np.linspace(0.1, 0.95, 10),

    'reg_lambda': np.linspace(0.1, 0.95, 10),

}



lgbm_fixed_params = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting_type': 'gbdt',

    'num_leaves': 31,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'learning_rate': 0.05,

    'verbose': 0

}
# Setting up classifiers

set_classifiers = {

    'LogisticRegression': {

        'model': LogisticRegression(),

        'params': logreg_param_grid

    },

    'DecisionTrees': {

        'model': DecisionTreeClassifier(),

        'params': tree_param_grid

    },

    'RandomForest': {

        'model': RandomForestClassifier(),

        'params': forest_param_grid

    },

    'LightGBM': {

        'model': lgb.LGBMClassifier(**lgbm_fixed_params),

        'params': lgbm_param_grid

    }

}
# Creating an instance for the homemade class BinaryClassifiersAnalysis

clf_tool = BinaryClassifiersAnalysis()

clf_tool.fit(set_classifiers, X_train_prep, y_train, random_search=True, cv=5, verbose=5)
# Evaluating metrics

df_performances = clf_tool.evaluate_performance(X_train_prep, y_train, X_test_prep, y_test, cv=5)

df_performances.reset_index(drop=True).style.background_gradient(cmap='Blues')
clf_tool.plot_roc_curve()
clf_tool.plot_confusion_matrix(classes=['Good', 'Bad'])
fig, ax = plt.subplots(figsize=(13, 11))

forest_feature_importance = clf_tool.feature_importance_analysis(model_features, specific_model='LightGBM', ax=ax)

plt.show()
clf_tool.plot_score_distribution('LightGBM', shade=True)
clf_tool.plot_score_bins('LightGBM', bin_range=0.1)
# Returning the LightGBM model and applying shap value

model = clf_tool.classifiers_info['LightGBM']['estimator']

explainer = shap.TreeExplainer(model)

df_X_train_prep = pd.DataFrame(X_train_prep, columns=model_features)

shap_values = explainer.shap_values(df_X_train_prep)



# Plotting a summary plot using shap

shap.summary_plot(shap_values, df_X_train_prep)
# Applying the full pipeline into new restaurants

new_restaurants_prep = full_pipeline.fit_transform(new_restaurants.drop('target', axis=1))



# Returning the best model and predicting the rate for new restaurants

model = clf_tool.classifiers_info['LightGBM']['estimator']

y_pred = model.predict(new_restaurants_prep)

y_probas = model.predict_proba(new_restaurants_prep)

y_scores = y_probas[:, 1]



# Labelling new data

new_restaurants['success_class'] = y_pred

new_restaurants['success_proba'] = y_scores

new_restaurants.head()
# Looking at the score distribution for new restaurants

fig, ax = plt.subplots(figsize=(16, 9))

sns.kdeplot(new_restaurants['success_proba'], ax=ax, shade=True, color='mediumseagreen')

format_spines(ax, right_border=False)

ax.set_title('Score Distribution of Success for New Restaurants on the Dataset', size=16, color='dimgrey')

plt.show()
# Ordering new restaurants by proba score

new_restaurants_data = new_restaurants.reset_index().merge(raw_data.reset_index()[['name', 'index']], how='left', on='index')

top_new = new_restaurants_data.sort_values(by='success_proba', ascending=False).head(10)

top_new = top_new.loc[:, ['name', 'success_proba', 'online_order', 'book_table', 'listed_in(type)',

                          'approx_cost', 'multiple_types', 'total_cuisines', 'location_feature',

                          'city_feature', 'type_feature']]

top_new
# Ordering new restaurants by proba score

bottom_new = new_restaurants_data.sort_values(by='success_proba', ascending=True).head(10)

bottom_new = bottom_new.loc[:, ['name', 'success_proba', 'online_order', 'book_table', 'listed_in(type)',

                          'approx_cost', 'multiple_types', 'total_cuisines', 'location_feature',

                          'city_feature', 'type_feature']]

bottom_new