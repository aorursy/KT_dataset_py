# for basic mathematics operation 
import numpy as np
import pandas as pd
from pandas import plotting
import datetime

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import datetime as dt
import missingno as msno

# for interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

#Word Cloud
from PIL import Image
import requests
from io import BytesIO
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from textblob import TextBlob


# for path
import os
print(os.listdir('../input/'))
#PYCARET
!pip install pycaret
from pycaret.regression import *
data = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
data.head()
unique_cat = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.csv')
unique_cat.head()
cat_sorted = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv')
cat_sorted.head()
#drop merchant_title, merchant_info_subtitle
data.drop(['merchant_title', 'merchant_info_subtitle','title'], axis = 1, inplace = True)
print("Columns drop successfully...")
#most units sold
most_sold_product = data['units_sold'].idxmax()
print("Prodcut with highest sales irrespective of price: \n")
print(data.iloc[most_sold_product].head())
print('\n')
response1 = requests.get(data[data['units_sold'] == 100000].product_picture.tolist()[0]) #printing the thumbnail of that video
Image.open(BytesIO(response1.content))
response2 = requests.get(data[data['units_sold'] == 100000].product_picture.tolist()[1]) #printing the thumbnail of that video
Image.open(BytesIO(response2.content))
response3 = requests.get(data[data['units_sold'] == 100000].product_picture.tolist()[2]) #printing the thumbnail of that video
Image.open(BytesIO(response3.content))
#Male and Female 
Total_male = round(data.title_orig.str.count("Men").sum()/len(data)*100, 3)
Total_female = round(data.title_orig.str.count("Women").sum()/len(data)*100, 3) 

fig = go.Figure()
fig.add_trace(go.Indicator(mode = "number+delta",
                             value = Total_male,
                             title = {"text": "Men Collection in %",
                                      "font" : {'color': 'rgb(58, 171, 163)', 'size': 25, 'family': 'Raleway'}},
                             number = {'font': {'color': 'rgb(58, 171, 163)', 'size': 25, 'family': 'Raleway'}},
                             domain = {'row': 0, 'column': 0}))

fig.add_trace(go.Indicator(mode = "number+delta",
                             value = Total_female,
                             title = {"text": "Women Collection in %",
                                      "font" : {'color': 'rgb(0, 0, 0)', 'size': 35, 'family': 'Raleway'}},
                             number = {'font': {'color': 'rgb(0, 0, 0)', 'size': 35, 'family': 'Raleway'}},
                             domain = {'row': 0, 'column': 1}))
fig.update_layout(grid = {'rows': 1, 'columns': 2, 'pattern': 'independent'})
fig.show()
#RETAIL PRICE #FINAL PRICE- CHARGED TO CUSTOMER
plt.subplot(1, 2, 1)
(data['retail_price']).plot.hist(bins=50, figsize=(12, 6), edgecolor = 'white', range = [0, 250])
plt.xlabel('retail_price', fontsize=12)
plt.title('Retail Price Distribution', fontsize=12)
plt.subplot(1, 2, 2)
np.log(data['retail_price']+1).plot.hist(bins=50, figsize=(12,6), edgecolor='white')
plt.xlabel('log(retail_price+1)', fontsize=12)
plt.title('Retail Price Distribution', fontsize=12)
#IS THERE ANY RELATION BETWEEN REATIL PRICE AND AD_BOOST
#uses_ad_boosts

ad_boost_by_buyer_no = data.loc[data['uses_ad_boosts'] == 0, 'retail_price']
ad_boost_by_seller_yes = data.loc[data['uses_ad_boosts'] == 1, 'retail_price']
fig, ax = plt.subplots(figsize=(18,8))
ax.hist(np.log(ad_boost_by_seller_yes+1), color='#0b2adb', alpha=1.0, bins=50,
       label='if there is ad-boost')
ax.hist(np.log(ad_boost_by_buyer_no+1), color='#d4db0b', alpha=0.7, bins=50,
       label='if there is no ad-boost')
plt.xlabel('log(Retail price + 1)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Retail Price Distribution by Ad-Boost', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend()
plt.show()
print('The average price is EUR {}'.format(round(ad_boost_by_seller_yes.mean(), 2)), 'if there is ad-boost');
print('The average price is EUR {}'.format(round(ad_boost_by_buyer_no.mean(), 2)), 'if there is no ad-boost')
#badge_product_quality
#IS THERE ANY RELATION BETWEEN REATIL PRICE AND AD_BOOST

product_quality_no = data.loc[data['badge_product_quality'] == 0, 'retail_price']
product_quality_yes = data.loc[data['badge_product_quality'] == 1, 'retail_price']
fig, ax = plt.subplots(figsize=(18,8))
ax.hist(product_quality_yes, color='#111212', alpha=1.0, bins=50, range = [0, 100],
       label='Prodcut Quality Badge Present')
ax.hist(product_quality_no, color='#02f0c8', alpha=0.7, bins=50, range = [0, 100],
       label='Prodcut Quality Badge Absent')
plt.xlabel('Retail Price', fontsize=12)
plt.ylabel('frequency', fontsize=12)
plt.title('Price Distribution by Product Quality Badge', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend()
plt.show()
data['success_product'] = np.nan
for i in data.index:
    if data['units_sold'].loc[i] >= 1000:
        data['success_product'].loc[i] = 1
    else:
        data['success_product'].loc[i] = 0
#FIND AVG RATING FOR SUCCESSFUL PRODUCT AND UNSUCCESSFUL PRODUCT
success_rating = float("{:.3f}".format(data[data["success_product"] == 1.0].rating.mean()))
unsuccess_rating = float("{:.3f}".format(data[data["success_product"] == 0.0].rating.mean()))

print("The average rating of the successful product is: "+ str(success_rating) + '\n')
print("The average rating of the unsuccessful product is: "+ str(unsuccess_rating))
#DISTRIBUTION OF RETAIL PRICE WHICH ARE SUCCESSFUL AND NOT
#IS THERE ANY RELATION BETWEEN REATIL PRICE AND AD_BOOST
#uses_ad_boosts

unsuccessful_product_by_seller = data.loc[data['success_product'] == 0.0, 'retail_price']
success_product_by_seller = data.loc[data['success_product'] == 1.0, 'retail_price']
fig, ax = plt.subplots(figsize=(18,8))
ax.hist(success_product_by_seller, color='#690232', alpha=1.0, bins=50, range = [0, 100],
       label='Successful Products: Retail Price')
ax.hist(unsuccessful_product_by_seller, color='#044a4a', alpha=0.7, bins=50, range = [0, 100],
       label='Unsuccessful Products: Retail Price')
plt.xlabel('Retail Price', fontsize=12)
plt.ylabel('frequency', fontsize=12)
plt.title('Successful And Unsuccessful Products Retail Price', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend()
plt.show()
#average merchant rating and the product is successful or not

def avg_merchant_rating(merchant_rating):
    merchant_rating = int(merchant_rating)
    
    bucket = ''
    if merchant_rating in range(0,3):
        bucket = 'Rating(< 3)'
    if merchant_rating in range(3,4):
        bucket = 'Rating(3-4)'
    if merchant_rating in range(4,5):
        bucket = 'Rating(4-5)'
    
    return bucket
#BOXPLOTS REGARDING MERCHANT REVIEWS: WHICH ARE SUCCESSFUL OR NOT?
data['merchant_rating_category'] = data['merchant_rating'].apply(avg_merchant_rating)
fig = px.box(data, x="merchant_rating_category", y="retail_price", color="success_product")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()
#UNITS SOLD CATEGORISATION, THEN PLOT THE RATINGS AS FOLLOWS:
def units_sold_category(units_sold):
    units_sold = int(units_sold)
    
    bucket = ''
    if units_sold in range(0,100):
        bucket = '< 100'
    if units_sold in range(100,1000):
        bucket = '100 - 1000'
    if units_sold in range(1000,5000):
        bucket = '1000 - 5000'
    if units_sold in range(5000,10000):
        bucket = '5000 - 10000'
    if units_sold in range(10000,20000):
        bucket = '10000 - 20000'
    if units_sold in range(20000,100000):
        bucket = '> 20000'
    return bucket
data['units_sold_category'] = data['units_sold'].apply(units_sold_category)
units_sold = data['units_sold_category'].unique().tolist()
grouped_by_units_sold_ratings = data.groupby('units_sold_category').agg({'rating_five_count': 'sum',
                                                                         'rating_four_count': 'sum',
                                                                         'rating_three_count': 'sum',
                                                                         'rating_two_count': 'sum',
                                                                         'rating_one_count': 'sum'})
grouped_by_units_sold_ratings.reset_index()
grouped_by_units_sold_ratings = grouped_by_units_sold_ratings.iloc[1:]
grouped_by_units_sold_ratings
grouped_by_units_sold_ratings.reset_index(inplace = True)
lis1, lis2, lis3, lis4, lis5 = [], [], [], [], []
for i in grouped_by_units_sold_ratings.index:
    lis1.append(grouped_by_units_sold_ratings.iloc[i][1])

for i in grouped_by_units_sold_ratings.index:
    lis2.append(grouped_by_units_sold_ratings.iloc[i][2])

for i in grouped_by_units_sold_ratings.index:
    lis3.append(grouped_by_units_sold_ratings.iloc[i][3])
    
for i in grouped_by_units_sold_ratings.index:
    lis4.append(grouped_by_units_sold_ratings.iloc[i][4])
    
for i in grouped_by_units_sold_ratings.index:
    lis5.append(grouped_by_units_sold_ratings.iloc[i][5])
Units_Sold=["<100", "100-1000", "1000-5000", "5000-10000", "10000-20000", ">20000"]

fig = go.Figure(data=[
    go.Bar(name='rating_five_count', x=Units_Sold, y=lis1),
    go.Bar(name='rating_four_count', x=Units_Sold, y=lis2),
    go.Bar(name='rating_three_count', x=Units_Sold, y=lis3),
    go.Bar(name='rating_two_count', x=Units_Sold, y=lis4),
    go.Bar(name='rating_one_count', x=Units_Sold, y=lis5)
])
# Change the bar mode
fig.update_layout(barmode='stack')
fig.show()
#CATEGORICAL FEATURE PLOT
fig = px.parallel_categories(data,
                             dimensions = ['origin_country', 'units_sold_category','merchant_rating_category', 'uses_ad_boosts'],
                             
                             labels = {'units_sold_category': 'Units Sold',
                                       'merchant_rating_category': 'Merchant Ratings',
                                       'uses_ad_boosts': 'Using Ads ?'})
fig.show()
data['count'] = 1
badge_count = data.groupby(['badges_count']).sum().reset_index()[['badges_count', 'count']]
badge_local_product = data.groupby(['badge_local_product']).sum().reset_index()[['badge_local_product', 'count']]
badge_product_quality = data.groupby(['badge_product_quality']).sum().reset_index()[['badge_product_quality', 'count']]
badge_fast_shipping = data.groupby(['badge_fast_shipping']).sum().reset_index()[['badge_fast_shipping', 'count']]
success_product = data.groupby(['success_product']).sum().reset_index()[['success_product', 'count']]
product_variation_size_id = data.groupby(['product_variation_size_id']).sum().reset_index()[['product_variation_size_id', 'count']].sort_values(by = 'count', ascending = False)[:7]
units_sold_category_total = data.groupby(['units_sold_category']).sum().reset_index()[['units_sold_category', 'count']]
merchant_rating_category_total = data.groupby(['merchant_rating_category']).sum().reset_index()[['merchant_rating_category', 'count']]
fig = make_subplots(rows=4, cols=2, shared_yaxes=True, subplot_titles=("Badge Count", "Local Products", "Product Quality",
                                                                       "Fast Shipping", "Successful Products", "Product Variation",
                                                                       "Units Sold", "Merchant Rating Category"))

fig.add_trace(go.Bar(x = badge_count['badges_count'].tolist(),
                     y = badge_count['count'].tolist(), 
                     marker=dict(color=badge_count['count'].tolist(), coloraxis="coloraxis")), 1,1)

fig.add_trace(go.Bar(x = badge_local_product['badge_local_product'].tolist(),
                     y = badge_local_product['count'].tolist(), 
                     marker=dict(color=badge_local_product['count'].tolist(), coloraxis="coloraxis")), 1,2)

fig.add_trace(go.Bar(x = badge_product_quality['badge_product_quality'].tolist(),
                     y = badge_product_quality['count'].tolist(), 
                     marker=dict(color=badge_product_quality['count'].tolist(), coloraxis="coloraxis")), 2,1)

fig.add_trace(go.Bar(x = badge_fast_shipping['badge_fast_shipping'].tolist(),
                     y = badge_fast_shipping['count'].tolist(), 
                     marker=dict(color=badge_fast_shipping['count'].tolist(), coloraxis="coloraxis")), 2,2)

fig.add_trace(go.Bar(x = success_product['success_product'].tolist(),
                     y = success_product['count'].tolist(), 
                     marker=dict(color=success_product['count'].tolist(), coloraxis="coloraxis")), 3,1)

fig.add_trace(go.Bar(x = product_variation_size_id['product_variation_size_id'].tolist(),
                     y = product_variation_size_id['count'].tolist(), 
                     marker=dict(color=product_variation_size_id['count'].tolist(), coloraxis="coloraxis")), 3,2)

fig.add_trace(go.Bar(x = units_sold_category_total['units_sold_category'].tolist(),
                     y = units_sold_category_total['count'].tolist(), 
                     marker=dict(color=units_sold_category_total['count'].tolist(), coloraxis="coloraxis")), 4,1)

fig.add_trace(go.Bar(x = merchant_rating_category_total['merchant_rating_category'].tolist(),
                     y = merchant_rating_category_total['count'].tolist(), 
                     marker=dict(color=merchant_rating_category_total['count'].tolist(), coloraxis="coloraxis")), 4,2)

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False, title_text='Visualisations:')
fig.show()
#WORD CLOUD
tags_total = data['tags'].dropna().tolist()
tags_final=(" ").join(tags_total)
response = requests.get('https://www.incrediblelab.com/wp-content/uploads/2017/04/wish-logo.jpg')
char_mask = np.array(Image.open(BytesIO(response.content)))
image_colors = ImageColorGenerator(char_mask)
plt.figure(figsize = (13,13))

wc = WordCloud(background_color="black", max_words=200,
               width=400, height=400, mask=char_mask, random_state=1).generate(tags_final)
# to recolour the image
plt.imshow(wc.recolor(color_func=image_colors))
data.corr().iplot(kind='heatmap', colorscale="Reds", title="Feature Correlation Matrix")
origin_country = data.groupby(['origin_country']).sum().reset_index()[['origin_country', 'count']]
origin_country_retail = data.groupby(['origin_country']).mean().reset_index()[['origin_country','retail_price', 'count']]
labels = origin_country['origin_country']
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=labels, values=origin_country['count'], name="Products based on origin country"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=origin_country_retail['retail_price'], name="Retail Price based on origin country"),
              1, 2)
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    annotations=[dict(text='Products', x=0.17, y=0.5, font_size=20, showarrow=False),
                 dict(text='Retail Price', x=0.84, y=0.5, font_size=20, showarrow=False)])
fig.show()
#take few columns and made new df
df=data[['title_orig','tags', 'retail_price','units_sold', 'price', 'rating', 'rating_count',
         'rating_five_count', 'rating_one_count', 'rating_three_count', 'rating_two_count', 'rating_four_count',
         'badge_local_product', 'product_variation_inventory', 'shipping_option_price', 
         'merchant_rating_count', 'merchant_rating']]
df.head(3)
exp_reg = setup(df, target = 'units_sold', silent = True)
best_model = compare_models(sort = 'MAPE', exclude= ['lightgbm'])
rf = create_model('et', fold = 5)
tuned_dt = tune_model(rf, optimize = 'MAPE')
plot_model(tuned_dt)