%matplotlib inline
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from wordcloud import WordCloud
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import re
df1 = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")
df2 = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.csv")
df3 = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv")
stop = list(stopwords.words("english"))
df1.dtypes
df1.head()
df1.isna().any()
df1['has_urgency_banner'] = df1['has_urgency_banner'].fillna(0)
df1['rating_one_count'] = df1['rating_one_count'].fillna(0)
df1['rating_two_count'] = df1['rating_two_count'].fillna(0)
df1['rating_three_count'] = df1['rating_three_count'].fillna(0)
df1['rating_four_count'] = df1['rating_four_count'].fillna(0)
df1['rating_five_count'] = df1['rating_five_count'].fillna(0)

df1['product_color'] = df1['product_color'].fillna("not specified")
df1['origin_country'] = df1['origin_country'].fillna("not specified")
bool_cols = ['uses_ad_boosts', 
             'badge_local_product',
             'badge_product_quality',
             'badge_fast_shipping',
             'shipping_is_express',
             'has_urgency_banner',
             'merchant_has_profile_picture']

num_cols = list(df1.select_dtypes(include=[np.number]).columns.values)
clean_num = list(set(num_cols) - set(bool_cols))
df1[clean_num].describe()
sold = df1['units_sold']
sold_count = dict(Counter(list(sold.values)))
sold_count = {k: v for k,v in sorted(sold_count.items(), key = lambda item : item[0])}

sold_xaxis = list(sold_count.keys())
sold_xaxis = ["Units sold: " + str(x) for x in sold_xaxis]
sold_yaxis = list(sold_count.values())

fig = go.Figure()
fig.add_trace(go.Bar(y = sold_yaxis, x = sold_xaxis))
fig.update_layout(yaxis_title = "Number of products")
fig.show()
corr_mat = df1[clean_num].corr()

plt.figure(figsize = (15, 10))
sns.heatmap(corr_mat, annot = True)
plt.show()
rating_df = df1[['rating_one_count',
                  'rating_two_count',
                  'rating_three_count',
                  'rating_four_count',
                  'rating_five_count', 
                  'rating_count']]

one_star = list(rating_df['rating_one_count'].values)
five_star = list(rating_df['rating_five_count'].values)
rg = sns.regplot(one_star, five_star)
x_rg = rg.get_lines()[0].get_xdata()
y_rg = rg.get_lines()[0].get_ydata()
plt.clf()

fig = go.Figure()
fig.add_trace(go.Scatter(x = one_star, y = five_star, mode='markers', name = 'Rating num'))
fig.add_trace(go.Scatter(x = x_rg, y = y_rg, mode='lines', name = 'Fitted Regression'))
fig.update_layout(xaxis_title = "1-star rating",
                  yaxis_title = "5-star rating",
                  title = "Scatterplot between number of 1-star and 5-star ratings",
                  showlegend = False)
sold = df1['units_sold']

fig = make_subplots(rows =1 , cols = 2,
                    subplot_titles = ["Scatterplot between number of 1-star ratings and units sold",
                                      "Scatterplot between number of 5-star ratings and units sold"])
fig.add_trace(go.Scatter(x = one_star, y = sold, mode = 'markers'),
              row = 1, col = 1)
fig.add_trace(go.Scatter(x = five_star, y = sold, mode = 'markers'), row = 1, col = 2)
fig.update_layout(showlegend = False)
df1['prop_good'] = (df1['rating_four_count'] + df1['rating_five_count']) / df1['rating_count']
df1['prop_bad'] =  (df1['rating_one_count'] + df1['rating_two_count']) / df1['rating_count']
df1['prop_neutral'] =  df1['rating_three_count'] / df1['rating_count']

corr_mat2 = df1[['prop_good', 'prop_bad', 'prop_neutral', 'rating_count',  'units_sold']].corr()
plt.figure(figsize = (15, 10))
sns.heatmap(corr_mat2, annot = True)
plt.show()
temp_df = df1.groupby("merchant_id")[['prop_good', 'prop_bad', 'prop_neutral', 'merchant_rating']].mean()
corr_mat3 = temp_df.corr()

plt.figure(figsize = (15, 10))
sns.heatmap(corr_mat3, annot = True)
plt.show()
price = df1['price']
retail_price = df1['retail_price']

fig = go.Figure()
fig.add_trace(go.Box(x = price, name = "Price"))
fig.add_trace(go.Box(x = retail_price, name = "Retail Price"))
fig.update_traces(opacity = 0.75)
fig.update_layout(barmode = "overlay", 
                  title = "Distribution of Retail Prices")
diff = price-retail_price
df1['diff_prices'] = diff

fig = go.Figure()
fig.add_trace(go.Histogram(x = diff, marker = dict(color = 'red')))
fig.update_layout(title = "Difference in retail and Wish prices")
min(diff)
sold.value_counts()
temp_df = df1[['units_sold', 'diff_prices']]
temp_df['units_sold'] = temp_df['units_sold'].apply(lambda x: 10 if (x < 10) else x)
temp_df['units_sold'].value_counts()
groups = list(temp_df['units_sold'].value_counts().index)
groups.sort()

fig = go.Figure()
for g in groups: 
    subset_df = temp_df[temp_df['units_sold'] == g]
    subset_x = subset_df['diff_prices']
    set_name = "Units sold: " + str(g)
    fig.add_trace(go.Box(x = subset_x, name = set_name))

fig.update_layout(title = "Box plot of differences in Wish and retail prices over number of units sold")
fig.show()
title = list(df1['title_orig'].values)
word_len = dict(Counter(list(map(lambda x: len(x.split(" ")), title))))
word_len = {k: v for k,v in sorted(word_len.items() , key = lambda item: item[1], reverse = True)}

def clean_text(txt):
    c = txt.lower()
    c = re.sub(r'[^\w\s]', '', c) # remove punctuation
    return c

df1['cleaned_title'] = df1['title_orig'].apply(lambda x: clean_text(x))
fig = go.Figure()
fig.add_trace(go.Bar(x = list(word_len.keys()), y = list(word_len.values())))
fig.update_layout(title = "Number of words in the title of a product listing", 
                  xaxis_title = "Number of words in title",
                  yaxis_title = "Count")
def top_n_freq(all_txt, n):
    freq = defaultdict(int)
    
    for txt in all_txt:
        words = txt.split(" ")
        for w in words:
            freq[w] += 1
    
    freq = dict(freq)
    del freq['']
    freq = {k: v for k, v in sorted(freq.items(), key = lambda item: item[1], reverse = True)}
    
    
    y = list(freq.keys())[:(n+1)]
    x = list(freq.values())[:(n+1)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x = x, y = y, orientation = "h"))
    fig.update_layout(title = "Top " + str(n) + " most common words",
                      xaxis_title = "Count")
    fig.show()
    
top_n_freq(list(df1['cleaned_title'].values), 10)
keywords = dict(zip(df3['keyword'], df3['count']))

def create_wordcloud(kw):
    wc = WordCloud(background_color = 'white',
                   height = 800,
                   width = 800)
    wc.generate_from_frequencies(kw)
    
    plt.figure(figsize = (10, 10))
    plt.imshow(wc)
    plt.axis("off")
    
create_wordcloud(keywords)