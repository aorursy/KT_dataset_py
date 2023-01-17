import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots
from wordcloud import WordCloud,STOPWORDS, ImageColorGenerator
import squarify as sq
import matplotlib.colors
from PIL import Image
import requests

df = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv',index_col = 0)

# examine first 20 rows
df.head(20)
# Check data types 
df.info()
df.isna().sum()/df.shape[0]*100
msno.bar(df, color = sns.color_palette('RdBu'))
reviews = df.dropna(how = 'any',subset = ['country','price'])
reviews
#confirm missing values has been dropped
reviews[['country','price']].isna().sum()
reviews.fillna('Unknown', inplace = True)
reviews

reviews.drop(['taster_twitter_handle'],axis = 1,inplace = True)
reviews
 

#Lets values counts on points
reviews['points'].value_counts()
# Price of a wine

reviews['price'].describe()
# how many varieties of wine we have
reviews['variety'].nunique()
# how often the country appears in a dataset
reviews['country'].value_counts()
#What are the minimum and maximum prices for each variety of wine?

top_varieties = reviews.groupby('variety')['price'].agg([min,max]).sort_values(by= 'max' ,ascending = False).reset_index()
top_varieties

plt.figure(figsize = (15,10))
ax= sns.countplot(x = 'points',data = reviews,color = 'pink')
plt.title('Points Count',fontsize=20)
plt.xlabel("Points")

plt.figure(figsize = (12,8))
ax = sns.boxplot(x = reviews['price'],orient = 'v')
plt.title('Price Distribution',fontsize = 20)
plt.ylabel('Price',fontsize = 20)
below200 = reviews[reviews['price']< 200]['price']
plt.figure(figsize = (12,8))
ax = sns.distplot(below200,rug = True)
plt.title('Wine price below 200',fontsize=25)


reviews['country'].value_counts()[:10].plot(kind='bar',color = 'magenta',figsize=(10,10))
plt.title("Top Ten Wine Producing Countries",fontsize=25)
plt.xlabel('Country' ,fontsize = 20)

top10 = reviews['country'].value_counts()[:10]

plt.figure(figsize = (12,8))
cmap = matplotlib.cm.Oranges
norm = matplotlib.colors.Normalize(vmin=0, vmax=15)
colors = [cmap(norm(value)) for value in range(10)]
np.random.shuffle(colors)

sq.plot(sizes =top10, label=top10.index, color = colors)
plt.title('Top Ten Countries',fontsize = 20)
plt.figure(figsize = (15,10))
ax = sns.scatterplot(x = 'price',y = 'points',data = reviews, alpha = 0.6, color = 'green')
plt.title('Points and Price Comparision',fontsize=25)
below200 = reviews[reviews['price']< 200].sample(500)

plt.figure(figsize = (12,8))
ax = sns.scatterplot(x = 'price',y = 'points',data = below200, alpha = 0.6, color = 'purple')
plt.title('Wine price vs points- Below $200',fontsize =25)
plt.xlabel('Price', fontsize = 25)
plt.ylabel('Points', fontsize = 25)

#Expensive wines by country top 20
exp_wine = reviews.groupby('country')['price'].max().sort_values(ascending = False)[:20].reset_index()

plt.figure(figsize= (10,12))
ax = sns.barplot('country', 'price',data = exp_wine,dodge = False,palette ='ocean')
plt.title('Most Expensive Wine by Country- Top 20',fontsize = 25)
plt.xlabel('Country', fontsize = 25)
plt.ylabel('Price', fontsize = 25)
plt.xticks(rotation = 90,size = 12)
plt.yticks(size = 15)



plt.figure(figsize = (12,8))
ax = sns.barplot(x = 'max',y = 'variety',data = top_varieties[:20],hue = 'max',dodge = False, palette = 'RdPu')
plt.ylabel('Variety',fontsize = 25)
plt.xlabel('Price',fontsize = 25)
plt.title('Top variety wine(by price)- Top 20', fontsize = 20)

           
    
province = reviews['province'].value_counts()[:20]

sns.set_style('darkgrid')
plt.figure(figsize=(14,15))
ax = sns.countplot(x ='province',data = reviews.loc[(reviews['province'].isin(province.index.values))],palette ='CMRmap',order = province.index)
plt.xticks(rotation = 90,size = 12)
plt.yticks(size = 12)
plt.xlabel('Province',fontsize = 25)
plt.title('Province of Wine Origin- Top 20',fontsize =25)
plt.ylabel('Number Counts', fontsize = 25)






stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = "white",stopwords = stopwords,max_words = 300,max_font_size = 200,
                     random_state = 42,).generate("".join(reviews['description'].astype(str)))

print(wordcloud)
plt.figure(figsize = (15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Word Cloud of Description',fontsize=25)
stopwords = set(STOPWORDS)

variety =  " ".join(review for review in reviews['variety'])

wordcloud = WordCloud(background_color = "white",stopwords = stopwords,max_words = 300,max_font_size = 200,
                     random_state = 42,).generate(variety)

print(wordcloud)
plt.figure(figsize = (15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Word Cloud of Variety',fontsize=25)
stopwords = set(STOPWORDS)

US = reviews[reviews['country']== 'US']


wordcloud = WordCloud(background_color = "white",stopwords = stopwords,max_words = 500,max_font_size = 200,
                     random_state = 42,).generate("".join(US['description'].astype(str)))

plt.figure(figsize = (15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('USA - Word Cloud of Description',fontsize=20)
stopwords = set(STOPWORDS)

Italy = reviews[reviews['country']== 'Italy']

wordcloud = WordCloud(background_color = "white",stopwords = stopwords,max_words = 500,max_font_size = 200,
                     random_state = 42,).generate("".join(Italy['description'].astype(str)))

plt.figure(figsize = (15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Italy - Word Cloud of Description',fontsize=20)
stopwords = set(STOPWORDS)

France = reviews[reviews['country']== 'France']
France

wordcloud = WordCloud(background_color = "white",stopwords = stopwords,max_words = 300,max_font_size = 200,
                     random_state = 42,).generate("".join(France['description'].astype(str)))

plt.figure(figsize = (15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('France - Word Cloud of Description',fontsize=20)