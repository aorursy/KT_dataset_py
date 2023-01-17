import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS
data = pd.read_csv("../input/winemag-data-130k-v2.csv", encoding = "ISO-8859-1")
data.head()
#Vintage is the year that the wine was manufactured.

# Lets retrieve the vintage from the title

def get_Vintage(row):

    vintage = 0

    words = []

    words = row['title'].split()

    for word in words:

        if word.isdigit():

            vintage = int(word)

            break;

    return vintage
data['vintage'] = data.apply (lambda row: get_Vintage (row),axis=1)
ausWines = data[(data.country == 'Australia')]

ausWines.info()
topAusWines = ausWines[(ausWines.points >= 90)]

df = topAusWines.groupby('province')['Unnamed: 0'].nunique()

df.sort_values(ascending = False)

fig = plt.figure(figsize=(20,10))

fig.suptitle('Top Australian Provinces', fontsize = 20)

plt.xlabel('Province', fontsize = 20)

plt.ylabel('Count', fontsize = 20)

df.plot.bar(legend = False)

plt.show()
df = topAusWines.groupby('variety')['Unnamed: 0'].nunique()

df.sort_values(ascending = False)

fig = plt.figure(figsize=(20,10))

fig.suptitle('Top Australian Varieties', fontsize = 20)

plt.xlabel('variety', fontsize = 20)

plt.ylabel('Count', fontsize = 20)



df.plot.bar(legend = False)

plt.show()
df = topAusWines.groupby('region_1')['Unnamed: 0'].nunique()

df.sort_values(ascending = True)



fig = plt.figure(figsize=(20,10))

fig.suptitle('Top Australian Regions', fontsize = 20)

plt.xlabel('region', fontsize = 20)

plt.ylabel('Count', fontsize = 20)



df.plot.bar(legend = False, width = 1)

plt.show()
topSAWines = topAusWines[(topAusWines.province == 'South Australia')]



df = topSAWines.groupby('region_1')['Unnamed: 0'].nunique()

df.sort_values(ascending = True)



fig = plt.figure(figsize=(20,10))

fig.suptitle('Top South Australian Regions', fontsize = 20)

plt.xlabel('region', fontsize = 20)

plt.ylabel('Count', fontsize = 20)



df.plot.bar(legend = False, width = 1)

plt.show()
df = topSAWines.groupby('variety')['Unnamed: 0'].nunique()

df.sort_values(ascending = True)



fig = plt.figure(figsize=(20,10))

fig.suptitle('Top South Australian Varieties', fontsize = 20)

plt.xlabel('variety', fontsize = 20)

plt.ylabel('Count', fontsize = 20)



df.plot.bar(legend = False, width = 1)

plt.show()
topMcLarenWines = topSAWines[(topSAWines.region_1 == 'McLaren Vale')]



df = topMcLarenWines.groupby('winery', as_index= True)['Unnamed: 0'].nunique()

df.sort_values(ascending = True)



fig = plt.figure(figsize=(20,10))

fig.suptitle('Top McLaren Vale Wineries', fontsize = 20)

plt.xlabel('Winery', fontsize = 20)

plt.ylabel('Count', fontsize = 20)



df.plot.bar(legend = False, width = 1)

plt.show()
df = topMcLarenWines.groupby('variety', as_index= True)['Unnamed: 0'].nunique()

df.sort_values(ascending = True)



fig = plt.figure(figsize=(20,10))

fig.suptitle('Top McLaren Vale Varieties', fontsize = 20)

plt.xlabel('Variety', fontsize = 20)

plt.ylabel('Count', fontsize = 20)



df.plot.bar(legend = False, width = 1)

plt.show()
df = ausWines.sort_values('price', ascending=False).head(20)

fig, axes = plt.subplots(nrows=3, ncols=2)

df.groupby('province')['Unnamed: 0'].nunique().plot.bar(ax = axes[0,0], figsize=(20,20), 

                                                        title = "Provinces - Top 20 Expensive Wines", rot = 0, x = "province",

                                                       y = "count")



df.groupby('region_1')['Unnamed: 0'].nunique().plot.bar(ax = axes[0,1], figsize=(20,20), 

                                                        title = "Regions - Top 20 Expensive Wines", rot = 0, x = "Regions",

                                                       y = "count")



df.groupby('variety')['Unnamed: 0'].nunique().plot.bar(ax = axes[1,0], figsize=(20,20), 

                                                        title = "Variety - Top 20 Expensive Wines", rot = 300, x = "Variety",

                                                       y = "count")



df.groupby('winery')['Unnamed: 0'].nunique().plot.bar(ax = axes[1, 1], figsize=(20,20), 

                                                        title = "Winery - Top 20 Expensive Wines", rot = 300, x = "Winery",

                                                       y = "count")



df.hist('points', ax = axes[2,0], figsize=(20,20))

df.hist('price', ax = axes[2,1], figsize=(20,20))

df = ausWines.sort_values('points', ascending=False).head(20)

fig, axes = plt.subplots(nrows=3, ncols=2)

df.groupby('province')['Unnamed: 0'].nunique().plot.bar(ax = axes[0,0], figsize=(20,20), 

                                                        title = "Provinces - Top 20 Rated Wines", rot = 0, x = "province",

                                                       y = "count")



df.groupby('region_1')['Unnamed: 0'].nunique().plot.bar(ax = axes[0,1], figsize=(20,20), 

                                                        title = "Regions - Top 20 Rated Wines", rot = 0, x = "Regions",

                                                       y = "count")



df.groupby('variety')['Unnamed: 0'].nunique().plot.bar(ax = axes[1,0], figsize=(20,20), 

                                                        title = "Variety - Top 20 Rated Wines", rot = 300, x = "Variety",

                                                       y = "count")



df.groupby('winery')['Unnamed: 0'].nunique().plot.bar(ax = axes[1, 1], figsize=(20,20), 

                                                        title = "Winery - Top 20 Rated Wines", rot = 300, x = "Winery",

                                                       y = "count")



df.hist('points', ax = axes[2,0], figsize=(20,20))

df.hist('price', ax = axes[2,1], figsize=(20,20), bins = 20)
ausWines[['price']].quantile([0.25, 0.5, 0.75, 1])

ausWines.groupby('variety', as_index = True)['Unnamed: 0'].nunique().sort_values(ascending = False).head(5)
fig, ax = plt.subplots(figsize=(20,10))

df_wide=ausWines[((ausWines.price <= 38) & ((ausWines.variety == 'Shiraz') | (ausWines.variety == 'Chardonnay') | 

                                            (ausWines.variety == 'Cabernet Sauvignon') | (ausWines.variety == 'Riesling') | 

                                            (ausWines.variety == 'Pinot Noir')))].pivot_table( index='variety', columns='points', values='price')



p2=sns.heatmap( df_wide, ax = ax )
stopwords = set(STOPWORDS)

STOP_WORDS = frozenset([ # http://www.nltk.org/book/ch02.html#stopwords_index_term

    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',

    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',

    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',

    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',

    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',

    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',

    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',

    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',

    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',

    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',

    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',

    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'along',

    'also', 'comes', 'cru', 'de', 'delicious', 'drink', 'edge', 'fine', 'finish', 'years', 'wine', 

    'Shiraz', 'Chardonnay', 'It', 'slightly', 'yet', 'still', 'ample', 'Cabernet', 'Pinot', 'Noir', 'note'

    , 'notes'

    ])

#fig, axes = plt.subplots(nrows=2, ncols=2)



stopwords.update(STOP_WORDS)

fig = plt.figure(figsize=[20,10])

fig.add_subplot(2, 2, 1)

shirazDesc = ''.join(topAusWines[topAusWines.variety == 'Shiraz']['description'])

wordcloud = WordCloud(stopwords=stopwords,max_font_size=50, max_words=50, background_color="white").generate(shirazDesc)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Shiraz')

plt.axis("off")

###############################

fig.add_subplot(2, 2, 2)

shirazDesc = ''.join(topAusWines[topAusWines.variety == 'Chardonnay']['description'])

wordcloud = WordCloud(stopwords=stopwords,max_font_size=50, max_words=50, background_color="white").generate(shirazDesc)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Chardonnay')

plt.axis("off")

################################

fig.add_subplot(2, 2, 3)

shirazDesc = ''.join(topAusWines[topAusWines.variety == 'Cabernet Sauvignon']['description'])

wordcloud = WordCloud(stopwords=stopwords,max_font_size=50, max_words=50, background_color="white").generate(shirazDesc)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Cabernet Sauvignon')

plt.axis("off")

################################

fig.add_subplot(2, 2, 4)

shirazDesc = ''.join(topAusWines[topAusWines.variety == 'Pinot Noir']['description'])

wordcloud = WordCloud(stopwords=stopwords,max_font_size=50, max_words=50, background_color="white").generate(shirazDesc)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Pinot Noir')

plt.axis("off")



plt.show()