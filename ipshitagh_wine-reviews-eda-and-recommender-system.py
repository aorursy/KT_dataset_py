import pandas as pd

import numpy as np

from wordcloud import WordCloud,STOPWORDS

import seaborn as sns

from scipy.stats import kurtosis, skew

from scipy import stats

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(color_codes=True)

print("Setup Complete")
wine = pd.read_csv(r'../input/wine-reviews/winemag-data-130k-v2.csv')

wine1 = wine
wine.head(5)
wine.shape
wine.describe()
wine.isnull().values.any()
sns.heatmap(wine.isnull(),yticklabels=False,cbar=False,cmap='viridis')
wine = wine.drop(columns=['Unnamed: 0','region_1','region_2','taster_twitter_handle','designation'])

wine1 = wine

sns.heatmap(wine.isnull(),yticklabels=False,cbar=False,cmap='viridis')
wine1.price.fillna(wine.price.dropna().median(),inplace =True)

wine1 = wine.dropna()

sns.heatmap(wine1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(30,5))

sns.boxplot(x=wine['price'],palette = 'colorblind')
plt.figure(figsize=(30,5))

sns.boxplot(x=wine1['price'],palette = 'colorblind')
plt.figure(figsize=(30,5))

sns.boxplot(x=wine['points'],palette = 'colorblind')
for feature in wine1.columns:

    uniq = np.unique(wine1[feature])

    print('{}: {} distinct values\n'.format(feature,len(uniq)))
sns.set_context('talk')

plt.figure(figsize=(20,10))

cnt = wine['country'].value_counts().to_frame()[0:20]

#plt.xscale('log')

sns.barplot(x= cnt['country'], y =cnt.index, data=cnt, palette='colorblind',orient='h')

plt.title('Distribution of Wine Reviews of Top 20 Countries');
cnt = wine1.groupby(['country',]).median()['price'].sort_values(ascending=False).to_frame()



plt.figure(figsize=(20,15))

sns.pointplot(x = cnt['price'] ,y = cnt.index ,color='r',orient='h',markers='o')

plt.title('Country wise average wine price')

plt.xlabel('Price')

plt.ylabel('Country');
cnt = wine.groupby(['country',]).median()['price'].sort_values(ascending=False).to_frame()



plt.figure(figsize=(20,15))

sns.pointplot(x = cnt['price'] ,y = cnt.index ,color='r',orient='h',markers='o')

plt.title('Country wise average wine price')

plt.xlabel('Price')

plt.ylabel('Country');
cnt = wine.groupby(['country',]).median()['price'].sort_values(ascending=False).to_frame()

plt.figure(figsize = (50,10))

g2 = sns.stripplot(y='price', x= 'country', 

                   data=wine, 

                   jitter=True,

                   dodge=True,

                   marker='o', 

                   alpha=0.5)

g2.set_title("Country X Price Distribution", fontsize=30)

plt.show()
plt.figure(figsize=(16,8))



cnt = wine.groupby(['country'])['price'].max().sort_values(ascending=False).to_frame()[:20]

g2 = sns.barplot(x = cnt['price'], y = cnt.index, palette= 'colorblind')

g2.set_title('Most expensive wine in country')

g2.set_ylabel('Country')

g2.set_xlabel('')
cnt = wine.groupby(['country',]).mean()['points'].sort_values(ascending=False).to_frame()



plt.figure(figsize=(20,15))

sns.pointplot(x = cnt['points'] ,y = cnt.index ,color='r',orient='h',markers='o')

plt.title('Country wise average wine points')

plt.xlabel('Points')

plt.ylabel('Country')

plt.figure(figsize = (30,20))

g2 = sns.stripplot(y='points', x= 'country', 

                   data=wine, 

                   jitter=True,

                   dodge=True,

                   marker='o', 

                   alpha=0.5)

g2.set_title("Country X Points Distribution", fontsize=30)

plt.show()
sns.set_context("talk")

plt.figure(figsize=(20,10))

cnt = wine['variety'].value_counts().to_frame()[0:20]

sns.barplot(x= cnt['variety'], y =cnt.index, data=cnt, palette='colorblind',orient='h')

plt.title('Distribution of Wine Reviews of Top 20 Varieties');
sns.set_context("talk")

plt.figure(figsize=(20,18))

cnt = wine.groupby(['variety'])['price'].min().sort_values(ascending=True).to_frame()[:20]

g2 = sns.barplot(x = cnt['price'], y = cnt.index, palette= 'colorblind')

g2.set_title('The grapes used the cheap wines')

g2.set_ylabel('Variety')

g2.set_xlabel('')

plt.show()
plt.figure(figsize=(20,18))

cnt = wine.groupby(['variety'])['price'].max().sort_values(ascending=False).to_frame()[:25]

g2 = sns.barplot(x = cnt['price'], y = cnt.index, palette= 'colorblind')

g2.set_title('The grapes used for most expensive wine')

g2.set_ylabel('Variety')

g2.set_xlabel('')

plt.show()
plt.figure(figsize=(20,15))

cnt = wine.groupby(['variety'])['points'].max().sort_values(ascending=False).to_frame()[:20]

g2 = sns.barplot(x = cnt['points'], y = cnt.index, palette= 'colorblind')

g2.set_title('Varieties who got highest point')

g2.set_ylabel('Variety')

g2.set_xlabel('')

plt.show()
plt.figure(figsize=(16,5))



plt.subplot(1,2,1)

cnt = wine.groupby(['variety'])['points'].max().sort_values(ascending=False).to_frame()[:15]

g2 = sns.barplot(x = cnt['points'], y = cnt.index, palette= 'colorblind')

g2.set_title('Varieties who got highest point')

g2.set_ylabel('Variety')

g2.set_xlabel('')



plt.subplot(1,2,2)

cnt = wine.groupby(['variety'])['price'].min().sort_values(ascending=True).to_frame()[:15]

g2 = sns.barplot(x = cnt['price'], y = cnt.index, palette= 'colorblind')

g2.set_title('The grapes used the cheap wines')

g2.set_xlabel('')

plt.show()
highest_point = wine.groupby(['variety'])['points'].max().sort_values(ascending=False).to_frame()[:15]

cheap = wine.groupby(['variety'])['price'].min().sort_values(ascending=True).to_frame()[:25]



s1 = pd.merge(highest_point, cheap, how='inner', on=['variety'])

print(s1)
plt.figure(figsize=(20,10))

cnt = wine['taster_name'].value_counts().to_frame()[0:20]

sns.barplot(x= cnt['taster_name'], y =cnt.index, data=cnt, palette='colorblind',orient='h')

plt.title('Top 20 tasters')

plt.show()
wine.groupby("taster_name")["points"].describe()
plt.figure(figsize = (30,10))

g2 = sns.stripplot(y='points', x='taster_name', 

                   data=wine, 

                   jitter=True,

                   dodge=True,

                   marker='o', 

                   alpha=0.5)

g2.set_title("Taster Name Points Distribuition", fontsize=25)

plt.show()
plt.figure(figsize = (50,10))

sns.boxplot(y='points', x='taster_name', 

                 data=wine )

sns.stripplot(y='points', x='taster_name', 

                   data=wine, 

                   jitter=True,

                   dodge=True,

                   marker='o', 

                   alpha=0.5)

g2.set_title("Taster Name Points Distribuition", fontsize=25)

plt.show()
cnt1 = np.where(wine['points']==100)

for locs in cnt1:

    print(wine.loc[locs,['taster_name','variety']])
sns.set_context("talk")

plt.figure(figsize= (16,8))

plt.title('Word cloud of Description of lowest rated wine')

wc = WordCloud(max_words=1000,max_font_size=40,background_color='black', stopwords = STOPWORDS,colormap='Set1')

wc.generate(' '.join(wine[wine['points']==80]['description']))

plt.imshow(wc,interpolation="bilinear")

plt.axis('off')

plt.show()
plt.figure(figsize= (16,8))

plt.title('Word cloud of Description of highest rated wines')

wc = WordCloud(max_words=1000,max_font_size=40,background_color='black', stopwords = STOPWORDS,colormap='Set1')

wc.generate(' '.join(wine[wine['points']>=97]['description']))

plt.imshow(wc,interpolation="bilinear")

plt.axis('off')

plt.show()
plt.figure(figsize= (16,8))

plt.title('Word cloud of Description of most expensive wines')

wc = WordCloud(max_words=1000,max_font_size=40,background_color='black', stopwords = STOPWORDS,colormap='Set1')

wc.generate(' '.join(wine[wine['price']>=108]['description']))

plt.imshow(wc,interpolation="bilinear")

plt.axis('off')

plt.show()
plt.figure(figsize= (16,8))

plt.title('Word cloud of Description of most expensive wines')

wc = WordCloud(max_words=1000,max_font_size=40,background_color='black', stopwords = STOPWORDS,colormap='Set1')

wc.generate(' '.join(wine[wine['price']<=10]['description']))

plt.imshow(wc,interpolation="bilinear")

plt.axis('off')

plt.show()
sns.set_context('poster')

plt.figure(figsize= (16,8))

plt.title('Word cloud of Description')

wc = WordCloud(max_words=1000,max_font_size=40,background_color='black', stopwords = STOPWORDS,colormap='Set1')

wc.generate(' '.join(wine['description']))

plt.imshow(wc,interpolation="bilinear")

plt.axis('off')

plt.show()
wine = wine.assign(desc_length = wine['description'].apply(len))
sns.set_context('paper')

plt.figure(figsize=(14,6))



g = sns.regplot(x='desc_length', y='price',

                data=wine, fit_reg=True,  line_kws={'color':'black'},color = 'red' )

g.set_title('Price by Description Length', fontsize=20)

g.set_ylabel('Price(USD)', fontsize = 16) 

g.set_xlabel('Description Length', fontsize = 16)

g.set_xticklabels(g.get_xticklabels(),rotation=45)



plt.show()
g = sns.countplot(x='points', data=wine, palette = 'colorblind') # seting the seaborn countplot to known the points distribuition

g.set_title("Points Count distribuition ", fontsize=20) # seting title and size of font

g.set_xlabel("Points", fontsize=15) # seting xlabel and size of font

g.set_ylabel("Count", fontsize=15) # seting ylabel and size of font





plt.show() #rendering the graphs
# Finding the relations between the variables.

plt.figure(figsize=(10,5))

c= wine.corr()

sns.heatmap(c,cmap="coolwarm",annot=True) #BrBG, RdGy, coolwarm

c
plt.figure(figsize=(20,8))



g = sns.regplot(x='points', y='price', 

                data=wine, line_kws={'color':'black'},

                x_jitter=True, fit_reg=True, color = 'red')

g.set_title("Points x Price Distribuition", fontsize=20)

g.set_xlabel("Points", fontsize= 15)

g.set_ylabel("Price", fontsize= 15)



plt.show()
wine = wine.assign(description_length = wine['description'].apply(len))

fig, ax = plt.subplots(figsize=(30,10))

sns.boxplot(x='points', y='description_length', data=wine)

plt.xticks(fontsize=20) # X Ticks

plt.yticks(fontsize=20) # Y Ticks

ax.set_title('Description Length per Points', fontweight="bold", size=25) # Title

ax.set_ylabel('Description Length', fontsize = 25) # Y label

ax.set_xlabel('Points', fontsize = 25) # X label

plt.show()
def cat_points(points):

    if points in list(range(80,83)):

        return 0

    elif points in list(range(83,87)):

        return 1

    elif points in list(range(87,90)):

        return 2

    elif points in list(range(90,94)):

        return 3

    elif points in list(range(94,98)):

        return 4

    else:

        return 5



wine["rating_cat"] = wine["points"].apply(cat_points)
fig, ax = plt.subplots(figsize=(30,10))

plt.xticks(fontsize=20) # X Ticks

plt.yticks(fontsize=20) # Y Ticks

ax.set_title('Number of wines per points', fontweight="bold", size=25) # Title

ax.set_ylabel('Number of wines', fontsize = 25) # Y label

ax.set_xlabel('Points', fontsize = 25) # X label

wine.groupby(['rating_cat']).count()['description'].plot(ax=ax, kind='bar')
#given_point = int(input("Your preferred point: "))

#given_price = float(input("Whats your budget, in dollars?: "))



given_point = 96 #dummy value

given_price = 20 #dummy value

found = False

for row_index,row in wine.iterrows():

    if row['points']==given_point and row['price']<given_price:

        print(row['variety'], "   ", row['price'], "   ", row['points'])

        found = True

if(not found):

    print("Sorry, not found.")
fig, ax = plt.subplots(figsize=(30,10))

sns.boxplot(x='points', y='desc_length', data=wine)

plt.xticks(fontsize=20) # X Ticks

plt.yticks(fontsize=20) # Y Ticks

ax.set_title('Description Length per Points', fontweight="bold", size=25) # Title

ax.set_ylabel('Description Length', fontsize = 25) # Y label

ax.set_xlabel('Points', fontsize = 25) # X label

plt.show()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.gridspec as gridspec # to do the grid of plots

country = wine.country.value_counts()[:20]



grid = gridspec.GridSpec(5, 2)

plt.figure(figsize=(16,7*4))



for n, cat in enumerate(country.index[:10]):

    

    ax = plt.subplot(grid[n])   



    vectorizer = TfidfVectorizer(ngram_range = (2, 3), min_df=5, 

                                 stop_words='english',

                                 max_df=.5) 

    

    X2 = vectorizer.fit_transform(wine.loc[(wine.country == cat)]['description']) 

    features = (vectorizer.get_feature_names()) 

    scores = (X2.toarray()) 

    

    # Getting top ranking features 

    sums = X2.sum(axis = 0) 

    data1 = [] 

    

    for col, term in enumerate(features): 

        data1.append( (term, sums[0,col] )) 



    ranking = pd.DataFrame(data1, columns = ['term','rank']) 

    words = (ranking.sort_values('rank', ascending = False))[:15]

    

    sns.barplot(x='term', y='rank', data=words, ax=ax, 

                color='blue', orient='v')

    ax.set_title(f"Wine's from {cat} N-grams", fontsize=19)

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

    ax.set_ylabel(' ')

    ax.set_xlabel(" ")



plt.subplots_adjust(top = 0.95, hspace=.9, wspace=.1)



plt.show()
wine['price_log'] = np.log(wine['price'])

wine.head(2)
from nltk.sentiment.vader import SentimentIntensityAnalyzer



SIA = SentimentIntensityAnalyzer()



# Applying Model, Variable Creation

sentiment = wine.sample(15000).copy()

sentiment['polarity_score']=sentiment.description.apply(lambda x:SIA.polarity_scores(x)['compound'])

sentiment['neutral_score']=sentiment.description.apply(lambda x:SIA.polarity_scores(x)['neu'])

sentiment['negative_score']=sentiment.description.apply(lambda x:SIA.polarity_scores(x)['neg'])

sentiment['positive_score']=sentiment.description.apply(lambda x:SIA.polarity_scores(x)['pos'])



sentiment['sentiment']= np.nan

sentiment.loc[sentiment.polarity_score>0,'sentiment']='POSITIVE'

sentiment.loc[sentiment.polarity_score==0,'sentiment']='NEUTRAL'

sentiment.loc[sentiment.polarity_score<0,'sentiment']='NEGATIVE'
def sentiment_analyzer_scores(sentence):

    score = SIA.polarity_scores(sentence)

    print("{:-<40} {}".format(sentence, str(score)))
print(sentiment_analyzer_scores("yaaaaay"))

print(sentiment_analyzer_scores("Today is a sunny day #love"))

print(sentiment_analyzer_scores("UGGHHH SUCH A BORING DAY"))

print(sentiment_analyzer_scores("i like kaggle a lot lol"))

print(sentiment_analyzer_scores("i like kaggle a lot!!"))
plt.figure(figsize=(14,5))



plt.suptitle('Sentiment of the reviews by: \n- Points and Price(log) -', size=22)



plt.subplot(121)

ax = sns.boxplot(x='sentiment', y='points', data=sentiment,palette = 'pastel')

ax.set_title("Sentiment by Points Distribution", fontsize=19)

ax.set_ylabel("Points ", fontsize=17)

ax.set_xlabel("Sentiment Label", fontsize=17)



plt.subplot(122)

ax1= sns.boxplot(x='sentiment', y='price_log', data=sentiment,palette = 'pastel')

ax1.set_title("Sentiment by Price Distribution", fontsize=19)

ax1.set_ylabel("Price (log) ", fontsize=17)

ax1.set_xlabel("Sentiment Label", fontsize=17)



plt.subplots_adjust(top = 0.75, wspace=.2)

plt.show()
from sklearn.neighbors import NearestNeighbors # KNN Clustering 

from scipy.sparse import csr_matrix # Compressed Sparse Row matrix

from sklearn.decomposition import TruncatedSVD # Dimensional Reduction
# Lets choice rating of wine is points, title as user_id, and variety,

col = ['province','variety','points']



wine1 = wine[col]

wine1 = wine1.dropna(axis=0)

wine1 = wine1.drop_duplicates(['province','variety'])

wine1 = wine1[wine1['points'] > 85]



wine_pivot = wine1.pivot(index= 'variety',columns='province',values='points').fillna(0)

wine_pivot_matrix = csr_matrix(wine_pivot)
wine_pivot_matrix
from sklearn.cluster import KMeans

from scipy.cluster.vq import kmeans, vq
trial = wine[['price', 'points']]

data = np.asarray([np.asarray(trial['price']), np.asarray(trial['points'])]).T
X = data

distortions = []

for k in range(2,30):

    k_means = KMeans(n_clusters = k)

    k_means.fit(X)

    distortions.append(k_means.inertia_)



fig = plt.figure(figsize=(15,10))

plt.plot(range(2,30), distortions, 'bx-')

plt.title("Elbow Curve")

plt.show()
knn = NearestNeighbors(n_neighbors=7, algorithm = 'brute',metric = 'cosine')

model_knn = knn.fit(wine_pivot_matrix)
for n in range(5):

    query_index = np.random.choice(wine_pivot.shape[0])

    #print(n, query_index)

    distance, indice = model_knn.kneighbors(wine_pivot.iloc[query_index,:].values.reshape(1,-1), n_neighbors=6)

    for i in range(0, len(distance.flatten())):

        if  i == 0:

            print('Recommendation for ## {0} ##:'.format(wine_pivot.index[query_index]))

        else:

            print('{0}: {1} with distance: {2}'.format(i,wine_pivot.index[indice.flatten()[i]],distance.flatten()[i]))

    print('\n')