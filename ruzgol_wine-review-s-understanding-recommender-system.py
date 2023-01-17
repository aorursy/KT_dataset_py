import pandas as pd #Library to handle with dataframes

import matplotlib.pyplot as plt # Library to plot graphics

import numpy as np # To handle with matrices

import seaborn as sns # to build modern graphics

from scipy.stats import kurtosis, skew # it's to explore some statistics of numerical values

from scipy import stats
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary



def CalcOutliers(df_num): 

    '''

    

    Leonardo Ferreira 20/10/2018

    Set a numerical value and it will calculate the upper, lower and total number of outliers

    It will print a lot of statistics of the numerical feature that you set on input

    

    '''

    # calculating mean and std of the array

    data_mean, data_std = np.mean(df_num), np.std(df_num)



    # seting the cut line to both higher and lower values

    # You can change this value

    cut = data_std * 3



    #Calculating the higher and lower cut values

    lower, upper = data_mean - cut, data_mean + cut



    # creating an array of lower, higher and total outlier values 

    outliers_lower = [x for x in df_num if x < lower]

    outliers_higher = [x for x in df_num if x > upper]

    outliers_total = [x for x in df_num if x < lower or x > upper]



    # array without outlier values

    outliers_removed = [x for x in df_num if x > lower and x < upper]

    

    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers

    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers

    print('Identified outliers: %d' % len(outliers_total)) # printing total number of values outliers of both sides

    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values

    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points

    

    return



# Importing our dataset in variable df_wine1

df_wine1 = pd.read_csv('../input/winemag-data-130k-v2.csv', index_col=0)
df_wine1
resumetable(df_wine1)
# The function describe is focused on numerical features

# in this case are points and price

print("Statistics of numerical data: ")

print(df_wine1.describe())
# define the size of figures that I will build

plt.figure(figsize=(16,5))



plt.subplot(1,2,1) # this will create a grid of 1 row and 2 columns; this is the first graphic

g = sns.countplot(x='points', data=df_wine1, color='forestgreen') # seting the seaborn countplot to known the points distribuition

g.set_title("Points Count distribuition ", fontsize=20) # seting title and size of font

g.set_xlabel("Points", fontsize=15) # seting xlabel and size of font

g.set_ylabel("Count", fontsize=15) # seting ylabel and size of font



plt.subplot(1,2,2)  # this will set the second graphic of our grid

plt.scatter(range(df_wine1.shape[0]), np.sort(df_wine1.points.values), color='forestgreen') # creating a cumulative distribution

plt.xlabel('Index', fontsize=15)  # seting xlabel and size of font

plt.ylabel('Points Dist(US)', fontsize=15)  # seting ylabel and size of font

plt.title("Points Distribuition", fontsize=20) # seting title and size of font



plt.show() #rendering the graphs
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



df_wine1["rating_cat"] = df_wine1["points"].apply(cat_points)
total = len(df_wine1)

plt.figure(figsize=(14,6))



g = sns.countplot(x='rating_cat', color='darkgreen',

                  data=df_wine1)

g.set_title("Point Categories Counting Distribution", fontsize=20)

g.set_xlabel("Categories ", fontsize=15)

g.set_ylabel("Total Count", fontsize=15)



sizes=[]



for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format((height/total)*100),

            ha="center", fontsize=14) 

    

g.set_ylim(0, max(sizes) * 1.15)



plt.show()
CalcOutliers(df_wine1['points'])
plt.figure(figsize=(14,5))



g1 = plt.subplot(121)

g1 = sns.distplot(np.log(df_wine1['price'].dropna() + 1),

                  color='darkgreen')

g1.set_title("Price Log distribuition  ", fontsize=20)

g1.set_xlabel("Price(Log)", fontsize=15)

g1.set_ylabel("Frequency LOG", fontsize=15)



plt.subplot(122)

plt.scatter(range(df_wine1.shape[0]), np.sort(df_wine1.price.values), 

            color='darkgreen')

plt.xlabel('Index', fontsize=15)

plt.ylabel('Prices(US)', fontsize=15)

plt.title("Distribuition of prices", fontsize=20)





plt.show()
CalcOutliers(df_wine1['price'])
plt.figure(figsize=(12,5))



g = sns.distplot(df_wine1[df_wine1['price'] < 300]['price'], color='darkgreen')

g.set_title("Price Distribuition Filtered 300", fontsize=20)

g.set_xlabel("Prices(US)", fontsize=15)

g.set_ylabel("Frequency Distribuition", fontsize=15)





plt.show()
# Let's get tehe price_log to better work with this feature

df_wine1['price_log'] = np.log(df_wine1['price'])
plt.figure(figsize=(10,4))



g = sns.regplot(x='points', y='price_log', 

                data=df_wine1, line_kws={'color':'red'},

                x_jitter=True, fit_reg=True, color='darkgreen')

g.set_title("Points x Price Distribuition", fontsize=20)

g.set_xlabel("Points", fontsize= 15)

g.set_ylabel("Price (log)", fontsize= 15)



plt.show()
plt.figure(figsize=(14,6))



country = df_wine1.country.value_counts()[:20]



g = sns.countplot(x='country', 

                  data=df_wine1[df_wine1.country.isin(country.index.values)],

                 color='darkgreen')

g.set_title("Country Of Wine Origin Count", fontsize=20)

g.set_xlabel("Country's ", fontsize=15)

g.set_ylabel("Count", fontsize=15)

g.set_xticklabels(g.get_xticklabels(),rotation=45)



plt.show()
plt.figure(figsize=(16,12))



plt.subplot(2,1,1)

g = sns.boxplot(x='country', y='price_log',

                  data=df_wine1.loc[(df_wine1.country.isin(country.index.values))],

                 color='darkgreen')

g.set_title("Price by Country Of Wine Origin", fontsize=20)

g.set_xlabel("Country's ", fontsize=15)

g.set_ylabel("Price Dist(US)", fontsize=15)

g.set_xticklabels(g.get_xticklabels(),rotation=45)



plt.subplot(2,1,2)

g1 = sns.boxplot(x='country', y='points',

                   data=df_wine1[df_wine1.country.isin(country.index.values)],

                 color='darkgreen')

g1.set_title("Points by Country Of Wine Origin", fontsize=20)

g1.set_xlabel("Country's ", fontsize=15)

g1.set_ylabel("Points", fontsize=15)

g1.set_xticklabels(g1.get_xticklabels(),rotation=45)



plt.subplots_adjust(hspace = 0.6,top = 0.9)



plt.show()
plt.figure(figsize=(15,5))

g = sns.boxplot(x='country', y='price_log', color='darkgreen',

                  data=df_wine1)

g.set_title("Price by Country Of Wine Origin", fontsize=20)

g.set_xlabel("Country's ", fontsize=15)

g.set_ylabel("Price Dist(US)", fontsize=15)

g.set_xticklabels(g.get_xticklabels(),rotation=90)



plt.show()
plt.figure(figsize=(14,15))



provinces = df_wine1['province'].value_counts()[:20]



plt.subplot(3,1,1)

g = sns.countplot(x='province', 

                  data=df_wine1.loc[(df_wine1.province.isin(provinces.index.values))], 

                  color='darkgreen')

g.set_title("Province Of Wine Origin ", fontsize=20)

g.set_xlabel("Provinces", fontsize=15)

g.set_ylabel("Count", fontsize=15)

g.set_xticklabels(g.get_xticklabels(),rotation=45)



plt.subplot(3,1,2)

g1 = sns.boxplot(y='price', x='province',

                  data=df_wine1.loc[(df_wine1.province.isin(provinces.index.values))], 

                  color='darkgreen')

g1.set_title("Province Of Wine Origin ", fontsize=20)

g1.set_xlabel("Province", fontsize=15)

g1.set_ylabel("Price", fontsize=15)

g1.set_xticklabels(g1.get_xticklabels(),rotation=45)



plt.subplot(3,1,3)

g2 = sns.boxplot(y='points', x='province',

                  data=df_wine1.loc[(df_wine1.province.isin(provinces.index.values))], 

                  color='darkgreen')

g2.set_title("Province Of Wine Origin", fontsize=20)

g2.set_xlabel("Provinces", fontsize=15)

g2.set_ylabel("Points", fontsize=15)

g2.set_xticklabels(g2.get_xticklabels(),rotation=45)



plt.subplots_adjust(hspace = 0.6,top = 0.9)



plt.show()
plt.figure(figsize=(14,16))



provinces = df_wine1['province'].value_counts()[:20]



plt.subplot(3,1,1)

g = sns.countplot(x='taster_name', data=df_wine1, color='darkgreen')

g.set_title("Taster Name Count - TOP 20 ", fontsize=20)

g.set_xlabel("Taster Name", fontsize=15)

g.set_ylabel("Count", fontsize=15)

g.set_xticklabels(g.get_xticklabels(),rotation=45)



plt.subplot(3,1,2)

g1 = sns.boxplot(y='price_log', x='taster_name', data=df_wine1, 

                 color='darkgreen')

g1.set_title("Taster Name Wine Values Distribuition ", fontsize=20)

g1.set_xlabel("Taster Name", fontsize=15)

g1.set_ylabel("Price", fontsize=15)

g1.set_xticklabels(g1.get_xticklabels(),rotation=45)



plt.subplot(3,1,3)

g2 = sns.boxplot(y='points', x='taster_name',

                  data=df_wine1, color='darkgreen')

g2.set_title("Taster Name Points Distribuition", fontsize=20)

g2.set_xlabel("Taster Name", fontsize=15)

g2.set_ylabel("Points", fontsize=15)

g2.set_xticklabels(g2.get_xticklabels(),rotation=45)



plt.subplots_adjust(hspace = 0.6,top = 0.9)



plt.show()
plt.figure(figsize=(14,16))



designation = df_wine1.designation.value_counts()[:20]



plt.subplot(3,1,1)

g = sns.countplot(x='designation', 

                  data=df_wine1.loc[(df_wine1.designation.isin(designation.index.values))],

                  color='darkgreen')

g.set_title("Province Of Wine Origin ", fontsize=20)

g.set_xlabel("Country's ", fontsize=15)

g.set_ylabel("Count", fontsize=15)

g.set_xticklabels(g.get_xticklabels(),rotation=45)



plt.subplot(3,1,2)

g1 = sns.boxplot(y='price_log', x='designation',

                  data=df_wine1.loc[(df_wine1.designation.isin(designation.index.values))], 

                 color='darkgreen')

g1.set_title("Province Of Wine Origin ", fontsize=20)

g1.set_xlabel("Province", fontsize=15)

g1.set_ylabel("Price", fontsize=15)

g1.set_xticklabels(g1.get_xticklabels(),rotation=45)



plt.subplot(3,1,3)

g2 = sns.boxplot(y='points', x='designation',

                  data=df_wine1.loc[(df_wine1.designation.isin(designation.index.values))], 

                 color='darkgreen')

g2.set_title("Province Of Wine Origin", fontsize=20)

g2.set_xlabel("Provinces", fontsize=15)

g2.set_ylabel("Points", fontsize=15)

g2.set_xticklabels(g2.get_xticklabels(),rotation=45)



plt.subplots_adjust(hspace = 0.6,top = 0.9)



plt.show()
plt.figure(figsize=(14,16))



variety = df_wine1.variety.value_counts()[:20]



plt.subplot(3,1,1)

g = sns.countplot(x='variety', 

                  data=df_wine1.loc[(df_wine1.variety.isin(variety.index.values))], 

                  color='darkgreen')

g.set_title("TOP 20 Variety ", fontsize=20)

g.set_xlabel(" ", fontsize=15)

g.set_ylabel("Count", fontsize=15)

g.set_xticklabels(g.get_xticklabels(),rotation=45)



plt.subplot(3,1,2)

g1 = sns.boxplot(y='price_log', x='variety',

                  data=df_wine1.loc[(df_wine1.variety.isin(variety.index.values))], 

                 color='darkgreen')

g1.set_title("Price by Variety's", fontsize=20)

g1.set_xlabel("", fontsize=15)

g1.set_ylabel("Price", fontsize=15)

g1.set_xticklabels(g1.get_xticklabels(),rotation=45)



plt.subplot(3,1,3)

g2 = sns.boxplot(y='points', x='variety',

                  data=df_wine1.loc[(df_wine1.variety.isin(variety.index.values))], 

                 color='darkgreen')

g2.set_title("Points by Variety's", fontsize=20)

g2.set_xlabel("Variety's", fontsize=15)

g2.set_ylabel("Points", fontsize=15)

g2.set_xticklabels(g2.get_xticklabels(),rotation=90)



plt.subplots_adjust(hspace = 0.7,top = 0.9)



plt.show()
plt.figure(figsize=(14,16))



winery = df_wine1.winery.value_counts()[:20]



plt.subplot(3,1,1)

g = sns.countplot(x='winery', 

                  data=df_wine1.loc[(df_wine1.winery.isin(winery.index.values))], 

                  color='darkgreen')

g.set_title("TOP 20 most frequent Winery's", fontsize=20)

g.set_xlabel(" ", fontsize=15)

g.set_ylabel("Count", fontsize=15)

g.set_xticklabels(g.get_xticklabels(),rotation=45)



plt.subplot(3,1,2)

g1 = sns.boxplot(y='price_log', x='winery',

                  data=df_wine1.loc[(df_wine1.winery.isin(winery.index.values))],

                 color='darkgreen')

g1.set_title("Price by Winery's", fontsize=20)

g1.set_xlabel("", fontsize=15)

g1.set_ylabel("Price", fontsize=15)

g1.set_xticklabels(g1.get_xticklabels(),rotation=45)



plt.subplot(3,1,3)

g2 = sns.boxplot(y='points', x='winery',

                  data=df_wine1.loc[(df_wine1.winery.isin(winery.index.values))],

                 color='darkgreen')

g2.set_title("Points by Winery's", fontsize=20)

g2.set_xlabel("Winery's", fontsize=15)

g2.set_ylabel("Points", fontsize=15)

g2.set_xticklabels(g2.get_xticklabels(),rotation=90)



plt.subplots_adjust(hspace = 0.7,top = 0.9)



plt.show()
df_wine1 = df_wine1.assign(desc_length = df_wine1['description'].apply(len))



plt.figure(figsize=(14,6))

g = sns.boxplot(x='points', y='desc_length', data=df_wine1,

                color='darkgreen')

g.set_title('Description Length by Points', fontsize=20)

g.set_ylabel('Description Length', fontsize = 16) # Y label

g.set_xlabel('Points', fontsize = 16) # X label

plt.show()
plt.figure(figsize=(14,6))



g = sns.boxplot(x='taster_name', y='desc_length', 

                data=df_wine1, color='darkgreen')

g.set_title('Description Length by Taster Name', fontsize=20)

g.set_ylabel('Description Length', fontsize = 16) # Y label

g.set_xlabel('Taster Name', fontsize = 16) # X label

g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.show()
plt.figure(figsize=(14,6))



g = sns.regplot(x='desc_length', y='price_log', line_kws={'color':'red'},

                data=df_wine1, fit_reg=True, color='darkgreen', )

g.set_title('Price by Description Length', fontsize=20)

g.set_ylabel('Price(USD)', fontsize = 16) 

g.set_xlabel('Description Length', fontsize = 16)

g.set_xticklabels(g.get_xticklabels(),rotation=45)



plt.show()


from wordcloud import WordCloud, STOPWORDS



stopwords = set(STOPWORDS)



newStopWords = ['fruit', "Drink", "black", 'wine', 'drink']



stopwords.update(newStopWords)



wordcloud = WordCloud(

    background_color='white',

    stopwords=stopwords,

    max_words=300,

    max_font_size=200, 

    width=1000, height=800,

    random_state=42,

).generate(" ".join(df_wine1['description'].astype(str)))



print(wordcloud)

fig = plt.figure(figsize = (12,14))

plt.imshow(wordcloud)

plt.title("WORD CLOUD - DESCRIPTION",fontsize=25)

plt.axis('off')

plt.show()
wordcloud = WordCloud(

    background_color='white',

    stopwords=stopwords,

    max_words=300,

    max_font_size=200, 

    width=1000, height=800,

    random_state=42,

).generate(" ".join(df_wine1['title'].astype(str)))



print(wordcloud)

fig = plt.figure(figsize = (12,14))

plt.imshow(wordcloud)

plt.title("WORD CLOUD - TITLES",fontsize=25)

plt.axis('off')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import matplotlib.gridspec as gridspec # to do the grid of plots



grid = gridspec.GridSpec(5, 2)

plt.figure(figsize=(16,7*4))



for n, cat in enumerate(country.index[:10]):

    

    ax = plt.subplot(grid[n])   

    # print(f'PRINCIPAL WORDS CATEGORY: {cat}')

    # vectorizer = CountVectorizer(ngram_range = (3,3)) 

    # X1 = vectorizer.fit_transform(df_train[df_train['host_cat'] == cat]['answer'])  

    # print(cat)

    # Applying TFIDF 

    vectorizer = TfidfVectorizer(ngram_range = (2, 3), min_df=5, 

                                 stop_words='english',

                                 max_df=.5) 

    

    X2 = vectorizer.fit_transform(df_wine1.loc[(df_wine1.country == cat)]['description']) 

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
from nltk.sentiment.vader import SentimentIntensityAnalyzer



SIA = SentimentIntensityAnalyzer()



# Applying Model, Variable Creation

sentiment = df_wine1.sample(15000).copy()

sentiment['polarity_score']=sentiment.description.apply(lambda x:SIA.polarity_scores(x)['compound'])

sentiment['neutral_score']=sentiment.description.apply(lambda x:SIA.polarity_scores(x)['neu'])

sentiment['negative_score']=sentiment.description.apply(lambda x:SIA.polarity_scores(x)['neg'])

sentiment['positive_score']=sentiment.description.apply(lambda x:SIA.polarity_scores(x)['pos'])



sentiment['sentiment']= np.nan

sentiment.loc[sentiment.polarity_score>0,'sentiment']='POSITIVE'

sentiment.loc[sentiment.polarity_score==0,'sentiment']='NEUTRAL'

sentiment.loc[sentiment.polarity_score<0,'sentiment']='NEGATIVE'

plt.figure(figsize=(14,5))



plt.suptitle('Sentiment of the reviews by: \n- Points and Price(log) -', size=22)



plt.subplot(121)

ax = sns.boxplot(x='sentiment', y='points', data=sentiment)

ax.set_title("Sentiment by Points Distribution", fontsize=19)

ax.set_ylabel("Points ", fontsize=17)

ax.set_xlabel("Sentiment Label", fontsize=17)



plt.subplot(122)

ax1= sns.boxplot(x='sentiment', y='price_log', data=sentiment)

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



wine1 = df_wine1[col]

wine1 = wine1.dropna(axis=0)

wine1 = wine1.drop_duplicates(['province','variety'])

wine1 = wine1[wine1['points'] > 85]



wine_pivot = wine1.pivot(index= 'variety',columns='province',values='points').fillna(0)

wine_pivot_matrix = csr_matrix(wine_pivot)
knn = NearestNeighbors(n_neighbors=10, algorithm= 'brute', metric= 'cosine')

model_knn = knn.fit(wine_pivot_matrix)
for n in range(5):

    query_index = np.random.choice(wine_pivot.shape[0])

    #print(n, query_index)

    distance, indice = model_knn.kneighbors(wine_pivot.iloc[query_index,:].values.reshape(1,-1), n_neighbors=6)

    for i in range(0, len(distance.flatten())):

        if  i == 0:

            print('Recmmendation for ## {0} ##:'.format(wine_pivot.index[query_index]))

        else:

            print('{0}: {1} with distance: {2}'.format(i,wine_pivot.index[indice.flatten()[i]],distance.flatten()[i]))

    print('\n')