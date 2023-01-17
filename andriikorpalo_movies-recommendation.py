import numpy as np

import pandas as pd

from ast import literal_eval

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from surprise import Reader, Dataset, SVD, KNNBaseline

from surprise.model_selection import cross_validate

import seaborn as sn

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import plotly.graph_objs as go 

import pylab as pl

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn import preprocessing

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import datetime

from fbprophet import Prophet

import matplotlib.pyplot as plt

import seaborn as sns
credits = pd.read_csv('../input/the-movies-dataset/credits.csv')

moviesMetaData = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv',low_memory=False)

keywords = pd.read_csv('../input/the-movies-dataset/keywords.csv')

ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
ratings.rename(columns={'movieId': 'id'}, inplace = True)
moviesMetaData['id'] = moviesMetaData['id'].astype(str)

credits['id'] = credits['id'].astype(str)

keywords['id'] = keywords['id'].astype(str)

ratings['id'] = ratings['id'].astype(str)
ratings = ratings[['id']]

ratings = ratings.drop_duplicates()



moviesMetaData = pd.merge(moviesMetaData,ratings, on='id')
mainList= pd.merge(moviesMetaData, credits, on='id')

mainList= pd.merge(mainList,keywords, on='id')

corrMatrix = mainList.corr()

sn.heatmap(mainList.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
def missingDF(data):

    missing_df = data.isnull().sum(axis=0).reset_index()

    missing_df.columns = ['column_name', 'missing_count']

    missing_df['filling_factor'] = (mainList.shape[0] 

                                    - missing_df['missing_count']) / mainList.shape[0] * 100

    missing_df.sort_values('filling_factor').reset_index(drop = True)

    

    missing_df = missing_df.sort_values('filling_factor').reset_index(drop = True)

    y_axis = missing_df['filling_factor'] 

    x_label = missing_df['column_name']

    x_axis = missing_df.index



    fig = plt.figure(figsize=(11, 4))

    plt.xticks(rotation=80, fontsize = 14)

    plt.yticks(fontsize = 13)



    plt.xticks(x_axis, x_label,family='fantasy', fontsize = 14 )

    plt.ylabel('Filling factor (%)', family='fantasy', fontsize = 16)

    plt.bar(x_axis, y_axis);

    

    return missing_df
table = missingDF(mainList)
mainList['release_date'] =  pd.to_datetime(mainList['release_date']) 

mainList['years'] = mainList['release_date'].apply(lambda x: x.year)



mainList[(mainList['years'] < 2019) & (mainList['years'] >= 1950)].groupby(by = 'years').mean()['vote_count'].plot()
mainList['budget'] = mainList['budget'].astype(float)

mainList['popularity'] = mainList['popularity'].astype(float)

sn.heatmap(mainList.corr(), annot=True)
mainList = mainList[['id', 'title', 'cast', 'crew', 'keywords', 'genres']]
mainList.head(5)


features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    mainList[feature] = mainList[feature].apply(literal_eval)

    

mainList.head()
def getDirector(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan
def getFirstThree(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        

        if len(names) > 3:

            names = names[:3]

        return names



    return []
mainList['director'] = mainList['crew'].apply(getDirector)



features = ['cast', 'keywords', 'genres']

for feature in features:

    mainList[feature] = mainList[feature].apply(getFirstThree)
mainList[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
def counting_values(df, column):

    value_count = {}

    for row in df[column].dropna():

        if len(row) > 0:

            for key in row:

                if key in value_count:

                    value_count[key] += 1

                else:

                    value_count[key] = 1

        else:

            pass

    return value_count
def count_director(df, column):

    value_count = {}

    for key in df[column].dropna():

        if key in value_count:

            value_count[key] += 1

        else:

            value_count[key] = 1

        

    return value_count

    
sn.heatmap(mainList.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
def hist(data):

    iData = dict(sorted(data.items(), key=lambda x: x[1],reverse=True)[:20])

    pos = np.arange(len(iData.keys()))

    width = 1.0

    

    ax = plt.axes()

    ax.set_xticks(pos + (width / 2))

    ax.set_xticklabels(iData.keys())

    plt.yticks(fontsize = 15)

    plt.xticks(rotation=85, fontsize = 15)

    plt.grid()

    plt.bar(iData.keys(), iData.values(), width, align = 'center', color='g')

    plt.show()
def createWordCloud(data):

    wordcloud = WordCloud(max_font_size=100)



    wordcloud.generate_from_frequencies(data)

     

    plt.figure(figsize=[10.1,10.1])

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")



    plt.show()

    genres_count = pd.Series(data)

    genres_count.sort_values(ascending = False).head(20).plot(kind = 'bar', grid='True')
createWordCloud(counting_values(mainList, 'genres'))
createWordCloud(counting_values(mainList, 'cast'))
createWordCloud(counting_values(mainList, 'keywords'))
createWordCloud(count_director(mainList, 'director'))
def deletingSpaces(x):

    if isinstance(x, list):

        return [str.lower(i.replace(" ", "")) for i in x]

    else:

        

        if isinstance(x, str):

            return str.lower(x.replace(" ", ""))

        else:

            return ''

        

features = ['cast', 'keywords', 'director']



for feature in features:

    mainList[feature] = mainList[feature].apply(deletingSpaces)

    

mainList.head(5)
def combineKeywords(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])



mainList['myKeywords'] = mainList.apply(combineKeywords, axis=1)



mainList['myKeywords'].head(5)
table = missingDF(mainList)
count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(mainList['myKeywords'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)
mainList = mainList.reset_index()

indices = pd.Series(mainList.index, index=mainList['title'])
def get_recommendations(title, cosine_sim=cosine_sim):

    idx = indices[title]



    sim_scores = list(enumerate(cosine_sim[idx]))



    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    sim_scores = sim_scores[1:11]



    movie_indices = [i[0] for i in sim_scores]

    

    sim_value = [x[1] for x in sim_scores]

    

    result = indices.iloc[movie_indices]

    

    result[0:10] = sim_value



    return(result)
def show(title):

    result = get_recommendations(title)



    plt.figure(figsize=(10,5))

    sn.barplot(x = result[0:10], y=result.index)

    plt.title("Recommended Movies from " + str.upper(title), fontdict= {'fontsize' :20})

    plt.xlabel("Cosine Similarities")

    plt.show()

show('Twelve Monkeys')
show('Twelve Monkeys')
get_recommendations('Twelve Monkeys')
reader = Reader()

ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

corrMatrix = ratings.corr()

ratings.head()
corrMatrix = ratings.corr()

sn.heatmap(corrMatrix, annot=True)
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
alg = KNNBaseline()

cross_validate(alg, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()

alg.fit(trainset)
alg.predict(1, 39)
def getForecast(userId, title):

    idx = indices[title]



    sim_scores = list(enumerate(cosine_sim[idx]))



    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    

    sim_scores = sim_scores[1:11]

    

    show(title)

    

    movie_indices = [i[0] for i in sim_scores]

    

    movies = mainList.iloc[movie_indices][['id', 'title']]

    movies['id'] = movies['id'].astype(int)

    

    def getEst(item):

        return alg.predict(userId, item['id']).est

    

    movies['est'] = movies.apply(getEst, axis=1)

    

    return movies.head(10)
getForecast(5,'From Dusk Till Dawn')
getForecast(78,'From Dusk Till Dawn')
getForecast(76,'Twelve Monkeys')
getForecast(32,'Twelve Monkeys')
df = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)
df.drop(df.index[19730],inplace=True)

df.drop(df.index[29502],inplace=True)

df.drop(df.index[35585],inplace=True)



df_numeric = df[['budget','popularity','revenue','runtime','vote_average','vote_count','title']]
df_numeric.head()
df_numeric.dropna(inplace=True)



df_numeric.isnull().sum()
df_numeric = df_numeric[df_numeric['vote_count']>30]
minmax_processed = preprocessing.MinMaxScaler().fit_transform(df_numeric.drop('title',axis=1))

df_numeric_scaled = pd.DataFrame(minmax_processed, index=df_numeric.index, columns=df_numeric.columns[:-1])
Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]



score = [kmeans[i].fit(df_numeric_scaled).score(df_numeric_scaled) for i in range(len(kmeans))]



pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()
kmeans = KMeans(n_clusters=5)

kmeans.fit(df_numeric_scaled)

df_numeric['cluster'] = kmeans.labels_



df_numeric.head(20)
plt.figure(figsize=(12,7))

axis = sn.barplot(x=np.arange(0,5,1),y=df_numeric.groupby(['cluster']).count()['budget'].values)

x=axis.set_xlabel("Cluster Number")

x=axis.set_ylabel("Number of movies")
df_numeric.groupby(['cluster']).mean()
df = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)



df_numeric = df[['revenue','runtime','vote_average','vote_count']]

df_numeric.dropna(inplace=True)

x_data = df_numeric[['revenue','runtime','vote_count']]

y_data = df_numeric['vote_average']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)



D_train = xgb.DMatrix(X_train, label=y_train)

D_test = xgb.DMatrix(X_test, label=y_test)



param = { 

    "silent":True,"eta":0.01,'subsample': 0.75,'colsample_bytree': 0.7,"max_depth":7, 'metric': 'rmse'} 



steps = 20 

model = xgb.train(param, D_train, steps)



preds = model.predict(D_test)

df = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)

df = df.dropna()

df = df[['vote_average','revenue']]

df = df.loc[df['vote_average'] > 5]

df = df.loc[df['revenue'] < 1500000000]

df = df.loc[df['revenue'] > 1000000]



sns.lmplot(x ="vote_average", y ="revenue", data = df, order = 2, ci = None) 
X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 20, test_size = 0.5)



model = LinearRegression()

model.fit(X_train, y_train)



model.score(X_test, y_test)

pred1 = model.predict(X_test) 

plt.scatter(X_test, y_test, color ='b') 

plt.plot(X_test, pred1, color ='k') 

  

plt.show() 
pred1 = model.predict(X_test) 

plt.scatter(X_test, y_test, color ='b') 

plt.plot(X_test, pred1, color ='k') 

  

plt.show() 
df = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)

df = df[['release_date', 'revenue']];

df = df.loc[df['revenue'] > 1000000]

df = df.loc[df['revenue'] < 1500000000]

df = df.loc[df['release_date'] > "2012-01-21"]

df = df.sort_values(by='release_date',ascending=True)

df.plot(figsize=(12,6), x='release_date',y='revenue')

df.reset_index(drop=True)

df.head(5)
split_date = '2015-01-01'

train_data = df[df['release_date'] <= split_date].copy()

test_data = df[df['release_date'] > split_date].copy()
model = Prophet()

prophetData = train_data.reset_index().rename(columns={'release_date':'ds','revenue':'y'})

model.fit(prophetData)



prophetTestData = test_data.reset_index().rename(columns={'release_date':'ds','revenue':'y'})

forecast = model.predict(df=prophetTestData)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)