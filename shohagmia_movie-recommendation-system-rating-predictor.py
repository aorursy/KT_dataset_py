%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from ast import literal_eval

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import wordnet

from surprise import Reader, Dataset, SVD

from sklearn import svm

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

import datetime

import string





import warnings; warnings.simplefilter('ignore')

THRESHOLD_PREDICTION = 1
def clean_sentence(s, concat=None):

    s = s.translate(str.maketrans('', '', string.punctuation))

    s = s.lower()

    if concat:

        s = concat.join(s.split())

    return s



def _make_in_format(df):

    y = np.array(df['rating'])

    temp_x = df.drop('rating', axis=1)  

#     print(temp_x)

    #min-max normalization

#     temp_x = (temp_x-temp_x.mean())/(temp_x.max()-temp_x.min())

    X = np.array(temp_x)



    return X,y



def accuracy_score(y_test,predictions):

        correct = []

        for i in range(len(y_test)):

            if predictions[i]>=y_test[i]-THRESHOLD_PREDICTION and predictions[i]<=y_test[i]+THRESHOLD_PREDICTION:

                correct.append(1)

            else:

                correct.append(0)



        accuracy = sum(map(int,correct))*1.0/len(correct)

        return accuracy
md = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')

md.head()
credits = pd.read_csv('../input/the-movies-dataset/credits.csv')

keywords = pd.read_csv('../input/the-movies-dataset/keywords.csv')

ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
credits.head(5)
keywords.head(5)
ratings.head(5)
keywords_len = len(keywords)

keywords_dict = {}

keywords_id_dict = {}



for it in range(keywords_len):

    keywords_arr = keywords.iloc[it]['keywords']

    keywords_arr = eval(keywords_arr)

    keywords_id_dict[keywords.iloc[it]['id']]=""

    for iit in range(len(keywords_arr)):

        keywords_id_dict[keywords.iloc[it]['id']] = keywords_id_dict[keywords.iloc[it]['id']] + clean_sentence(keywords_arr[iit]['name']) + " "

        if keywords_dict.get(keywords_arr[iit]['name']):

            keywords_dict[keywords_arr[iit]['name']] = keywords_dict[keywords_arr[iit]['name']]+1

        else:

            keywords_dict[keywords_arr[iit]['name']]=1
# sort in ascending order of occurence

keyword_occurences = []

for k,v in keywords_dict.items():

    keyword_occurences.append([k,v])

keyword_occurences.sort(key = lambda x:x[1], reverse = True)
# HISTOGRAMS

fig = plt.figure(1, figsize=(18,13))

ax = fig.add_subplot(1,1,1)

trunc_occurences = keyword_occurences[0:50]

y_axis = [i[1] for i in trunc_occurences]

x_axis = [k for k,i in enumerate(trunc_occurences)]

x_label = [i[0] for i in trunc_occurences]

plt.xticks(rotation=85, fontsize = 15)

plt.yticks(fontsize = 15)

plt.xticks(x_axis, x_label)

plt.ylabel("Nb. of occurences", fontsize = 18, labelpad = 10)

ax.bar(x_axis, y_axis, align = 'center', color='g')

#_______________________

plt.title("Keywords popularity",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 25)

plt.show()
def missing_factor(p_df):

    missing_df = p_df.isnull().sum(axis=0).reset_index()

    missing_df.columns = ['column_name', 'missing_count']

    missing_df['filling_factor'] = (p_df.shape[0] 

                                - missing_df['missing_count']) / p_df.shape[0] * 100

    

    return missing_df
meta_missing = missing_factor(md)

meta_missing.sort_values('filling_factor').reset_index(drop = True)
keywords_missing = missing_factor(keywords)

keywords_missing.sort_values('filling_factor').reset_index(drop = True)
credits_missing = missing_factor(credits)

credits_missing.sort_values('filling_factor').reset_index(drop = True)
ratings_missing = missing_factor(ratings)

ratings_missing.sort_values('filling_factor').reset_index(drop = True)
md.replace(r'^\s*$', np.NaN, regex=True)

md = md.dropna(subset=['overview'])

md = md.dropna(subset=['vote_count'])

md = md.dropna(subset=['release_date'])

md = md.dropna(subset=['revenue'])

md = md.dropna(subset=['title'])

md = md.dropna(subset=['vote_average'])
md.shape
movie_id_dict = {}

movies_data_len = md.shape[0]

train_dataset = pd.DataFrame()

avg_popularity = 0

avg_vote_count = 0

avg_vote_average = 0

avg_revenue = 0



totally_filled_data_count = 1



popularity_total = 0

vote_count_total = 0

vote_average_total = 0

revenue_total = 0



flag = False



for it in range(movies_data_len):

    if md.iloc[it]['popularity'] and isinstance(md.iloc[it]['popularity'], float) and np.isnan(md.iloc[it]['popularity'])==False:

        if md.iloc[it]['vote_count'] and isinstance(md.iloc[it]['vote_count'], float) and np.isnan(md.iloc[it]['vote_count'])==False:

            if md.iloc[it]['vote_average']  and isinstance(md.iloc[it]['vote_average'], float) and np.isnan(md.iloc[it]['vote_average'])==False:

                if md.iloc[it]['revenue'] and isinstance(md.iloc[it]['revenue'], float) and np.isnan(md.iloc[it]['revenue'])==False:

                    popularity_total += md.iloc[it]['popularity']

                    vote_count_total += md.iloc[it]['vote_count']

                    vote_average_total += md.iloc[it]['vote_average']

                    revenue_total += md.iloc[it]['revenue']

                    totally_filled_data_count += 1

                    

    cur_genres = eval(md.iloc[it]['genres'])

    concated_genres = "" 

    for git in range(len(cur_genres)):

        item = cur_genres[git]['name']

        concated_genres += clean_sentence(item, "_")+' '

        

    

    movie_id_dict[int(md.iloc[it]['id'])] = {'popularity': md.iloc[it]['popularity'], 

                                        'vote_count': md.iloc[it]['vote_count'], 

                                        'vote_average': md.iloc[it]['vote_average'],

                                        'revenue': md.iloc[it]['revenue'],

                                        'genres': concated_genres,

                                        'overview': clean_sentence(md.iloc[it]['overview']),

                                        'title': clean_sentence(md.iloc[it]['title'])}

    

avg_popularity = popularity_total/totally_filled_data_count

avg_vote_count = vote_count_total/totally_filled_data_count

avg_vote_average = vote_average_total/totally_filled_data_count

avg_revenue = revenue_total/totally_filled_data_count
movie_id_dict[15602]
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')

vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')

C = vote_averages.mean()

C
m = vote_counts.quantile(0.90)

m 
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]

qualified['vote_count'] = qualified['vote_count'].astype('int')

qualified['vote_average'] = qualified['vote_average'].astype('int')

qualified.shape
def weighted_rating(x):

    v = x['vote_count']

    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)
qualified.head(10)
s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'genre'

gen_md = md.drop('genres', axis=1).join(s)
def build_chart(genre, percentile=0.85):

    df = gen_md[gen_md['genre'] == genre]

    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')

    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')

    C = vote_averages.mean()

    m = vote_counts.quantile(percentile)

    

    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]

    qualified['vote_count'] = qualified['vote_count'].astype('int')

    qualified['vote_average'] = qualified['vote_average'].astype('int')

    

    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)

    qualified = qualified.sort_values('wr', ascending=False).head(250)

    

    return qualified
build_chart('Romance').head(15)
links_small = pd.read_csv('../input/the-movies-dataset/links_small.csv')

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

# md.shape
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]

smd.shape
#Check EDA Notebook for how and why I got these indices.

smd['id'] = smd['id'].astype('int')
credits_id_dict = {}



for cit in range(len(credits)):

    cast_item = eval(credits.iloc[cit]['cast'])

    crew_item = eval(credits.iloc[cit]['crew'])

    concat_cast_crew = ""

    

    for cast_it in range(len(cast_item)):

        concat_cast_crew += clean_sentence(cast_item[cast_it]['name'], "_")+" "

        

    for crew_it in range(len(crew_item)):

        concat_cast_crew += clean_sentence(crew_item[crew_it]['name'], "_")+" "

        

    credits_id_dict[credits.iloc[cit]['id']] = concat_cast_crew
smd['description'] = smd['overview']



for mvit in range(len(smd)):

    mid = smd.iloc[mvit]['id']

    smd.iloc[mvit]['description'] = ""

    

    if movie_id_dict.get(mid):

        smd.iloc[mvit]['description'] += movie_id_dict[mid]['genres'] + " " + movie_id_dict[mid]['overview']

        

    if keywords_id_dict.get(mid):

        smd.iloc[mvit]['description'] += keywords_id_dict[mid] + " "

        

    if credits_id_dict.get(mid):

        smd.iloc[mvit]['description'] += credits_id_dict[mid] + " "
smd.shape
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

tfidf_matrix = tf.fit_transform(smd['description'])
# tfidf_matrix.shape

tfidf_matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim.shape
smd = smd.reset_index()

titles = smd['title']

indices = pd.Series(smd.index, index=smd['title'])
def get_recommendations(title):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    movie_indices = [i[0] for i in sim_scores]

    return titles.iloc[movie_indices]
get_recommendations('The Godfather').head(20)
get_recommendations('The Dark Knight').head(10)
reader = Reader()
from numpy import nan
rating_data_len = ratings.shape[0]



modified_trainset = pd.DataFrame(index=range(rating_data_len), 

                                columns=['userId', 'movieId','popularity',

                                        'vote_count','vote_average','revenue', 

                                        'rating'])

popularityArr = [0]*rating_data_len

voteCountArr = [0]*rating_data_len

voteAverageArr = [0]*rating_data_len

revenueArr = [0]*rating_data_len





for it in range(rating_data_len):

    movie_id = int(ratings.iloc[it]['movieId'])

    movie_metadata = movie_id_dict.get(movie_id)

    

    if movie_metadata:

        temp = movie_metadata['popularity']

        if isinstance(temp, str):

            temp = float(temp)

            

        if np.isnan(temp):

            popularityArr[it] = avg_popularity

        else:

            popularityArr[it] = temp



            

            

        temp = movie_metadata['vote_count']

        if isinstance(temp, str):

            temp = float(temp)

            

        if np.isnan(temp):

            voteCountArr[it] =  avg_vote_count

        else:

            voteCountArr[it] = temp



            

            

        temp = movie_metadata['vote_average']

        if isinstance(temp, str):

            temp = float(temp)

            

        if np.isnan(temp):

            voteAverageArr[it] = avg_vote_average

        else:

            voteAverageArr[it] = temp



        

        

        temp = movie_metadata['revenue']

        if isinstance(temp, str):

            temp = float(temp)

            

        if np.isnan(temp):

            revenueArr[it] =  avg_revenue

        else:

            revenueArr[it] = temp

    else:

        popularityArr[it] = avg_popularity

        voteCountArr[it] =  avg_vote_count

        voteAverageArr[it] = avg_vote_average

        revenueArr[it] =  avg_revenue
modified_trainset['userId'] = ratings['userId']*100

modified_trainset['movieId'] = ratings['movieId']

modified_trainset['popularity'] = popularityArr

modified_trainset['vote_count'] = voteCountArr

modified_trainset['vote_average'] = voteAverageArr

modified_trainset['revenue'] = revenueArr

modified_trainset['rating'] = ratings['rating']
modified_trainset.head(12)
X,y = _make_in_format(modified_trainset)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

svr_system = svr_rbf.fit(X_train, y_train)
y_pred_svr = svr_system.predict(X_test)

accuracy_score(y_test, y_pred_svr)
def make_prediction_svr(userId, movieId):

    x_test = modified_trainset.loc[modified_trainset['movieId'] == movieId].head(1)

    if len(x_test)==0:

        print('No movie found')

    else:

        y_prediction = svr_system.predict([[userId, movieId, x_test['popularity'], x_test['vote_count'], x_test['vote_average'], x_test['revenue']]])

        print(y_prediction)
make_prediction_svr(100, 1029)