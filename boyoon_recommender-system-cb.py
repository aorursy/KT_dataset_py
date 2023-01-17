%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
pd.options.display.max_columns = 100
pd.options.display.max_rows = 300
from ast import literal_eval

credits = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv").drop(index=4553).reset_index(drop=True)
movies = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv", parse_dates=["release_date"]).drop(index=4553).reset_index(drop=True)

# parse dictionaries of list in dataframe
credits["cast"] = credits["cast"].apply(literal_eval)
credits["crew"] = credits["crew"].apply(literal_eval)
movies["genres"] = movies["genres"].apply(literal_eval)
movies["keywords"] = movies["keywords"].apply(literal_eval)
movies["production_companies"] = movies["production_companies"].apply(literal_eval)
movies["production_countries"] = movies["production_countries"].apply(literal_eval)
movies["spoken_languages"] = movies["spoken_languages"].apply(literal_eval)

display(credits.head(), movies.head(3))
def dict_onehot(df, key="name", n=None):
    if n:
        result = df.apply(lambda x: "|".join(map(lambda y: str(y[key]), x[:n]))).str.get_dummies()
    else:
        result = df.apply(lambda x: "|".join(map(lambda y: str(y[key]), x))).str.get_dummies()
    return result
genres = dict_onehot(movies["genres"])
cast = dict_onehot(credits['cast'], "id", 15)
crew_ratio = pd.DataFrame(columns=["music", "action", "style", "art"])
directors = pd.DataFrame(columns=["director"])

for idx in range(len(credits)):
    total = len(credits['crew'][idx])
    if total == 0:
        crew_ratio.loc[idx] = [0, 0, 0, 0]
        directors.loc[idx] = ""
        continue
        
    music, action, style, art, director = 0, 0, 0, 0, ""
    for i, job in enumerate(map(lambda x: x['job'].lower(), credits['crew'][idx])):
        if ("music" in job) or ("orchestra" in job):
            music += 1
            
        elif ("stunt" in job) or ("helicopter" in job) or ("underwater" in job):
            action += 1
            
        elif ("costume" in job)  or ("make" in job) or ("hair" in job):
            style += 1

        elif ("art" in job) or ("visual" in job) or ("cg " in job) or ("designer" in job) or ("animation" in job) or ("effect" in job):
            art += 1
            
        elif job == "director":
            director = credits['crew'][idx][i]["name"]
    
    crew_ratio.loc[idx] = [style/total, art/total, music/total, action/total]
    directors.loc[idx] = director
dirs = directors["director"].str.get_dummies()
movies["year"] = movies["release_date"].dt.year
movies["month"] = movies["release_date"].dt.month

year_cat = movies["year"].apply(lambda x: str(x//10*10) + "'s").str.get_dummies()
season = movies["month"].apply(lambda x: 'Spring' if 3 <= x <= 5 else('Summer' if x <= 8 else('Fall' if x <= 11 else 'winter'))).str.get_dummies()
prod_countries = dict_onehot(movies["production_countries"])
spoken_lang = dict_onehot(movies["spoken_languages"])
original_lang = movies["original_language"].str.get_dummies()
from datetime import datetime as dt

movies["homepage"] = movies["homepage"].apply(lambda x: 0 if pd.isnull(x) else 1)

num1 = movies[["budget", "homepage", "revenue", "runtime"]]
num2 = movies[["popularity", "vote_average", "vote_count", "year"]]
prod_companies = dict_onehot(movies["production_companies"])

# 제작사별 평균 인지도 대비 평점
prod_vm = prod_companies.mul((movies["vote_average"] / movies["popularity"]).values, axis=0).replace(0, np.nan)
prod_vm_mean = prod_companies.mul(prod_vm.mean().fillna(0))

# 제작사별 (영화별)평균 예산
prod_b = prod_companies.mul(movies["budget"].values, axis=0).replace(0, np.nan)
prod_b_mean = prod_companies.mul(prod_b.mean().fillna(0))
# crew_ratio
keywords = movies["keywords"].apply(lambda x: " ".join(map(lambda y: y["name"], x)))
overview = movies["overview"].fillna("")
tagline = movies["tagline"].fillna("")
from sklearn.metrics.pairwise import cosine_similarity

def get_cosim(matrix):
    return pd.DataFrame(cosine_similarity(matrix), columns=movies["title"], index=movies["title"])
from sklearn.decomposition import PCA

def pca(df, ndim):
    pca = PCA(n_components=ndim)
    return pca.fit_transform(df.values)
    
def plot_pca(pca_array, ndim, ylim=[-1.3,1.8]):
    var_exp = sorted([np.var(pca_array[:,i]) for i in range(ndim)], reverse=True)
    cum_var_exp = np.cumsum(var_exp)
    
    # plot
    plt.figure(figsize = (12,5))

    plt.subplot(121)
    plt.bar(range(ndim), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')
    plt.step(range(ndim), cum_var_exp, where='mid',label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')

    plt.subplot(122)
    plt.scatter(pca_array[:,0],pca_array[:,1], c='orange',alpha=0.4)
    plt.ylim(ylim[0], ylim[1])

    plt.show()
cosim1 = get_cosim(genres)
cosim1.head()
cosim2 = get_cosim(year_cat) * 0.8 + get_cosim(season) * 0.2
cosim2.head()
prod_countries_pca = pca(prod_countries, 10)
plot_pca(prod_countries_pca, 10, ylim=[-0.8, 1.2])
spoken_lang_pca = pca(spoken_lang, 17)
plot_pca(spoken_lang_pca, 17, ylim=[-0.5, 1.1])
original_lang_pca = pca(original_lang, 6)
plot_pca(original_lang_pca, 6, ylim=[-0.5, 0.8])
get_cosim(original_lang) * 0.1
get_cosim(original_lang_pca)
cosim3 = get_cosim(prod_countries_pca) * 0.3 + get_cosim(spoken_lang_pca) * 0.3  + get_cosim(original_lang_pca) * 0.4
cosim3.head()
cosim8 = get_cosim(cast)
cosim8.head()
dirs_pca = pca(dirs, 3)
plot_pca(dirs_pca, 3, ylim=[-0.1, 1.1])
from sklearn.preprocessing import StandardScaler

def scaler(df):
    std = StandardScaler()
    df = df - df.mean()
    return std.fit_transform(df.values)
(num1 - num1.mean()).loc[4801][0]
pd.DataFrame([[1,2,3],[4,5,6]]).mean()
pd.DataFrame([[1,2,3],[6,1,3],[2,2,6]])
temp = pd.DataFrame([[1,2,3],[6,1,3],[2,2,6]])
temp = temp - temp.mean()
std = StandardScaler()
std.fit_transform(temp.values).std(0)
(num1 - num1.mean())
cosim4 = cosim3
# cosim4.head()
cosim5 = get_cosim(scaler(num2))
# cosim5.head()
prod_vm_pca = pca(np.sqrt(prod_vm_mean), 20)
plot_pca(prod_vm_pca, 20, ylim=[-0.025, 0.075])
prod_vm_scaled = scaler(pd.DataFrame(prod_vm_pca))
cosim6 = get_cosim(prod_vm_scaled)
cosim6
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

def lemmatier(sentense):
    sentense = re.sub(r'[^0-9a-zA-Z ]', '', sentense)
    wnl = WordNetLemmatizer()
    
    result = ""
    for word, tag in pos_tag(word_tokenize(sentense)):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        lemma = wnl.lemmatize(word, wntag) if wntag else word
        result += lemma + " "
        
    return result.lower()
keywords = keywords.apply(lemmatier)
overview = overview.apply(lemmatier)
tagline = tagline.apply(lemmatier)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=3, stop_words="english")
keywords_tfidf = tfidf.fit_transform(keywords).toarray()
keywords_tfidf.shape
tfidf = TfidfVectorizer(min_df=5, stop_words="english")
overview_tfidf = tfidf.fit_transform(overview).toarray()
overview_tfidf.shape
tfidf = TfidfVectorizer(min_df=4, stop_words="english")
tagline_tfidf = tfidf.fit_transform(tagline).toarray()
tagline_tfidf.shape
keywords_pca = pca(pd.DataFrame(np.sqrt(keywords_tfidf)), 1300)
plot_pca(keywords_pca, 1300, ylim=[-0.8, 1])
overview_pca = pca(pd.DataFrame(np.sqrt(overview_tfidf)), 2000)
plot_pca(overview_pca, 2000, ylim=[-0.6, 0.7])
tagline_pca = pca(pd.DataFrame(np.log1p(tagline_tfidf)), 450)
plot_pca(tagline_pca, 450, ylim=[-0.2, 0.9])
cosim7 = (get_cosim(scaler(pd.DataFrame(keywords))) * 0.3) + (get_cosim(scaler(pd.DataFrame(overview_pca))) * 0.6) + (get_cosim(scaler(pd.DataFrame(tagline_pca))) * 0.1)
cosim7
def get_recommendation(title, cosims:list, weights:list):
    indices = pd.Series(movies.index, index=movies["title"])
    idx = indices[title]
    avg = np.sum([np.array(c)*w for c, w in zip(cosims, weights)], axis=0) / len(cosims)
    score = list(enumerate(avg[idx]))
    score = sorted(score, key=lambda x: x[1], reverse=True)
    score = score[1:21] # top 20
    movie_index = [i[0] for i in score]
    
    return movies["title"].iloc[movie_index]
import requests
from urllib.request import urlopen
from PIL import Image

def movie_poster(titles):
    data_URL = 'http://www.omdbapi.com/?i=tt3896198&apikey=f9cdaffd'
    
    fig, axes = plt.subplots(2, 10, figsize=(30,9))
    
    for i, ax in enumerate(axes.flatten()):
        w_title = titles[i].strip().split()
        params = {
            's':titles[i],
            'type':'movie',
            'y':''    
        }
        response = requests.get(data_URL,params=params).json()
        
        if response["Response"] == 'True':
            poster_URL = response["Search"][0]["Poster"]
            img = Image.open(urlopen(poster_URL))
            ax.imshow(img)
            
        ax.axis("off")
        if len(w_title) >= 10:
            ax.set_title(f"{i+1}. {' '.join(w_title[:5])}\n{' '.join(w_title[5:10])}\n{' '.join(w_title[10:])}", fontsize=10)
        elif len(w_title) >= 5:
            ax.set_title(f"{i+1}. {' '.join(w_title[:5])}\n{' '.join(w_title[5:])}", fontsize=10)
        else:
            ax.set_title(f"{i+1}. {titles[i]}", fontsize=10)
        
    plt.show()
# rec1 = get_recommendation("Begin Again", cosims=[cosim1, cosim7], weights=[0.8, 0.2])
# movie_poster(list(rec1))
rec2 = get_recommendation("Batman Begins", cosims=[cosim1, cosim2, cosim3, cosim4, cosim5, cosim6], weights=[0.16, 0.01, 0.01, 0.13, 0.23, 0.23])
movie_poster(list(rec2))







