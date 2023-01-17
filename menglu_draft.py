!pip install textstat

!pip install chart_studio
import random

import math

import time

import string

import nltk

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import WordNetLemmatizer,PorterStemmer

from nltk.tokenize import word_tokenize

from sklearn.cluster import DBSCAN

from nltk.corpus import stopwords

from collections import  Counter

import matplotlib.pyplot as plt

import tensorflow_hub as hub

import tensorflow as tf

import pyLDAvis.gensim

from tqdm import tqdm

import seaborn as sns

import pandas as pd

import numpy as np

import pyLDAvis

import gensim

import spacy

import json

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from pprint import pprint

from copy import deepcopy

from tqdm.notebook import tqdm

plt.style.use('seaborn')

%matplotlib inline 

from collections import defaultdict

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette() 

remove_words = set(stopwords.words('english')) 

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import pyLDAvis.sklearn

from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig

import textstat

import matplotlib.colors as mcolors

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from statistics import *

import concurrent.futures

import geopandas as gpd

pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



punctuations = string.punctuation

stopwords = list(STOP_WORDS)



parser = English()



def plot_readability(a,title,bins=0.1,colors=['#3A4750']):

    trace1 = ff.create_distplot([a], [" Abstract "], bin_size=bins, colors=colors, show_rug=False)

    trace1['layout'].update(title=title)

    iplot(trace1, filename='Distplot')

    table_data= [["Statistical Measures","Abstract"],

                ["Mean",mean(a)],

                ["Standard Deviation",pstdev(a)],

                ["Variance",pvariance(a)],

                ["Median",median(a)],

                ["Maximum value",max(a)],

                ["Minimum value",min(a)]]

    trace2 = ff.create_table(table_data)

    iplot(trace2, filename='Table')

    

punctuations = string.punctuation

stopwords = list(STOP_WORDS)



parser = English()

def spacy_tokenizer(sentence):

    #reference : https://www.kaggle.com/thebrownviking20/analyzing-quora-for-the-insinceres

    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]

    mytokens = " ".join([i for i in mytokens])

    return mytokens

import warnings

warnings.filterwarnings('ignore')
### functions for clean data

def format_name(author):

    middle_name = " ".join(author['middle'])

    

    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])





def format_affiliation(affiliation):

    text = []

    location = affiliation.get('location')

    if location:

        text.extend(list(affiliation['location'].values()))

    

    institution = affiliation.get('institution')

    if institution:

        text = [institution] + text

    return ", ".join(text)



def format_authors(authors, with_affiliation=False):

    name_ls = []

    

    for author in authors:

        name = format_name(author)

        if with_affiliation:

            affiliation = format_affiliation(author['affiliation'])

            if affiliation:

                name_ls.append(f"{name} ({affiliation})")

            else:

                name_ls.append(name)

        else:

            name_ls.append(name)

    

    return ", ".join(name_ls)



def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    

    return body



def format_bib(bibs):

    if type(bibs) == dict:

        bibs = list(bibs.values())

    bibs = deepcopy(bibs)

    formatted = []

    

    for bib in bibs:

        bib['authors'] = format_authors(

            bib['authors'], 

            with_affiliation=False

        )

        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]

        formatted.append(", ".join(formatted_ls))



    return "; ".join(formatted)

def load_files(dirname):

    filenames = os.listdir(dirname)

    raw_files = []



    for filename in tqdm(filenames):

        filename = dirname + filename

        file = json.load(open(filename, 'rb'))

        raw_files.append(file)

    

    return raw_files



def generate_clean_df(all_files):

    cleaned_files = []

    

    for file in tqdm(all_files):

        features = [

            file['paper_id'],

            file['metadata']['title'],

            format_authors(file['metadata']['authors']),

            format_authors(file['metadata']['authors'], 

                           with_affiliation=True),

            format_body(file['abstract']),

            format_body(file['body_text']),

            format_bib(file['bib_entries']),

            file['metadata']['authors'],

            file['bib_entries']

        ]



        cleaned_files.append(features)



    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()

    

    return clean_df



##################################### clean all the dataset #########################

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'

filenames = os.listdir(biorxiv_dir)

all_files = []

for filename in filenames:

    filename = biorxiv_dir + filename

    file = json.load(open(filename, 'rb'))

    all_files.append(file)

file = all_files[0]

texts = [(di['section'], di['text']) for di in file['body_text']]

texts_di = {di['section']: "" for di in file['body_text']}



body = ""

for section, text in texts_di.items():

    body += section

    body += "\n\n"

    body += text

    body += "\n\n"

authors = all_files[0]['metadata']['authors']



bibs = list(file['bib_entries'].values())

format_authors(bibs[1]['authors'], with_affiliation=False)

bib_formatted = format_bib(bibs[:5])

cleaned_files = []



for file in tqdm(all_files):

    features = [

        file['paper_id'],

        file['metadata']['title'],

        format_authors(file['metadata']['authors']),

        format_authors(file['metadata']['authors'], 

                       with_affiliation=True),

        format_body(file['abstract']),

        format_body(file['body_text']),

        format_bib(file['bib_entries']),

        file['metadata']['authors'],

        file['bib_entries']

    ]

    

    cleaned_files.append(features)

    

    col_names = [

    'paper_id', 

    'title', 

    'authors',

    'affiliations', 

    'abstract', 

    'text', 

    'bibliography',

    'raw_authors',

    'raw_bibliography'

]



clean_df = pd.DataFrame(cleaned_files, columns=col_names)

clean_df.to_csv('biorxiv_clean.csv', index=False)



comm_dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/'

comm_files = load_files(comm_dir)

comm_df = generate_clean_df(comm_files)

comm_df.to_csv('clean_comm_use.csv', index=False)

noncomm_dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/'

noncomm_files = load_files(noncomm_dir)

noncomm_df = generate_clean_df(noncomm_files)

noncomm_df.to_csv('clean_noncomm_use.csv', index=False)

from wordcloud import WordCloud, STOPWORDS



def plot_wordcloud(text, mask=None, max_words=200, max_font_size=50, figure_size=(15.0,15.0), 

                   title = None, title_size=20, image_color=False,color = color):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color=color,

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    
plot_wordcloud(clean_df['abstract'].values, title="Word Cloud of Authors in biorxiv medrxiv Data",color = 'white')
plot_wordcloud(comm_df['abstract'].values, title="Word Cloud of Authors in comm use subset Data",color = 'white')
plot_wordcloud(noncomm_df['abstract'].values, title="Word Cloud of Authors in non comm use subset Data",color = 'white')
df1 = clean_df['abstract'].dropna()

df3 = comm_df["abstract"].dropna()

df2 = noncomm_df["abstract"].dropna()



## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    #Reference and credits: https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



## Get the bar chart from sincere questions ##

freq_dict = defaultdict(int)

for sent in df1:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in df2:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'orange')



freq_dict = defaultdict(int)

for sent in df3:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'black')





# Creating two subplots

fig = tools.make_subplots(rows=2, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words in biorxiv_data", 

                                          "Frequent words in comm_data",

                                          "Frequent words in noncomm_data"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)









fig['layout'].update(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots of Abstracts")

iplot(fig, filename='word-plots')
%%time

text_1 = clean_df["abstract"].dropna().apply(spacy_tokenizer)

text_2 = comm_df["abstract"].dropna().apply(spacy_tokenizer)

text_3 = noncomm_df['abstract'].dropna().apply(spacy_tokenizer)

#count vectorization

vectorizer_1= CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

vectorizer_2= CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

vectorizer_3 =  CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')



text1_vectorized = vectorizer_1.fit_transform(text_1)

text2_vectorized = vectorizer_2.fit_transform(text_2)

text3_vectorized = vectorizer_3.fit_transform(text_3)

%%time

lda1 = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online',verbose=True)

lda2= LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online',verbose=True)

lda3 = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online',verbose=True)



lda_1 = lda1.fit_transform(text1_vectorized)

lda_2 = lda2.fit_transform(text2_vectorized)

lda_3 = lda3.fit_transform(text3_vectorized)

def selected_topics(model, vectorizer, top_n=10):

    for idx, topic in enumerate(model.components_):

        print("Topic %d:" % (idx))

        print([(vectorizer.get_feature_names()[i], topic[i])

                        for i in topic.argsort()[:-top_n - 1:-1]]) 
print("LDA Model of Bioarvix data Abstracts:")

selected_topics(lda1, vectorizer_1)
print("LDA Model of clean_comm data Abstracts:")

selected_topics(lda2, vectorizer_2)
print("LDA Model of clean_noncomm data Abstracts:")

selected_topics(lda3, vectorizer_3)
import pandas as pd

COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
import pandas as pd

Case = pd.read_csv("../input/coronavirusdataset/Case.csv")

PatientInfo = pd.read_csv("../input/coronavirusdataset/PatientInfo.csv")

PatientRoute = pd.read_csv("../input/coronavirusdataset/PatientRoute.csv")

Region = pd.read_csv("../input/coronavirusdataset/Region.csv")

SearchTrend = pd.read_csv("../input/coronavirusdataset/SearchTrend.csv")

Time = pd.read_csv("../input/coronavirusdataset/Time.csv")

TimeAge = pd.read_csv("../input/coronavirusdataset/TimeAge.csv")

TimeGender = pd.read_csv("../input/coronavirusdataset/TimeGender.csv")

TimeProvince = pd.read_csv("../input/coronavirusdataset/TimeProvince.csv")

Weather = pd.read_csv("../input/coronavirusdataset/Weather.csv")
##### take a look at data 

print(covid_19_data.head())



covid_19_data.describe()

covid_19_data.info()

covid_19_data.isnull().sum()

covid_19_data.head()



print(time_series_covid_19_confirmed.head())

# Grouping confirmed, recovered and death cases per country

grouped_country = covid_19_data.groupby(["Country/Region"],as_index=False)["Confirmed","Recovered","Deaths"].last().sort_values(by="Confirmed",ascending=False)



# Using just first 10 countries with most cases

most_common_countries = grouped_country.head(10)
# FUNCTION TO SHOW ACTUAL VALUES ON BARPLOT



def show_valushowes_on_bars(axs):

    def _show_on_single_plot(ax):

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.2f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center")



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)

# Function returns interactive lineplot of confirmed cases on specific country





def getGrowthPerCountryInteractive(countryName):

    country = covid_19_data[covid_19_data["Country/Region"] == countryName]



    fig_1 = px.line(country, x="observation_date", y="Confirmed", title=(countryName + " confirmed cases."))

    fig_2 = px.line(country, x="observation_date", y="Deaths", title=(countryName +" death cases"))

    fig_3 = px.line(country, x="observation_date", y="Recovered", title=(countryName + " recovered cases"))



    fig_1.show()

    fig_2.show()

    fig_3.show()
china = covid_19_data[covid_19_data["Country/Region"] == "Mainland China"]

fig_1 = px.bar(china,x="Province/State",y="Confirmed",color="Recovered",text="Deaths")

fig_1.show()

china_states = china[china["Province/State"] != "Hubei"]

fig_2 = px.bar(china_states,x="Province/State",y="Confirmed", title="Confirmed cases in other states of China.")

fig_2.show()

fig_3 = px.bar(china_states, x="Province/State",y="Recovered",color="Deaths",title="Recovered vs Deaths in other states of China.")

fig_3.show()
cols = time_series_covid_19_confirmed.keys()
confirmed = time_series_covid_19_confirmed.loc[:, cols[4]:cols[-1]]

deaths = time_series_covid_19_deaths.loc[:, cols[4]:cols[-1]]

recoveries = time_series_covid_19_recovered.loc[:, cols[4]:cols[-1]]

dates = confirmed.keys()

world_cases = []

total_deaths = [] 

mortality_rate = []

total_recovered = [] 

for i in dates:

    confirmed_sum = confirmed[i].sum()

    death_sum = deaths[i].sum()

    recovered_sum = recoveries[i].sum()

    world_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    mortality_rate.append(death_sum/confirmed_sum)

    total_recovered.append(recovered_sum)

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)

total_deaths = np.array(total_deaths).reshape(-1, 1)

total_recovered = np.array(total_recovered).reshape(-1, 1)

days_in_future = 10

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-10]

start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False) 
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=4, C=0.1)

svm_confirmed.fit(X_train_confirmed, y_train_confirmed)

svm_pred = svm_confirmed.predict(future_forcast)

# check against testing data

svm_test_pred = svm_confirmed.predict(X_test_confirmed)

print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))

print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))


linear_model = LinearRegression(normalize=True, fit_intercept=False)

linear_model.fit(X_train_confirmed, y_train_confirmed)

test_linear_pred = linear_model.predict(X_test_confirmed)

linear_pred = linear_model.predict(future_forcast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))


tol = [1e-4, 1e-3, 1e-2]

alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]

alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]



bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}



bayesian = BayesianRidge()

bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

bayesian_search.fit(X_train_confirmed, y_train_confirmed)



bayesian_search.best_params_

bayesian_confirmed = bayesian_search.best_estimator_

test_bayesian_pred = bayesian_confirmed.predict(X_test_confirmed)

bayesian_pred = bayesian_confirmed.predict(future_forcast)

print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))



# Future predictions using SVM 

print('SVM future predictions:')

set(zip(future_forcast_dates[-10:], svm_pred[-10:]))

# Future predictions using Linear Regression 

print('Ridge regression future predictions:')

set(zip(future_forcast_dates[-10:], bayesian_pred[-10:]))

# Future predictions using Linear Regression 

print('Linear regression future predictions:')

print(linear_pred[-10:])



latest_confirmed = time_series_covid_19_confirmed[dates[-1]]

latest_deaths = time_series_covid_19_deaths[dates[-1]]

latest_recoveries = time_series_covid_19_recovered[dates[-1]]

unique_countries =  list(time_series_covid_19_confirmed['Country/Region'].unique())

country_confirmed_cases = []

no_cases = []

for i in unique_countries:

    cases = latest_confirmed[time_series_covid_19_confirmed['Country/Region']==i].sum()

    if cases > 0:

        country_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

        

for i in no_cases:

    unique_countries.remove(i)

    

unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]

for i in range(len(unique_countries)):

    country_confirmed_cases[i] = latest_confirmed[time_series_covid_19_confirmed['Country/Region']==unique_countries[i]].sum()

    

    # number of cases per country/region

print('Confirmed Cases by Countries/Regions:')

for i in range(len(unique_countries)):

    print(f'{unique_countries[i]}: {country_confirmed_cases[i]} cases')

unique_provinces =  list(time_series_covid_19_confirmed['Province/State'].unique())

# those are countries, which are not provinces/states.

outliers = ['United Kingdom', 'Denmark', 'France']

for i in outliers:

    unique_provinces.remove(i)

    province_confirmed_cases = []

no_cases = [] 

for i in unique_provinces:

    cases = latest_confirmed[time_series_covid_19_confirmed['Province/State']==i].sum()

    if cases > 0:

        province_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

 

# remove areas with no confirmed cases

for i in no_cases:

    unique_provinces.remove(i)

    

unique_provinces = [k for k, v in sorted(zip(unique_provinces, province_confirmed_cases), key=operator.itemgetter(1), reverse=True)]

for i in range(len(unique_provinces)):

    province_confirmed_cases[i] = latest_confirmed[time_series_covid_19_confirmed['Province/State']==unique_provinces[i]].sum()

    

    # number of cases per province/state/city

print('Confirmed Cases by Province/States (US, China, Australia, Canada):')

for i in range(len(unique_provinces)):

    print(f'{unique_provinces[i]}: {province_confirmed_cases[i]} cases')

    

    nan_indices = [] 



# handle nan if there is any, it is usually a float: float('nan')



for i in range(len(unique_provinces)):

    if type(unique_provinces[i]) == float:

        nan_indices.append(i)



unique_provinces = list(unique_provinces)

province_confirmed_cases = list(province_confirmed_cases)



for i in nan_indices:

    unique_provinces.pop(i)

    province_confirmed_cases.pop(i)

    

china_confirmed = latest_confirmed[time_series_covid_19_confirmed['Country/Region']=='China'].sum()

outside_mainland_china_confirmed = np.sum(country_confirmed_cases) - china_confirmed




print('Outside Mainland China {} cases:'.format(outside_mainland_china_confirmed))

print('Mainland China: {} cases'.format(china_confirmed))

print('Total: {} cases'.format(china_confirmed+outside_mainland_china_confirmed))

# Only show 10 countries with the most confirmed cases, the rest are grouped into the other category

visual_unique_countries = [] 

visual_confirmed_cases = []

others = np.sum(country_confirmed_cases[10:])

for i in range(len(country_confirmed_cases[:10])):

    visual_unique_countries.append(unique_countries[i])

    visual_confirmed_cases.append(country_confirmed_cases[i])



visual_unique_countries.append('Others')

visual_confirmed_cases.append(others)

# lets look at it in a logarithmic scale 

log_country_confirmed_cases = [math.log10(i) for i in visual_confirmed_cases]



# Only show 10 provinces with the most confirmed cases, the rest are grouped into the other category

visual_unique_provinces = [] 

visual_confirmed_cases2 = []

others = np.sum(province_confirmed_cases[10:])

for i in range(len(province_confirmed_cases[:10])):

    visual_unique_provinces.append(unique_provinces[i])

    visual_confirmed_cases2.append(province_confirmed_cases[i])



visual_unique_provinces.append('Others')

visual_confirmed_cases2.append(others)





plt.figure(figsize=(32, 18))

plt.barh(visual_unique_provinces, visual_confirmed_cases2)

plt.title('# of Coronavirus Confirmed Cases in Provinces/States', size=20)

plt.show()



c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases per Country')

plt.pie(visual_confirmed_cases, colors=c)

plt.legend(visual_unique_countries, loc='best')

plt.show()





c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases in Countries Outside of Mainland China')

plt.pie(visual_confirmed_cases[1:], colors=c)

plt.legend(visual_unique_countries[1:], loc='best')

plt.show()



us_regions = list(time_series_covid_19_confirmed[time_series_covid_19_confirmed['Country/Region']=='US']['Province/State'].unique())

us_confirmed_cases = []

no_cases = [] 

for i in us_regions:

    cases = latest_confirmed[time_series_covid_19_confirmed['Province/State']==i].sum()

    if cases > 0:

        us_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

 

## remove areas with no confirmed cases

for i in no_cases:

    us_regions.remove(i)



c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases in the United States')

plt.pie(us_confirmed_cases, colors=c)

plt.legend(us_regions, loc='best')

plt.show()



china_regions = list(time_series_covid_19_confirmed[time_series_covid_19_confirmed['Country/Region']=='China']['Province/State'].unique())

china_confirmed_cases = []

no_cases = [] 

for i in china_regions:

    cases = latest_confirmed[time_series_covid_19_confirmed['Province/State']==i].sum()

    if cases > 0:

        china_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

 

# remove areas with no confirmed cases

for i in no_cases:

    china_confirmed_cases.remove(i)

c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases in the Mainland China')

plt.pie(china_confirmed_cases, colors=c)

plt.legend(china_regions, loc='best')

plt.show()    
time_series_covid_19_confirmed.info()
sns.jointplot(x="Long", y="Lat", data=time_series_covid_19_confirmed);
import mplleaflet as mpll

# comment out the last line of this cell for rendering Leaflet map.

rids = np.arange(time_series_covid_19_confirmed.shape[0])

np.random.shuffle(rids)

f, ax = plt.subplots(1, figsize=(6, 6))

time_series_covid_19_confirmed.iloc[rids[:100], :].plot(kind='scatter', x='Long', y='Lat', \

                      s=30, linewidth=0, ax=ax);

mpll.display(fig=f,)
from shapely.geometry import Point

xys_wb = gpd.GeoSeries(time_series_covid_19_confirmed[['Long', 'Lat']].apply(Point, axis=1), \

                      crs="+init=epsg:4326")

xys_wb = xys_wb.to_crs(epsg=3857)

x_wb = xys_wb.apply(lambda i: i.x)

y_wb = xys_wb.apply(lambda i: i.y)


from bokeh.plotting import figure, output_notebook, show

from bokeh.tile_providers import STAMEN_TONER

output_notebook()

from bokeh.plotting import figure, output_notebook, show, ColumnDataSource

output_notebook()

minx, miny, maxx, maxy = xys_wb.total_bounds

y_range = miny, maxy

x_range = minx, maxx



def base_plot(tools='pan,wheel_zoom,reset',plot_width=600, plot_height=400, **plot_args):

    p = figure(tools=tools, plot_width=plot_width, plot_height=plot_height,

        x_range=x_range, y_range=y_range, outline_line_color=None,

        min_border=0, min_border_left=0, min_border_right=0,

        min_border_top=0, min_border_bottom=0, **plot_args)



    p.axis.visible = False

    p.xgrid.grid_line_color = None

    p.ygrid.grid_line_color = None

    return p



options = dict(line_color=None, fill_color='#800080', size=4)



p = base_plot()

p.add_tile(STAMEN_TONER)

p.circle(x=x_wb, y=y_wb, **options)

#<div class="bk-banner">

#<a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>

#<span id="efa98bda-2ccf-4dbf-ae97-94033d60c79b">Loading BokehJS ...</span>

#</div>

#<bokeh.models.renderers.GlyphRenderer at 0x1052bb5f8>
import datashader as ds

#from datashader.callbacks import InteractiveImage

from datashader.colors import viridis

from datashader import transfer_functions as tf

from bokeh.tile_providers import STAMEN_TONER



p = base_plot()

p.add_tile(STAMEN_TONER)



pts = pd.DataFrame({'x': x_wb, 'y': y_wb})

pts['count'] = 1

def create_image90(x_range, y_range, w, h):

    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)

    agg = cvs.points(pts, 'x', 'y',  ds.count('count'))

    img = tf.interpolate(agg.where(agg > np.percentile(agg,90)), \

                         cmap=viridis, how='eq_hist')

    return tf.dynspread(img, threshold=0.1, max_px=4)

sns.kdeplot(time_series_covid_19_confirmed['Long'], time_series_covid_19_confirmed['Lat'], shade=True, cmap='viridis');

f, ax = plt.subplots(1, figsize=(9, 9))

sns.kdeplot(time_series_covid_19_confirmed['Long'], time_series_covid_19_confirmed['Lat'], \

            shade=True, cmap='Purples', \

            ax=ax);



ax.set_axis_off()

plt.axis('equal')

plt.show()