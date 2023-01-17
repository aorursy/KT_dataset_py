!pip install -q pycountry
import os
import gc
import re
import folium
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import scipy as sp
import pandas as pd

import pycountry
from sklearn import metrics
from sklearn.utils import shuffle
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import nltk
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import random
import networkx as nx
from pandas import Timestamp

import requests
from IPython.display import HTML
import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

tqdm.pandas()
np.random.seed(0)
%env PYTHONHASHSEED=0

import warnings
warnings.filterwarnings("ignore")
DATA_PATH = "../input/CORD-19-research-challenge/"
CLEAN_DATA_PATH = "../input/cord-19-eda-parse-json-and-generate-clean-csv/"

pmc_df = pd.read_csv(CLEAN_DATA_PATH + "clean_pmc.csv")
biorxiv_df = pd.read_csv(CLEAN_DATA_PATH + "biorxiv_clean.csv")
comm_use_df = pd.read_csv(CLEAN_DATA_PATH + "clean_comm_use.csv")
noncomm_use_df = pd.read_csv(CLEAN_DATA_PATH + "clean_noncomm_use.csv")

papers_df = pd.concat([pmc_df,
                       biorxiv_df,
                       comm_use_df,
                       noncomm_use_df], axis=0).reset_index(drop=True)
CORONA_FILE = "../input/corona-virus-report/covid_19_clean_complete.csv"

full_table = pd.read_csv(CORONA_FILE, parse_dates=['Date'])

full_table[['Province/State']] = full_table[['Province/State']].fillna('')
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')
full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']
full_table[cases] = full_table[cases].fillna(0)
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']
full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

# replacing Mainland china with just China
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

# filling missing values 
full_table[['Province/State']] = full_table[['Province/State']].fillna('')
full_table[cases] = full_table[cases].fillna(0)

# cases in the ships
ship = full_table[full_table['Province/State'].str.contains('Grand Princess')|full_table['Country/Region'].str.contains('Cruise Ship')]

# china and the row
china = full_table[full_table['Country/Region']=='China']
row = full_table[full_table['Country/Region']!='China']

# latest
full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()
china_latest = full_latest[full_latest['Country/Region']=='China']
row_latest = full_latest[full_latest['Country/Region']!='China']

# latest condensed
full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
# temp.style.background_gradient(cmap='Reds')

temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
papers_df.head(5)
authors1 = papers_df['authors']
#authors1.dropna()
authors2 = str(authors1.values.tolist())
authors3 = authors2.replace('\'', '').replace(';', '').replace('•', '').replace('  ', '').replace(')','').replace('.','').replace(' · ','').replace(' nan,','').replace(' † ','')
authors4 = authors3.replace('-', ' ') 
#print(authors4)
authors5 = authors4.split(",")
#print(authors5)
      

# List of month names separated with a space
#seperator = ','
#print("Scenario#1: ", converttostr(authors2, seperator))

#for auth_list in authors1:
#list1 = auth_list.str.split (",")

!pip install git+https://github.com/namsor/namsor-python-sdk2.git
from __future__ import print_function
import time
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

from __future__ import print_function
import time
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: api_key
configuration = openapi_client.Configuration()
configuration.api_key['X-API-KEY'] = 'e62b88e51641b73e20e52a24d85dd9d8'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['X-API-KEY'] = 'Bearer'

# create an instance of the API class
api_instance = openapi_client.PersonalApi(openapi_client.ApiClient(configuration))
personal_name_full = 'Abhijeeth Ray' # str | 

try:
    # [USES 10 UNITS PER NAME] Infer the likely country of residence of a personal full name, or one surname. Assumes names as they are in the country of residence OR the country of origin.
    api_response = api_instance.country(personal_name_full)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling PersonalApi->country: %s\n" % e)
#for ii in range(len(authors5)):
df = pd.DataFrame(columns = ['Name','Country'])
                                 
for ii in range(200):
    personal_name_full = authors5[ii]
    api_response = api_instance.country(personal_name_full)
    country1 = api_response.country
    df.loc[ii, ['Name']] = personal_name_full
    df.loc[ii, ['Country']] = country1
    
df.head(5)
df.tail(5)
    #pprint(api_response.get("country"))
    #pprint(api_response)
Country_list = df.groupby('Country').count()
print(Country_list)
Country_list.plot(kind='bar')


#pprint(api_response)
type(api_response)
print(len(authors5))

import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: api_key
configuration = openapi_client.Configuration()
configuration.api_key['X-API-KEY'] = ''
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['X-API-KEY'] = 'Bearer'

# create an instance of the API class
api_instance = openapi_client.PersonalApi(openapi_client.ApiClient(configuration))
batch_personal_name_in = openapi_client.BatchPersonalNameIn() # BatchPersonalNameIn | A list of personal names (optional)

#try:
    # [USES 10 UNITS PER NAME] Infer the likely country of residence of up to 100 personal full names, or surnames. Assumes names as they are in the country of residence OR the country of origin.
api_response = api_instance.country_batch(batch_personal_name_in=batch_personal_name_in)
pprint(api_response)
#except ApiException as e:
#    print("Exception when calling PersonalApi->country_batch: %s\n" % e)
    
def multiple_replace(dict, text):
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

def get_countries(names):
    alphabet = ["A", "B", "C", "D", "E", "F",\
                "G", "H", "I", "J", "K", "L",\
                "M", "N", "O", "P", "Q", "R",\
                "S", "T", "U", "V", "W", "X",\
                "Y", "Z"]

    repl_dict = dict(zip([a+" " for a in alphabet], [""]*26))
    repls = []
    for name in names.split(", "):
        repl = multiple_replace(repl_dict, name.strip().replace(") ", "").replace("( ", ""))
        if len(repl.split()) == 1:
            repl = name[0] + " " + repl
        repl = repl.replace(";", "").replace(":", "").replace(".", "").replace(",", "")
        repl = repl.split(" ")

        for idx in range(len(repl)):
            if len(repl[idx]) <= 1 and repl[idx] not in alphabet:
                repl[idx] = "A"

        response = client.origin(repl[0], repl[1])
        repls.append(response.country_origin)

    return repls

countries = pd.read_csv("../input/researcher-countries/countries.csv").values[:, 0].tolist()
cont_list = sorted(list(set(countries)))
counts = [countries.count(cont) for cont in cont_list]
df = pd.DataFrame(np.transpose([cont_list, counts]))
df.columns = ["Country of origin", "Count"]
fig = px.bar(df, x="Country of origin", y="Count", title="Country of origin of researchers", template="simple_white")
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.data[0].marker.line.width = 0.5
fig
codes = [pycountry.countries.get(alpha_2=con).name for con in cont_list]
df["Codes"] = codes
df["Count"] = df["Count"].apply(int)
fig = px.scatter_geo(df, locations="Codes", size='Count', hover_name="Country of origin",
                     projection="natural earth", locationmode="country names", title="Country of origin of researchers", color="Count",
                     template="plotly")
# fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
# fig.data[0].marker.line.width = 0.2
fig.show()
fig = px.choropleth(df, locations="Codes", hover_name="Country of origin",
                    projection="natural earth", locationmode="country names", title="Country of origin of European researchers", color="Count",
                    template="plotly", scope="europe", color_continuous_scale="aggrnyl")
fig
fig = px.choropleth(df, locations="Codes", hover_name="Country of origin",
                    projection="natural earth", locationmode="country names", title="Country of origin of Asian researchers", color="Count",
                    template="plotly", scope="asia", color_continuous_scale="agsunset")
fig
fig = px.choropleth(df, locations="Codes", hover_name="Country of origin",
                    projection="natural earth", locationmode="country names", title="Country of origin of African researchers", color="Count",
                    template="plotly", scope="africa", color_continuous_scale="spectral")
fig
def new_len(x):
    if type(x) is str:
        return len(x.split())
    else:
        return 0

papers_df["abstract_words"] = papers_df["abstract"].apply(new_len)
nums = papers_df.query("abstract_words != 0 and abstract_words < 500")["abstract_words"]
fig = ff.create_distplot(hist_data=[nums],
                         group_labels=["All abstracts"],
                         colors=["coral"])

fig.update_layout(title_text="Abstract words", xaxis_title="Abstract words", template="simple_white", showlegend=False)
fig.show()
biorxiv_df["abstract_words"] = biorxiv_df["abstract"].apply(new_len)
nums_1 = biorxiv_df.query("abstract_words != 0 and abstract_words < 500")["abstract_words"]
pmc_df["abstract_words"] = pmc_df["abstract"].apply(new_len)
nums_2 = pmc_df.query("abstract_words != 0 and abstract_words < 500")["abstract_words"]
comm_use_df["abstract_words"] = comm_use_df["abstract"].apply(new_len)
nums_3 = comm_use_df.query("abstract_words != 0 and abstract_words < 500")["abstract_words"]
noncomm_use_df["abstract_words"] = noncomm_use_df["abstract"].apply(new_len)
nums_4 = noncomm_use_df.query("abstract_words != 0 and abstract_words < 500")["abstract_words"]
fig = ff.create_distplot(hist_data=[nums_1, nums_2, nums_3, nums_4],
                         group_labels=["Biorxiv", "PMC", "Commerical", "Non-commercial"],
                         colors=px.colors.qualitative.Plotly[4:], show_hist=False)

fig.update_layout(title_text="Abstract words vs. Paper type", xaxis_title="Abstract words", template="plotly_white")
fig.show()
def polarity(x):
    if type(x) == str:
        return SIA.polarity_scores(x)
    else:
        return 1000
    
SIA = SentimentIntensityAnalyzer()
polarity_0 = [pol for pol in papers_df["abstract"].apply(lambda x: polarity(x)) if pol != 1000]
polarity_1 = [pol for pol in biorxiv_df["abstract"].apply(lambda x: polarity(x)) if pol != 1000]
polarity_2 = [pol for pol in pmc_df["abstract"].apply(lambda x: polarity(x)) if pol != 1000]
polarity_3 = [pol for pol in comm_use_df["abstract"].apply(lambda x: polarity(x)) if pol != 1000]
polarity_4 = [pol for pol in noncomm_use_df["abstract"].apply(lambda x: polarity(x)) if pol != 1000]
fig = go.Figure(go.Histogram(x=[pols["neg"] for pols in polarity_0 if pols["neg"] < 0.15], marker=dict(
            color='seagreen', line=dict(color="rgb(0, 0, 0)",
                      width=0.5))
    ))

fig.update_layout(xaxis_title="Negativity sentiment", title_text="Negativity sentiment", template="simple_white")
fig.show()
fig = ff.create_distplot(hist_data=[[pol["neg"] for pol in pols if pol["neg"] < 0.15] for pols in [polarity_1, polarity_2, polarity_3, polarity_4]],
                         group_labels=["Biorxiv", "PMC", "Commerical", "Non-commercial"],
                         colors=px.colors.qualitative.Plotly[4:], show_hist=False)

fig.update_layout(title_text="Negativity sentiment vs. Paper type", xaxis_title="Negativity sentiment", template="plotly_white")
fig.show()
fig = go.Figure(go.Bar(y=["Biorxiv", "PMC", "Commercial", "Non-commercial"], x=[np.mean(x) - 0.042 for x in [[pol["neg"] for pol in pols] for pols in [polarity_1, polarity_2, polarity_3, polarity_4]]], orientation="h", marker=dict(color=px.colors.qualitative.Plotly[4:], line=dict(color="rgb(0, 0, 0)",
                      width=1))))
fig.update_layout(xaxis_title="Paper type", yaxis_title="Average negativity", title_text="Average negativity vs. Paper type", template="plotly_white")
fig.show()
fig = go.Figure(go.Histogram(x=[pols["pos"] for pols in polarity_0 if pols["pos"] < 0.15], marker=dict(
        color='indianred', line=dict(color="rgb(0, 0, 0)",
                      width=0.5)
    )))
fig.update_layout(xaxis_title="Positivity sentiment", title_text="Positivity sentiment", template="simple_white")
fig.show()
fig = ff.create_distplot(hist_data=[[pol["pos"] for pol in pols if pol["pos"] < 0.15] for pols in [polarity_1, polarity_2, polarity_3, polarity_4]],
                         group_labels=["Biorxiv", "PMC", "Commerical", "Non-commercial"],
                         colors=px.colors.qualitative.Plotly[4:], show_hist=False)
fig.update_layout(title_text="Positivity sentiment vs. Paper type", xaxis_title="Positivity sentiment", template="plotly_white")
fig.show()
fig = go.Figure(go.Bar(y=["Biorxiv", "PMC", "Commercial", "Non-commercial"], x=[np.mean(x) - 0.055 for x in [[pol["pos"] for pol in pols] for pols in [polarity_1, polarity_2, polarity_3, polarity_4]]], orientation="h", marker=dict(color=px.colors.qualitative.Plotly[4:], line=dict(color="rgb(0, 0, 0)",
                      width=1))))
fig.update_layout(xaxis_title="Paper type", yaxis_title="Average positivity", title_text="Average positivity vs. Paper type", template="plotly_white")
fig.show()
fig = go.Figure(go.Histogram(x=[pols["neu"] for pols in polarity_0], marker=dict(
        color='dodgerblue', line=dict(color="rgb(0, 0, 0)",
                      width=0.15)
    )))
fig.update_layout(xaxis_title="Neutrality sentiment", title_text="Neutrality sentiment", template="simple_white")
fig.show()
fig = ff.create_distplot(hist_data=[[pol["neu"] for pol in pols if pol["neu"]] for pols in [polarity_1, polarity_2, polarity_3, polarity_4]],
                         group_labels=["Biorxiv", "PMC", "Commerical", "Non-commercial"],
                         colors=px.colors.qualitative.Plotly[4:], show_hist=False)
fig.update_layout(title_text="Neutrality sentiment vs. Paper type", xaxis_title="Neutrality sentiment", template="plotly_white")
fig.show()
fig = go.Figure(go.Bar(y=["Biorxiv", "PMC", "Commercial", "Non-commercial"], x=[np.mean(x) - 0.88 for x in [[pol["neu"] for pol in pols] for pols in [polarity_1, polarity_2, polarity_3, polarity_4]]], orientation="h", marker=dict(color=px.colors.qualitative.Plotly[4:], line=dict(color="rgb(0, 0, 0)",
                      width=1))))
fig.update_layout(xaxis_title="Paper type", yaxis_title="Average neutrality", title_text="Average neutrality vs. Paper type", template="plotly_white")
fig.show()
fig = go.Figure(go.Histogram(x=[pols["compound"] for pols in polarity_0], marker=dict(
        color='orchid', line=dict(color="rgb(0, 0, 0)",
                      width=0.5)
    )))
fig.update_layout(xaxis_title="Compoundness sentiment", title_text="Compoundness sentiment", template="simple_white")
fig.show()
fig = ff.create_distplot(hist_data=[[pol["compound"] for pol in pols] for pols in [polarity_1, polarity_2, polarity_3, polarity_4]],
                         group_labels=["Biorxiv", "PMC", "Commerical", "Non-commercial"],
                         colors=px.colors.qualitative.Plotly[4:], show_hist=False)
fig.update_layout(title_text="Compoundness sentiment vs. Paper type", xaxis_title="Compoundness sentiment", template="plotly_white")
fig.show()
fig = go.Figure(go.Bar(y=["Biorxiv", "PMC", "Commercial", "Non-commercial"], x=[np.mean(x) for x in [[pol["compound"] for pol in pols] for pols in [polarity_1, polarity_2, polarity_3, polarity_4]]], orientation="h", marker=dict(color=px.colors.qualitative.Plotly[4:], line=dict(color="rgb(0, 0, 0)",
                      width=1))))
fig.update_layout(xaxis_title="Paper type", yaxis_title="Average compoundness", title_text="Average compoundness vs. Paper type", template="plotly_white")
fig.show()
def nonan(x):
    if type(x) == str:
        return x.replace("\n", "")
    else:
        return ""

text = ' '.join([nonan(abstract) for abstract in papers_df["abstract"]])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
fig = px.imshow(wordcloud)
fig.update_layout(title_text='Common words in abstracts')
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ

    elif nltk_tag.startswith('V'):
        return wordnet.VERB

    elif nltk_tag.startswith('N'):
        return wordnet.NOUN

    elif nltk_tag.startswith('R'):
        return wordnet.ADV

    else:          
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []

    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    return " ".join(lemmatized_sentence)

def clean_text(abstract):
    abstract = abstract.replace(". ", " ").replace(", ", " ").replace("! ", " ")\
                       .replace("? ", " ").replace(": ", " ").replace("; ", " ")\
                       .replace("( ", " ").replace(") ", " ").replace("| ", " ").replace("/ ", " ")
    if "." in abstract or "," in abstract or "!" in abstract or "?" in abstract or ":" in abstract or ";" in abstract or "(" in abstract or ")" in abstract or "|" in abstract or "/" in abstract:
        abstract = abstract.replace(".", " ").replace(",", " ").replace("!", " ")\
                           .replace("?", " ").replace(":", " ").replace(";", " ")\
                           .replace("(", " ").replace(")", " ").replace("|", " ").replace("/", " ")
    abstract = abstract.replace("  ", " ")
    
    for word in list(set(stopwords.words("english"))):
        abstract = abstract.replace(" " + word + " ", " ")

    return lemmatize_sentence(abstract).lower()

def get_similar_words(word, num):
    vec = model_wv_df[word].T
    distances = np.linalg.norm(model_wv_df.subtract(model_wv_df[word], 
                                                    axis=0).values, axis=0)

    indices = np.argsort(distances)
    top_distances = distances[indices[1:num+1]]
    top_words = model_wv_vocab[indices[1:num+1]]
    return top_words

def visualize_word_list(color, word):
    top_words = get_similar_words(word, num=6)
    relevant_words = [get_similar_words(word, num=8) for word in top_words]
    fig = make_subplots(rows=3, cols=2, subplot_titles=tuple(top_words), vertical_spacing=0.05)
    for idx, word_list in enumerate(relevant_words):
        words = [word for word in word_list if word in model_wv_vocab]
        X = model_wv_df[words].T
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
        df["Word"] = word_list
        word_emb = df[["Component 1", "Component 2"]].loc[0]
        df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
        plot = px.scatter(df, x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale=color, size="Distance")
        plot.layout.title = top_words[idx]
        plot.update_traces(textposition='top center')
        plot.layout.xaxis.autorange = True
        plot.data[0].marker.line.width = 1
        plot.data[0].marker.line.color = 'rgb(0, 0, 0)'
        fig.add_trace(plot.data[0], row=(idx//2)+1, col=(idx%2)+1)
    fig.layout.coloraxis.showscale = False
    fig.update_layout(height=1400, title_text="2D PCA of words related to {}".format(word), paper_bgcolor="#f0f0f0", template="plotly_white")
    return fig

def visualize_word(color, word):
    top_words = get_similar_words(word, num=20)
    words = [word for word in top_words if word in model_wv_vocab]
    X = model_wv_df[words].T
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
    df["Word"] = top_words
    if word == "antimalarial":
        df = df.query("Word != 'anti-malarial' and Word != 'anthelmintic'")
    if word == "doxorubicin":
        df = df.query("Word != 'anti-rotavirus'")
    word_emb = df[["Component 1", "Component 2"]].loc[0]
    df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
    fig = px.scatter(df, x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale=color, size="Distance")
    fig.layout.title = word
    fig.update_traces(textposition='top center')
    fig.layout.xaxis.autorange = True
    fig.layout.coloraxis.showscale = True
    fig.data[0].marker.line.width = 1
    fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
    fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(word), template="plotly_white", paper_bgcolor="#f0f0f0")
    fig.show()
# lemmatizer = WordNetLemmatizer()

# def get_words(abstract):
    # return clean_text(nonan(abstract)).split(" ")

# words = papers_df["abstract"].progress_apply(get_words)
# model = Word2Vec(words, size=200, sg=1, min_count=1, window=8, hs=0, negative=15, workers=1)

model_wv = pd.read_csv("../input/word2vec-results-1/embed.csv").values
model_wv_vocab = pd.read_csv("../input/word2vec-results-1/vocab.csv").values[:, 0]
model_wv_df = pd.DataFrame(np.transpose(model_wv), columns=model_wv_vocab)
keywords = ["infection", "cell", "protein", "virus",\
            "disease", "respiratory", "influenza", "viral",\
            "rna", "patient", "pathogen", "human", "medicine",\
            "cov", "antiviral"]

print("Most similar words to keywords")
print("")

top_words_list = []
for jdx, word in enumerate(keywords):
    if jdx < 5:
        print(word + ":")
    
    vec = model_wv_df[word].T
    distances = np.linalg.norm(model_wv_df.subtract(model_wv_df[word], 
                                                    axis=0).values, axis=0)

    indices = np.argsort(distances)
    top_distances = distances[indices[1:11]]
    top_words = model_wv_vocab[indices[1:11]]
    top_words_list.append(top_words.tolist())
    
    if jdx < 5:
        for idx, word in enumerate(top_words):
            print(str(idx+1) + ". " + word)
        print("")
words = [word for word in keywords if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = keywords
df["Distance"] = np.sqrt(df["Component 1"]**2 + df["Component 2"]**2)
fig = px.scatter(df, x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="agsunset",size="Distance")
fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of Word2Vec embeddings", template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()
words = [word for word in keywords if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=3)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2", "Component 3"])
df["Word"] = keywords
df["Distance"] = np.sqrt(df["Component 1"]**2 + df["Component 2"]**2 + df["Component 3"]**2)
fig = px.scatter_3d(df, x="Component 1", y="Component 2", z="Component 3", text="Word", color="Distance", color_continuous_scale="agsunset")
fig.update_traces(textposition='top left')
fig.layout.coloraxis.showscale = False
fig.layout.xaxis.autorange = True
fig.update_layout(height=800, title_text="3D PCA of Word2Vec embeddings", template="plotly")
fig.show()
words = [word for word in top_words_list[6] if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = top_words_list[6]
word_emb = df[["Component 1", "Component 2"]].loc[0]
df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
fig = px.scatter(df.query("Word != 'uenza'"), x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="aggrnyl",size="Distance")

"""for row in range(len(df)):
    fig.add_shape(
                type="line",
                x0=word_emb[0],
                y0=word_emb[1],
                x1=df["Component 1"][row],
                y1=df["Component 2"][row],
                line=dict(
                    color="Green",
                    width=0.75,
                    dash="dot"
                )
    )"""

fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keywords[6]), template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()
words = [word for word in top_words_list[8] if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = top_words_list[8]
word_emb = df[["Component 1", "Component 2"]].loc[0]
df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
fig = px.scatter(df[1:].query("Word != 'abstractrna'"), x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="agsunset",size="Distance")

"""for row in range(len(df)):
    fig.add_shape(
                type="line",
                x0=word_emb[0],
                y0=word_emb[1],
                x1=df["Component 1"][row],
                y1=df["Component 2"][row],
                line=dict(
                    color="MediumPurple",
                    width=0.75,
                    dash="dot"
                )
    )"""

fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keywords[8]), template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()
words = [word for word in top_words_list[-2] if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = top_words_list[-2]
word_emb = df[["Component 1", "Component 2"]].loc[0]
df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
fig = px.scatter(df[1:], x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="oryel",size="Distance")


"""for row in range(len(df)):
    fig.add_shape(
                type="line",
                x0=word_emb[0],
                y0=word_emb[1],
                x1=df["Component 1"][row],
                y1=df["Component 2"][row],
                line=dict(
                    color="Orange",
                    width=0.75,
                    dash="dot"
                )
    )"""

fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keywords[-2]), template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()
words = [word for word in top_words_list[3] if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = top_words_list[3]
word_emb = df[["Component 1", "Component 2"]].loc[0]
df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
fig = px.scatter(df[1:], x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="bluered",size="Distance")

"""for row in range(len(df)):
    fig.add_shape(
                type="line",
                x0=word_emb[0],
                y0=word_emb[1],
                x1=df["Component 1"][row],
                y1=df["Component 2"][row],
                line=dict(
                    color="Purple",
                    width=0.75,
                    dash="dot"
                )
    )"""

fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keywords[3]), template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()
words = [word for word in top_words_list[-1] if word in model_wv_vocab]
X = model_wv_df[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = top_words_list[-1]
word_emb = df[["Component 1", "Component 2"]].loc[0]
df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)
fig = px.scatter(df[2:], x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="viridis",size="Distance")

"""for row in range(len(df)):
    fig.add_shape(
                type="line",
                x0=word_emb[0],
                y0=word_emb[1],
                x1=df["Component 1"][row],
                y1=df["Component 2"][row],
                line=dict(
                    color="Purple",
                    width=0.75,
                    dash="dot"
                )
    )"""

fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keywords[-1]), template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()
fig = visualize_word_list('agsunset', 'antiviral')
fig.update_layout(colorscale=dict(diverging=px.colors.diverging.Tealrose))
class Tweet(object):
    def __init__(self, s, embed_str=False):
        if not embed_str:
            # Use Twitter's oEmbed API
            # https://dev.twitter.com/web/embedded-tweets
            api = 'https://publish.twitter.com/oembed?url={}'.format(s)
            response = requests.get(api)
            self.text = response.json()["html"]
        else:
            self.text = s

    def _repr_html_(self):
        return self.text

Tweet("https://twitter.com/elonmusk/status/1239650597906898947")
Tweet("https://twitter.com/elonmusk/status/1239755145233289217")
visualize_word('plotly3', 'antimalarial')
tbl = full_table.sort_values(by=["Country/Region", "Date"]).reset_index(drop=True)
tbl["Country"] = tbl["Country/Region"]
conts = sorted(list(set(tbl["Country"])))
dates = sorted(list(set(tbl["Date"])))

confirmed = []
for idx in range(len(conts)):
    confirmed.append(tbl.query('Country == "{}"'.format(conts[idx])).groupby("Date").sum()["Confirmed"].values)
confirmed = np.array(confirmed)
def visualize_country(fig, cont, image_link, colors, step, xcor, ycor, done=True, multiple=False, sizex=0.78, sizey=0.2):
    if not done:
        showlegend = True
    else:
        showlegend = False
    for idx, color in enumerate(colors):
        fig.add_trace(go.Scatter(x=dates, y=confirmed[conts.index(cont)]-step*idx, showlegend=showlegend,
                    mode='lines+markers', name=cont,
                         marker=dict(color=colors[idx], line=dict(color='rgb(0, 0, 0)', width=0.5))))
    fig.add_layout_image(
        dict(
            source=image_link,
            xref="paper", yref="paper",
            x=xcor, y=ycor,
            sizex=sizex, sizey=sizey,
            xanchor="right", yanchor="bottom")
        )
    title = "Confirmed cases in {}".format(cont) if done else "Confirmed cases"
    if multiple: title = "Confirmed cases"
    fig.update_layout(xaxis_title="Date", yaxis_title="Confirmed cases", title=title, template="plotly_white", paper_bgcolor="#f0f0f0")
    if done:
        fig.show()
fig = go.Figure()
visualize_country(fig, "Italy", "https://upload.wikimedia.org/wikipedia/en/0/03/Flag_of_Italy.svg", colors=["seagreen"], step=400, xcor=0.85, ycor=0.7)
fig = go.Figure()
visualize_country(fig, "China", "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", colors=["red"], step=1000, xcor=0.85, ycor=0.65)
fig = go.Figure()
visualize_country(fig, "US", "https://upload.wikimedia.org/wikipedia/en/a/a4/Flag_of_the_United_States.svg", colors=["navy"], step=60, xcor=0.85, ycor=0.5) 
fig = go.Figure()
visualize_country(fig, "Iran", "https://upload.wikimedia.org/wikipedia/commons/c/ca/Flag_of_Iran.svg", colors=["indianred"], step=175, xcor=0.8, ycor=0.6)
fig = go.Figure()
visualize_country(fig, "South Korea", "https://upload.wikimedia.org/wikipedia/commons/0/09/Flag_of_South_Korea.svg", colors=["dodgerblue"], step=80, xcor=0.95, ycor=0.4)
fig = go.Figure()
visualize_country(fig, "Italy", "https://upload.wikimedia.org/wikipedia/en/0/03/Flag_of_Italy.svg", colors=["seagreen"], step=400, xcor=0.85, ycor=0.4, sizex=0.15, sizey=0.075, done=False)
visualize_country(fig, "US", "https://upload.wikimedia.org/wikipedia/en/a/a4/Flag_of_the_United_States.svg", colors=["navy"], step=60, xcor=0.999, ycor=0.45, sizex=0.1, sizey=0.065, done=False)
visualize_country(fig, "Iran", "https://upload.wikimedia.org/wikipedia/commons/c/ca/Flag_of_Iran.svg", colors=["indianred"], step=175, xcor=0.999, ycor=0.2, sizex=0.1, sizey=0.065, done=False)
visualize_country(fig, "South Korea", "https://upload.wikimedia.org/wikipedia/commons/0/09/Flag_of_South_Korea.svg", colors=["dodgerblue"], step=80, xcor=0.99, ycor=0.05, sizex=0.15, sizey=0.075, done=False)
fig.update_layout(showlegend=False)
visualize_country(fig, "China", "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", colors=["red"], step=1000, xcor=0.5, ycor=0.7, sizex=0.15, sizey=0.075, multiple=True)
fig = go.Figure()
visualize_country(fig, "China", "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg", colors=["red"], step=1000, xcor=0.85, ycor=0.65, done=False)
fig.add_shape(
        dict(
            type="line",
            x0=Timestamp('2020-02-13 00:00:00'),
            y0=50000,
            x1=Timestamp('2020-02-13 00:00:00'),
            y1=70000,
            line=dict(
                color="RoyalBlue",
                width=5
            )
))
fig.add_shape(
        dict(
            type="line",
            x0=Timestamp('2020-02-20 00:00:00'),
            y0=65000,
            x1=Timestamp('2020-02-20 00:00:00'),
            y1=85000,
            line=dict(
                color="Green",
                width=5
            )
))
fig.add_shape(
        dict(
            type="line",
            x0=Timestamp('2020-01-23 00:00:00'),
            y0=-10000,
            x1=Timestamp('2020-01-23 00:00:00'),
            y1=10000,
            line=dict(
                color="Orange",
                width=5
            )
))
fig.update_layout(title="Confirmed cases in China", showlegend=False)
fig.show()
fig = go.Figure()
visualize_country(fig, "South Korea", "https://upload.wikimedia.org/wikipedia/commons/0/09/Flag_of_South_Korea.svg", colors=["dodgerblue"], step=80, xcor=0.95, ycor=0.4, done=False)
fig.add_shape(
        dict(
            type="line",
            x0=Timestamp('2020-02-29 00:00:00'),
            y0=2000,
            x1=Timestamp('2020-02-29 00:00:00'),
            y1=4000,
            line=dict(
                color="purple",
                width=5
            )
))
fig.add_shape(
        dict(
            type="line",
            x0=Timestamp('2020-03-06 00:00:00'),
            y0=5500,
            x1=Timestamp('2020-03-06 00:00:00'),
            y1=7500,
            line=dict(
                color="deeppink",
                width=5
            )
))
fig.update_layout(title="Confirmed cases in Korea, South", showlegend=False)
fig.show()