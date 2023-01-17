import numpy as np 

import random

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from collections import Counter

from nltk.corpus import stopwords

from tqdm import tqdm

import os

import re

import string

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv")

df.head()
df.columns
df.info()
df.describe(include = "all")
df.isnull().sum()
df.duplicated(subset=None, keep='first').value_counts()
def get_tech_keys(tag):

    if(not tag):

        return tag

    tag = tag.replace('><', ',')

    tag = tag.replace('<', '')

    tag = tag.replace('>', '')

    return tag
df['TechKeys'] = df['Tags'].apply(get_tech_keys)
tech_key_list   = []

tech_key_values = None

index_counter = 0

tech_key_index_list = []



for item in df['TechKeys']:

    item_parts = item.split(',')

    

    for item_ in item_parts:

        

        tech_key_index_list.append(index_counter)

        tech_key_list.append(item_)

        index_counter += 1

    

df_tech_key_new = pd.DataFrame({'id' : tech_key_index_list, 'tech_key' : tech_key_list}) 
plt.figure(figsize=(8,8))

df_tech_key_new.tech_key.value_counts().nlargest(10).plot(kind='barh')
def get_tags_counts(col):

    if(not col):

        return 0

    tags_count = len(col.split(','))

    return tags_count
def show_donut_plot(col):

    

    rating_data = df.groupby(col)[['Id']].count().head(10)

    plt.figure(figsize = (12, 8))

    plt.pie(rating_data[['Id']], autopct = '%1.2f%%', startangle = 140, pctdistance = 1.1, shadow = True)



    # create a center circle for more aesthetics to make it better

    gap = plt.Circle((0, 0), 0.5, fc = 'white')

    fig = plt.gcf()

    fig.gca().add_artist(gap)

    

    plt.axis('equal')

    

    cols = []

    for index, row in rating_data.iterrows():

        cols.append(index)

    plt.legend(cols)

    

    plt.title('Donut Plot: SOF Questions by ' +str(col), loc='center')

    

    plt.show()
df['TagsCount'] = df['TechKeys'].apply(get_tags_counts)
show_donut_plot('TagsCount')
print(df['Y'].value_counts())

fig = go.Figure(go.Funnelarea(

    text = df.Y,

    values = df.Y.value_counts(),

    title = {"position": "top center", "text": "Funnel-Chart of Question Quality Distribution"}

    ))

fig.show()
df['Num_words_body'] = df['Body'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text

df['Num_words_title'] = df['Title'].apply(lambda x:len(str(x).split())) #Number Of words in main text
plt.figure(figsize=(12,6))

p1=sns.kdeplot(df[df['Y']=='HQ']['Num_words_body'], shade=True, color="b").set_title('Kernel Distribution of Number Of words')

p2=sns.kdeplot(df[df['Y']=='LQ_CLOSE']['Num_words_body'], shade=True, color="r")

p2=sns.kdeplot(df[df['Y']=='LQ_EDIT']['Num_words_body'], shade=True, color="g")

plt.legend(labels=['HQ','LQ_CLOSE','LQ_EDIT'])

plt.ylabel("Probability Density")

plt.xlabel("Number of words")

plt.xlim(-20,500)
def code_available(content):

    

    if('<code>' in content):

        return True

    

    return False
df['code_available'] = df['Body'].apply(code_available)
show_donut_plot('code_available')
def get_week(col):

    return col.strftime("%V")
df['CreationDatetime'] = pd.to_datetime(df['CreationDate']) 

df['CreationYear'] = df['CreationDatetime'].dt.year.astype(int)

df['CreationMonth'] = df['CreationDatetime'].dt.month.astype(int)
show_donut_plot('CreationYear')
import squarify

def show_treemap_tech_key(col):

    df_type_series = df_tech_key_new.groupby(col)['id'].count().sort_values(ascending = False).head(50)



    type_sizes = []

    type_labels = []

    for i, v in df_type_series.items():

        type_sizes.append(v)

        

        type_labels.append(str(i) + ' ('+str(v)+')')





    fig, ax = plt.subplots(1, figsize = (12,12))

    squarify.plot(sizes=type_sizes, 

                  label=type_labels[:25],  # show labels for only first 10 items

                  alpha=.2 )

    plt.title('TreeMap by '+ str(col))

    plt.axis('off')

    plt.show()
show_treemap_tech_key('tech_key')
code_start = '<code>'

code_end   = '</code>'



def get_codes(content):

    

    if('<code>' not in content):

        return None

    

    code_list = []

    

    loop_counter = 0

    while(code_start in content):



        code_start_index = content.index(code_start)

        if(code_end not in content):

            code_end_index = len(content)

        else:

            code_end_index = content.index(code_end)



        substring_1 = content[code_start_index : (code_end_index + len(code_end) )]

 

        code_list.append(substring_1)

        

        content = content.replace(substring_1, '')

        

        loop_counter += 1



    

    return ' '.join(code_list)



def  clean_text(content):

    

    content = content.lower()

    

    content = re.sub('<.*?>+', '', content)

    

    content = re.sub(r"(@[A-Za-z0-9]+)|^rt|http.+?", "", content)

    content = re.sub(r"(\w+:\/\/\S+)", "", content)

    content = re.sub(r"([^0-9A-Za-z \t])", " ", content)

    content = re.sub(r"^rt|http.+?", "", content)

    content = re.sub(" +", " ", content)



    # remove numbers

    content = re.sub(r"\d+", "", content)

    

    return content



def get_non_codes(content):

    

    loop_counter = 0

    while(code_start in content):



        code_start_index = content.index(code_start)

        if(code_end not in content):

            code_end_index = len(content)

        else:

            code_end_index = content.index(code_end)



        substring_1 = content[code_start_index : (code_end_index + len(code_end) )]



        content = content.replace(substring_1, ' ')

        

        loop_counter += 1

        

    content = clean_text(content)



    return content
%%time

df['Body_code'] = df['Body'].apply(get_codes)

df['Body_content'] = df['Body'].apply(get_non_codes)
%%time

stopwords1 = stopwords.words('english')

df['content_words'] = df['Body_content'].apply(lambda x:str(x).split())
def remove_short_words(content):



    new_content_list = []

    for item in content:

        

        if(len(item) > 2):

            new_content_list.append(item)

    

    return new_content_list
df['content_words'] = df['content_words'].apply(remove_short_words)
df.head()
words_collection = Counter([item for sublist in df['content_words'] for item in sublist if not item in stopwords1])

freq_word_df = pd.DataFrame(words_collection.most_common(30))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='YlGnBu', low=0, high=0, axis=0, subset=None)
fig = px.scatter(freq_word_df, x="frequently_used_word", y="count", color="count", title = 'Frequently used words - Scatter plot')

fig.show()
fig = px.pie(freq_word_df, values='count', names='frequently_used_word', title='Stackoverflow Questions - Frequently Used Word')

fig.show()
def get_question_level(level):

    if(not level):

        return level

    if(level == 'LQ_CLOSE'):

        return 3

    if(level == 'LQ_EDIT'):

        return 2

    if(level == 'HQ'):

        return 1

    return level



df['Level'] = df['Y'].apply(get_question_level)
fig = px.sunburst(df, path=['CreationYear', 'CreationMonth'], values='Level',

                  color='Level', hover_data=['Level'])

fig.show()
import spacy

import matplotlib.pyplot as plt

import seaborn as sns



# for advanced visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff
nlp = spacy.load('en')
from spacy import displacy

rand = df['Title'][1000]

doc = nlp(rand)

  

displacy.render(doc, style='dep', jupyter=True)
nlp = spacy.load('en_core_web_lg')
from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(stop_words = 'english')

words = cv.fit_transform(df.Title)

sum_words = words.sum(axis=0)



words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]

words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])



plt.style.use('fivethirtyeight')

color = plt.cm.ocean(np.linspace(0, 1, 20))

frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)

plt.title("Question Title - Most Frequently Occuring Words")

plt.show()
from wordcloud import WordCloud



wordcloud = WordCloud(background_color = 'lightcyan', width = 2000, height = 2000).generate_from_frequencies(dict(words_freq))

plt.figure(figsize=(10, 10))

plt.axis('off')

plt.imshow(wordcloud)

plt.title("Corpus of Question Text", fontsize = 20)

plt.show()
df.head()
trace = go.Scatter3d(

    x = df['Num_words_body'],

    y = df['Level'],

    z = df['TechKeys'].apply(lambda x:x.split(",")[0]),

    name = '3DPlot',

    mode='markers',

    marker=dict(

        size=10,

        color = df['Level'],

        colorscale = 'Viridis',

    )

)

df_dia = [trace]



layout = go.Layout(

    title = 'Number of words in body vs Quality vs Tags',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

)

fig = go.Figure(data = df_dia, layout = layout)

iplot(fig)