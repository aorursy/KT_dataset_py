import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mplt
from tqdm.notebook import tqdm
import plotly.graph_objects as go
import os
sns.set()
!apt-get install p7zip
!p7zip -d '../input/mercari-price-suggestion-challenge/train.tsv.7z'
import pandas as pd
data = pd.read_csv('train.tsv', sep='\t')
data.head()
data = data[data['price'] > 0].reset_index(drop=True)
def calc_char_len(x): # Calculating the character length of each text data
  try: return len(x)
  except: return 0

def calc_word_len(x): # Calculating the word length of each text data
  try: return len(x.split(' '))
  except: return 0
data['price_log'] = data['price'].apply(lambda x:np.log1p(x)) # creating new feature -> log(1+price)
data['item_description_word_length'] = data['item_description'].apply(lambda x:calc_word_len(x)) # creating new feature -> character length of item_description
data['item_description_char_length'] = data['item_description'].apply(lambda x:calc_char_len(x)) # creating new feature -> word length of item_description
data['log_item_description_word_length'] = data['item_description_word_length'].apply(lambda x:np.log1p(x)) # creating new feature -> log(1 + character length of item_description)
data['log_item_description_char_length'] = data['item_description_char_length'].apply(lambda x:np.log1p(x)) # creating new feature -> log(1 + word length of item_description)
data['name_length'] = data['name'].apply(lambda x:len(x)) # creating new feature -> character length of name_length
# data.head(5)
data['category_name'] = data['category_name'].fillna('no label/no label/no label') # substituting the rows of feature 'category_name' with NaN values with 'no label'
# this is to divide the category_name feature into 3 sub categories
sub_category_1 = []
sub_category_2 = []
sub_category_3 = []
for feature in tqdm(data['category_name'].values):
  fs = feature.split('/')
  a,b,c = fs[0], fs[1], ' '.join(fs[2:])
  sub_category_1.append(a)
  sub_category_2.append(b)
  sub_category_3.append(c)
data['sub_category_1'] = sub_category_1
data['sub_category_2'] = sub_category_2
data['sub_category_3'] = sub_category_3
data['brand_name'] = data['brand_name'].fillna('unknown') # replacing NaN values with 'unknown'
data['item_description'] = data['item_description'].fillna('') # replacing NaN values with ''
data.dtypes
# let's check the dataframe after adding new features
data.head(5)
data['price'].describe()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=data['price'])], 
                layout = go.Layout(title='histogram of length of question title in train data', 
                                  xaxis=dict(title='length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=data['price_log'])], 
                layout = go.Layout(title='histogram of log of price', 
                                  xaxis=dict(title='log of price'), 
                                  yaxis=dict(title='frequency')))
plt.show()
data['train_id'].describe()
data['item_condition_id'].value_counts()
data['item_condition_id'].value_counts()
# histogram of length of question titles
plt = go.Figure(data=[go.Bar(x=data['item_condition_id'].value_counts())], 
                layout = go.Layout(title='bar plot of item_condition_id', 
                                  xaxis=dict(title='item_condition_id'), 
                                  yaxis=dict(title='number of data points')))
plt.show()
sns.FacetGrid(data, hue="item_condition_id", height=10).map(sns.distplot, 'price').add_legend();
mplt.title('comparing the price distribution of products with different item_condition_id.\n')
mplt.ylabel('PDF of price')
mplt.show()
sns.FacetGrid(data, hue="item_condition_id", height=10).map(sns.distplot, 'price_log').add_legend();
mplt.title('comparing the log of price distribution of products with different item_condition_id.\n')
mplt.ylabel('PDF of log of price')
mplt.show()
import plotly.graph_objects as go
import pandas as pd

fig = go.Figure()

for c in data['item_condition_id'].unique():
    fig.add_trace(go.Violin(x=data['item_condition_id'][data['item_condition_id'] == c],
                            y=data['price'][data['item_condition_id'] == c],
                            name=f'item_condition_id = {c}',
                            box_visible=True,
                            meanline_visible=True))

fig.show()
import plotly.graph_objects as go
import pandas as pd

fig = go.Figure()

for c in data['item_condition_id'].unique():
    fig.add_trace(go.Violin(x=data['item_condition_id'][data['item_condition_id'] == c],
                            y=data['price_log'][data['item_condition_id'] == c],
                            name=f'item_condition_id = {c}',
                            box_visible=True,
                            meanline_visible=True))

fig.show()
data['category_name'].unique().shape
data['category_name'].describe()
data['sub_category_1'].describe()
categories = data['sub_category_1'].value_counts()
fig = go.Figure([go.Pie(labels=categories.keys(), values=categories)])
fig.update_traces(hole=.5, hoverinfo="label+percent+name")
fig.update_layout(title_text="'sub_category_1' Pie chart",
                  annotations=[dict(text='sub_category_1', x=0.5, y=0.5, 
                                    font_size=16, showarrow=False)])
fig.show()
sns.FacetGrid(data, hue="sub_category_1", height=8).map(sns.distplot, 'price').add_legend();
mplt.title('comparing the price distribution of products with sub_category_1\n')
mplt.ylabel('PDF of price')
mplt.show()
sns.FacetGrid(data, hue="sub_category_1", height=8).map(sns.distplot, 'price_log').add_legend();
mplt.title('comparing the log of price distribution of products with sub_category_1\n')
mplt.ylabel('PDF of log of price')
mplt.show()
import plotly.graph_objects as go
import pandas as pd

fig = go.Figure()

for c in data['sub_category_1'].unique():
    fig.add_trace(go.Violin(x=data['sub_category_1'][data['sub_category_1'] == c],
                            y=data['price'][data['sub_category_1'] == c],
                            name=f'sub_category_1 = {c}',
                            box_visible=True,
                            meanline_visible=True))

fig.show()
import plotly.graph_objects as go
import pandas as pd

fig = go.Figure()

for c in data['sub_category_1'].unique():
    fig.add_trace(go.Violin(x=data['sub_category_1'][data['sub_category_1'] == c],
                            y=data['price_log'][data['sub_category_1'] == c],
                            name=f'sub_category_1 = {c}',
                            box_visible=True,
                            meanline_visible=True))

fig.show()
data['sub_category_2'].value_counts()
categories = data['sub_category_2'].value_counts()[:20]
fig = go.Figure([go.Pie(labels=categories.keys(), values=categories)])
fig.update_traces(hole=.5, hoverinfo="label+percent+name")
fig.update_layout(title_text="'sub_category_2' Pie chart",
                  annotations=[dict(text='sub_category_2', x=0.5, y=0.5, 
                                    font_size=16, showarrow=False)])
fig.show()
data['sub_category_3'].value_counts()
mplt.figure(figsize=(16,12))
sns.barplot(x=data['sub_category_3'].value_counts().keys()[:20], y=data['sub_category_3'].value_counts()[:20])
mplt.ylabel('number of products')
locs, labels = mplt.xticks()
mplt.setp(labels, rotation=60)
mplt.title('bar-plot of top 20 sub_category_3')
mplt.show()
data['brand_name'].value_counts()
print('Number of NaN values in brand_name:')
data['brand_name'].isna().sum()
data['brand_name'] = data['brand_name'].fillna('unknown') # replacing NaN values with 'none'
mplt.figure(figsize=(16,12))
sns.barplot(x=data['brand_name'].value_counts().keys()[:20], y=data['brand_name'].value_counts()[:20])
mplt.ylabel('number of products')
locs, labels = mplt.xticks()
mplt.setp(labels, rotation=50)
mplt.title('bar-plot of top 20 brands (including products with unknown brand)')
mplt.show()
df = data.groupby('brand_name')['price'].mean().reset_index().sort_values(by='price', ascending=False)
df.head(5)
mplt.figure(figsize=(16,12))
sns.barplot(x=df['brand_name'].values[:20], y=df['price'].values[:20])
mplt.ylabel('average price of products')
locs, labels = mplt.xticks()
mplt.setp(labels, rotation=50)
mplt.title('bar-plot of top 20 brands with their mean product price')
mplt.show()
df = data.groupby('brand_name')['price'].max().reset_index().sort_values(by='price', ascending=False)
df.head(5)
mplt.figure(figsize=(16,12))
sns.barplot(x=df['brand_name'].values[:20], y=df['price'].values[:20])
mplt.ylabel('max. price of products')
locs, labels = mplt.xticks()
mplt.setp(labels, rotation=50)
mplt.title('bar-plot of top 20 brands with their priciest product price')
mplt.show()
data['shipping'].value_counts()
sns.FacetGrid(data, hue="shipping", height=8).map(sns.distplot, 'price').add_legend();
mplt.title('comparing the price distribution of products with different shipping.\n')
mplt.ylabel('PDF of price')
mplt.show()
sns.FacetGrid(data, hue="shipping", height=8).map(sns.distplot, 'price_log').add_legend();
mplt.title('comparing the price distribution of products with different shipping.\n')
mplt.ylabel('PDF of price')
mplt.show()
data['item_description']
print('Number of products with item_description not defined:')
data['item_description'].isna().sum()
data['item_description_char_length'].describe()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=data['item_description_char_length'])], 
                layout = go.Layout(title='histogram of character length item_description', 
                                  xaxis=dict(title='character length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=data['log_item_description_char_length'])], 
                layout = go.Layout(title='histogram of log of character length item_description', 
                                  xaxis=dict(title='log of length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
df = data.groupby('item_description_char_length')['price'].mean().reset_index()
sns.relplot(x="item_description_char_length", y="price", kind="line", data=df, height=8)
mplt.show()
df = data.groupby('item_description_char_length')['price_log'].mean().reset_index()
sns.relplot(x="item_description_char_length", y="price_log", kind="line", data=df, height=8)
mplt.show()
data['item_description_word_length'].describe()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=data['item_description_word_length'])], 
                layout = go.Layout(title='histogram of word length item_description', 
                                  xaxis=dict(title='length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
mplt.figure(figsize=(12,8))
mplt.hist(data['log_item_description_word_length'])
mplt.xlabel('log(item_description_word_length + 1)')
mplt.ylabel('frequency')
mplt.title('histogram of log of item_description_word_length')
mplt.show()
df = data.groupby('item_description_word_length')['price'].mean().reset_index()
sns.relplot(x="item_description_word_length", y="price", kind="line", data=df, height=8)
mplt.show()
df = data.groupby('item_description_word_length')['price_log'].mean().reset_index()
sns.relplot(x="item_description_word_length", y="price_log", kind="line", data=df, height=8)
mplt.show()
data.columns
data['name_length'].describe()
print('Number of products with name not defined:')
data['name'].isna().sum()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=data['name_length'])], 
                layout = go.Layout(title='histogram of length of answer in train data', 
                                  xaxis=dict(title='length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
df = data.groupby('name_length')['price_log'].mean().reset_index()
sns.relplot(x="name_length", y="price_log", kind="line", data=df, height=8)
mplt.show()
data['item_description'].isna().sum()
all_text_features = []
for i in zip(data['name'],data['brand_name'],data['item_description'],data['sub_category_3']):
  all_text_features.append(' '.join(i))
all_text_features[:3]
data['all_text_features'] = all_text_features
import re
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't", '•', '❤', '✨', '$', '❌', '♡', '☆', '✔', '⭐',
            '✅', '⚡', '‼', '—', '▪', '❗', '■', '●', '➡',
            '⛔', '♦', '〰', '×', '⚠', '°', '♥', '★', '®', '·', '☺', '–', '➖',
            '✴', '❣', '⚫', '✳', '➕', '™', 'ᴇ', '》', '✖', '▫', '¤',
            '⬆', '⃣', 'ᴀ', '❇', 'ᴏ', '《', '☞', '❄', '»', 'ô', '❎', 'ɴ', '⭕', 'ᴛ',
            '◇', 'ɪ', '½', 'ʀ', '❥', '⚜', '⋆', '⏺', '❕', 'ꕥ', '：', '◆', '✽',
            '…', '☑', '︎', '═', '▶', '⬇', 'ʟ', '！', '✈', '�', '☀', 'ғ']
def preprocess_text(text_data):
    preprocessed_text = []
    # tqdm is for printing the status bar
    for sentence in tqdm(text_data):
        sent = decontracted(sentence)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text
preprocessed_all_text_features = preprocess_text(data['all_text_features'].fillna('').values) # list of cleaned data in 'item_description'
y = data['price'].values
X = data[['item_condition_id', 'shipping', 'name_length', 'sub_category_1', 'sub_category_2', 'log_item_description_char_length']]
X['preprocessed_text'] = preprocessed_all_text_features
X.head(5)
X.to_csv('ipdated_features.csv', index=False)
