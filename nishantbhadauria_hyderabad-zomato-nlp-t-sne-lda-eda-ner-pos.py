import pandas as pd

hyd_rest=pd.read_csv('../input/zomato-restaurants-hyderabad/Restaurant names and Metadata.csv')

hyd_rev=pd.read_csv('../input/zomato-restaurants-hyderabad/Restaurant reviews.csv')
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer 

from nltk.stem import PorterStemmer, LancasterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from textblob import TextBlob

import warnings

warnings.filterwarnings('ignore')

from IPython.display import Image

%matplotlib inline
hyd_rev['Rating'].value_counts(normalize=True)
hyd_rev['Review']=hyd_rev['Review'].astype(str)

hyd_rev['Review_length'] = hyd_rev['Review'].apply(len)
import plotly.express as px

fig = px.scatter(hyd_rev, x=hyd_rev['Rating'], y=hyd_rev['Review_length'])

fig.update_layout(title_text="Rating vs Review Length")

fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='crimson',tickangle=45, ticklen=10)

fig.show()
hyd_rev['Polarity'] = hyd_rev['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
hyd_rev['Polarity'].plot(kind='hist', bins=100)
stop_words = stopwords.words('english')

print(stop_words)

rest_word=['order','restaurant','taste','ordered','good','food','table','place','one','also']

rest_word
import re

hyd_rev['Review']=hyd_rev['Review'].map(lambda x: re.sub('[,\.!?]','', x))

hyd_rev['Review']=hyd_rev['Review'].map(lambda x: x.lower())

hyd_rev['Review']=hyd_rev['Review'].map(lambda x: x.split())

hyd_rev['Review']=hyd_rev['Review'].apply(lambda x: [item for item in x if item not in stop_words])

hyd_rev['Review']=hyd_rev['Review'].apply(lambda x: [item for item in x if item not in rest_word])
from wordcloud import WordCloud

hyd_rev['Review']=hyd_rev['Review'].astype(str)



ps = PorterStemmer() 

hyd_rev['Review']=hyd_rev['Review'].map(lambda x: ps.stem(x))

long_string = ','.join(list(hyd_rev['Review'].values))

long_string

wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')

wordcloud.generate(long_string)

wordcloud.to_image()



hyd_rev['Rating']=pd.to_numeric(hyd_rev['Rating'],errors='coerce')

pos_rev = hyd_rev[hyd_rev.Rating>= 3]

neg_rev = hyd_rev[hyd_rev.Rating< 3]

long_string = ','.join(list(pos_rev['Review'].values))

long_string

wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')

wordcloud.generate(long_string)

wordcloud.to_image()
long_string = ','.join(list(neg_rev['Review'].values))

long_string

wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')

wordcloud.generate(long_string)

wordcloud.to_image()
from gensim.models import word2vec

pos_rev = hyd_rev[hyd_rev.Rating>= 3]

neg_rev = hyd_rev[hyd_rev.Rating< 3]
def build_corpus(data):

    "Creates a list of lists containing words from each sentence"

    corpus = []

    for col in ['Review']:

        for sentence in data[col].iteritems():

            word_list = sentence[1].split(" ")

            corpus.append(word_list)

            

    return corpus



corpus = build_corpus(neg_rev)        

corpus[0:2]
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)

model

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
tsne_plot(model)
def build_corpus(data):

    "Creates a list of lists containing words from each sentence"

    corpus = []

    for col in ['Review']:

        for sentence in data[col].iteritems():

            word_list = sentence[1].split(" ")

            corpus.append(word_list)

            

    return corpus



corpus = build_corpus(pos_rev)        

corpus[0:2]
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)

model

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
tsne_plot(model)
from nltk.tag import pos_tag

from nltk import pos_tag_sents
neg_texts = neg_rev['Review'].str.split().map(pos_tag)

neg_texts.head()

def count_tags(title_with_tags):

    tag_count = {}

    for word, tag in title_with_tags:

        if tag in tag_count:

            tag_count[tag] += 1

        else:

            tag_count[tag] = 1

    return(tag_count)

neg_texts.map(count_tags).head()

neg_texts = pd.DataFrame(neg_texts)

neg_texts['tag_counts'] = neg_texts['Review'].map(count_tags)

neg_texts.head()
tag_set = list(set([tag for tags in neg_texts['tag_counts'] for tag in tags]))

for tag in tag_set:

    neg_texts[tag] = neg_texts['tag_counts'].map(lambda x: x.get(tag, 0))

title = 'Frequency of POS Tags in Negative Reviews'    

neg_texts[tag_set].sum().sort_values().plot(kind='barh', logx=True, figsize=(12,8), title=title)

import numpy as np

import seaborn as sns

def plot_10_most_common_words(count_data, count_vectorizer):

    import matplotlib.pyplot as plt

    words = count_vectorizer.get_feature_names()

    total_counts = np.zeros(len(words))

    for t in count_data:

        total_counts+=t.toarray()[0]

        count_dict = (zip(words, total_counts))

    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]

    words = [w[0] for w in count_dict]

    counts = [w[1] for w in count_dict]

    x_pos = np.arange(len(words)) 

    

    plt.figure(2, figsize=(15, 15/1.6180))

    plt.subplot(title='10 most common words in Negative reviews')

    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})

    sns.barplot(x_pos, counts, palette='husl')

    plt.xticks(x_pos, words, rotation=90) 

    plt.xlabel('words')

    plt.ylabel('counts')

    plt.show()

text2=neg_rev['Review'].values

count_vectorizer = CountVectorizer(stop_words='english')



count_data = count_vectorizer.fit_transform(text2)

plot_10_most_common_words(count_data, count_vectorizer)





import warnings

warnings.simplefilter("ignore", DeprecationWarning)

# Load the LDA model from sk-learn

from sklearn.decomposition import LatentDirichletAllocation as LDA

 

# Helper function

def print_topics(model, count_vectorizer, n_top_words):

    words = count_vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(model.components_):

        print("\nTopic #%d:" % topic_idx)

        print(" ".join([words[i]

                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

        

# Tweak the two parameters below

number_topics = 10

number_words = 10

# Create and fit the LDA model

lda = LDA(n_components=number_topics, n_jobs=-1)

lda.fit(count_data)

# Print the topics found by the LDA model

print("10 Topics found via LDA for negative reviews:")

print_topics(lda, count_vectorizer, number_words)





import spacy

from spacy import displacy

from collections import Counter

import en_core_web_sm

nlp = en_core_web_sm.load()

def extract_named_ents(text):    

    return [ ent.label_ for ent in nlp(text).ents]  

neg_rev['named_ents'] = neg_rev['Review'].apply(extract_named_ents)   
text_ents=neg_rev[['named_ents','Review','Restaurant','Rating']]

text_ents['named_ents_new']=[','.join(map(str, l)) for l in text_ents['named_ents']]

text_ents
DATE_PROBLEM=text_ents['named_ents_new'] == 'DATE'

text_ents[DATE_PROBLEM]
avg_rating=hyd_rev.groupby('Restaurant',as_index=False)['Rating'].mean()

merged=hyd_rest.merge(avg_rating, how='inner',left_on='Name',right_on='Restaurant')
merged.head()
merged["North_indian"]= merged["Cuisines"].str.find("North Indian")  

merged["Chinese"]=merged["Cuisines"].str.find("Chinese")

merged["South_Indian"]=merged["Cuisines"].str.find("South Indian")
merged.loc[merged['North_indian'] == -1, 'North_Indian_menu'] = 0

merged.loc[merged['North_indian'] > -1, 'North_Indian_menu'] = 1

merged.loc[merged['Chinese'] == -1, 'Chinese_menu'] = 0

merged.loc[merged['Chinese'] > -1, 'Chinese_menu'] = 1

merged.loc[merged['South_Indian'] == -1, 'South_Indian_menu'] = 0

merged.loc[merged['South_Indian'] > -1, 'South_Indian_menu'] = 1
North=merged[merged['North_Indian_menu'] == 1]

mean_rating_N=North.groupby(['Name','Cost'],as_index=False).Rating.mean()
import plotly.express as px

fig = px.bar(mean_rating_N, x="Name", y="Cost",color="Rating")

fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='crimson',tickangle=45, ticklen=10)

fig.update_layout(title_text="North Indian restaurant cost vs rating")

fig.show()
South=merged[merged['South_Indian_menu'] == 1]



mean_rating_S=South.groupby(['Name','Cost'],as_index=False).Rating.mean()

fig = px.bar(mean_rating_S, x="Name", y="Cost",color="Rating")

fig.update_layout(title_text="South Indian restaurant cost vs rating")

fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='crimson',tickangle=45, ticklen=10)

fig.show()
Chinese=merged[merged['Chinese_menu'] == 1]

mean_rating_C=Chinese.groupby(['Name','Cost'],as_index=False).Rating.mean()

fig = px.bar(mean_rating_C, x="Name", y="Cost",color="Rating")

fig.update_layout(title_text="Chinese restaurant cost vs rating")

fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='crimson',tickangle=45, ticklen=10)

fig.show()
merged['Cost']=merged['Cost'].str.replace(',', '').astype(float)

merged['Cost']=merged['Cost'].astype(float)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(merged[['Cost', 'Rating']])

merged['Name'] = kmeans.labels_

with plt.style.context('bmh', after_reset=True):

    pal = sns.color_palette('Spectral', 7)

    plt.figure(figsize = (8,6))

    for i in range(2):

        ix = merged.Name == i

        plt.scatter(merged.loc[ix, 'Rating'], merged.loc[ix, 'Cost'], color = pal[i], label = str(i))

        plt.text(merged.loc[i, 'Rating'], merged.loc[i, 'Cost'], str(i) + ': '+str(merged.loc[i, 'Name'].round(2)), fontsize = 14, color = 'brown')

    plt.title('KMeans Hyderabad Restaurant for Cost and Rating')

    plt.legend()

    plt.show()
merged['Cuisines'] = merged['Cuisines'].astype(str)

merged['fusion_num'] = merged['Cuisines'].apply(lambda x: len(x.split(',')))



from collections import Counter

lst_cuisine = set()

Cnt_cuisine = Counter()

for cu_lst in merged['Cuisines']:

    cu_lst = cu_lst.split(',')

    lst_cuisine.update([cu.strip() for cu in cu_lst])

    for cu in cu_lst:

        Cnt_cuisine[cu.strip()] += 1



cnt = pd.DataFrame.from_dict(Cnt_cuisine, orient = 'index')

cnt.sort_values(0, ascending = False, inplace = True)





tmp_cnt = cnt.head(10)

tmp_cnt.rename(columns = {0:'cnt'}, inplace = True)

with plt.style.context('bmh'):

    f = plt.figure(figsize = (12, 8))

    ax = plt.subplot2grid((2,2), (0,0))

    sns.barplot(x = tmp_cnt.index, y = 'cnt', data = tmp_cnt, ax = ax, palette = sns.color_palette('Blues_d', 10))

    ax.set_title('# Cuisine')

    ax.tick_params(axis='x', rotation=70)

    ax = plt.subplot2grid((2,2), (0,1))

    sns.countplot(merged['fusion_num'], ax=ax, palette = sns.color_palette('Blues_d', merged.fusion_num.nunique()))

    ax.set_title('# Cuisine Provided')

    ax.set_ylabel('')



    ax = plt.subplot2grid((2,2), (1,0), colspan = 2)

    fusion_rate = merged[['fusion_num', 'Rating']].copy()

    fusion_rate.loc[fusion_rate['fusion_num'] > 5,'fusion_num'] = 5

    fusion_rate = fusion_rate.loc[fusion_rate.Rating != -1, :]

    pal = sns.color_palette('Oranges', 11)

    for i in range(1,6):

        num_ix = fusion_rate['fusion_num'] == i

        sns.distplot(fusion_rate.loc[num_ix, 'Rating'], color = pal[i*2], label = str(i), ax = ax)

        ax.legend()

        ax.set_title('Rating Distribution for fusion_number')



    plt.subplots_adjust(wspace = 0.5, hspace = 0.8, top = 0.85)

    plt.suptitle('Cuisine _ Rating')

    plt.show()        

print('# Unique Cuisine: ', len(lst_cuisine))

hyd_rev['total_reviews']=hyd_rev['Metadata'].str.extract('(\d+)')

hyd_rev
import plotly.express as px

fig = px.scatter_3d(hyd_rev, x='Review_length', y='total_reviews', z='Rating')

fig.update_layout(title_text="Review Length vs Rating vs Number of Reviews ")

fig.show()
reviewer_rating=hyd_rev.groupby(['Reviewer'],as_index=False).Rating.mean()

merged2=reviewer_rating.merge(hyd_rev[['Reviewer','total_reviews']],how='left',left_on='Reviewer',right_on='Reviewer')

merged2=merged2.drop_duplicates()

merged2['total_reviews']=merged2['total_reviews'].fillna(0)

merged2['total_reviews']=merged2['total_reviews'].astype(int)

reveiwer_300=merged2[merged2['total_reviews']>300]
fig = px.scatter_matrix(reveiwer_300,dimensions=["total_reviews", "Rating"], color="Reviewer")

fig.update_layout(title_text="Total Reviews vs Ratings for 300+ reviewers ")

fig.show()