



import pandas as pd

import numpy as np

import os

import re

import gensim

import spacy

import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("german")







data_test = pd.read_csv('../input/10k-german-news-articles/Articles.csv')

data_test.head()


body_new = [re.sub('<[^<]+?>|&amp', '', word) for word in data_test['Body'].values]

data_test['Body'] = body_new

#! python -m spacy download de_core_news_sm

! python -m spacy download de_core_news_sm

import spacy

import de_core_news_sm

nlp = de_core_news_sm.load()





#nlp = spacy.load('de_core_news_sm')



## Drop Duplicates and NA values



data_test_clean = data_test.dropna()

data_test_clean = data_test_clean.drop_duplicates()
article_data = data_test_clean[['ID_Article','Title','Body']].drop_duplicates()

article_data.shape


## Convert the body text into a series of words to be fed into the model



def text_clean_tokenize(article_data):

    

    review_lines = list()



    lines = article_data['Body'].values.astype(str).tolist()



    for line in lines:

        tokens = word_tokenize(line)

        tokens = [w.lower() for w in tokens]

        table = str.maketrans('','',string.punctuation)

        stripped = [w.translate(table) for w in tokens]

        # remove remaining tokens that are not alphabetic

        words = [word for word in stripped if word.isalpha()]

        stop_words = set(stopwords.words('german'))

        words = [w for w in words if not w in stop_words]

        words = [stemmer.stem(w) for w in words]



        review_lines.append(words)

    return(review_lines)

    

    

review_lines = text_clean_tokenize(article_data)
## Building the word2vec model



model =  gensim.models.Word2Vec(sentences = review_lines,

                               size=100,

                               window=2,

                               workers=4,

                               min_count=2,

                               seed=42,

                               iter= 50)



model.save("word2vec.model")

import spacy



def tokenize(sent):

    doc = nlp.tokenizer(sent)

    return [token.lower_ for token in doc if not token.is_punct]



new_df = (article_data['Body'].apply(tokenize).apply(pd.Series))



new_df = new_df.stack()

new_df = (new_df.reset_index(level=0)

                .set_index('level_0')

                .rename(columns={0: 'word'}))



new_df = new_df.join(article_data.drop('Body', 1), how='left')



new_df = new_df[['word','ID_Article','Title']]

word_list = list(model.wv.vocab)

vectors = model.wv[word_list]

vectors_df = pd.DataFrame(vectors)

vectors_df['word'] = word_list

merged_frame = pd.merge(vectors_df, new_df, on='word')

merged_frame_rolled_up = merged_frame.drop('word',axis=1).groupby(['ID_Article','Title']).mean().reset_index()

del merged_frame

del new_df

del vectors
merged_frame_rolled_up.columns


from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

data_subset = merged_frame_rolled_up.drop(['ID_Article','Title'],axis=1).values



from sklearn.cluster import KMeans

number_of_clusters = range(1,20,5)

kmeans = [KMeans(n_clusters=i,max_iter=5,init='k-means++') for i in number_of_clusters]

score = [kmeans[i].fit(data_subset).inertia_ for i in range(len(kmeans))]

tmp =0

best_k = 0

value_all = 0

diff1 = []

for i in range(len(score)-1):

    

    scores = (score[i] - score[i+1])

    diff1.append(scores)

    

diff2=[]

for i in range(len(diff1)-1):

    difference = diff1[i] - diff1[i+1]

    diff2.append(difference)

diff2.insert(0, 0) 

diff3 = [i-j for i,j in zip(diff2,diff1)]



m = max(i for i in diff3)

best_k = number_of_clusters[diff3.index(m)]

print(best_k)



kmeans = KMeans(n_clusters=best_k,init='k-means++',max_iter=5).fit(data_subset)

labels = kmeans.labels_

merged_frame_rolled_up['labels'] = labels







tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=3000,init='pca',random_state=42)

tsne_results = tsne.fit_transform(data_subset)

%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns



merged_frame_rolled_up['tsne-2d-one'] = tsne_results[:,0]

merged_frame_rolled_up['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    hue = 'labels',

    palette=sns.color_palette("hls", len(merged_frame_rolled_up['labels'].unique())),

    data=merged_frame_rolled_up,

    legend="full",

    alpha=0.3

)

## Interactive t-SNE mapping using https://www.kaggle.com/maksimeren/covid-19-literature-clustering



from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper

from bokeh.palettes import Category20

from bokeh.transform import linear_cmap

from bokeh.io import output_file, show

from bokeh.transform import transform

from bokeh.io import output_notebook

from bokeh.plotting import figure



output_notebook()



# data sources

source = ColumnDataSource(data=dict(

    x= merged_frame_rolled_up['tsne-2d-one'].values, 

    y= merged_frame_rolled_up['tsne-2d-two'].values, 

    desc= merged_frame_rolled_up['labels'].values, 

    titles= merged_frame_rolled_up['Title'].values

    ))



# hover over information

hover = HoverTool(tooltips=[

    ("Title", "@titles")

])

# map colors

mapper = linear_cmap(field_name='desc', 

                     palette=Category20[len(merged_frame_rolled_up['labels'].unique())],

                     low=min(merged_frame_rolled_up['labels'].values) ,high=max(merged_frame_rolled_up['labels'].values))

#prepare the figure

p = figure(plot_width=1000, plot_height=1000, 

           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 

           title="Clustering German Articles Based on K Means and Word2Vec", 

           toolbar_location="right")

# plot

p.scatter('x', 'y', size=5, 

          source=source,

          fill_color=mapper,

          line_alpha=0.3,

          line_color="black")



# output_file('plots/t-sne_covid-19_interactive.html')

show(p)
