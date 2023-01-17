# loading libraries - must have internet on



# Overall tools

import numpy as np

import pandas as pd

import json

import glob

from scipy.spatial.distance import cdist



# Progress bar for the loops

import time

import sys

import tqdm



# Text tools

import re, nltk, spacy, gensim

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.stem import PorterStemmer

nltk.download("punkt")

nltk.download("stopwords")



# Sklearn

from sklearn.feature_extraction.text import HashingVectorizer # Vectorizor for the words in the abstract

from sklearn.feature_extraction.text import TfidfVectorizer # Vectorizor for the text in the abstract (tf-idf)

from sklearn.manifold import TSNE

from sklearn.cluster import MiniBatchKMeans

from sklearn.cluster import KMeans

from sklearn import metrics



# Plotting tools

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Bokeh

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS

from bokeh.palettes import Category20

from bokeh.transform import linear_cmap

from bokeh.io import output_file, show

from bokeh.transform import transform

from bokeh.io import output_notebook

from bokeh.plotting import figure

from bokeh.layouts import column

from bokeh.models import RadioButtonGroup

from bokeh.models import TextInput

from bokeh.layouts import gridplot

from bokeh.models import Div

from bokeh.models import Paragraph

from bokeh.layouts import column, widgetbox
#loading metadata file



root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
# importing all json files



all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
# File Reader class



class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
# Checking if the File Reader Class worked



print(FileReader(all_json[0]))
# Function to add break every length characters



def get_breaks(content, length):

    data = ""

    words = content.split(' ')

    total_chars = 0



    for i in range(len(words)):

        total_chars += len(words[i])

        if total_chars > length:

            data = data + "<br>" + words[i]

            total_chars = 0

        else:

            data = data + " " + words[i]

    return data
# Input the research papers into a DataFrame



dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    content = FileReader(entry)

    

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided, we input the title

        dict_['abstract_summary'].append(meta_data['title'].values[0])

    else:

        dict_['abstract_summary'].append(content.abstract)

        

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    

    # if more than one author

    try:

        authors = str(meta_data['authors'].values[0]).split(';')

        authors1 = [i.split(',') for i in authors]    

        dict_['authors'].append(". ".join(authors))

    except Exception as e:

        dict_['authors'].append(". ".join(authors))

    

    # add the title information, add breaks when needed

    dict_['title'].append(meta_data['title'].values[0])

    

    # add the journal information

    dict_['journal'].append(meta_data['journal'].values[0])

    

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])

df_covid.head()
dict_ = None
# Adding word count column



df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))

df_covid.head()
# We will remove the duplicated papers

df_covid.shape
# Removing duplicated papers



duplicate_paper = ~(df_covid.title.isnull() | df_covid.abstract.isnull()) & (df_covid.duplicated(subset=['title', 'abstract']))

df_covid = df_covid[~duplicate_paper].reset_index(drop=True)

df_covid.shape
# Creating a list of stopwords in english



english_stopwords = list(set(stopwords.words('english')))
# Creating a lemmatizing function



lmtzr = WordNetLemmatizer()
# Creating a stem function



porter = PorterStemmer()
# Creating a function that cleans text of special characters



def strip_characters(text):

    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)

    t = re.sub('/', ' ', t)

    t = t.replace("'",'')

    return t
# Creating a function that makes text lowercase and uses the function created above



def clean(text):

    t = text.lower()

    t = strip_characters(t)

    return t
# Tokenize into individual tokens - words mostly



def tokenize(text):

    words = nltk.word_tokenize(text)

    return list(set([word for word in words 

                     if len(word) > 1

                     and not word in english_stopwords

                     and not (word.isnumeric() and len(word) is not 4)

                     and (not word.isnumeric() or word.isalpha())] )

               )
# Creating a function that cleans, lemmatize and tokenize texts



def preprocess(text):

    t = clean(text)

    tokens = tokenize(t)

    l = [lmtzr.lemmatize(word) for word in tokens]

    return tokens
def stemming(text):

    stem_sentence=[]

    for word in text:

        stem_sentence.append(porter.stem(word))

    return "".join(stem_sentence)
# Preprocessing all the strings inside the column abstract. It will make them lowercase, remove special characters, stopwords and tokenize them.

df_covid['abstract_processed'] = df_covid['abstract'].apply(lambda x: preprocess(x))
# Preprocessing all the strings inside the column abstract. It will make stem them.

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: stemming(x))
abstract = df_covid['abstract_processed'].tolist()

len(abstract)
# Creating vectors for each word

n_gram_all = []



for word in abstract:

    n_gram = []

    for i in range(len(word)-2+1):

        n_gram.append("".join(word[i:i+2]))

    n_gram_all.append(n_gram)
# hash vectorizer instance



hvec = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=2**12)
# Fit and Transforming hash vectorizer



X = hvec.fit_transform(n_gram_all)

X.shape
# THIS WILL TAKE A LONG, LONG TIME. Go watch some series. Go call your family. Catch up with your friends.



# Building the clustering model and calculating the values of the Distortion and Inertia



distortions = [] 

inertias = [] 

mapping1 = {} 

mapping2 = {} 

K = range(1,26) 



for k in tqdm.tqdm(K): 

    #Building and fitting the model 

    kmeanModel = KMeans(n_clusters=k).fit(X.toarray()) 

    kmeanModel.fit(X.toarray())     



    distortions.append(sum(np.min(cdist(X.toarray(), kmeanModel.cluster_centers_, 

                          'euclidean'),axis=1)) / X.toarray().shape[0]) 

    inertias.append(kmeanModel.inertia_)



    mapping1[k] = sum(np.min(cdist(X.toarray(), kmeanModel.cluster_centers_, 

                     'euclidean'),axis=1)) / X.toarray().shape[0] 

    mapping2[k] = kmeanModel.inertia_ 

    time.sleep(0.1)
# List of number of clusters and the decrease of value, this helps to see exactly where the elbow is flexing



for key,val in mapping1.items(): 

    print(str(key)+' : '+str(val.round(4)))
# Plotting the elbow graph



plt.plot(K, distortions, 'bx-') 

plt.xlabel('Values of K') 

plt.ylabel('Distortion') 

plt.title('The Elbow Method using Distortion') 

plt.show()
# Dimensionality Reduction with t-SNE



tsne = TSNE(verbose = 1, perplexity = 10, metric = 'cosine', early_exaggeration = 20, learning_rate = 300, random_state = 42)

X_embedded = tsne.fit_transform(X)
# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", 1)



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)



plt.title("t-SNE Covid-19 Articles")

# plt.savefig("plots/t-sne_covid19.png")

plt.show()
# determining the best number of clusters



k = 21

kmeans = MiniBatchKMeans(n_clusters=k)

y_pred = kmeans.fit_predict(X)

y = y_pred


output_notebook()

y_labels = y_pred



# data sources

source = ColumnDataSource(data=dict(

    x= X_embedded[:,0], 

    y= X_embedded[:,1],

    x_backup = X_embedded[:,0],

    y_backup = X_embedded[:,1],

    desc= y_labels, 

    titles= df_covid['title'],

    authors = df_covid['authors'],

    journal = df_covid['journal'],

    abstract = df_covid['abstract_summary'],

    labels = ["C-" + str(x) for x in y_labels]

    ))



# hover over information

hover = HoverTool(tooltips=[

    ("Title", "@titles{safe}"),

    ("Author(s)", "@authors"),

    ("Journal", "@journal"),

    ("Abstract", "@abstract{safe}"),

],

                 point_policy="follow_mouse")



# map colors

mapper = linear_cmap(field_name='desc', 

                     palette=Category20[20],

                     low=min(y_labels) ,high=max(y_labels))



# prepare the figure

p = figure(plot_width=800, plot_height=800, 

           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 

           title="t-SNE Covid-19 Articles, Clustered(K-Means), Abstracts Hash Vectorized", 

           toolbar_location="right")



# plot

p.scatter('x', 'y', size=5, 

          source=source,

          fill_color=mapper,

          line_alpha=0.3,

          line_color="black",

          legend = 'labels')



# add callback to control 

callback = CustomJS(args=dict(p=p, source=source), code="""

            

            var radio_value = cb_obj.active;

            var data = source.data; 

            

            x = data['x'];

            y = data['y'];

            

            x_backup = data['x_backup'];

            y_backup = data['y_backup'];

            

            labels = data['desc'];

            

            if (radio_value == '20') {

                for (i = 0; i < x.length; i++) {

                    x[i] = x_backup[i];

                    y[i] = y_backup[i];

                }

            }

            else {

                for (i = 0; i < x.length; i++) {

                    if(labels[i] == radio_value) {

                        x[i] = x_backup[i];

                        y[i] = y_backup[i];

                    } else {

                        x[i] = undefined;

                        y[i] = undefined;

                    }

                }

            }





        source.change.emit();

        """)



# callback for searchbar

keyword_callback = CustomJS(args=dict(p=p, source=source), code="""

            

            var text_value = cb_obj.value;

            var data = source.data; 

            

            x = data['x'];

            y = data['y'];

            

            x_backup = data['x_backup'];

            y_backup = data['y_backup'];

            

            abstract = data['abstract'];

            titles = data['titles'];

            authors = data['authors'];

            journal = data['journal'];



            for (i = 0; i < x.length; i++) {

                if(abstract[i].includes(text_value) || 

                   titles[i].includes(text_value) || 

                   authors[i].includes(text_value) || 

                   journal[i].includes(text_value)) {

                    x[i] = x_backup[i];

                    y[i] = y_backup[i];

                } else {

                    x[i] = undefined;

                    y[i] = undefined;

                }

            }

            





        source.change.emit();

        """)



# option

option = RadioButtonGroup(labels=["C-0", "C-1", "C-2",

                                  "C-3", "C-4", "C-5",

                                  "C-6", "C-7", "C-8",

                                  "C-9", "C-10", "C-11",

                                  "C-12", "C-13", "C-14",

                                  "C-15", "C-16", "C-17",

                                  "C-18", "C-19", "C-20", "C-21",

                                  "C-22", "C-22", "C-23", "C-24",

                                  "C-25", "C-26", "C-27", "C-28",

                                  "C-20", "C-29", "C-30", "C-31",

                                  "C-32", "C-33", "C-34", "C-35",

                                  "C-36", "C-37", "C-38", "C-39",

                                  "C-40", "All"], 

                          active=40, callback=callback)



# search box

keyword = TextInput(title="Search:", callback=keyword_callback)



#header

header = Div(text="""<h1>COVID-19 Research Papers Interactive Cluster Map</h1>""")



# show

show(column(header, widgetbox(option, keyword),p))

# Vectorizing with plain text and TD-IDF



vectorizer = TfidfVectorizer(max_features=2**12)

X1 = vectorizer.fit_transform(df_covid['abstract'].values)
# Dimension reduction



tsne = TSNE(verbose=1, perplexity = 10, metric = 'cosine', early_exaggeration = 20, learning_rate = 300, random_state = 42)

X_embedded1 = tsne.fit_transform(X1.toarray())
# THIS WILL TAKE A LONG, LONG TIME. Go watch some series. Go call your family. Catch up with your friends.



# Building the clustering model and calculating the values of the Distortion and Inertia



distortions = [] 

inertias = [] 

mapping1 = {} 

mapping2 = {} 

K = range(1,26) 



for k in tqdm.tqdm(K): 

    #Building and fitting the model 

    kmeanModel = KMeans(n_clusters=k).fit(X1.toarray()) 

    kmeanModel.fit(X1.toarray())     



    distortions.append(sum(np.min(cdist(X1.toarray(), kmeanModel.cluster_centers_, 

                          'euclidean'),axis=1)) / X1.toarray().shape[0]) 

    inertias.append(kmeanModel.inertia_)



    mapping1[k] = sum(np.min(cdist(X1.toarray(), kmeanModel.cluster_centers_, 

                     'euclidean'),axis=1)) / X1.toarray().shape[0] 

    mapping2[k] = kmeanModel.inertia_ 

    time.sleep(0.1)
# List of number of clusters and the decrease of value, this helps to see exactly where the elbow is flexing



for key,val in mapping1.items(): 

    print(str(key)+' : '+str(val.round(4)))
# Plotting the elbow graph



plt.plot(K, distortions, 'bx-') 

plt.xlabel('Values of K') 

plt.ylabel('Distortion') 

plt.title('The Elbow Method using Distortion') 

plt.show()
# determining the best number of clusters for TD IDF



k = 21

kmeans = MiniBatchKMeans(n_clusters=k)

y_pred1 = kmeans.fit_predict(X1)

y1 = y_pred1
# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", len(set(y1)))



# plot

sns.scatterplot(X_embedded1[:,0], X_embedded1[:,1], hue=y1, legend='full', palette=palette)

plt.title("t-SNE Covid-19 Articles - Clustered(K-Means) - Tf-idf with Plain Text")

# plt.savefig("plots/t-sne_covid19_label_TFID.png")

plt.show()


output_notebook()

y_labels = y_pred1



# data sources

source = ColumnDataSource(data=dict(

    x= X_embedded1[:,0], 

    y= X_embedded1[:,1],

    x_backup = X_embedded1[:,0],

    y_backup = X_embedded1[:,1],

    desc= y_labels, 

    titles= df_covid['title'],

    authors = df_covid['authors'],

    journal = df_covid['journal'],

    abstract = df_covid['abstract_summary'],

    labels = ["C-" + str(x) for x in y_labels]

    ))



# hover over information

hover = HoverTool(tooltips=[

    ("Title", "@titles{safe}"),

    ("Author(s)", "@authors"),

    ("Journal", "@journal"),

    ("Abstract", "@abstract{safe}"),

],

                 point_policy="follow_mouse")



# map colors

mapper = linear_cmap(field_name='desc', 

                     palette=Category20[20],

                     low=min(y_labels) ,high=max(y_labels))



# prepare the figure

p = figure(plot_width=800, plot_height=800, 

           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 

           title="t-SNE Covid-19 Articles, Clustered(K-Means), Tf-idf with Plain Text", 

           toolbar_location="right")



# plot

p.scatter('x', 'y', size=5, 

          source=source,

          fill_color=mapper,

          line_alpha=0.3,

          line_color="black",

          legend = 'labels')



# add callback to control 

callback = CustomJS(args=dict(p=p, source=source), code="""

            

            var radio_value = cb_obj.active;

            var data = source.data; 

            

            x = data['x'];

            y = data['y'];

            

            x_backup = data['x_backup'];

            y_backup = data['y_backup'];

            

            labels = data['desc'];

            

            if (radio_value == '20') {

                for (i = 0; i < x.length; i++) {

                    x[i] = x_backup[i];

                    y[i] = y_backup[i];

                }

            }

            else {

                for (i = 0; i < x.length; i++) {

                    if(labels[i] == radio_value) {

                        x[i] = x_backup[i];

                        y[i] = y_backup[i];

                    } else {

                        x[i] = undefined;

                        y[i] = undefined;

                    }

                }

            }





        source.change.emit();

        """)



# callback for searchbar

keyword_callback = CustomJS(args=dict(p=p, source=source), code="""

            

            var text_value = cb_obj.value;

            var data = source.data; 

            

            x = data['x'];

            y = data['y'];

            

            x_backup = data['x_backup'];

            y_backup = data['y_backup'];

            

            abstract = data['abstract'];

            titles = data['titles'];

            authors = data['authors'];

            journal = data['journal'];



            for (i = 0; i < x.length; i++) {

                if(abstract[i].includes(text_value) || 

                   titles[i].includes(text_value) || 

                   authors[i].includes(text_value) || 

                   journal[i].includes(text_value)) {

                    x[i] = x_backup[i];

                    y[i] = y_backup[i];

                } else {

                    x[i] = undefined;

                    y[i] = undefined;

                }

            }

            





        source.change.emit();

        """)



# option

option = RadioButtonGroup(labels=["C-0", "C-1", "C-2",

                                  "C-3", "C-4", "C-5",

                                  "C-6", "C-7", "C-8",

                                  "C-9", "C-10", "C-11",

                                  "C-12", "C-13", "C-14",

                                  "C-15", "C-16", "C-17",

                                  "C-18", "C-19", "C-20", "C-21",

                                  "C-22", "C-22", "C-23", "C-24",

                                  "C-25", "C-26", "C-27", "C-28",

                                  "C-20", "C-29", "C-30", "C-31",

                                  "C-32", "C-33", "C-34", "C-35",

                                  "C-36", "C-37", "C-38", "C-39",

                                  "C-40", "All"], 

                          active=40, callback=callback)



# search box

keyword = TextInput(title="Search:", callback=keyword_callback)



#header

header = Div(text="""<h1>COVID-19 Research Papers Interactive Cluster Map</h1>""")



# show

show(column(header, widgetbox(option, keyword),p))