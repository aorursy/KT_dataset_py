# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier

import pickle

from numpy import hstack

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score

import pandas as pd

import random, os

from pytorch_transformers import AdamW, WarmupLinearSchedule

from tqdm import tqdm, trange

from torch import nn

import torch

import numpy

from sklearn.preprocessing import LabelEncoder

from string import digits

from stemming.porter2 import stem

from gensim import corpora, models, similarities

import nltk

from transformers import BertTokenizer, BertModel

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')
def seed_torch(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    numpy.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

seed_torch(0)
lda = None

dictionary = None
vocab = pickle.load(open('/kaggle/input/localdata/vocab.pickle', 'rb'))

vectorizer = CountVectorizer(vocabulary=vocab)

labels = pickle.load(open('/kaggle/input/localdata/labels.pickle', 'rb'))
def load_LDA():

    lda = models.LdaModel.load('/kaggle/input/ldafiles/all_papers_model_sample.lda')

    dictionary = corpora.dictionary.Dictionary.load_from_text('/kaggle/input/ldafiles/all_papers_text.dict')

    return lda, dictionary



def stemall(documents):

    remove_digits = str.maketrans('', '', digits)

    return ([ [ stem(word) for word in line.translate(remove_digits).lower().split(" ")] for line in documents ])



def get_features_from_LDA(corpus):

    lda_features = []

    stem_docs = stemall(corpus)

    for doc in stem_docs:

        doc_lda = lda.get_document_topics(dictionary.doc2bow(doc))

        res = numpy.array(list(zip(*doc_lda))[1])

        lda_features.append(res)

        

    features_df = pd.DataFrame(lda_features).fillna(0)

    return features_df.to_numpy()
def pipeline(model, data, vectorizer, train=True):

    global lda, dictionary

    

    # sentece2features

    corpus = data['Sentence'].tolist()

    

    if lda == None:

        lda, dictionary = load_LDA()

    # count based feature

    vec_corpus = vectorizer.transform(corpus).todense()

    

    # LDA + vec_corpus

    lda_features = get_features_from_LDA(corpus)

    print(vec_corpus.shape, lda_features.shape)

    vec_corpus = hstack([vec_corpus, lda_features])

    

    predict_proba = model.predict_proba(vec_corpus)

    return predict_proba
trained_models = list()

model_location = '/kaggle/input/non-neural-network-model'

names = ['b_clf', 'rf_clf', 'et_clf', 'lr_clf', 'dt_clf', 'xgb_clf', 'meta_clf']

for m in names:

    m_ = pickle.load(open('{}/{}.bin'.format(model_location, m), 'rb'))

    trained_models.append(m_)
def get_prediction_from_ensemble(trained_models, vectorizer, df):

    predictions = list()

    for m in trained_models[:-1]:

        m_prediction = pipeline(m, df, vectorizer, train=False)

        predictions.append(m_prediction)



    meta_model = trained_models[-1]

    X_meta = hstack(predictions)

    y_pred = meta_model.predict(X_meta)

    return y_pred
df_val_adj = pickle.load(open('/kaggle/input/localdata/val_ajudicated.pickle', 'rb'))

true_val_adj = df_val_adj['Label']

y_pred_val_adj = get_prediction_from_ensemble(trained_models, vectorizer, df_val_adj)

acc = accuracy_score(true_val_adj, y_pred_val_adj)

print('Accuracy : %.3f' % (acc*100))
class SciBertClassification(torch.nn.Module):

    def __init__(self, base_model, nb_classes, tokenizer):

        super(SciBertClassification, self).__init__()

        

        self.scibert = base_model

        self.output = torch.nn.Linear(768, nb_classes)

        self.tokenizer = tokenizer

       

    def forward(self, text_sentences):

        representation = [] 

        for sentence in text_sentences:

            input_ids = torch.tensor(self.tokenizer.encode(sentence)).unsqueeze(0)

            outputs = self.scibert(input_ids)

            last_hidden_states = outputs[0][:, -1, :]

            output_logits = self.output(last_hidden_states)

            representation.append(output_logits)

            

        b_logits = torch.cat(representation, dim=0)

        return b_logits
import torch

from torch.utils import data



class Dataset_scibert(data.Dataset):

  def __init__(self, sentences, labels):

        self.labels = labels

        self.sentences = sentences



  def __len__(self):

        'Denotes the total number of samples'

        return len(self.sentences)



  def __getitem__(self, index):

        X = self.sentences[index]

        y = self.labels[index]

        

        return X, y
model_version = 'scibert_scivocab_uncased'

do_lower_case = True

Bert = BertModel.from_pretrained('/kaggle/input/scibert-scivocab-uncased/')

tokenizer = BertTokenizer.from_pretrained('/kaggle/input/scibert-scivocab-uncased', do_lower_case=do_lower_case)
encoder = LabelEncoder()

encoder.fit(labels)
model = SciBertClassification(Bert, len(encoder.classes_), tokenizer)

model.load_state_dict(torch.load('/kaggle/input/fine-tuned-scibert-clf/model.bin', map_location=torch.device('cpu')))

model.eval()

print('Model loaded')
def predict(model, prediction_dataloader):

    model.eval()

    sentences = {}

    counter = 0

    predictions , true_labels, indices = [], [], []

    for batch in prediction_dataloader:

        b_sentences, b_labels = batch

        

        for s in b_sentences:

            sentences[counter] = s

            counter = counter + 1

            

        with torch.no_grad():

            logits = model(b_sentences)



        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()



        predictions.append(logits)

        true_labels.append(label_ids)

        

    predictions = [item for sublist in predictions for item in sublist]

    true_labels = [item for sublist in true_labels for item in sublist]    

    sentences = [sentences[i] for i in range(len(sentences))]

    return predictions, true_labels, sentences
val_adj = pickle.load(open('/kaggle/input/localdata/val_ajudicated.pickle','rb'))

val_adj.head()

print(val_adj.Label.unique())
params = {'batch_size': 64,

          'shuffle': True,

          'num_workers': 6}



validation_adj_set = Dataset_scibert(val_adj['Sentence'], encoder.transform(val_adj['Label']))

validation_adj_generator = data.DataLoader(validation_adj_set, **params)

#evaluate(model, validation_adj_generator, encoder)
nn_predictions_val_adj, true_labels_val_adj, val_adj_sentences = predict(model, validation_adj_generator)

true_labels_text_val_adj = [encoder.classes_[i] for i in true_labels_val_adj]

nn_predictions_val_adj_text_label = [encoder.classes_[numpy.argmax(p)] for p in nn_predictions_val_adj]
from sklearn.metrics import confusion_matrix

import seaborn as sns



cnf = confusion_matrix(true_labels_text_val_adj, nn_predictions_val_adj_text_label, labels=val_adj['Label'].unique().tolist())

a = numpy.around(cnf)

print(a)

ax = sns.heatmap(a, annot=True,fmt = '.0f')

ax.set_xticklabels(labels=val_adj['Label'].unique().tolist(), rotation=90) #, ylabel=encoder.classes_)

ax.set_yticklabels(labels=val_adj['Label'].unique().tolist(), rotation=0) #, ylabel=encoder.classes_)

ax.set_ylim(len(val_adj['Label'].unique().tolist())+0.5, -0.5)
print(classification_report(true_labels_text_val_adj, nn_predictions_val_adj_text_label))
import numpy as np

import pandas as pd

import pickle

from subprocess import check_output

import os



# bokeh packages

#To create intractive plot we need this to add callback method.

#This is for creating layout

#from bokeh.models import CustomJS 

#from bokeh.layouts import column

#from bokeh.io import output_file, show, output_notebook, push_notebook

#from bokeh.plotting import *

#from bokeh.models import ColumnDataSource, Div, Select, Button, ColorBar, CustomJS, DataTable, DateFormatter, TableColumn

#from bokeh.layouts import row,column,gridplot,widgetbox, layout

#from bokeh.models.widgets import Tabs,Panel

#from bokeh.transform import cumsum, linear_cmap

#from bokeh.palettes import Blues8

#from bokeh.io import  output_notebook, show

#from bokeh.models import ColumnDataSource

#from bokeh.palettes import Spectral6

#from bokeh.plotting import figure



from ipywidgets import interact

import ipywidgets as widgets

from ipywidgets import interactive

import plotly.graph_objects as go

import os



#output_notebook()

import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_colwidth', 0)
def make_clickable_multi(url, name):

    return '<a href="http://doi.org/{}" target="_blank">{}</a>'.format(url,name)



def temp(title):

    return title
data_dir="/kaggle/input/localdata"

filename = os.path.join(data_dir,'jan_feb_march_scibert_predictions.csv')

base_df = pd.read_csv(filename)



base_df['year']= pd.to_datetime(base_df['year'], errors='coerce')

base_df['year'] = base_df['year'].dt.year



df = base_df[['title','authors','sentence', 'year', 'source_x', 'journal',  

       'predicted_label', 'doi']]





df['sentence']=df['sentence'].str.replace('`', '', regex=True)

df=df.replace(to_replace= r'\\', value= '', regex=True)

df['journal']= df['journal'].replace(np.nan, '', regex=True)



df = df[df['predicted_label'] != 'Irrevlant Sentence']



df = df[['title','authors','sentence', 'year', 'source_x', 'journal',  

       'predicted_label', 'doi']]



df['url'] = ['http://doi.org/'+str(i) for i in df['doi'].values.tolist()]

df = df[['title','authors','sentence', 'year', 'source_x', 'journal',  

       'predicted_label', 'url']]
df['predicted_label'].value_counts()


"""

Copyright 2019, Marek Cermak



Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:



The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""



def init_datatable_mode():

    """Initialize DataTable mode for pandas DataFrame represenation."""

    import pandas as pd

    from IPython.core.display import display, Javascript



    # configure path to the datatables library using requireJS

    # that way the library will become globally available

    display(Javascript("""

        require.config({

            paths: {

                DT: '//cdn.datatables.net/1.10.19/js/jquery.dataTables.min',

            }

        });



        $('head').append('<link rel="stylesheet" type="text/css" href="//cdn.datatables.net/1.10.19/css/jquery.dataTables.min.css">');

    """))



    def _repr_datatable_(self):

        """Return DataTable representation of pandas DataFrame."""

        # classes for dataframe table (optional)

        classes = ['table', 'table-striped', 'table-bordered']



        # create table DOM

        script = (

            f'$(element).html(`{self.to_html(index=False, classes=classes)}`);\n'

        )



        # execute jQuery to turn table into DataTable

        script += """

            require(["DT"], function(DT) {

                $(document).ready( () => {

                    // Turn existing table into datatable

                    $(element).find("table.dataframe").DataTable();

                })

            });

        """



        return script



    pd.DataFrame._repr_javascript_ = _repr_datatable_
def get_publications(df):

    df_paper = df.copy(deep=True)

    df_paper = df_paper[['title','authors','year','source_x','journal', 'url']]

    df_paper.drop_duplicates(inplace=True)

    df_paper['Clinical Topic(s)'] =pd.Series(['NA' for _ in range(len(df_paper))])

    groups = df.groupby(['title']).groups

    for k,v in groups.items():

        try:

            lb = df.iloc[v.tolist()]['predicted_label'].unique().tolist()

            topics =  ', '.join(x for x in lb)

            df_paper.at[df_paper['title']==k,'Clinical Topic(s)'] = topics

        except:

            continue

            

    return df_paper



def compute_pub_stats(df):

    # Bar Chart with 8 classes (count papers, sentences)

    labels = df.predicted_label.unique().tolist()

    stat={}

    for lb in labels:

        stat[lb]={'sent_count':df[df.predicted_label==lb].shape[0],'paper_count':df[df.predicted_label==lb]['title'].unique().shape[0]}

    return stat
def show_publication_stats(stats):

    

    labels = list(stats.keys())

#     print(labels)



   

    sentences = [stats[s]['sent_count'] for s in labels]

    papers = [stats[s]['paper_count'] for s in labels]

    fig = go.Figure()

    fig.add_trace(go.Bar(x=labels,

                    y=sentences,

                    name='# Sentences',

#                     marker_color='rgb(55, 83, 109)'

                    ))

#     fig.add_trace(go.Bar(x=labels,

#                     y=papers,

#                     name='# Papers',

# #                     marker_color='rgb(26, 118, 255)'

#                     ))



    fig.update_layout(

        title='Overall Sentences Statistics',

        xaxis_tickfont_size=14,

        yaxis=dict(

            title='Counts',

            titlefont_size=16,

            tickfont_size=14,

        ),

        legend=dict(

            x=0,

            y=1.0,

#             bgcolor='rgba(255, 255, 255, 0)',

            bordercolor='rgba(255, 255, 255, 0)'

        ),

        barmode='group',

        bargap=0.15, # gap between bars of adjacent location coordinates.

        bargroupgap=0.1 # gap between bars of the same location coordinate.

    )

    fig.show()
def show_paper_counts(stats):

    

    labels = list(stats.keys())

   

    #title=''Overall Paper Statistics'

   

    sentences = [stats[s]['sent_count'] for s in labels]

    papers = [stats[s]['paper_count'] for s in labels]

    fig = go.Figure()

#     fig.add_trace(go.Bar(x=labels,

#                     y=sentences,

#                     name='# Sentences',

# #                     marker_color='rgb(55, 83, 109)'

#                     ))

    fig.add_trace(go.Bar(x=labels,

                    y=papers,

                    name='# Papers',

                    marker_color='indianred'

                    ))



    fig.update_layout(

        title='Overall Paper Statistics',

        xaxis_tickfont_size=10,

        yaxis=dict(

            title='Number of Papers',

            titlefont_size=10,

            tickfont_size=10,

        ),

        legend=dict(

            x=0.0,

            y=1.0,

            bgcolor='rgba(255, 255, 255, 0)',

            bordercolor='rgba(255, 255, 255, 0)'

        ),

        barmode='group',

        bargap=0.15, # gap between bars of adjacent location coordinates.

        bargroupgap=0.1 # gap between bars of the same location coordinate.

    )

    fig.show()

    

    

    '''

    #title='Overall Sentences Statistics',



    labels = list(stats.keys())

#     print(labels)



   

    sentences = [stats[s]['sent_count'] for s in labels]

    papers = [stats[s]['paper_count'] for s in labels]

    fig = go.Figure()

    fig.add_trace(go.Bar(x=labels,

                    y=sentences,

                    name='# Sentences',

                    marker_color='rgb(55, 83, 109)'

                    ))



    #fig.add_trace(go.Bar(x=labels,

    #                 y2=papers,

    #                 name='# Papers',

    #                 marker_color='rgb(26, 118, 255)'

    #                 ))

        fig.update_layout(

        title='Overall Sentences Statistics',

        xaxis_tickfont_size=14,

        yaxis=dict(

            title='Counts',

            titlefont_size=16,

            tickfont_size=14,

        ),

        legend=dict(

            x=0,

            y=1.0,

            #bgcolor='rgba(255, 255, 255, 0)',

            bordercolor='rgba(255, 255, 255, 0)'

        ),

        barmode='group',

        bargap=0.15, # gap between bars of adjacent location coordinates.

        bargroupgap=0.1 # gap between bars of the same location coordinate.

    )

    fig.show()

    '''
def get_filtered_values(df_paper):

    labels = [(x,x) for x in df_paper['Clinical Topic(s)'].unique()]

    labels.insert(0,('All','All'))

    publish_year_min= min(df_paper.year.unique().tolist())

    publish_year_max= max(df_paper.year.unique().tolist())



    #publish_year.insert(0,('All','All'))



    #Source

    source = [(x,x) for x in df_paper.source_x.unique()]

    source.insert(0,('All','All'))

    #country

    #country = [(x,x) for x in df_paper.country.unique()]

    return labels,publish_year_min, publish_year_max,source

def get_filtered_values_details(df_paper):

    labels = [(x,x) for x in df_paper['journal'].unique()]

    labels.insert(0,('All','All'))

    publish_year_min= min(df_paper.year.unique().tolist())

    publish_year_max= max(df_paper.year.unique().tolist())



    #publish_year.insert(0,('All','All'))



    #Source

    source = [(x,x) for x in df_paper.source_x.unique()]

    source.insert(0,('All','All'))



    #country

    journal = [(x,x) for x in df_paper.journal.unique()]

    return journal,publish_year_min, publish_year_max,source

def filter_publication(Topic,Year,Source):

    if Source=='All':

        tmp_df = pub_df

    else:

        tmp_df = pub_df[pub_df.source_x==Source]



    if Topic=='All':

        if Year=='All':

            return tmp_df

        else:

            return tmp_df[(tmp_df.year==Year)]

    else:

        if Year=='All':

            return tmp_df[(tmp_df['Clinical Topic(s)']==Topic)]

        else:

            return tmp_df[((tmp_df['Clinical Topic(s)']==Topic)&(tmp_df.year==Year))]
def filter_publication_details(Topic,Journal,Year,Source):

    

    tmp_df = df[df['predicted_label']==Topic] 

    

    if Source=='All':

        pass

    else:

        tmp_df = tmp_df[tmp_df.source_x==Source]





    if Topic=='All':

        if Year=='All':

            return tmp_df

        else:

            return tmp_df[(tmp_df.year==Year)]

    else:

        if Year=='All':

            return tmp_df[(tmp_df['journal']==Journal)]

        else:

            return tmp_df[((tmp_df['journal']==Journal)&(tmp_df.year==Year))]
def filter_df(label,pyear,source):

    if journal!=None:

        tmp_df = df[(df.source_x==source)]

    else:

        tmp_df=df

        

    return 

    if label=='All':

        if pyear=='All':

            return tmp_df

        else:

            return tmp_df[(tmp_df.year==pyear)]

    else:

        if pyear=='All':

            return tmp_df[(tmp_df.predicted_label==label)]

        else:

            return tmp_df[((tmp_df.predicted_label==label)&(tmp_df.year==pyear))]
def display_label_wise_data(label,pyear,source):

    

    tmp_df = df[((df.source_x==source)&(df.predicted_label==label))]

    

    if pyear=='All':

        tmp_df = tmp_df[(tmp_df.predicted_label==label)]

#         tmp_df =tmp_df[['sentence','title','authors','country','year','journal','source_x',]]

#         return tmp_df

    else:

        tmp_df = tmp_df[((tmp_df.predicted_label==label)&(tmp_df.year==pyear))]

    tmp_df =tmp_df[['sentence','title','authors','url','year','journal','source_x',]]

    return tmp_df

def show_wordcloud(df, annot):

    df['sentence'] =  df['sentence'].astype(str)

    text = df[df['predicted_label']==annot]['sentence'].values 

    if len(text) < 1:

        print('No sentence found to form word cloud.')

        return 

    

    plt.figure(figsize=(8,8))

    wordcloud = WordCloud().generate(str(text))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()
#init_datatable_mode()

pub_df = get_publications(df)

stats = compute_pub_stats(df)

labels= stats.keys()
labels = list(stats.keys())

   

sentences = [stats[s]['sent_count'] for s in labels]

papers = [stats[s]['paper_count'] for s in labels]



tmp_s = pd.DataFrame({'labels': labels, 'sentences': sentences})

tmp_p = pd.DataFrame({'labels': labels, 'papers': papers})
ax = tmp_s.plot.bar(x='labels', y='sentences', rot=90)
#show_paper_counts(stats)

ax = tmp_p.plot.bar(x='labels', y='papers', rot=90)
pd.set_option('display.max_rows', 100)
labels,publish_year_min, publish_year_max,source = get_filtered_values(pub_df)

widget=interact(filter_publication, df=pub_df, Topic=labels,Year=(publish_year_min, publish_year_max,1),Source=source)
show_wordcloud(df, 'Death')
#labels,publish_year_min, publish_year_max,source = get_filtered_values(pub_df)

journal,publish_year_min, publish_year_max,source = get_filtered_values_details(df)

widget=interact(filter_publication_details, df=df, Topic=[('Death', 'Death')],Journal=journal, Year=(publish_year_min, publish_year_max,1),Source=source)
show_wordcloud(df, 'CriticallyIll')
#labels,publish_year_min, publish_year_max,source = get_filtered_values(pub_df)

journal,publish_year_min, publish_year_max,source = get_filtered_values_details(df)

widget=interact(filter_publication_details, df=df, Topic=[('CriticallyIll', 'CriticallyIll')],Journal=journal, Year=(publish_year_min, publish_year_max,1),Source=source)
show_wordcloud(df, 'Extrapulmonary_Manifestations')
journal,publish_year_min, publish_year_max,source = get_filtered_values_details(df)

widget=interact(filter_publication_details, df=df, Topic=[('Extrapulmonary_Manifestations', 'Extrapulmonary_Manifestations')],Journal=journal, Year=(publish_year_min, publish_year_max,1),Source=source)
show_wordcloud(df, 'Recovery')
journal,publish_year_min, publish_year_max,source = get_filtered_values_details(df)

widget=interact(filter_publication_details, df=df, Topic=[('Recovery', 'Recovery')],Journal=journal, Year=(publish_year_min, publish_year_max,1),Source=source)
show_wordcloud(df, 'Organ Damage and Needs Specialist')
journal,publish_year_min, publish_year_max,source = get_filtered_values_details(df)

widget=interact(filter_publication_details, df=df, Topic=[('Organ Damage and Needs Specialist', 'Organ Damage and Needs Specialist')],Journal=journal, Year=(publish_year_min, publish_year_max,1),Source=source)
show_wordcloud(df, 'MedicalJourney')
journal,publish_year_min, publish_year_max,source = get_filtered_values_details(df)

widget=interact(filter_publication_details, df=df, Topic=[('MedicalJourney', 'MedicalJourney')],Journal=journal, Year=(publish_year_min, publish_year_max,1),Source=source)