# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.



# Python libraries

# Classic,data manipulation and linear algebra

import pandas as pd

import numpy as np



# Plots

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.tools as tls

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import squarify



# Data processing, metrics and modeling

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score

import lightgbm as lgbm

from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve,auc

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_predict

from yellowbrick.classifier import DiscriminationThreshold



# Stats

import scipy.stats as ss

from scipy import interp

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform



# Time

from contextlib import contextmanager

@contextmanager

def timer(title):

    t0 = time.time()

    yield

    print("{} - done in {:.0f}s".format(title, time.time() - t0))



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore') 



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/superheroes-nlp-dataset/superheroes_nlp_dataset.csv")

data.head(2)
display(data.info(),data.head())
data['name'] = data['name'].astype(str)

data['real_name'] = data['real_name'].astype(str)

data['full_name'] = data['full_name'].astype(str)

data['history_text'] = data['history_text'].astype(str)

data['powers_text'] = data['powers_text'].astype(str)
# 2 datasets

D = data[(data['creator'] == "DC Comics")]

M = data[(data['creator'] == "Marvel Comics")]

data['creator'].loc[(data['creator']!="DC Comics") & (data['creator']!="Marvel Comics")]  = "Others"





#------------COUNT-----------------------

def creator_count():

    trace = go.Bar( x = data['creator'].value_counts().values.tolist(), 

                    y = ['Marvel Comics','DC Comics', 'Others'],

                    orientation = 'h', 

                    text=data['creator'].value_counts().values.tolist(), 

                    textfont=dict(size=15),

                    textposition = 'auto',

                    opacity = 0.8,marker=dict(

                    color=['red', 'blue','orchid'],

                    line=dict(color='#000000',width=1.5)))



    layout = dict(title =  'Count of Creator of Superheroes')



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)



def creator_percent():

    trace = go.Pie(labels = ['Marvel Comics','DC Comics', 'Others'], values = data['creator'].value_counts(), 

                   textfont=dict(size=15), opacity = 0.8,

                   marker=dict(colors=['red', 'blue','orchid'], 

                               line=dict(color='#000000', width=1.5)))





    layout = dict(title =  'Distribution of Creator of Superheroes')



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)
creator_count()

creator_percent()
# 2 datasets

M = data[(data['gender'] == "Male")]

W = data[(data['gender'] == "Female")]



#------------COUNT-----------------------

def gender_count():

    trace = go.Bar( x = data['gender'].value_counts().values.tolist(), 

                    y = ['Male','Female' ], 

                    orientation = 'h', 

                    text=data['gender'].value_counts().values.tolist(), 

                    textfont=dict(size=15),

                    textposition = 'auto',

                    opacity = 0.8,marker=dict(

                    color=['gold', 'deeppink'],

                    line=dict(color='#000000',width=1.5)))



    layout = dict(title =  'Count of Gender of Superheroes')



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)



#------------PERCENTAGE-------------------

def gender_percent():

    trace = go.Pie(labels = ['Male','Female'], values = data['gender'].value_counts(), 

                   textfont=dict(size=15), opacity = 0.8,

                   marker=dict(colors=['gold', 'deeppink'], 

                               line=dict(color='#000000', width=1.5)))





    layout = dict(title =  'Distribution of Gender of Superheroes')



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)
gender_count()

gender_percent()
W.sort_values('overall_score',ascending=False).head(10)
superpowers = data.loc[:, data.columns.str.startswith('has')].dropna()

superpowers.columns = superpowers.columns.str.replace(r'has_', '')

superpowers = superpowers.T.reset_index()

superpowers['Total'] = superpowers.sum(axis=1)

superpowers = superpowers.sort_values('Total',ascending=False)

superpowers.head(1)
plt.style.use('ggplot') # Using ggplot2 style visuals



fig, ax = plt.subplots()



fig.set_size_inches(20, 10)



sns.set_context("paper", font_scale=1.5)

f=sns.barplot(x=superpowers['index'], y=superpowers['Total'], data=superpowers)

f.set_xlabel("Name of Superpower",fontsize=18)

f.set_ylabel("No. of Superheroes with Superpower",fontsize=18)

f.set_title('Superpowers')

for item in f.get_xticklabels():

    item.set_rotation(90)
data_x = data[['intelligence_score','strength_score','speed_score','durability_score','power_score','combat_score']]
plt.style.use('ggplot') # Using ggplot2 style visuals 



f, ax = plt.subplots(figsize=(11, 15))



ax.set_facecolor('#fafafa')

ax.set(xlim=(-.05, 200))

plt.ylabel('Variables')

plt.title("Overview  of the Power Scores")

ax = sns.boxplot(data = data_x, 

  orient = 'h', 

  palette = 'Set2')
def correlation_plot():

    #correlation

    correlation = data_x.corr()

    #tick labels

    matrix_cols = correlation.columns.tolist()

    #convert to array

    corr_array  = np.array(correlation)

    trace = go.Heatmap(z = corr_array,

                       x = matrix_cols,

                       y = matrix_cols,

                       colorscale='Viridis',

                       colorbar   = dict() ,

                      )

    layout = go.Layout(dict(title = 'Correlation Matrix for variables',

                            #autosize = False,

                            #height  = 1400,

                            #width   = 1600,

                            margin  = dict(r = 0 ,l = 100,

                                           t = 0,b = 100,

                                         ),

                            yaxis   = dict(tickfont = dict(size = 9)),

                            xaxis   = dict(tickfont = dict(size = 9)),

                           )

                      )

    fig = go.Figure(data = [trace],layout = layout)

    py.iplot(fig)
correlation_plot()
data_y = data.drop(['overall_score','intelligence_score','strength_score','speed_score','durability_score','power_score','combat_score'],axis=1)

data.loc[:, 'total_superpowers'] = data_y.iloc[:, 1:].sum(axis=1)
data_powers_alignment=data[['name','total_superpowers','alignment','creator']].sort_values('total_superpowers',ascending=False)

data_powers_alignment.head(1)
plt.style.use('ggplot') # Using ggplot2 style visuals



fig, ax = plt.subplots()



fig.set_size_inches(20, 10)



sns.set_context("paper", font_scale=1.5)

f=sns.barplot(x=data_powers_alignment["name"].head(30), y=data_powers_alignment['total_superpowers'].head(30), data=data_powers_alignment)

f.set_xlabel("Name of Superhero",fontsize=18)

f.set_ylabel("No. of Superpowers",fontsize=18)

f.set_title('Top 30 Superheroes having highest no. powers')

for item in f.get_xticklabels():

    item.set_rotation(90)
plt.style.use('ggplot') # Using ggplot2 style visuals



fig, ax = plt.subplots()



fig.set_size_inches(20, 10)



sns.set_context("paper", font_scale=1.5)

f=sns.swarmplot(x=data_powers_alignment["creator"], y=data_powers_alignment['total_superpowers'],hue=data_powers_alignment["alignment"],data=data_powers_alignment)

f.set_xlabel("Comics",fontsize=18)

f.set_ylabel("No. of Superpowers",fontsize=18)

f.set_title('Distirbution of Good/Bad Superheroes, their creators and their Superpower')

for item in f.get_xticklabels():

    item.set_rotation(90)
data_powers_marvel = data_powers_alignment.loc[data_powers_alignment['creator'] == "Marvel Comics"]





plt.style.use('ggplot') # Using ggplot2 style visuals



fig, ax = plt.subplots()



fig.set_size_inches(20, 10)



sns.set_context("paper", font_scale=1.5)

f=sns.barplot(x=data_powers_marvel["name"].head(30), y=data_powers_marvel['total_superpowers'].head(30), data=data_powers_marvel)

f.set_xlabel("Name of Superhero",fontsize=18)

f.set_ylabel("No. of Superpowers",fontsize=18)

f.set_title('Top 30 Superheroes from Marvel Comics having highest no. powers')

for item in f.get_xticklabels():

    item.set_rotation(90)
data_powers_dc = data_powers_alignment.loc[data_powers_alignment['creator'] == "DC Comics"]





plt.style.use('ggplot') # Using ggplot2 style visuals



fig, ax = plt.subplots()



fig.set_size_inches(20, 10)



sns.set_context("paper", font_scale=1.5)

f=sns.barplot(x=data_powers_dc["name"].head(30), y=data_powers_dc['total_superpowers'].head(30), data=data_powers_dc)

f.set_xlabel("Name of Superhero",fontsize=18)

f.set_ylabel("No. of Superpowers",fontsize=18)

f.set_title('Top 30 Superheroes from DC Comics having highest no. powers')

for item in f.get_xticklabels():

    item.set_rotation(90)
data['total_score'] = data['intelligence_score'] + data['strength_score'] + data['speed_score'] + data['durability_score'] + data['power_score']+data['combat_score']
data.sort_values(['overall_score','total_score','total_superpowers'], ascending=[False,False, False]).head(1)
plt.style.use('ggplot') # Using ggplot2 style visuals



fig, ax = plt.subplots()



fig.set_size_inches(20, 10)



sns.set_context("paper", font_scale=1.5)

f=sns.swarmplot(x=data["creator"], y=data['total_score'],hue=data["alignment"],data=data)

f.set_xlabel("Comics",fontsize=18)

f.set_ylabel("Total Power Score",fontsize=18)

f.set_title('Distirbution of Good/Bad Superheroes, their creators and their Superpower Score')

for item in f.get_xticklabels():

    item.set_rotation(90)
data_good = data.sort_values(['overall_score', 'total_superpowers'], ascending=[False, False])

data_good.loc[data_good['alignment'] == 'Good'].head(1)
data_good[(data_good['alignment'] == 'Good')&(data_good['creator'] == 'Marvel Comics')].dropna().head(1)
data_good[(data_good['alignment'] == 'Good')&(data_good['creator'] == 'DC Comics')].dropna().head(1)
plt.style.use('ggplot') # Using ggplot2 style visuals





fig, ax = plt.subplots()



fig.set_size_inches(20, 10)



sns.set_context("paper", font_scale=1.5)

f = sns.countplot(x=data["type_race"], data=data, order=data.type_race.value_counts().index)

f.set_xlabel("Race of Superhero",fontsize=18)

f.set_ylabel("No. of Superheroes",fontsize=18)

f.set_title('Race of the Superheroes')

for item in f.get_xticklabels():

    item.set_rotation(90)
top_race = data.type_race.value_counts().head(15)

print(top_race)
top_race_names = ['Human','Mutant','God / Eternal','Metahuman','Alien','Animal','Demon','Android','Human / Radiation','Cyborg','Asgardian','Inhuman','Kryptonian','Demi-God','New God']

data_race = data[data['type_race'].isin(top_race_names)]
plt.style.use('ggplot') # Using ggplot2 style visuals





fig, ax = plt.subplots()



fig.set_size_inches(20, 10)



sns.set_context("paper", font_scale=1.5)

f=sns.barplot(x=data_race["type_race"],y=data_race["total_superpowers"],data=data_race,order=data_race.type_race.value_counts().index )

f.set_xlabel("Race of Superhero",fontsize=18)

f.set_ylabel("No. of Superpowers",fontsize=18)

f.set_title('Most Common Races v/s Number of Powers of a Superhero')

for item in f.get_xticklabels():

    item.set_rotation(90)
plt.style.use('ggplot') # Using ggplot2 style visuals





fig, ax = plt.subplots()



fig.set_size_inches(20, 10)



sns.set_context("paper", font_scale=1.5)

f=sns.barplot(x=data_race["type_race"],y=data_race["total_score"],data=data_race,order=data_race.type_race.value_counts().index )

f.set_xlabel("Race of Superhero",fontsize=18)

f.set_ylabel("Total Power Score",fontsize=18)

f.set_title('Most Common Races v/s Total Power Score of a Superhero')

for item in f.get_xticklabels():

    item.set_rotation(90)
data_race = data.groupby(['type_race'])['total_score'].mean().to_frame(name = 'mean_power_score').reset_index()

data_race = data_race.sort_values(by='mean_power_score', ascending=False)

data_race.head(5)
plt.style.use('ggplot') # Using ggplot2 style visuals





fig, ax = plt.subplots()



fig.set_size_inches(20, 10)



sns.set_context("paper", font_scale=1.5)

f=sns.barplot(x=data_race["type_race"],y=data_race["mean_power_score"],data=data_race )

f.set_xlabel("Race of Superhero",fontsize=18)

f.set_ylabel("Average Power Score",fontsize=18)

f.set_title('Most Powerful Races of Superheroes (based on powerscores)')

for item in f.get_xticklabels():

    item.set_rotation(90)
data_race = data.groupby(['type_race'])['total_superpowers'].mean().to_frame(name = 'mean_power_list').reset_index()

data_race = data_race.sort_values(by='mean_power_list', ascending=False)

data_race.head(5)
plt.style.use('ggplot') # Using ggplot2 style visuals





fig, ax = plt.subplots()



fig.set_size_inches(20, 10)



sns.set_context("paper", font_scale=1.5)

f=sns.barplot(x=data_race["type_race"],y=data_race["mean_power_list"],data=data_race )

f.set_xlabel("Race of Superhero",fontsize=18)

f.set_ylabel("Average Number of Powers",fontsize=18)

f.set_title('Most Powerful Races of Superheroes (based on number of superpowers)')

for item in f.get_xticklabels():

    item.set_rotation(90)
import spacy

from collections import Counter



def history_text_processing(history_text):

    frequency = {}

    list_of_entities = []

    listofmax = []

    most_common_key = " "

    max_key = "null"

    nlp = spacy.load("en_core_web_sm")

    doc = nlp(history_text)



    for ent in doc.ents:

        list_of_entities.append(str(ent.text))

    

    list = list_of_entities

    

    

    

    for item in list:

       if (item in frequency):

          frequency[item] += 1

       else:

          frequency[item] = 1



    if len(frequency) != 0:

        max_value = max(frequency.values())  # maximum value

        max_key = max(frequency, key=frequency.get) # getting key containing the `maximum`

    

    k = Counter(frequency)

    # Finding 3 highest values

    high = k.most_common(3)

    empty = []

    

    for i in high:

        empty.append(i[0])

    

    most_common = " ".join(empty)

    

    return str(most_common)
from tqdm.auto import tqdm

tqdm.pandas()



data['name_entity'] = data['history_text'].progress_apply(lambda x:history_text_processing(str(x)) if x != None else x )

    
data.head(1)
from difflib import SequenceMatcher

def similiarity_ratio(row,col1,col2):

    return SequenceMatcher(None, row[col1].lower(), row[col2].lower()).ratio() 

data['name_match'] = data.apply(lambda x:similiarity_ratio(x,col1='name_entity',col2='name'),axis=1)
print("Mean of the name_match :" + str(data['name_match'].mean()))

sum_of_name_match = (data['name_match'] > 0).values.sum()

percentage_matched = (sum_of_name_match/len(data))*100

print("Number of of name entites which has a match is in names  :" + str(sum_of_name_match))

print("Percentage of name entites which has a match is in names :" + str(percentage_matched))
plt.style.use('ggplot') # Using ggplot2 style visuals



fig, ax = plt.subplots()



fig.set_size_inches(20, 10)

colors = ['#FFD700', '#7EC0EE']



sns.set_context("paper", font_scale=1.5)

f = sns.distplot(data['name_match'], kde=True);

f.set_xlabel("Similarity Score",fontsize=18)

f.set_title('Distribution of Similarity Scores based Superhero name and Prediction')
data_text = data[['history_text', 'creator']]

#we will only select comics by Marvel or DC as there's too many comic creators

data_text = data_text.loc[data_text['creator'].isin(['Marvel Comics','DC Comics'])]

data_text.head(1)
# Importing necessary libraries

import string,re

import nltk

import gc

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

lemmatiser = WordNetLemmatizer()

# Defining modules for Text Processing



from nltk.corpus import stopwords

", ".join(stopwords.words('english'))

stopwords_list = set(stopwords.words('english'))





puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]



def clean_text(x):

    x = str(x)

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x



def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x



def remove_stopwords(text):

    return " ".join([word for word in str(text).split() if word not in stopwords_list])



def stem_text(text):    

    lemma = nltk.wordnet.WordNetLemmatizer()

    class FasterStemmer(object):

        def __init__(self):

            self.words = {}



        def stem(self, x):

            if x in self.words:

                return self.words[x]

            t = lemma.lemmatize(x)

            self.words[x] = t

            return t

    faster_stemmer = FasterStemmer()

    text = text.split()

    stemmed_words = [faster_stemmer.stem(word) for word in text]

    text = " ".join(stemmed_words)

    del faster_stemmer

    gc.collect

    return text
data_text['history_text'] = data_text['history_text'].progress_apply(lambda x: x.lower())

data_text['history_text'] = data_text['history_text'].progress_apply(lambda x: clean_text(x))

data_text['history_text'] = data_text['history_text'].progress_apply(lambda x: clean_numbers(x))

data_text['history_text'] = data_text['history_text'].progress_apply(lambda x: remove_stopwords(x))

data_text['history_text'] = data_text['history_text'].progress_apply(lambda x: stem_text(x))   
data_text.head(1)
# Importing necessary libraries

from sklearn.preprocessing import LabelEncoder

y = data_text['creator']

labelencoder = LabelEncoder()

y = labelencoder.fit_transform(y)

X = data_text['history_text']
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

# 80-20 splitting the dataset 

X_train, X_test, y_train, y_test = train_test_split(X, y

                                  ,test_size=0.35, random_state=1234)

# defining the bag-of-words transformer on the text-processed corpus 

bow_transformer=CountVectorizer(analyzer='word').fit(X_train)

# transforming into Bag-of-Words and hence textual data to numeric..

text_bow_train=bow_transformer.transform(X_train)

# transforming into Bag-of-Words and hence textual data to numeric..

text_bow_test=bow_transformer.transform(X_test)
from sklearn.linear_model import LogisticRegression

# instantiating the model with simple Logistic Regression..

model = LogisticRegression()

# training the model...

model = model.fit(text_bow_train, y_train)
model.score(text_bow_train, y_train)
model.score(text_bow_test, y_test)
target_names = ['Marvel', 'DC']



from sklearn.metrics import classification_report

 

# getting the predictions of the Validation Set...

predictions = model.predict(text_bow_test)

# getting the Precision, Recall, F1-Score

print(classification_report(y_test,predictions,target_names=target_names))
data_text = data[['history_text', 'creator']]

#we will only select comics by Marvel or DC as there's too many comic creators

data_text = data_text.loc[data_text['creator'].isin(['Marvel Comics','DC Comics'])]

data_text['history_text'] = data_text['history_text'].progress_apply(lambda x: x.lower())

data_text['history_text'] = data_text['history_text'].progress_apply(lambda x: clean_text(x))

data_text['history_text'] = data_text['history_text'].progress_apply(lambda x: clean_numbers(x))

data_text['history_text'] = data_text['history_text'].progress_apply(lambda x: remove_stopwords(x))

data_text['history_text'] = data_text['history_text'].progress_apply(lambda x: stem_text(x))   

from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(stop_words='english', analyzer='word', strip_accents='unicode', sublinear_tf=True,

                           token_pattern=r'\w{1,}', max_features=10000, ngram_range=(1,2))

tfidf.fit(data_text.history_text);
features = tfidf.transform(data_text.history_text)
from sklearn import model_selection, preprocessing

# split the dataset into training and validation datasets

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data_text.history_text, data_text.creator, test_size=0.30, random_state=1)



# label encode the target variable

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

valid_y = encoder.fit_transform(valid_y)
xtrain_tfidf =  tfidf.transform(train_x)

xvalid_tfidf =  tfidf.transform(valid_x)
from sklearn import metrics, linear_model, naive_bayes, metrics, svm, ensemble

def train_model(classifier, trains, t_labels, valids, v_labels):

    # fit the training dataset on the classifier

    classifier.fit(trains, t_labels)



    # predict the labels on validation dataset

    predictions = classifier.predict(valids)

    target_names = ['Marvel', 'DC']

    print(metrics.classification_report(v_labels, predictions,target_names=target_names))

    return metrics.accuracy_score(predictions, v_labels)

# Naive Bayes

print ("Naive Bayes")

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y);

print ("Accuracy: ", accuracy)

# Logistic Regression

print ("Logistic Regression")

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y);

print ("Accuracy: ", accuracy)

# SVM

print ("SVM")

accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y);

print ("Accuracy: ", accuracy)

# Random Forest

print ("Random Forest")

accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)

print ("Accuracy: ", accuracy )