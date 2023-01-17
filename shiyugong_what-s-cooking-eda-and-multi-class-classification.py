# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import re



# Visualization

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

from wordcloud import WordCloud, STOPWORDS

%matplotlib inline



import random

import plotly

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

init_notebook_mode(connected=True)

import plotly.offline as offline

import plotly.graph_objs as go



# Model

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.metrics import log_loss

from scipy.optimize import minimize

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import KFold

from sklearn import model_selection 



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
train_data = pd.read_json('../input/train.json') # store as dataframe objects

test_data = pd.read_json('../input/test.json')
train_data.info()
train_data.shape # 39774 observations, 3 columns
print("The training data consists of {} recipes".format(len(train_data)))
print("First five elements in our training sample:")

train_data.head()
test_data.info()
test_data.shape # 9944 observations, 2 columns
print("The test data consists of {} recipes".format(len(test_data)))
print("First five elements in our test sample:")

test_data.head()
print("Number of cuisine categories: {}".format(len(train_data.cuisine.unique())))

train_data.cuisine.unique()
#Define a function to generate randoms colors for further visualizations

def random_colours(number_of_colors):

    colors = []

    for i in range(number_of_colors):

        colors.append("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))

    return colors
trace = go.Table(

                header=dict(values=['Cuisine','Number of recipes'],

                fill = dict(color=['#EABEB0']), 

                align = ['left'] * 5),

                cells=dict(values=[train_data.cuisine.value_counts().index,train_data.cuisine.value_counts()],

                align = ['left'] * 5))



layout = go.Layout(title='Number of recipes in each cuisine category',

                   titlefont = dict(size = 20),

                   width=500, height=650, 

                   paper_bgcolor =  'rgba(0,0,0,0)',

                   plot_bgcolor = 'rgba(0,0,0,0)',

                   autosize = False,

                   margin=dict(l=30,r=30,b=1,t=50,pad=1),

                   )

data = [trace]

fig = dict(data=data, layout=layout)

iplot(fig)
#  Label distribution in percents

labelpercents = []

for i in train_data.cuisine.value_counts():

    percent = (i/sum(train_data.cuisine.value_counts()))*100

    percent = "%.2f" % percent

    percent = str(percent + '%')

    labelpercents.append(percent)
trace = go.Bar(

            x=train_data.cuisine.value_counts().values[::-1],

            y= [i for i in train_data.cuisine.value_counts().index][::-1],

            text =labelpercents[::-1],  textposition = 'outside', 

            orientation = 'h',marker = dict(color = random_colours(20)))

layout = go.Layout(title='Number of recipes in each cuisine category',

                   titlefont = dict(size = 25),

                   width=1030, height=450, 

                   plot_bgcolor = 'rgba(0,0,0,0)',

                   

                   margin=dict(l=75,r=110,b=50,t=60),

                   )

data = [trace]

fig = dict(data=data, layout=layout)

iplot(fig, filename='horizontal-bar')
print('Maximum Number of Ingredients in a Dish: ',train_data['ingredients'].str.len().max())

print('Minimum Number of Ingredients in a Dish: ',train_data['ingredients'].str.len().min())
trace = go.Histogram(

    x= train_data['ingredients'].str.len(),

    xbins=dict(start=0,end=80,size=1),

   marker=dict(color='#fbca5f'),

    opacity=0.75)

data = [trace]

layout = go.Layout(

    title='Distribution of Recipe Length',

    xaxis=dict(title='Number of ingredients'),

    yaxis=dict(title='Count of recipes'),

    bargap=0.1,

    bargroupgap=0.2)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
boxplotcolors = random_colours(21)

labels = [i for i in train_data.cuisine.value_counts().index][::-1]

data = []

for i in range(20):

    trace = {

            "type": 'violin',

            "y": train_data[train_data['cuisine'] == labels[i]]['ingredients'].str.len(),

            "name": labels[i],

            "box": {

                "visible": True

            },

            "meanline": {

                "visible": True

            }

        }

    data.append(trace)

layout = go.Layout(

    title = "Recipe Length Distribution by cuisine"

)



fig = go.Figure(data=data,layout=layout)

iplot(fig, filename = "Box Plot Styling Outliers")
print("Word Cloud Function..")

stopwords = set(STOPWORDS)

size = (20,10)



def cloud(text, title, stopwords=stopwords, size=size):

    """

    Function to plot WordCloud

    Includes: 

    """

    # Setting figure parameters

    mpl.rcParams['figure.figsize']=(10.0,10.0)

    mpl.rcParams['font.size']=12

    mpl.rcParams['savefig.dpi']=100

    mpl.rcParams['figure.subplot.bottom']=.1 

    

    # Processing Text

    wordcloud = WordCloud(width=1600, height=800,

                          background_color='black',

                          stopwords=stopwords,

                         ).generate(str(text))

    

    # Output Visualization

    fig = plt.figure(figsize=size, dpi=80, facecolor='k',edgecolor='k')

    plt.imshow(wordcloud,interpolation='bilinear')

    plt.axis('off')

    plt.title(title, fontsize=50,color='y')

    plt.tight_layout(pad=0)

    plt.show()

    

# Data Set for Word Clouds

train_data["ing"] = train_data.ingredients.apply(lambda x: list(map(str, x)), 1).str.join(' ')

# All

cloud(train_data["ing"].values, title="All Cuisine", size=[8,5])
y = train_data.cuisine.copy()

print("Cuisine WordClouds")

cloud_df = pd.concat([train_data.loc[train_data.index,'ing'], y],axis=1)

for cuisine_x in y.unique():

    cloud(cloud_df.loc[cloud_df.cuisine == cuisine_x, "ing"].values, title="{} Cuisine".format(cuisine_x.capitalize()), size=[8,5])

train_data.drop('ing',axis=1,inplace=True)
train_data['seperated_ingredients'] = train_data['ingredients'].apply(','.join)

test_data['seperated_ingredients'] = test_data['ingredients'].apply(','.join)
import nltk

from collections import Counter
train_data['for ngrams']=train_data['seperated_ingredients'].str.replace(',',' ')
import networkx as nx

def generate_ngrams(text, n):

    words = text.split(' ')

    iterations = len(words) - n + 1

    for i in range(iterations):

       yield words[i:i + n]

def net_diagram(*cuisines):

    ngrams = {}

    for title in train_data[train_data.cuisine==cuisines[0]]['for ngrams']:

            for ngram in generate_ngrams(title, 2):

                ngram = ','.join(ngram)

                if ngram in ngrams:

                    ngrams[ngram] += 1

                else:

                    ngrams[ngram] = 1

    ngrams_mws_df = pd.DataFrame.from_dict(ngrams, orient='index')

    ngrams_mws_df.columns = ['count']

    ngrams_mws_df['cusine'] = cuisines[0]

    ngrams_mws_df.reset_index(level=0, inplace=True)



    ngrams = {}

    for title in train_data[train_data.cuisine==cuisines[1]]['for ngrams']:

            for ngram in generate_ngrams(title, 2):

                ngram = ','.join(ngram)

                if ngram in ngrams:

                    ngrams[ngram] += 1

                else:

                    ngrams[ngram] = 1

    

    ngrams_mws_df1 = pd.DataFrame.from_dict(ngrams, orient='index')

    ngrams_mws_df1.columns = ['count']

    ngrams_mws_df1['cusine'] = cuisines[1]

    ngrams_mws_df1.reset_index(level=0, inplace=True)

    cuisine1=ngrams_mws_df.sort_values('count',ascending=False)[:25]

    cuisine2=ngrams_mws_df1.sort_values('count',ascending=False)[:25]

    df_final=pd.concat([cuisine1,cuisine2])

    g = nx.from_pandas_edgelist(df_final,source='cusine',target='index')

    cmap = plt.cm.RdYlGn

    colors = [n for n in range(len(g.nodes()))]

    k = 0.35

    pos=nx.spring_layout(g, k=k)

    nx.draw_networkx(g,pos, node_size=df_final['count'].values*8, cmap = cmap, node_color=colors, edge_color='grey', font_size=15, width=3)

    plt.title("Top 25 Bigrams for %s and %s" %(cuisines[0],cuisines[1]), fontsize=30)

    plt.gcf().set_size_inches(30,30)

    plt.show()

    plt.savefig('network.png')
net_diagram('chinese','thai')
net_diagram('indian','chinese')
# Prepare the data 

features = [] # list of list containg the recipes

for item in train_data['ingredients']:

    features.append(item)

    

# Test Sample - only features - the target variable is not provided.

features_test = [] # list of lists containg the recipes

for item in test_data['ingredients']:

    features_test.append(item)
# Both train and test samples are processed in the exact same way

# Train

features_processed= [] # here we will store the preprocessed training features

for item in features:

    newitem = []

    for ingr in item:

        ingr.lower() # Case Normalization - convert all to lower case 

        ingr = re.sub("[^a-zA-Z]"," ",ingr) # Remove punctuation, digits or special characters 

        ingr = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', ingr) # Remove different units  

        newitem.append(ingr)

    features_processed.append(newitem)



# Test 

features_test_processed= [] 

for item in features_test:

    newitem = []

    for ingr in item:

        ingr.lower() 

        ingr = re.sub("[^a-zA-Z]"," ",ingr)

        ingr = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', ingr) 

        newitem.append(ingr)

    features_test_processed.append(newitem) 
# Check for empty instances in train and test samples after processing before proceeding to next stage of the analysis    

count_m = []    

for recipe in features_processed:

    if not recipe:

        count_m.append([recipe])

    else: pass

print("Empty instances in the preprocessed training sample: " + str(len(count_m))) 
count_m = []    

for recipe in features_test_processed:

    if not recipe:

        count_m.append([recipe])

    else: pass

print("Empty instances in the preprocessed test sample: " + str(len(count_m)))    
# Binary representation of the training set will be employed

vectorizer = CountVectorizer(analyzer = "word",

                             ngram_range = (1,1), # unigrams

                             binary = True, #  (the default is counts)

                             tokenizer = None,    

                             preprocessor = None, 

                             stop_words = None,  

                             max_df = 0.99) # any word appearing in more than 99% of the sample will be discarded
# Fit the vectorizer on the training data and transform the test sample

train_X = vectorizer.fit_transform([str(i) for i in features_processed])

test_X =  vectorizer.transform([str(i) for i in features_test_processed])
# Extract the target variable

target = train_data['cuisine']
# Apply label encoding on the target variable (before model development)

lb = LabelEncoder()

train_Y = lb.fit_transform(target)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y , random_state = 0)
### building the classifiers

clfs = []



rfc = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

rfc.fit(X_train, y_train)

print('RFC LogLoss {score}'.format(score=log_loss(y_test, rfc.predict_proba(X_test))))

clfs.append(rfc)



logreg = LogisticRegression(random_state = 42)

logreg.fit(X_train, y_train)

print('LogisticRegression LogLoss {score}'.format(score=log_loss(y_test, logreg.predict_proba(X_test))))

clfs.append(logreg)



svc = SVC(random_state=42,probability=True, kernel='linear')

svc.fit(X_train, y_train)

print('SVC LogLoss {score}'.format(score=log_loss(y_test, svc.predict_proba(X_test))))

clfs.append(svc)

### finding the optimum weights



predictions = []

for clf in clfs:

    predictions.append(clf.predict_proba(X_test))



def log_loss_func(weights):

    ''' scipy minimize will pass the weights as a numpy array '''

    final_prediction = 0

    for weight, prediction in zip(weights, predictions):

            final_prediction += weight*prediction



    return log_loss(y_test, final_prediction)

    

#the algorithms need a starting value, right not we chose 0.5 for all weights

starting_values = [0.5]*len(predictions)



cons = ({'type':'eq','fun':lambda w: 1-sum(w)})

#our weights are bound between 0 and 1

bounds = [(0,1)]*len(predictions)



res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)



print('Ensamble Score: {best_score}'.format(best_score=res['fun']))

print('Best Weights: {weights}'.format(weights=res['x']))
# Ensemble Unigram model (baseline model) 

vclf=VotingClassifier(estimators=[('clf1',RandomForestClassifier(n_estimators = 50,random_state = 42)),

                                  ('clf2',LogisticRegression(random_state = 42)),

                                  ('clf3',SVC(kernel='linear',random_state = 42,probability=True))

                                  ], 

                                    voting='soft', weights = [0.05607363, 0.70759724, 0.23632913]) 

vclf.fit(train_X, train_Y)
# 5-fold Cross validation of  the results

kfold = model_selection.KFold(n_splits=5, random_state=42)

valscores = model_selection.cross_val_score(vclf, train_X, train_Y, cv=kfold)

print('Mean accuracy on 5-fold cross validation: ' + str(np.mean(valscores)))
# Generate predictions on test sample

predictions = vclf.predict(test_X) 

predictions = lb.inverse_transform(predictions)

predictions_final = pd.DataFrame({'cuisine' : predictions , 'id' : test_data.id }, columns=['id', 'cuisine'])

predictions_final.to_csv('Final_submission.csv', index = False)