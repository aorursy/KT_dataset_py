# Import libraries

import pandas as pd

import numpy as np

import ast

import re

import math



# libraries for displaying images

from IPython.display import Image 

from IPython.core.display import HTML 



import matplotlib.pyplot as plt

import seaborn as sns



from plotly.offline import init_notebook_mode, plot, iplot

import plotly.graph_objs as go



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
zomato_df = pd.read_csv(r"../input/zomato-bangalore-restaurants/zomato.csv", encoding='utf-8')
zomato_df.info()
zomato_df = zomato_df.drop_duplicates(subset=['address', 'name']).reset_index().drop('index', axis=1)
zomato_df.head()
histo = zomato_df.groupby('name')['address'].count()

# Lets take the top 50 restaurants and visualize in plot bar graph

histo = histo.sort_values()[-50:]
ax = histo.plot(kind='bar', figsize=(20, 8), rot=90, width = 0.8, color=[ 'blue'])

rects = ax.patches

labels = list(histo)

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width() / 2, height + 1, label,

            ha='center', va='bottom', fontsize=14)

ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on') # remove borders

ax.xaxis.set_tick_params(labelsize=15) # set xticks as 14

ax.legend(fontsize=14) # set legend sie as 14

ax.set_title('No of restaurants', fontsize=16) # set title and add font size as 16

ax.set_xlabel('Restaurant Name', fontsize=16)

#ax.grid(False)  # remove grid

ax.set_facecolor("white") # set bg color white

ax.legend(['#Restaurants'])
histo = zomato_df.groupby('location')['url'].count().sort_values(ascending=False)[:50]

ax = histo.plot(kind='bar', figsize=(20, 8), rot=90, width = 0.8, color=[ 'blue'])

rects = ax.patches

labels = list(histo)

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width() / 2, height + .05, label,

            ha='center', va='bottom', fontsize=10)

ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on') # remove borders

ax.xaxis.set_tick_params(labelsize=15) # set xticks as 14

ax.legend(fontsize=14) # set legend sie as 14

ax.set_title('No of restaurants', fontsize=16) # set title and add font size as 16

ax.set_xlabel('Neighborhood', fontsize=16)

#ax.grid(False)  # remove grid

ax.set_facecolor("white") # set bg color white

ax.legend(['#Restaurants'])
total = zomato_df.groupby('location')['url'].count().sort_values(ascending=False)[:50]

percent = (zomato_df.groupby('location')['url'].count().sort_values(ascending=False)[:50]/zomato_df.shape[0]).sort_values(ascending=False)

loction_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

loction_data.sort_values(ascending=False,by= ['Percent'])[:20]
types = set()

def func(x):

    if(type(x) == list):

        print(x)

        for y in x:

            types.add(y.strip())

_ = zomato_df['rest_type'].str.split(',').apply(func)
column_names = list(types)

# instantiate the dataframe

neighborhood = pd.DataFrame(columns=column_names)

neighborhood
neighborhood['neighborhood'] = zomato_df.groupby('location').groups.keys()

neighborhood = neighborhood.set_index('neighborhood').fillna(0)

#neighborhood
i=0

for i in range(0,len(zomato_df)):

    for x in types:

        if type(zomato_df.loc[i, 'rest_type']) == str and x in zomato_df.loc[i, 'rest_type']:

            neighborhood.loc[zomato_df.loc[i, 'location'], x] = neighborhood.loc[zomato_df.loc[i, 'location'], x]+1
neighborhood
dfs = neighborhood.reset_index().melt('neighborhood', var_name='cols',  value_name='vals')
plt.figure(figsize=(15,20))

ax = sns.swarmplot(x="vals", y="cols", data=dfs)

ax.set_xlabel('Number of restaurant', fontsize=16)

ax.set_ylabel('Type of restaurant', fontsize=16)

ax.set_title('Distribution of different types of restaurant')

plt.savefig("swarm.png")
types = set()

def func(x):

    if(type(x) == list):

        for y in x:

            types.add(y.strip())

_ = zomato_df['cuisines'].str.split(',').apply(func)
column_names = list(types)

neighborhood_cns = pd.DataFrame(columns=column_names)

neighborhood_cns
neighborhood_cns['neighborhood'] = zomato_df.groupby('location').groups.keys()

neighborhood_cns = neighborhood_cns.set_index('neighborhood').fillna(0)

#neighborhood_cns
i=0

for i in range(0,len(zomato_df)):

    for x in types:

        if type(zomato_df.loc[i, 'cuisines']) == str and x in zomato_df.loc[i, 'cuisines']:

            neighborhood_cns.loc[zomato_df.loc[i, 'location'], x] = neighborhood_cns.loc[zomato_df.loc[i, 'location'], x]+1
neighborhood_cns
plt.figure(figsize=(30,30))

sns.heatmap(neighborhood_cns,cmap="BuPu")
Cuisine_data = pd.DataFrame(neighborhood_cns.sum(axis=0))

Cuisine_data.reset_index(inplace=True)

Cuisine_data.columns = ['Cuisines', 'Number of Resturants']

Top15= (Cuisine_data.sort_values(['Number of Resturants'],ascending=False)).head(15)

sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.barplot(Top15['Cuisines'], Top15['Number of Resturants'])

plt.xlabel('Cuisines', fontsize=20)

plt.ylabel('Number of Resturants', fontsize=20)

plt.title('Top 15 Cuisines on Zomato', fontsize=30)

plt.xticks(rotation = 90)

plt.show()
listed = zomato_df['listed_in(type)'].unique()

column_names = list(listed)

# instantiate the dataframe

neighborhood_lst = pd.DataFrame(columns=column_names)

neighborhood_lst
neighborhood_lst['neighborhood'] = zomato_df.groupby('location').groups.keys()

neighborhood_lst = neighborhood_lst.set_index('neighborhood').fillna(0)

#neighborhood_lst
i=0

for i in range(0,len(zomato_df)):

    for x in listed:

        if type(zomato_df.loc[i, 'listed_in(type)']) == str and x in zomato_df.loc[i, 'listed_in(type)'] and type(zomato_df.loc[i, 'location']) == str:

            neighborhood_lst.loc[zomato_df.loc[i, 'location'], x] = neighborhood_lst.loc[zomato_df.loc[i, 'location'], x]+1
neighborhood_lst
category = neighborhood_lst.columns

fig, ax = plt.subplots(nrows=4, ncols=2, figsize = (25,25))

fig.delaxes(ax[3,1])



plt.subplots_adjust(wspace=0.2, hspace=0.5)

ax = ax.flatten()



for i in range(0,len(category)):

    d= neighborhood_lst[[category[i]]].sort_values(by=category[i], ascending=False).head()

    d.plot(ax=ax[i],kind='bar')

    ax[i].set_title(category[i])

    ax[i].set_xticklabels(d.index, rotation='vertical')

    

plt.show()
zomato_df['approx_cost(for two people)'] = zomato_df['approx_cost(for two people)'].str.replace(",","").astype(float)
#choosen neighbourhood which has more than fifty restaurant

above_50 = zomato_df.groupby('location')['url'].count()[zomato_df.groupby('location')['url'].count() >= 50].index
zomato_df[zomato_df['location'].isin(above_50)].groupby('location')['approx_cost(for two people)'].mean().sort_values(ascending= False)
bins = pd.IntervalIndex.from_tuples([(0, 500), (501, 1000), (1001, 2000), (2001, 3000), (3001, 4000), (4001, 5000), (5001, 6000)])

zomato_df['cost_cat'] = pd.cut(zomato_df['approx_cost(for two people)'], bins)
zomato_df['rate'] = zomato_df['rate'].str.split('/').str[0]

zomato_df.loc[zomato_df['rate']=="NEW", 'rate'] = np.nan

zomato_df.loc[zomato_df['rate']=="-", 'rate'] = np.nan

zomato_df['rate'] = zomato_df['rate'].astype('float')
plt.figure(figsize=(15,15))

ax = sns.boxplot(x="cost_cat", y="rate", data=zomato_df)

ax.set_xlabel('Cost', fontsize=16)

ax.set_ylabel('Rating', fontsize=16)

ax.set_title('Price and Distribution')

plt.savefig("box.png")

zomato_df.drop('cost_cat', axis=1, inplace=True)
bins = pd.IntervalIndex.from_tuples([(0, 250),(251, 500), (501, 750),(751, 1000),(1001, 1500), (1501, 2000), (2001, 3000), (3001, 4000), (4001, 5000), (5001, 6000)])

zomato_df['cost_cat'] = pd.cut(zomato_df['approx_cost(for two people)'], bins)
plt.figure(figsize=(15,15))

ax = sns.boxplot(x="cost_cat", y="rate", data=zomato_df)

ax.set_xlabel('Cost', fontsize=16)

ax.set_ylabel('Rating', fontsize=16)

ax.set_title('Price and Distribution')

plt.savefig("box.png")

zomato_df.drop('cost_cat', axis=1, inplace=True)
data = [

    go.Scatter(x = zomato_df['approx_cost(for two people)'],

              y = zomato_df['rate'],

              mode = "markers",

               text = zomato_df['name'],

               marker = dict(opacity = 0.7,

                            size = 10,

                            color = zomato_df['rate'], #Set color equalivant to rating

                            colorscale= 'Viridis',

                            showscale=True,

                             maxdisplayed=2500,

                            ),

                hoverinfo="text+x+y",

              )

]

layout = go.Layout(autosize=True,

                   xaxis=dict(title="Average Cost of Two (INR)",

                             #titlefont=dict(size=20,),

                             #tickmode="linear",

                             ),

                   yaxis=dict(title="Rating",

                             #titlefont=dict(size=17,),

                             ),

                  )

iplot(dict(data=data, layout=layout))
zomato_df['reviews_list'] =  zomato_df['reviews_list'].apply(ast.literal_eval)
ind = zomato_df[zomato_df['reviews_list']!="[]"].index

for i in ind:

    review_list = []

    for review in zomato_df.loc[i,'reviews_list']:

        if(review[0]!=None and float(review[0].replace('Rated ',''))>=0):

            review_list.append(float(review[0].replace('Rated ','')))

    #zomato_df.loc[i,'total_review'] = sum(review_list)

    #zomato_df.loc[i,'count_review'] = len(review_list)

    if(len(review_list)>0):

        zomato_df.loc[i,'avg_review'] = sum(review_list)/len(review_list)  #calulated the average rating of each restaurant
rest_review = zomato_df.groupby('name')['avg_review'].sum().round().sort_values(ascending=False)

rest_review.head()
rest_review_above_50 = zomato_df.groupby('name')['avg_review'].sum().round().sort_values(ascending=False)[:50]

ax = rest_review_above_50.plot(kind='bar', figsize=(20, 8), rot=90, width = 0.8, color=[ 'blue'])

rects = ax.patches

labels = list(rest_review_above_50)

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width() / 2, height + .05, label,

            ha='center', va='bottom', fontsize=10)

ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on') # remove borders

ax.xaxis.set_tick_params(labelsize=15) # set xticks as 14

ax.legend(fontsize=14) # set legend sie as 14

ax.set_title('Ranked on Reviews', fontsize=16) # set title and add font size as 16

ax.set_xlabel('Restaurant', fontsize=16)

#ax.grid(False)  # remove grid

ax.set_facecolor("white") # set bg color white

ax.legend(['#Restaurants'])
#NLP Libraries

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en')

from collections import Counter

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from spacy.tokens import Doc



from tqdm import tqdm
sent_analyzer = SentimentIntensityAnalyzer()



def sentiment_scores(docx):

    return sent_analyzer.polarity_scores(docx.text)



Doc.set_extension("sentimenter",getter=sentiment_scores)
ind = zomato_df[zomato_df['reviews_list']!="[]"].index

for i in tqdm(range(20)):#ind

    review_list = []

    pos_score = []

    neg_score = []

    for review in zomato_df.loc[i,'reviews_list']:

        if(review[0]!=None and float(review[0].replace('Rated ',''))>=0):

            comment = nlp(str(review[1]))

            verbs = [ word for word in comment if word.is_stop == False and not word.is_punct and (word.pos_ == 'VERB'or word.pos_ == 'ADJ')]

            values  = ' '.join(str(v) for v in verbs)

            pos_score.append(nlp(values)._.sentimenter['pos'])

            neg_score.append(nlp(values)._.sentimenter['neg'])

    if(len(pos_score)>0):

        #zomato_df.loc[i,'pos_score'] = sum(pos_score)/len(pos_score)

        #zomato_df.loc[i,'neg_score'] = sum(neg_score)/len(neg_score)

        zomato_df.loc[i,'sent_score'] = (sum(pos_score)/len(pos_score))-(sum(neg_score)/len(neg_score)) #overall rating of feedback
rest_review_above_50 = zomato_df.groupby('name')['sent_score'].sum().round().sort_values(ascending=False)[:50]

ax = rest_review_above_50.plot(kind='bar', figsize=(20, 8), rot=90, width = 0.8, color=[ 'blue'])

rects = ax.patches

labels = list(rest_review_above_50)

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width() / 2, height + .05, label,

            ha='center', va='bottom', fontsize=10)

ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on') # remove borders

ax.xaxis.set_tick_params(labelsize=15) # set xticks as 14

ax.legend(fontsize=14) # set legend sie as 14

ax.set_title('Ranked on Reviews feadback', fontsize=16) # set title and add font size as 16

ax.set_xlabel('Restaurant', fontsize=16)

#ax.grid(False)  # remove grid

ax.set_facecolor("white") # set bg color white

ax.legend(['#Restaurants'])


from IPython.display import Image

Image("../input/images/sentiment.png")