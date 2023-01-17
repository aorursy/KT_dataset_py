# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read data

items = pd.read_csv('/kaggle/input/animal-crossing/items.csv')

user_reviews = pd.read_csv('/kaggle/input/animal-crossing/user_reviews.csv')

critic = pd.read_csv('/kaggle/input/animal-crossing/critic.csv')

villagers = pd.read_csv('/kaggle/input/animal-crossing/villagers.csv')
# Standard plotly imports

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode



import plotly.express as px

import plotly.io as pio



# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)



pd.set_option('display.max_columns',500)
items.head()
items[items.name == 'Acoustic Guitar'].head()
items[items.name == 'Rusted Part'].head()
items.shape[0], items.name.nunique()
missing_values = items.isna().sum().sort_values(ascending=False)

missing_values = pd.DataFrame({'Feature':missing_values.index, 'Missing Value Count':missing_values.values})
# plot missing values

fig = px.bar(missing_values[::-1], x= 'Missing Value Count', y='Feature', orientation='h',text='Missing Value Count',

             title='Missing Value Count in Features - Items Dataset',template="plotly_dark")

fig.show()
# drop the recipe column 

items = items.drop(['recipe','recipe_id','sources'], axis=1)



# drop duplicates 

items = items.drop_duplicates()
items.shape[0], items.name.nunique()
items['customizable'] = np.where(items['customizable'] == True,'Customizable','Non Customizable')

items['orderable'] = np.where(items['orderable'] == True,'Orderable','Not Orderable')

items['buy_currency'] = items['buy_currency'].fillna('None')

items['buy_value'] = items['buy_value'].fillna(0)

items['sell_value'] = np.where(items['sell_value'].isnull(), items['buy_value']/4,items['sell_value'])
cat_counts = items.groupby('category')['category'].count()

cat_counts = pd.DataFrame({'Category':cat_counts.index,'Count':cat_counts.values})
fig = px.pie(cat_counts, values='Count', names='Category', title='Category Distribution for the Items', 

             template="plotly_dark",height=500)

fig.show()
# cutomizable counts

customizable = items[items.customizable == 'Customizable'].groupby('category')['category'].count()

non_customizable = items[items.customizable == 'Non Customizable'].groupby('category')['category'].count().sort_values(ascending=False)



# orderable counts

orderable = items[items.orderable == 'Orderable'].groupby('category')['category'].count()

non_orderable = items[items.orderable == 'Not Orderable'].groupby('category')['category'].count().sort_values(ascending=False)
fig = go.Figure(data=[

    go.Bar(name='Non Customizable', x=non_customizable.index, y=non_customizable.values,marker_color='cyan'),

    go.Bar(name='Customizable', x=customizable.index, y=customizable.values, marker_color='chartreuse')

])

# Change the bar mode

fig.update_layout(barmode='group',template="plotly_dark",title_text='Customizable and Non-Customizable Category Counts',height=400)

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Non Orderable', x=non_orderable.index, y=non_orderable.values,marker_color='cyan'),

    go.Bar(name='Orderable', x=orderable.index, y=orderable.values, marker_color='chartreuse')

])

# Change the bar mode

fig.update_layout(barmode='group',template="plotly_dark",title_text='Orderable and Non-Orderable Category Counts',height=400)

fig.show()
expensive_items = items[['name','buy_value']].sort_values(by='buy_value', ascending=False).head(10)

cheapest_items = items[items.buy_value > 0][['name','buy_value']].sort_values(by='buy_value').head(10)
fig = go.Figure(data=[

    go.Bar(name='Most Expensive Items', x=expensive_items[::-1].buy_value, y=expensive_items[::-1].name,marker_color='cornsilk',

           orientation='h'),

])

# Change the bar mode

fig.update_layout(template="plotly_dark",title_text='Top 10 Expensive Items',height=400)

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Cheapest Items', x=cheapest_items.buy_value, y=cheapest_items.name,marker_color='coral',

           orientation='h'),

])

# Change the bar mode

fig.update_layout(template="plotly_dark",title_text='Top 10 Cheapest Items',height=400)

fig.show()
cat_buy_value = items.groupby('category')['buy_value'].median().sort_values(ascending=False)

cat_sale_value = items.groupby('category')['sell_value'].median().sort_values(ascending=False)



categories = items.category.unique()
fig = go.Figure()

for cats in categories:

    fig.add_trace(go.Violin(x=items['category'][items['category'] == cats],

                            y=items['buy_value'][items['category'] == cats],

                            name=cats,

                            box_visible=False,

                            meanline_visible=False,jitter=0.05))



fig.update_layout(template="plotly_dark",title_text='Buy Value Distribution by Category',height=400)



fig.show()
fig = go.Figure()

for cats in categories:

    fig.add_trace(go.Violin(x=items['category'][items['category'] == cats],

                            y=items['sell_value'][items['category'] == cats],

                            name=cats,

                            box_visible=False,

                            meanline_visible=False,jitter=0.05))



fig.update_layout(template="plotly_dark",title_text='Sell Value Distribution by Category',height=400)



fig.show()
fig = go.Figure(data=[

    go.Bar(name='Category', x=cat_buy_value.index, y=cat_buy_value.values,marker_color='yellow')

])

# Change the bar mode

fig.update_layout(template="plotly_dark",title_text='Buy Value (Median)',height=400)

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Category', x=cat_sale_value.index, y=cat_sale_value.values,marker_color='red')

])

# Change the bar mode

fig.update_layout(template="plotly_dark",title_text='Sale Value (Median)',height=400)

fig.show()
villagers.head()
missing_values_villagers = villagers.isna().sum().sort_values(ascending=False)

missing_values_villagers = pd.DataFrame({'Feature':missing_values_villagers.index, 'Missing Value Count':missing_values_villagers.values})
# plot missing values

fig = px.bar(missing_values_villagers[::-1], x= 'Missing Value Count', y='Feature', orientation='h',text='Missing Value Count',

             title='Missing Value Count in Features - Villagers Dataset',template="plotly_dark",height=400)

fig.show()
species_count = villagers.groupby('species')['species'].count()

species_count = pd.DataFrame({'Species':species_count.index,'Count':species_count.values})



fig = px.pie(species_count, values='Count', names='Species', title='Species Distribution for the Villagers', 

             template="plotly_dark",height=500)

fig.show()
males = villagers[villagers.gender == 'male'].groupby('species')['species'].count().sort_values(ascending=False)

females = villagers[villagers.gender == 'female'].groupby('species')['species'].count().sort_values(ascending=False)



fig = go.Figure(data=[

    go.Bar(name='Males', x=males[::-1].index, y=males[::-1].values,marker_color='lightskyblue'),

    go.Bar(name='FeMales', x=females.index, y=males.values,marker_color='lightsalmon')

])

# Change the bar mode

fig.update_layout(barmode='group',template="plotly_dark",title_text='Species Counts - Males & Females',height=400)

fig.show()
personality = villagers.groupby('personality')['personality'].count().sort_values()

males = villagers[villagers.gender == 'male'].groupby('personality')['personality'].count().sort_values(ascending = False)

females = villagers[villagers.gender == 'female'].groupby('personality')['personality'].count().sort_values(ascending = False)



fig = go.Figure(data=[

    go.Bar(name='Personality', x=personality[::-1].index, y=personality[::-1].values,marker_color=' steelblue')

])

fig.update_layout(barmode='group',template="plotly_dark",title_text='Personality Types for Villagers',height=400)

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Males', x=males[::-1].index, y=males[::-1].values,marker_color='lightskyblue'),

    go.Bar(name='FeMales', x=females.index, y=males.values,marker_color='lightsalmon')

])

fig.update_layout(barmode='group',template="plotly_dark",title_text='Personality - Males and Females',height=400)

fig.show()
user_reviews['date'] = pd.to_datetime(user_reviews['date'], format='%Y-%m-%d', errors='ignore')
user_review_counts = user_reviews.groupby('date')['user_name'].count()



fig = px.line(x=user_review_counts.index, y=user_review_counts.values, range_x=['2020-03-20','2020-05-03'])



fig.update_layout(

    xaxis = dict(title_text = "Date"),

    yaxis = dict(title_text='Review Count'),height=350,title_text='Review Counts')

    

fig.show()
review_grade_count = user_reviews.groupby('grade')['user_name'].count()



fig = go.Figure(data=[

    go.Bar(name='Count', x=review_grade_count[::-1].index, y=review_grade_count[::-1].values,marker_color='red')

])

fig.update_layout(title_text='Review Counts by Score',height=400)

fig.show()
high_rank_trend = user_reviews[user_reviews.grade >= 9].groupby('date')['user_name'].count()



fig = px.line(x=high_rank_trend.index, y=high_rank_trend.values, range_x=['2020-03-20','2020-05-03'])



fig.update_layout(

    xaxis = dict(title_text = "Date"),

    yaxis = dict(title_text='Review Count'),height=400,title_text='Review Counts - Grade >= 9')

    

fig.show()
# create a column to for bad reviews and good reviews - any review above >7 is good and the rest is bad. 

user_reviews['is_bad_review'] = user_reviews['grade'].apply(lambda x: 1 if x < 7 else 0)
# create a separate dataset with the text column and the newly created 'is_bad_review' column.

reviews_df = user_reviews[['text','is_bad_review']].rename(columns={'text':'review'})
# define functions for cleaning data

from nltk.corpus import wordnet

import string

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.tokenize import WhitespaceTokenizer

from nltk.stem import WordNetLemmatizer



def get_wordnet_pos(pos_tag):

    if pos_tag.startswith('J'):

        return wordnet.ADJ

    elif pos_tag.startswith('V'):

        return wordnet.VERB

    elif pos_tag.startswith('N'):

        return wordnet.NOUN

    elif pos_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN



def clean_text(text):

    # lower text

    text = text.lower()

    # tokenize text and remove puncutation

    text = [word.strip(string.punctuation) for word in text.split(" ")]

    # remove words that contain numbers

    text = [word for word in text if not any(c.isdigit() for c in word)]

    # remove stop words

    stop = stopwords.words('english')

    text = [x for x in text if x not in stop]

    # remove empty tokens

    text = [t for t in text if len(t) > 0]

    # pos tag text

    pos_tags = pos_tag(text)

    # lemmatize text

    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    # remove words with only one letter

    text = [t for t in text if len(t) > 1]

    # join all

    text = " ".join(text)

    return(text)



# clean data

reviews_df["review_clean"] = reviews_df["review"].apply(lambda x: clean_text(x))
# add character count column

reviews_df["Char_Count"] = reviews_df["review"].apply(lambda x: len(x))



# add number of words column

reviews_df["Word_Count"] = reviews_df["review"].apply(lambda x: len(x.split(" ")))
# add sentiment anaylsis columns

from nltk.sentiment.vader import SentimentIntensityAnalyzer



sid = SentimentIntensityAnalyzer()

reviews_df["sentiments"] = reviews_df["review"].apply(lambda x: sid.polarity_scores(x))

reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)
reviews_df[reviews_df["Word_Count"] >= 5].sort_values("pos", ascending = False)[["review", "pos"]].head(10)
reviews_df[reviews_df["Word_Count"] >= 5].sort_values("neg", ascending = False)[["review", "neg"]].head(10)
import seaborn as sns



for x in [0, 1]:

    subset = reviews_df[reviews_df['is_bad_review'] == x]

    

    # Draw the density plot

    if x == 0:

        label = "Good reviews"

    else:

        label = "Bad reviews"

    sns.distplot(subset['compound'], hist = False, label = label)