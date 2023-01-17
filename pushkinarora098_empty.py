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
df = pd.read_csv('../input/knightsaddy/train.csv')

cc = pd.read_csv("../input/gdp-country/2014_world_gdp_with_codes.csv")
dictr = cc.to_dict()

codedict = {}

codedict1 = {}

for i in range(1,222):

    codedict[dictr['COUNTRY'][i]] = dictr['CODE'][i]

    codedict1[dictr['CODE'][i]] = dictr['COUNTRY'][i]
df['country'] = df['country'].apply(lambda x: "United States" if(x=="US") else x)

df['country'] = df['country'].fillna('United States')

countryplot = pd.DataFrame((df.loc[(df['country']!='Macedonia')&((df['country']!='England')&(df['country']!='Moldova'))]['country'].apply(lambda x: codedict[x])).value_counts())

countryplot['Name'] = countryplot.index
data = dict(type = 'choropleth',

            locations = countryplot['Name'],

            z = countryplot['country'],

            text = countryplot['Name'].apply(lambda x: codedict1[x]),

            colorscale = 'Reds',

            autocolorscale=False,

            reversescale=False,

            marker_line_color='darkgray',

            marker_line_width=0.5,

            colorbar = {'title':'Number of sales'})

layout = dict(title = "total sales",

             geo = dict(showframe = False,

                       projection = {'type':'equirectangular'}))
import plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

choromap = go.Figure([data],layout)

plot(choromap)

iplot(choromap)
import seaborn as sns

sns.distplot(df['points'])
reviews_df = pd.DataFrame(df[['review_description','points']])
reviews_df
%%time

from nltk.corpus import wordnet



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

    

import string

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.tokenize import WhitespaceTokenizer

from nltk.stem import WordNetLemmatizer



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



# clean text data

reviews_df["review_clean"] = reviews_df["review_description"].apply(lambda x: clean_text(x))
%%time

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

reviews_df["sentiments"] = reviews_df["review_description"].apply(lambda x: sid.polarity_scores(x))

reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)
reviews_df.columns
reviews_df["nb_chars"] = reviews_df["review_description"].apply(lambda x: len(x))

reviews_df["nb_words"] = reviews_df["review_description"].apply(lambda x: len(x.split(" ")))
# wordcloud function



from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



stopwords = set(STOPWORDS)



newStopWords = ['wine','fruit','parts','drinks','drink','not']



stopwords.update(newStopWords)





def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color = 'white',

        stopwords = stopwords,

        max_words = 200,

        max_font_size = 40, 

        scale = 3,

        random_state = 42

    ).generate(str(data))



    fig = plt.figure(1, figsize = (20, 20))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize = 20)

        fig.subplots_adjust(top = 2.3)



    plt.imshow(wordcloud)

    plt.show()

show_wordcloud(reviews_df["review_description"])
reviews_df.columns 
# ['review_description','points','review_clean','neg','neg1','neu','neu1','pos','pos1','neg2','neu2','pos2','compound','nb_chars','nb_words']

# reviews_df = reviews_df.drop(['neg1','neu1','pos1','neg2','neu2','pos2'],axis=1)
reviews_df['price'] = df['price']
sns.heatmap(reviews_df[['pos','points','nb_words','compound','neg','price']].corr(),annot=True)
show_wordcloud(df["review_title"])
reviews_df[reviews_df["nb_words"] >= 4].sort_values("compound", ascending = False)[["review_description", "pos","points","compound","neg"]].head(10)
reviews_df[reviews_df["nb_words"] >= 4].sort_values("compound", ascending = True)[["review_description", "pos","points","compound","neg"]].head(10)
reviews_df['user_name'] = df['user_name']

reviews_df['review_title'] = df['review_title']

reviews_df['designation'] = df['designation']

reviews_df['country'] = df['country']

reviews_df['region_1'] = df['region_1']

reviews_df['region_2'] = df['region_2']

reviews_df['winery'] = df['winery']

reviews_df['variety'] = df['variety']
reviews_df.to_csv('saddy.csv')
%%time

from gensim.test.utils import common_texts

from gensim.models.doc2vec import Doc2Vec, TaggedDocument



documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews_df["review_clean"].apply(lambda x: x.split(" ")))]



# train a Doc2Vec model with our text data

model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)



# transform each document into a vector data

doc2vec_df = reviews_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)

doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]

reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)
# import pandas as pd

# reviews_df = pd.read_csv('../input/reviewdf/saddy (2).csv')
%%time

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df = 10)

tfidf_result = tfidf.fit_transform(reviews_df["review_clean"]).toarray()

tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())

tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]

tfidf_df.index = reviews_df.index

reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)
h = reviews_df
h.drop("review_description",axis=1,inplace=True)
h.drop("review_clean",axis=1,inplace=True)
h.drop("region_1",axis=1,inplace=True)
h.drop("region_2",axis=1,inplace=True)

h.drop("user_name",axis=1,inplace=True)

h.drop("designation",axis=1,inplace=True)
h['price'] = h['price'].fillna(20)
k = (h.isnull().sum()/h.shape[0])*100

for i in range(len(k)):

    print(h.columns[i])

    print((k[i]))
j = pd.DataFrame(h['variety'].value_counts())

j['name'] = j.index

j.index = range(1,29)

j.columns = ['count','name']

d = {}

j = j.to_dict()

for i in range(1,29):

    d[j['name'][i]] = i

d
reviews_df['variety'] = reviews_df['variety'].apply(lambda x: d[x])
# feature selection

label = "variety"

ignore_cols = [label]

features = [c for c in reviews_df.columns if c not in ignore_cols]



# split the data into train and test

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(reviews_df[features], reviews_df[label], test_size = 0.20, random_state = 42)
# train a random forest classifier

rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
# rf.fit(X_train, y_train)
# # show feature importance

# feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)

# feature_importances_df.head(20)