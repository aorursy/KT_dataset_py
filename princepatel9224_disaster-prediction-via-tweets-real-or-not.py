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
import plotly 

import matplotlib.pyplot as plt

import re 

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import joblib

from nltk.tokenize import RegexpTokenizer

import nltk

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import confusion_matrix 

from imblearn.combine import SMOTETomek

from wordcloud import WordCloud

import plotly.graph_objects as go



nltk.download('stopwords')





stop_words = set(stopwords.words("english"))

default_stemmer = PorterStemmer()

default_stopwords = stopwords.words('english')

default_tokenizer=RegexpTokenizer(r"\w+")
df = pd.read_csv("../input/nlp-getting-started/train.csv") 

df.head()
df.shape
df.columns
df.dtypes
df_target = df["target"].value_counts()

fig = go.Figure([go.Pie(labels=df_target.index, values=df_target.values

                        ,hole=0.5)])  # can change the size of hole 



fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=15)

fig.update_layout(title="Disaster Tweets target columns distribution",title_x=0.5)

fig.show()
df_columns = df.columns.tolist()

null_value_counts = df[df_columns].isnull().sum()



fig = go.Figure(go.Bar(

    x=null_value_counts.index,y=null_value_counts.values,text=null_value_counts.values,

    textposition = "outside",

))

fig.update_layout(title_text='Null value counts',xaxis_title="Column name",yaxis_title="Counts of null values")

fig.show()
df['keyword'] = df['keyword'].astype(str)

df['text'] = df[['keyword', 'text']].apply(lambda x: ' '.join(x), axis = 1) 
df = df.drop(["location","keyword"],axis = 1)
def clean_text(text, ):

        if text is not None:

        #exclusions = ['RE:', 'Re:', 're:']

        #exclusions = '|'.join(exclusions)

                text = re.sub(r'[0-9]+','',text)

                text =  text.lower()

                text = re.sub('re:', '', text)

                text = re.sub('-', '', text)

                text = re.sub('_', '', text)

                text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

                text = re.sub(r'\S*@\S*\s?', '', text, flags=re.MULTILINE)

        # Remove text between square brackets

                text =re.sub('\[[^]]*\]', '', text)

        # removes punctuation

                text = re.sub(r'[^\w\s]','',text)

                text = re.sub(r'\n',' ',text)

                text = re.sub(r'[0-9]+','',text)

                #text = re.sub(r'[0-9]+','',text)

        # strip html 

                p = re.compile(r'<.*?>')

                text = re.sub(r"\'ve", " have ", text)

                text = re.sub(r"can't", "cannot ", text)

                text = re.sub(r"n't", " not ", text)

                text = re.sub(r"I'm", "I am", text)

                text = re.sub(r" m ", " am ", text)

                text = re.sub(r"\'re", " are ", text)

                text = re.sub(r"\'d", " would ", text)

                text = re.sub(r"\'ll", " will ", text)

        

                text = p.sub('', text)



        def tokenize_text(text,tokenizer=default_tokenizer):

            token = default_tokenizer.tokenize(text)

            return token

        

        def remove_stopwords(text, stop_words=default_stopwords):

            tokens = [w for w in tokenize_text(text) if w not in stop_words]

            return ' '.join(tokens)



        def stem_text(text, stemmer=default_stemmer):

            tokens = tokenize_text(text)

            return ' '.join([stemmer.stem(t) for t in tokens])



        text = stem_text(text) # stemming

        text = remove_stopwords(text) # remove stopwords

        #text.strip(' ') # strip whitespaces again?



        return text
df['text'] = df['text'].apply(clean_text)

tweet_text_list = df.text.tolist()

tweet_text_string = ''.join(tweet_text_list)
high_freq_word = pd.Series(' '.join(df['text']).split()).value_counts()[:20]
fig = go.Figure(go.Bar(y=high_freq_word.index, x=high_freq_word.values,orientation="h",marker={'color': high_freq_word.values,'colorscale': 'Viridis'} ))

fig.update_layout(title_text='Search most frequent word use in text column',xaxis_title="Count",yaxis_title="Words")

fig.show()
wordcloud_ip = WordCloud(

                      background_color='black',

                      margin=3,

                      width=1800,

                      height=1400,

                      max_words=200

                     ).generate(tweet_text_string)



plt.figure( figsize=(20,10) )

plt.imshow(wordcloud_ip)
cv = TfidfVectorizer(max_features = 1000)

x = cv.fit_transform(df['text'])

df1 = pd.DataFrame(x.toarray(), columns=cv.get_feature_names())

df.drop(["text"], axis=1, inplace=True)

main_df = pd.concat([df,df1], axis=1)
main_df.head()
Y = main_df.iloc[:,1]

X = main_df.iloc[:,2:]
rfc = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=2000,criterion="entropy")

model=rfc.fit(X,Y)
rfc.score(X,Y)
rfc_predict = rfc.predict(X)
print(classification_report(Y, rfc_predict))
confusion_matrix(Y, rfc_predict)
test = pd.read_csv("../input/nlp-getting-started/test.csv",usecols=["text","id"])

test.shape
test['text'] = test['text'].apply(clean_text)

vect = cv.transform(test['text']).toarray()

test["target"] = rfc.predict(vect)
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv",usecols=["id"])

submission
submission = submission.set_index('id').join(test.set_index('id'))

submission = submission.drop(["text"],axis=1)
submission["target"].value_counts()
submission.to_csv("submission.csv")