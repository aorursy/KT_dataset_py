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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder
le = LabelEncoder()
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import missingno as miss
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import iplot
import cufflinks as cf
cf.go_offline()
from tqdm import tqdm

from nltk.corpus import stopwords    
from nltk.tokenize import word_tokenize
from textblob import TextBlob

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, MaxPooling1D, Conv1D, Concatenate, Bidirectional, GlobalMaxPool1D, ActivityRegularization, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
df = pd.read_csv("/kaggle/input/sentiment-analysis-for-financial-news/all-data.csv", encoding='latin-1', header = None)
df.columns = ["Sentiment", "News Headline"]
df.head()
df.shape
df.info()
df.describe(include = "all")
df.isna().sum()
miss.bar(df)
plt.show()
#check for duplicates

len(df[df.duplicated()])
df = df.drop_duplicates()
print(df.head())
print(df.shape)
df['nr_of_char'] = df['News Headline'].str.len()
df['nr_of_char'] = df['nr_of_char'] / df['nr_of_char'].max()
df[['Sentiment', 'nr_of_char']].pivot(columns = 'Sentiment', values = 'nr_of_char').iplot(kind = 'box')
df['nr_of_words'] = df['News Headline'].str.split().str.len()
df['nr_of_words'] = df['nr_of_words'] / df['nr_of_char'].max()
df[['Sentiment', 'nr_of_words']].pivot(columns = 'Sentiment', values = 'nr_of_words').iplot(kind = 'box')
df['nr_of_unique_words'] = df['News Headline'].apply(lambda x: len(set(x.split())))
df['nr_of_unique_words'] = df['nr_of_unique_words'] / df['nr_of_unique_words'].max()
df[['Sentiment', 'nr_of_unique_words']].pivot(columns = 'Sentiment', values = 'nr_of_unique_words').iplot(kind='box')
df['nr_of_punctuation'] = df['News Headline'].str.split(r"\?|,|\.|\!|\"|'").str.len()
df['nr_of_punctuation'] = df['nr_of_punctuation'] / df['nr_of_punctuation'].max()
df[['Sentiment', 'nr_of_punctuation']].pivot(columns = 'Sentiment', values = 'nr_of_punctuation').iplot(kind = 'box')
stop_words = set(stopwords.words('english'))
df['nr_of_stopwords'] = df['News Headline'].str.split().apply(lambda x: len(set(x) & stop_words))
df['nr_of_stopwords'] = df['nr_of_stopwords'] / df['nr_of_stopwords'].max()
df[['Sentiment', 'nr_of_stopwords']].pivot(columns = 'Sentiment', values = 'nr_of_stopwords').iplot(kind = 'box')
df.corr().iplot(kind='heatmap',colorscale="YlGnBu",title="Feature Correlation Matrix")
df.insert(0, 'Id', range(1, 1 + len(df))) #defining custom Id column
def show_donut_plot(col): #donut plot function
    
    rating_data = df.groupby(col)[['Id']].count().head(10)
    plt.figure(figsize = (12, 8))
    plt.pie(rating_data[['Id']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)

    # create a center circle for more aesthetics to make it better
    gap = plt.Circle((0, 0), 0.5, fc = 'white')
    fig = plt.gcf()
    fig.gca().add_artist(gap)
    
    plt.axis('equal')
    
    cols = []
    for index, row in rating_data.iterrows():
        cols.append(index)
    plt.legend(cols)
    
    plt.title('Donut Plot: Reviews \n', loc='center')
    plt.show()
show_donut_plot('Sentiment')
import re
import spacy
nlp = spacy.load('en')

def normalize(msg):
    
    msg = re.sub('[^A-Za-z]+', ' ', msg) #remove special character and intergers
    doc = nlp(msg)
    res=[]
    for token in doc:
        if(token.is_stop or token.is_punct or token.is_currency or token.is_space or len(token.text) <= 2): #word filteration
            pass
        else:
            res.append(token.lemma_.lower())
    return res
df["News Headline"] = df["News Headline"].apply(normalize)
df.head()
words_collection = Counter([item for sublist in df['News Headline'] for item in sublist])
freq_word_df = pd.DataFrame(words_collection.most_common(20))
freq_word_df.columns = ['frequently_used_word','count']
fig = px.scatter(freq_word_df, x="frequently_used_word", y="count", color="count", title = 'Frequently used words - Scatter plot')
fig.show()
df["News Headline"] = df["News Headline"].apply(lambda x : " ".join(x))
df = df[["News Headline", "Sentiment"]]
df["Sentiment"] = le.fit_transform(df["Sentiment"])
df.head()
rename = {"News Headline": "text", "Sentiment": "labels"}
df.rename(columns = rename, inplace=True)
!pip install transformers
!pip install simpletransformers
train_x_y = df.sample(frac = 0.75, random_state = 42)
test_x_y = pd.concat([df, train_x_y]).drop_duplicates(keep=False)
print(train_x_y.shape)
print(test_x_y.shape)
from simpletransformers.classification import ClassificationModel, ClassificationArgs


model_args = ClassificationArgs()
model_args.train_batch_size = 2
model_args.gradient_accumulation_steps = 8
model_args.learning_rate = 3e-5
model_args.num_train_epochs = 1

model_bert = ClassificationModel("bert", "bert-base-uncased", num_labels=3, args=model_args, use_cuda=False)
model_bert.train_model(train_x_y)
pred_bert, out_bert = model_bert.predict(test_x_y['text'].values)

acc_bert = accuracy_score(test_x_y['labels'].to_numpy(), pred_bert)
f1_bert = f1_score(test_x_y['labels'].to_numpy(), pred_bert, average='micro')

print("Accuracy score -->", acc_bert)
print("F1 score -->", f1_bert)
#graph with confusion matrix

cm = confusion_matrix(pred_bert, test_x_y['labels'].to_numpy())
#group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(3,3)
sns.heatmap(cm, annot=labels, fmt="", cmap='Blues')
plt.show()
fig,ax=plt.subplots(figsize=(10,5))
sns.regplot(x=pred_bert, y=test_x_y['labels'].to_numpy(),marker="*")
plt.show()