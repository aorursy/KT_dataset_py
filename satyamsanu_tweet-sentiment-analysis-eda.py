#Importing Libararies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Reading Files
df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
df_ss = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
#Checking size
print(df_train.size)
print(df_test.size)
#Checking Shape
print(df_train.shape)
print(df_test.shape)
#Some basic info about data
print(df_train.info())
print(df_test.info())
df_train.head()
df_test.head()
#Cehcking the null values
df_train.isnull().sum()
#Removing the null values
df_train.dropna(inplace=True)
df_train.isnull().sum()
#Visualization of distribution of sentiments
f, axes = plt.subplots(ncols=2 ,figsize=(15, 5))
sns.countplot(df_train['sentiment'],ax=axes[0]).set_title('Sentiment')
sns.countplot(df_test['sentiment'],ax=axes[1]).set_title('Sentiment')

#Percenatge wise distribution of sentiments
print(df_train['sentiment'].value_counts(normalize=True))
print(df_test['sentiment'].value_counts(normalize=True))
df = df_train
ax = sns.countplot(y="sentiment", data=df )
plt.title('Distribution of  Configurations')
plt.xlabel('Number of Axles')

total = len(df['sentiment'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.show()
df = df_test
ax = sns.countplot(y="sentiment", data=df )
plt.title('Distribution of  Configurations')
plt.xlabel('Number of Axles')

total = len(df['sentiment'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.show()
#Text Preprocessing
#Text preprocessing helper functions
import nltk
import re
import string
from nltk.corpus import stopwords

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(tokenized_text)
    return combined_text
# Applying the cleaning function to both test and training datasets
df_train['text_clean'] = df_train['text'].apply(str).apply(lambda x: text_preprocessing(x))
df_test['text_clean'] = df_train['text'].apply(str).apply(lambda x: text_preprocessing(x))

df_train['selected_text_clean'] = df_train['selected_text'].apply(str).apply(lambda x: text_preprocessing(x))
df_test['selected_text_clean'] = df_train['selected_text'].apply(str).apply(lambda x: text_preprocessing(x))
df_train.head()
#Adding new Columns
df_train['text_len'] = df_train['text'].astype(str).apply(len)
df_train['Num_words_ST'] = df_train['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text
df_train['Num_word_text'] = df_train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text
df_train['difference_in_words'] = df_train['Num_word_text'] - df_train['Num_words_ST'] #Difference in Number of words text and Selected Text
df_train.head()
#Text length distribution Visualization
f, axes = plt.subplots(nrows=3 ,figsize=(15, 12))
sns.distplot(df_train[df_train['sentiment'] == 'positive']["text_len"],ax=axes[0],bins=75, color="#009c1a", axlabel=False).set_title('Positive Sentiment')
sns.distplot(df_train[df_train['sentiment'] == 'negative']["text_len"],ax=axes[1],bins=75, color="#e3170a", axlabel=False).set_title('Negative Sentiment')
sns.distplot(df_train[df_train['sentiment'] == 'neutral']["text_len"],ax=axes[2],bins=75, color="#011f4b", axlabel=False).set_title('Neutral Sentiment')
#Color palettes
colors = ["#e9d758", "#ff8552", "#338e8e", "#e6e6e6", "#39393a"]
Red = ["#e3170a", "#e04135", "#d69e9a", "#d6c2c0", "#d6cecd"]
Blue = ["#011f4b","#03396c","#005b96","#6497b1","#b3cde0"]
Green = ["#009c1a","#22b600","#26cc00","#7be382","#d2f2d4"]
#Number of words in Text  Visualization
f, axes = plt.subplots(nrows=3 ,figsize=(15, 10))
sns.countplot(df_train[df_train['sentiment'] == 'positive']["Num_word_text"],ax=axes[0], palette=Green)
sns.countplot(df_train[df_train['sentiment'] == 'negative']["Num_word_text"],ax=axes[1], palette=Red)
sns.countplot(df_train[df_train['sentiment'] == 'neutral']["Num_word_text"],ax=axes[2], palette=Blue)
#Number of words in Selected Text  Visualization
f, axes = plt.subplots(nrows=3 ,figsize=(15, 10))
sns.countplot(df_train[df_train['sentiment'] == 'positive']["Num_words_ST"],ax=axes[0], color="#009c1a")
sns.countplot(df_train[df_train['sentiment'] == 'negative']["Num_words_ST"],ax=axes[1], color="#e3170a")
sns.countplot(df_train[df_train['sentiment'] == 'neutral']["Num_words_ST"],ax=axes[2], color="#011f4b")
#Number of words in Text and Selected Text Visulaization
fig = plt.figure(figsize=(20, 5))
sns.kdeplot(df_train["Num_words_ST"],shade=True, color="#011f4b")
sns.kdeplot(df_train["Num_word_text"], shade=True, color="#e3170a")
plt.legend()
plt.show()
#Common Words from Text_Clean

from collections import Counter
temp_list = df_train['text_clean'].apply(lambda x:str(x).split())
top = Counter([item for sublist in temp_list for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap="bone_r")
#Common Words from selected_text_clean

from collections import Counter
temp_list = df_train['selected_text_clean'].apply(lambda x:str(x).split())
top = Counter([item for sublist in temp_list for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='bone_r')
#positive
from collections import Counter
temp_list = df_train[df_train['sentiment']=='positive']['text_clean'].apply(lambda x:str(x).split())
top = Counter([item for sublist in temp_list for item in sublist])
temp1 = pd.DataFrame(top.most_common(20))
temp1.columns = ['Common_words','count']
temp1.style.background_gradient(cmap='Greens')
#neagtive
temp_list = df_train[df_train['sentiment']=='negative']['text_clean'].apply(lambda x:str(x).split())
top = Counter([item for sublist in temp_list for item in sublist])
temp1 = pd.DataFrame(top.most_common(20))
temp1.columns = ['Common_words','count']
temp1.style.background_gradient(cmap='Reds')
#neagtive
temp_list = df_train[df_train['sentiment']=='neutral']['text_clean'].apply(lambda x:str(x).split())
top = Counter([item for sublist in temp_list for item in sublist])
temp1 = pd.DataFrame(top.most_common(20))
temp1.columns = ['Common_words','count']
temp1.style.background_gradient(cmap='Blues')
#Wordcloud
f, ax = plt.subplots(nrows=3 ,figsize=(20,35))

#Positive
from wordcloud import WordCloud
cloud1 = WordCloud(width=1440, height=1080).generate(str(df_train[df_train['sentiment']=='positive']["text_clean"]))
ax[0].imshow(cloud1)
ax[0].axis('off')
ax[0].set_title('Positive',fontsize=25);

#Neutral
cloud2 = WordCloud(width=1440, height=1080).generate(str(df_train[df_train['sentiment']=='neutral']["text_clean"]))
ax[1].imshow(cloud2)
ax[1].axis('off')
ax[1].set_title('Neutral',fontsize=25);

#Negative
cloud3 = WordCloud(width=1440, height=1080).generate(str(df_train[df_train['sentiment']=='negative']["text_clean"]))
ax[2].imshow(cloud3)
ax[2].axis('off')
ax[2].set_title('Negative',fontsize=25);
#
pos = df_train[df_train['sentiment']=='positive']
neg = df_train[df_train['sentiment']=='negative']
neutral = df_train[df_train['sentiment']=='neutral']

##source of code : https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]





#for word, freq in top_unigrams:
    #print(word, freq)

pos_unigrams = get_top_n_words(pos['text_clean'],30)
neg_unigrams = get_top_n_words(neg['text_clean'],30)
neutral_unigrams = get_top_n_words(neutral['text_clean'],30)    
    
plt.subplots(figsize=(15,5))
df1 = pd.DataFrame(pos_unigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(
    kind='bar',color='#009c1a', title='Top 20 Unigrams in positve text')

plt.subplots(figsize=(15,5))
df2 = pd.DataFrame(neg_unigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(
     kind='bar',  color='#e3170a',title='Top 20 Unigrams in negative text')

plt.subplots(figsize=(15,5))
df3 = pd.DataFrame(neutral_unigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(
     kind='bar',color="#011f4b" ,title='Top 20 Unigrams in neutral text')
def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


pos_bigrams = get_top_n_gram(pos['text_clean'],(2,2),30)
neg_bigrams = get_top_n_gram(neg['text_clean'],(2,2),30)
neutral_bigrams = get_top_n_gram(neutral['text_clean'],(2,2),30)


#for word, freq in top_bigrams:
    #print(word, freq)
    
plt.subplots(figsize=(15,5))
df1 = pd.DataFrame(pos_bigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(
    kind='bar',color='#009c1a', title='Top 20 Bigrams in positve text')
   
    
plt.subplots(figsize=(15,5))
df2 = pd.DataFrame(neg_bigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(
     kind='bar',  color='#e3170a',title='Top 20 Bigrams in negative text')

plt.subplots(figsize=(15,5))
df3 = pd.DataFrame(neutral_bigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(
     kind='bar',color="#011f4b" ,title='Top 20 Bigrams in neutral text')

