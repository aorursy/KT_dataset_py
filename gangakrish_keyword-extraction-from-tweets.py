# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from textblob import TextBlob

import re

import itertools

import datetime

import csv



# Download Wordnet through NLTK in python console:

import nltk

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer 

from nltk.tokenize import word_tokenize



import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

import unidecode

import string



from nltk.probability import FreqDist



import matplotlib.pyplot as plt

import seaborn as sns

import string

%matplotlib inline

#from plotly import graph_objs as go

#import plotly.express as px

#import plotly.figure_factory as ff



#sentiment analyser packages



from nltk.sentiment.vader import SentimentIntensityAnalyzer



from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC  

from sklearn.datasets import load_files

from sklearn.model_selection import GridSearchCV

import numpy as np

#import mglearn

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline



from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import f1_score, accuracy_score

from sklearn.metrics import roc_auc_score



#import fasttext



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



'''import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

train_df = train_df[train_df['text'].notna()]

#train_df = train_df.head(1000)

train_df = train_df.reset_index()

train_df
train_df.info()
def word_check(word, list):

    if word in list:

        return 1

    else:

        return 0

    

def word_cooccurance(word1,word2,candi_kw_lst):

    value = 0

    for k in range(len(candi_kw_lst)) :

        value  = value + check_both(word1,word2,candi_kw_lst[k])

    

    return value



def check_both(word1, word2 , list): 

    if word1 in list:

        if word2 in list:

            return 1

        else:

            return 0

    else:

        return 0

    

def word_freq(word,list1):

    return list1.count(word)

    



def strip_links(text):

    text = str(text)

    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)

    links         = re.findall(link_regex, text)

    for link in links:

        text = text.replace(link[0], '')    

    return text







def get_tweet_sentiment(tweet): 

    ''' 

    Utility function to classify sentiment of passed tweet 

    using textblob's sentiment method 

    '''

    # create TextBlob object of passed tweet text 

    analysis = TextBlob(clean_tweet(tweet)) 

    # set sentiment 

    if analysis.sentiment.polarity > 0: 

        return 'positive'

    elif analysis.sentiment.polarity == 0: 

        return 'neutral'

    else: 

        return 'negative'

    

def clean_tweet(tweet): 

    ''' 

    Utility function to clean tweet text by removing links, special characters 

    using simple regex statements. 

    '''

    tweet = str(tweet)

    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())



def textblob_sentiment(tweet):

    pol_score = TextBlob(tweet).sentiment.polarity

    if pol_score > 0: 

        return 'positive'

    elif pol_score == 0: 

        return 'neutral'

    else: 

        return 'negative'



def vader_sentiment(tweet):

    senti = SentimentIntensityAnalyzer()

    compound_score = senti.polarity_scores(tweet)['compound']

    

    # set sentiment 

    if compound_score >= 0.05: 

        return 'positive'

    elif (compound_score > -0.05) and (compound_score < 0.05): 

        return 'neutral'

    else: 

        return 'negative'

    







def document_features(document):

    document_words = set(document)

    features = {}

    for word in word_features:

        features['contains({})'.format(word)] = (word in document_words)

    return features

stop_words = set(stopwords.words('english'))



appos = {

"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "i would",

"i'd" : "i had",

"i'll" : "i will",

"i'm" : "i am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "i have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"that's" : "that is",

"there's" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not"

}







def text_preprocess(text):

    lemma = nltk.wordnet.WordNetLemmatizer()

    

    text = str(text)

    

    #removing mentions and hashtags



    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", text).split())

    

    #remove http links from tweets

    

    

    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)

    links         = re.findall(link_regex, text)

    for link in links:

        text = text.replace(link[0], '')  

    

    text_pattern = re.sub("`", "'", text)

    

    #fix misspelled words



    '''Here we are not actually building any complex function to correct the misspelled words but just checking that each character 

    should occur not more than 2 times in every word. It’s a very basic misspelling check.'''



    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))

    

    

   # print(text_pattern)

    

    #Convert to lower and negation handling

    

    text_lr = text_pattern.lower()

    

   # print(text_lr)

    

    words = text_lr.split()

    text_neg = [appos[word] if word in appos else word for word in words]

    text_neg = " ".join(text_neg) 

   # print(text_neg)

    

    #remove stopwords

    

    tokens = word_tokenize(text_neg)

    text_nsw = [i for i in tokens if i not in stop_words]

    text_nsw = " ".join(text_nsw) 

   # print(text_nsw)

    

    

    #remove tags

    

    text_tags=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text_nsw)



    # remove special characters and digits

    text_alpha=re.sub("(\\d|\\W)+"," ",text_tags)

    

    #Remove accented characters

    text = unidecode.unidecode(text_alpha)

    

    '''#Remove punctuation

    table = str.maketrans('', '', string.punctuation)

    text = [w.translate(table) for w in text.split()]'''

    

    sent = TextBlob(text)

    tag_dict = {"J": 'a', 

                "N": 'n', 

                "V": 'v', 

                "R": 'r'}

    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    

    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]

   

    return " ".join(lemmatized_list)

   
train_df['processed_text'] = None

#train_df['clean_text2'] = None



for i in range(len(train_df)):

    train_df.processed_text[i] = text_preprocess(train_df.text[i])

import matplotlib.pyplot as plt

ax = train_df['sentiment'].value_counts(sort=False).plot(kind='barh')

ax.set_xlabel('Number of Samples in training Set')

ax.set_ylabel('Label')
from wordcloud import WordCloud

import matplotlib.pyplot as plt



# Polarity ==  negative

train_s0 = train_df[train_df.sentiment == 'negative']

all_text = ' '.join(word for word in train_s0.processed_text)

wordcloud_neg = WordCloud(colormap='Reds', width=1000, height=1000, background_color='white').generate(all_text) #mode='RGBA'

plt.figure(figsize=(20,10))

plt.title('Negative sentiment - Wordcloud')

plt.imshow(wordcloud_neg, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()



wordcloud_neg.to_file('negative_senti_wordcloud.jpg')



# Polarity ==  neutral

train_s1 = train_df[train_df.sentiment == 'neutral']

all_text = ' '.join(word for word in train_s1.processed_text)

wordcloud_neu = WordCloud(width=1000, height=1000, colormap='Blues', background_color='white').generate(all_text)

plt.figure( figsize=(20,10))

plt.title('Neutral sentiment - Wordcloud')

plt.imshow(wordcloud_neu, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()



wordcloud_neu.to_file('neutral_senti_wordcloud.jpg')



# Polarity ==  positive

train_s2 = train_df[train_df.sentiment  == 'positive']

all_text = ' '.join(word for word in train_s2.processed_text)

wordcloud_pos = WordCloud(width=1000, height=1000, colormap='Wistia',background_color='white').generate(all_text)

plt.figure(figsize=(20,10))

plt.title('Positive sentiment - Wordcloud')

plt.imshow(wordcloud_pos, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()



wordcloud_pos.to_file('positive_senti_wordcloud.jpg')

# wordcloud for frequently occuring bigrams



import nltk

from string import digits



# Load default stop words and add a few more.



stopwordsList = []

 

# Load default stop words and add a few more specific to my text.

stopwordsList = stopwords.words('english')

stopwordsList.append('dont')

stopwordsList.append('didnt')

stopwordsList.append('doesnt')

stopwordsList.append('cant')

stopwordsList.append('couldnt')

stopwordsList.append('couldve')

stopwordsList.append('im')

stopwordsList.append('ive')

stopwordsList.append('isnt')

stopwordsList.append('theres')

stopwordsList.append('wasnt')

stopwordsList.append('wouldnt')

stopwordsList.append('a')

stopwordsList.append('also')

stopwordsList.append('rt')





WNL = nltk.WordNetLemmatizer()



text_content = train_df['processed_text']



# After the punctuation above is removed it still leaves empty entries in the list.

text_content = [s for s in text_content if len(s) != 0]



# Best to get the lemmas of each word to reduce the number of similar words

text_content = [WNL.lemmatize(t) for t in text_content]





#nltk_tokens = nltk.word_tokenize(text)  



bigrams_list = list(nltk.bigrams(text_content))

#print(bigrams_list)



dictionary2 = [' '.join(tup) for tup in bigrams_list]

#print (dictionary2)



vectorizer = CountVectorizer(ngram_range=(2, 2))

bag_of_words = vectorizer.fit_transform(dictionary2)

vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0) 

words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

#print (words_freq[:100])

words_dict = dict(words_freq)



WC_height = 1000

WC_width = 1500

WC_max_words = 200



wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,stopwords=stopwordsList,background_color='white')



wordCloud.generate_from_frequencies(words_dict)



plt.figure(figsize=(20,10))

plt.title('Most frequently occurring bigrams connected by colour')

plt.imshow(wordCloud, interpolation='bilinear')

plt.axis("off")

plt.show()



wordCloud.to_file('wordcloud_freq_bigrams.jpg')
# function to collect hashtags

def hashtag_extract(x):

    hashtags = []

    # Loop over the words in the tweet

    for i in x:

        ht = re.findall(r"#(\w+)", i)

        hashtags.append(ht)



    return hashtags
# extracting hashtags from positive tweets



HT_positive = hashtag_extract(train_df['text'][train_df['sentiment'] == 'positive'])



# extracting hashtags from negative tweets

HT_negative = hashtag_extract(train_df['text'][train_df['sentiment'] == 'negative'])



# extracting hashtags from neutral tweets

HT_neutral = hashtag_extract(train_df['text'][train_df['sentiment'] == 'neutral'])





# unnesting list

HT_positive = sum(HT_positive,[])

HT_negative = sum(HT_negative,[])

HT_neutral = sum(HT_neutral,[])

# hashtags contributing to positive tweets



a = nltk.FreqDist(HT_positive)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})

# selecting top 10 most frequent hashtags     

d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(16,5))

plt.title('Hashtags contributing to positive tweets')

ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()
# hashtags contributing to negative tweets



b = nltk.FreqDist(HT_negative)

e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

# selecting top 10 most frequent hashtags

e = e.nlargest(columns="Count", n = 10)   

plt.figure(figsize=(16,5))

plt.title('Hashtags contributing to negative tweets')

ax = sns.barplot(data=e, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()

# hashtags contributing to neutral tweets



b = nltk.FreqDist(HT_neutral)

e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})



# selecting top 10 most frequent hashtags

e = e.nlargest(columns="Count", n = 10)   

plt.figure(figsize=(16,5))

plt.title('Hashtags contributing to neutral tweets')

ax = sns.barplot(data=e, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()

from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# bag-of-words feature matrix

bow = bow_vectorizer.fit_transform(train_df['text'])



top_sum=bow.toarray().sum(axis=0)

top_sum_cv=[top_sum]#to let pandas know that these are rows

columns_cv = bow_vectorizer.get_feature_names()

x_traincvdf = pd.DataFrame(top_sum_cv,columns=columns_cv)





import operator

dic = {}

for i in range(len(top_sum_cv[0])):

    dic[columns_cv[i]]=top_sum_cv[0][i]

sorted_dic=sorted(dic.items(),reverse=True,key=operator.itemgetter(1))

print(sorted_dic[1:])

from matplotlib import pyplot as plt



sorted_dic = sorted_dic[:15]



xs, ys = [*zip(*sorted_dic)]





plt.figure(figsize=(10,8))

plt.bar(xs, ys)

plt.xlabel('Words')

plt.ylabel('Frequency')

plt.title('Top words - Count Vectorizer')

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# TF-IDF feature matrix

tfidf = tfidf_vectorizer.fit_transform(train_df['text'])



top_sum=tfidf.toarray().sum(axis=0)

top_sum_tfidf=[top_sum]#to let pandas know that these are rows

columns_tfidf = tfidf_vectorizer.get_feature_names()

x_traintfidf_df = pd.DataFrame(top_sum_tfidf,columns=columns_tfidf)





import operator

dic = {}

for i in range(len(top_sum_tfidf[0])):

    dic[columns_cv[i]]=top_sum_tfidf[0][i]

sorted_dic=sorted(dic.items(),reverse=True,key=operator.itemgetter(1))

print(sorted_dic[1:])
from matplotlib import pyplot as plt



sorted_dic = sorted_dic[:15]



xs, ys = [*zip(*sorted_dic)]





plt.figure(figsize=(10,8))

plt.bar(xs, ys)

plt.xlabel('Words')

plt.ylabel('Frequency')

plt.title('Top words - Count Vectorizer')

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x='sentiment',data=train_df)
# loading test data

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

test_df = test_df[test_df['text'].notna()]

test_df = test_df.reset_index()

test_df

from nltk.tokenize import word_tokenize

from collections import Counter

from collections import OrderedDict 



import nltk

from nltk.corpus import stopwords

stopwords_en = set(stopwords.words('english'))





text = 'my boss is bullying me...'

print (text)
'''step 1 : lower each word and tokenize each word in sentences (text)'''



text = text.lower()

text = clean_tweet(text)



print (text)



tokenized_sents = word_tokenize(text)

print (tokenized_sents)



'''step 2 : a)Split by delimiters       b)Split by stop word        c)Candidate Keyword'''



candi_kw = []

candi_kw_lst = []



for i in tokenized_sents:

    

    if i not in stopwords_en:

        candi_kw.append(i)

    else:

        if len(candi_kw) == 0:

            print ('stopword')

        else:

            candi_kw_lst.append(candi_kw) 

            candi_kw = []

candi_kw_lst.append(candi_kw)
flat_list_keyword = [item for sublist in candi_kw_lst for item in sublist]

flat_unique_keyword = set(flat_list_keyword)



flat_unique_list = [] 

      

# traverse for all elements 

for x in flat_list_keyword: 

    # check if exists in unique_list or not 

    if x not in flat_unique_list: 

        flat_unique_list.append(x) 

# initializing cooccurance matrix



matrix_df = pd.DataFrame(0, columns=flat_unique_list, index=flat_unique_list)

matrix_df
candi_kw_lst
# creating co-occurance matrix



j = 0



for i in range(len(matrix_df)):

    for j in range(len(matrix_df.columns)):

        if (matrix_df.index[i] == matrix_df.columns[j]):

            matrix_df.iloc[i,j] = flat_list_keyword.count(matrix_df.index[j])

        else:

            value = 0

            for k in range(len(candi_kw_lst)) :

                value  = value + check_both(matrix_df.index[i],matrix_df.columns[j],candi_kw_lst[k])

            matrix_df.iloc[i,j] = value
# sum row wise and create column for word degree



'''# find degree of word 



Word Degree (deg(w)) = word_freq+ # howmany times a word has a interaction with other words



'''



matrix_df['degree'] = matrix_df.sum(axis=1)
# word_frequency



'''# find frequency of word



Word Frequency (freq(w)) # how many times a particular word appeared among all candidate keywords.



Keyword score = (deg(w)/freq(w))'''



matrix_df['word_frequency'] = None



for i in range(len(matrix_df)):

    

    matrix_df.word_frequency[i] = word_freq(matrix_df.index[i])

    

# calculate keyword score



matrix_df['keyword_score'] = None



for i in range(len(matrix_df)):

    

    matrix_df.keyword_score[i] = matrix_df.degree[i] / matrix_df.word_frequency[i]



matrix_df
#getting keyword_score for each word



matrix_dict = matrix_df.to_dict()['keyword_score']

matrix_dict
#calculate keyword score for candidate keywords

candi_kw_score = {}



for i in range(len(candi_kw_lst)):

    score = 0

    for j in range(len(candi_kw_lst[i])):

        key_name = str(candi_kw_lst[i])

        score = score + matrix_dict[candi_kw_lst[i][j]]

    

    candi_kw_score [key_name] = score

    



candi_kw_score = {k: v for k, v in sorted(candi_kw_score.items(), key=lambda item: item[1],reverse=True)}



candi_kw_score
# extract top 3 scored candidate keywords



n_items = dict(itertools.islice(candi_kw_score.items(), 4)) 



final_phrase = []



for x in list(n_items)[0:3]:

    final_phrase.append(x)



# removing special characters



removetable = str.maketrans('', '', "@,#%[]'")

out_list = [s.translate(removetable) for s in final_phrase]



extracted_keywords = ' '.join(out_list)

extracted_keywords
def extract_keywords(text):

    

    global key_name

    

    text = text.lower()

    

    text = clean_tweet(text)



    tokenized_sents = word_tokenize(text)

    

    candi_kw = []

    candi_kw_lst = []



    for i in tokenized_sents:



        if i not in stopwords_en:

            candi_kw.append(i)

        else:

            if len(candi_kw) == 0:

                pass 

            else:

                candi_kw_lst.append(candi_kw) 

                candi_kw = []

    candi_kw_lst.append(candi_kw)

    

    flat_list_keyword = [item for sublist in candi_kw_lst for item in sublist]

    flat_unique_keyword = set(flat_list_keyword)



    flat_unique_list = [] 



    # traverse for all elements 

    for x in flat_list_keyword: 

        # check if exists in unique_list or not 

        if x not in flat_unique_list: 

            flat_unique_list.append(x) 

            

    # initializing cooccurance matrix



    matrix_df = pd.DataFrame(0, columns=flat_unique_list, index=flat_unique_list)



    

    # creating co-occurance matrix



    j = 0



    for i in range(len(matrix_df)):

        for j in range(len(matrix_df.columns)):

            if (matrix_df.index[i] == matrix_df.columns[j]):

                matrix_df.iloc[i,j] = flat_list_keyword.count(matrix_df.index[j])

            else:

                value = 0

                for k in range(len(candi_kw_lst)) :

                    value  = value + check_both(matrix_df.index[i],matrix_df.columns[j],candi_kw_lst[k])

                matrix_df.iloc[i,j] = value



        # sum row wise and create column for word degree



    '''# find degree of word 



    Word Degree (deg(w)) = word_freq+ # howmany times a word has a interaction with other words



    '''



    matrix_df['degree'] = matrix_df.sum(axis=1)

    

    # word_frequency



    '''# find frequency of word



    Word Frequency (freq(w)) # how many times a particular word appeared among all candidate keywords.



    Keyword score = (deg(w)/freq(w))'''



    matrix_df['word_frequency'] = None



    for i in range(len(matrix_df)):



        matrix_df.word_frequency[i] = word_freq(matrix_df.index[i],flat_list_keyword)



    

        # calculate keyword score



    matrix_df['keyword_score'] = None



    for i in range(len(matrix_df)):



        matrix_df.keyword_score[i] = matrix_df.degree[i] / matrix_df.word_frequency[i]



   

    #getting keyword_score for each word



    matrix_dict = matrix_df.to_dict()['keyword_score']

    

    #calculate keyword score for candidate keywords

    

    candi_kw_score = {}



    for i in range(len(candi_kw_lst)):

        score = 0

        for j in range(len(candi_kw_lst[i])):

            key_name = str(candi_kw_lst[i])

            score = score + matrix_dict[candi_kw_lst[i][j]]

        candi_kw_score [key_name] = score





    candi_kw_score = {k: v for k, v in sorted(candi_kw_score.items(), key=lambda item: item[1],reverse=True)}

    

    # extract top 3 scored candidate keywords



    n_items = dict(itertools.islice(candi_kw_score.items(), 10)) 



    final_phrase = []



    for x in list(n_items)[0:7]:

        final_phrase.append(x)



    # removing special characters



    removetable = str.maketrans('', '', "@,#%[]'")

    out_list = [s.translate(removetable) for s in final_phrase]



    extracted_keywords = ' '.join(out_list)

    

    return extracted_keywords



  
# train set prediction based on RAKE algo:



# Select the top 5 rows of the Dataframe

train_df = train_df.head(4000)

train_df.reset_index()



train_df['predicted_text'] = None



for i in range(len(train_df)):

    train_df.predicted_text[i] = extract_keywords(train_df.text[i])
train_df
#evaluation on train set



def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



train_df['jaccard'] = None



for i in range(len(train_df)):

    train_df.jaccard[i] = jaccard(train_df.selected_text[i],train_df.predicted_text[i])
train_df["jaccard"].mean()
# test set prediction 



test_df ['selected_text'] = None



for i in range(len(test_df)):

    test_df.selected_text[i] = extract_keywords(test_df.text[i])
test_df
# Remove column name 'A' 

test_df1 = test_df.drop(['index','text','sentiment'], axis = 1) 

test_df1
test_df1.to_csv('submission.csv', index = False)