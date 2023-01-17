####Import Libraries#####
import pandas as pd
import numpy as np
import nltk
import warnings
warnings.filterwarnings('ignore')
####Data Loading#####
path = "D:\\Unstructed L0\\Kaggle Problem\\"
df = pd.read_csv(path+"data.csv")
df.head()
####Data Cleaning####
####Import Libraries###
import nltk
import re
import string
from string import punctuation
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize, pos_tag, pos_tag_sents
import collections
from nltk.corpus import stopwords # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
from nltk.tokenize import RegexpTokenizer
# ##Remove URL's#####
# def remove_urls(text):
#     new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
#     return new_text

# ##Lower Case######
# def text_lowercase(text):
#     return text.lower()

# ##make strings##
# def text_string(text):
#     return text.astype('str')

# ##remove numbers##
# def remove_numbers(text):
#     result = re.sub(r'\d+',' ',text)
#     return result

# ##remove punctutation##
# def remove_punctuation(text):
#     translator = str.maketrans('','',string.punctuation)
#     return text.translate(translator)

# ##tokenize####
# def tokenize(text):
#     text = word_tokenize(text)
#     return text

# ##remove stopwords###
# my_stop_words = list(punctuation) + list(text.ENGLISH_STOP_WORDS) +list('abcdefghijklmnopqrstuvwxyz')
# def remove_stopwords(text):
#     text = [i for i in text if not i in my_stop_words]
#     return text

# ##Lemmatize##
# lematizer = WordNetLemmatizer()
# def lemmatize(text):
#     text = [lematizer.lemmatize(token) for token in text]
#     return text
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# def preprocessing(text):
#     text = text_lowercase(text)
#     text = remove_urls(text)
#     text = remove_numbers(text)
#     text = remove_punctuation(text)
#     text = tokenize(text)
#     text = remove_stopwords(text)
#     text = lemmatize(text)
#     text = ' '.join(text)
#     return text.strip()
# df = df[df['text'].notnull()]
# df.shape
my_stop_words = list(punctuation) + list(text.ENGLISH_STOP_WORDS) +list('abcdefghijklmnopqrstuvwxyz')
def text_preprocessing(text):
    text = re.sub(r"http\S+", "", text) #remove URL \S+ matches all non-whitespace characters (the end of the url)
    text = re.sub(r"@\S+", "", text) #remove Username
    text = re.sub(r"\d+", "", text) #remove number
    text = text.translate(str.maketrans('', '', string.punctuation)) #Punctuation Removal
    text = text.encode('ascii', 'ignore').decode('ascii') # remove emojis
    text = [i for i in word_tokenize(text) if not i in my_stop_words] # Stopword Removal; Adding username here will take time
    text = ' '.join(text)
    return text.strip()
import string
from string import punctuation
from nltk.tokenize import word_tokenize
df['clean_text'] = df['text'].apply(lambda x: text_preprocessing(x))
df.head(3)
#df['Clean_text'] = df['text'].apply(lambda x: preprocessing(str(x)))
df.head()
df['clean_text']
df['text'][0]
len(df)
#############LDA using SKLearn#################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
cv = CountVectorizer(max_df = 0.95,min_df = 2, stop_words = 'english')
dtm = cv.fit_transform(df['clean_text'])
dtm
from sklearn.decomposition import LatentDirichletAllocation
# LDA = LatentDirichletAllocation(n_components = 5,random_state = 42, max_iter = 50, batch_size = 100)
LDA = LatentDirichletAllocation(n_components = 5,random_state = 120, max_iter = 50, batch_size = 148,learning_decay=0.5)
LDA.fit(dtm)
###########################GRID SEARCH APPROACH#############################
# Log Likelyhood: Higher the better
print("Log Likelihood: ", LDA.score(dtm))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", LDA.perplexity(dtm))

# See model parameters
print(LDA.get_params())
###Grid Search####
# Define Search Param
search_params = {'n_components': [5], 'learning_decay': [.5, .7, .9]}

# Init the Model
LDA = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(LDA, param_grid=search_params)

# Do the Grid Search
model.fit(dtm)
# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(dtm))

#Batch size
#print("Batch Size: ", model.batch_size(dtm))
#Grab vocab of words
len(cv.get_feature_names())
type(cv.get_feature_names())
cv.get_feature_names()[10000]
import random

random_word_id = random.randint(0,12623)
cv.get_feature_names()[random_word_id]
#Grab the topics
len(LDA.components_)
type(LDA.components_)
LDA.components_.shape
LDA.components_
###Consider a single topic###
single_topic = LDA.components_[0]
single_topic.argsort() #argsort arranges by index value from lowest to largest
###Top 10 greatest values of Argsort()###
single_topic.argsort()[-10:] ##Grab last 10 values of the index positions of argsort
top_ten_words = single_topic.argsort()[-10:]
for index in top_ten_words:
    print(cv.get_feature_names()[index])
topic_values = LDA.transform(dtm)
df['Topic'] = topic_values.argmax(axis=1)
df
#Grab highest prob words per topic
for index,topic in enumerate(LDA.components_):
    print(f"THE TOP 15 WORDS FOR TOPIC #{index}")
    print([cv.get_feature_names()[index] for index in topic.argsort()[-15:]])
    print('\n')
    print('\n')
topic_results = LDA.transform(dtm)
topic_results.shape
topic_results[0].round(2)
topic_results[0].argmax()
df['Topic'] = topic_results.argmax(axis = 1)
df.head()
# LDAtopic_dict = {0:'tech_news',1:'tech_news',2:'Automobiles',3:'Automobiles',4:'tech_news',5:'glassdoor_reviews',
#                 6:'sports_news',7:'room_rentals',8:'tech_news',9:'Automobiles',10:'tech_news',11:'Automobiles',12:'sports_news',
#                 13:'Automobiles',14:'Automobiles',15:'glassdoor_reviews',16:'tech_news',17:'sports_news',18:'glassdoor_reviews',
#                 19:'Automobiles'}
# df['Topic Label'] = df['Topic'].map(LDAtopic_dict)
LDAtopic_dict = {0:'room_rentals',1:'glassdoor_reviews',2:'sports_news',3:'Automobiles',4:'tech_news'}
df['Topic Label'] = df['Topic'].map(LDAtopic_dict)
# def f1(text1):
#     text1 = text1.lower()
#     if re.findall(r'(employee)|(job)|(opportunity)|(organization)|(company)|(honeywell)|(pay)|(people)|(work)|(benefit)|(layoff)|(global)|(change)|(growth)|(analyst)|(manager)|(portfolio)|(salary)|(balance)|(environment)|(analytics)|(business)|(management)',text1,flags = re.I):
#         return "glassdoor_reviews"
#     elif re.findall(r'(car)|(engine)|(oil)|(drive)|(gas)|(park)|(garrage)|(track)|(speed)|(fluid)|(tire)|(driver)|(road)|(cars)|(cur)|(pockets)|(volvo)',text1,flags = re.I):
#         return "Automobiles"
#     elif re.findall(r'(sports)|(manchester)|(chelsea)|(win)|(player)|(game)|(club)|(premier)|(season)|(league)|(race)|(united)|(city)|(cup)|(england)|(goal)|(play)|(adventure)|(talksport)|(points)',text1,flags = re.I):
#         return "sports_news"
#     elif re.findall(r'(temporary)|(university)|(retirement)|(article)|(deal)|(subject)|(writes)|(schedule)|(questionable)|(globally)|(strategy)|(corporate)|(industrial)|(competitive)|(retirement)|(phone)|(list)|(sign)|(deal)|(contract)|(insurance)|(email)|(read)|(distribution)|(rights)|(conslots)|(discount)|(internet)',text1,flags = re.I):
#         return "tech_news"
#     else:
#         return "room_rentals"
#df['topic'] = df['clean_text'].apply(lambda x: f1(str(x)))
df
df['Topic Label'].unique()
df1 = df[['Id','Topic Label']]
df1.rename(columns = {'Topic Label':'topic'},inplace = True)
df1.head()
df1.to_csv("D:\\Unstructed L0\\Kaggle Problem\\Output\\Final_LDA_new12.csv",index = False)
####NMF Model#####
df = df[['Id','text','clean_text']]
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df = 0.95,min_df = 2,stop_words = 'english')
dtm = tfidf.fit_transform(df['clean_text'])
dtm
from sklearn.decomposition import NMF
nmf_model = NMF(n_components = 5, random_state = 42,max_iter = 50)
nmf_model.fit(dtm)
for index,topic in enumerate(nmf_model.components_):
    print(f"THE TOP 15 WORDS FOR TOPIC # {index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
topic_results = nmf_model.transform(dtm)
df['topic'] = topic_results.argmax(axis = 1)
df
mytopic_dict = {0:'room_rentals',1:'glassdoor_reviews',2:'sports_news',3:'Automobiles',4:'tech_news'}
df['NMF_Label'] = df['topic'].map(mytopic_dict)
df.head()
df['NMF_Label'].unique()
df_NMF = df[['Id','NMF_Label']]
df_NMF.rename(columns = {'NMF_Label':'topic'},inplace = True)
df_NMF
df_NMF['topic'].unique()
df_NMF.to_csv("D:\\Unstructed L0\\Kaggle Problem\\Output\\Final_NMF_new6.csv",index = False)
#########LDA Using Gensim##########
df = df[['Id','text']]
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import ldamodel
import gensim.corpora
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import pyLDAvis.gensim as gensimvis
import pyLDAvis
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import datetime
import time
import os
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
#######Data Cleaning#######
def basic_clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub('[^A-z0-9]', ' ', text)
    text = ''.join([letter for letter in text if ord(letter) < 128])
    text = re.sub('\s+',' ',text).strip()
    return text

def remove_stopwords(text):
    text = text.split()
    text = [x for x in text if ((x not in my_stop_words) and (x not in st2))]
    return ' '.join(text)

def split_char_num(text):
    text = re.sub('([A-z]+)([0-9]+)',r'\1\2',text)
    return text

def remove_numbers(text):
    text = re.sub(r'[0-9]+',' ',text)
    return text

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    text = text.split()
    text = [lemmatizer.lemmatize(x) for x in text]
    text = [x for x in text if len(x)>2]
    text = list(set(text))
    return ' '.join(text)
my_stop_words = list(punctuation) + list(text.ENGLISH_STOP_WORDS) +list('abcdefghijklmnopqrstuvwxyz')
st2 = ['Disney']
new_text = list(df['text'])
new_text = [basic_clean_text(x) for x in new_text]
new_text = [basic_clean_text(remove_stopwords(x)) for x in new_text]
new_text = [basic_clean_text(split_char_num(x)) for x in new_text]
new_text = [basic_clean_text(remove_numbers(x)) for x in new_text]
new_text = [basic_clean_text(lemmatize_text(x)) for x in new_text]
new_text_1 = [x.split() for x in new_text]
id2word = gensim.corpora.Dictionary(new_text_1)
corpus = [id2word.doc2bow(text) for text in new_text_1]
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LDA(corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
print(datetime.datetime.now())
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=new_text_1, start=2, limit=40, step=6)
print(datetime.datetime.now())
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
num_topics = 10
LDA = gensim.models.ldamodel.LdaModel
start = time.time()
print (datetime.datetime.now())
lda = LDA(corpus, num_topics=num_topics, id2word = id2word, passes=11)
print ('time taken to run the model ' , time.time()-start)
print (datetime.datetime.now())
#####Getting LDA topics######
def get_lda_topics(model, num_topics):
    word_dict = {}
    topics = model.show_topics(10,20)
    word_dict = {'Topic '+str(i):[x.split('*') for x in words.split('+')] \
                 for i,words in lda.show_topics(10,20)}
    return pd.DataFrame.from_dict(word_dict)
get_lda_topics(lda,10)
df_top_cont = pd.DataFrame()
final_list = []
for tc in range(num_topics):
    word_id_list = lda.get_topic_terms(tc)
    word_score_list = []
    for key, score in word_id_list:
        word_score_list.append(str(id2word[key]))
    final_list.append(word_score_list)
df_top_cont = pd.DataFrame(final_list)
df_top_cont.columns = ['contributor' + str(i) for i in range(1,11)]
df_top_cont.index = ['Topic ' + str(i) for i in range(1, num_topics+1)]
#df_top_cont.to_csv("D:\\Unstructed L0\\Topic Modelling\\Report1.csv")
df_top_cont.head()
# def f1(text1):
#     text1 = text1.lower()
#     if re.findall(r'(employee)|(job)|(opportunity)|(organization)|(company)|(honeywell)|(pay)|(people)|(work)|(benefit)|(insurance)|(layoff)|(global)|(change)|(growth)|(analyst)|(manager)|(portfolio)',text1,flags = re.I):
#         return "glassdoor_reviews"
#     elif re.findall(r'(car)|(engine)|(oil)|(drive)|(gas)|(park)|(garrage)|(track)|(speed)|(fluid)|(tire)',text1,flags = re.I):
#         return "Automobiles"
#     elif re.findall(r'(sports)|(manchester)|(chelsea)|(win)|(play)|(game)|(club)|(premier)|(season)|(league)|(race)|(united)|(city)|(cup)',text1,flags = re.I):
#         return "sports_news"
#     elif re.findall(r'(analytics)|(business)|(management)|(temporary)|(retirement)|(article)|(deal)|(subject)|(writes)|(schedule)|(questionable)|(globally)|(strategy)|(corporate)|(industrial)|(competitive)|(retirement)',text1,flags = re.I):
#         return "tech_news"
#     else:
#         "room_rentals"
df.head()
df['topic'] = df['text'].apply(lambda x: f1(str(x)))
df.head()
df2 = df[['Id','topic']]
#df2.to_csv("D:\Unstructed L0\Kaggle Problem\Output\Final4.csv",index = False)
df2
df2.to_csv("D:\\Unstructed L0\\Kaggle Problem\\Output\\Final4.csv",index = False)