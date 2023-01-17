#! pip install comet-ml

#! pip install wordcloud

#! pip install tsne

#! pip install textblob

#! pip install gensim

#! pip install scikitplot
#Deploying the model

#from comet_ml import Experiment
# Api key

#experiment = Experiment(api_key="NMdrE2Fvv00bzfhwE99pCjGSq",

                        #project_name="team-4-climate-change", workspace="primmk", log_code=True)


#Standard Imports

import numpy as np

import pandas as pd

import re



#Visualisations 

import matplotlib.pyplot as plt

import seaborn as sns 

import scikitplot as skplt



#Data Cleaning

from nltk.stem import PorterStemmer

import nltk

from nltk.corpus import stopwords

from textblob import Word





#import dependencies

from gensim.models import word2vec

from sklearn.manifold import TSNE



#import for imbalance

from sklearn.utils import resample





# imports for N- Grams

from sklearn.feature_extraction.text import CountVectorizer





#Modeling

from sklearn import preprocessing

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer





from sklearn.linear_model import LogisticRegression

from sklearn import metrics 

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

df_train=pd.read_csv("../input/climate-change-belief-analysis/train.csv")

df_test =pd.read_csv("../input/climate-change-belief-analysis/test.csv")

df_train.head()
df_test.head()
# check null values in the train dataframe

df_train.isnull().sum()
# check null values in the test dataframe

df_test.isnull().sum()
blanks = []  # start with an empty list



for i,sen,mes,twe in df_train.itertuples():  # iterate over the DataFrame

    if type(mes)==str:            # avoid NaN values

        if mes.isspace():         # test 'review' for whitespace

            blanks.append(i)     # add matching index numbers to the list

        

print(len(blanks), 'blanks: ', blanks)
dist_class = df_train['sentiment'].value_counts()

labels = ['1', '2','0','-1']



fig, (ax1 )= plt.subplots(1, figsize=(8,4))



sns.barplot(x=dist_class.index, y=dist_class, ax=ax1).set_title("Tweet message distribution over the sentiments")

plt.show()
df_train.info()
def clean_text(df):

    """

    This function cleans tweets on the 'messages' column.



    Parameters: 

    df (obj): Data frame.



    Returns:

    Dataframe with cleaned tweets.



    """

    # Lowering all the text

    df.message = df.message.apply(lambda x: " ".join(x.lower() for x in x.split()))

    # Removing mentions

    df.message = df.message.apply(lambda x: re.sub("(@[A-Za-z0-9]+)","",x))

    # Removing short words

    df.message = df.message.apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

    # Removing https/http links

    df.message = df.message.apply(lambda x: re.sub('http[s]?://\S+', '', x))

    # Removing punctuation, with the exception of hashtags

    df.message = df.message.str.replace("[^a-zA-Z#]", " ")

    # Removing numbers

    df.message = df.message.apply(lambda x: re.sub('\d+','',x.lower()))

    return df
clean_text(df_train)

df_train.head()
clean_text(df_test)

df_test.head()
stop = stopwords.words('english')

df_test['message'] = df_test['message'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

stop = stopwords.words('english')

df_train['message'] = df_train['message'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#Code for lemmatize

df_train['message'] = df_train['message'].apply(lambda x: " ".join([Word(word).

lemmatize() for word in x.split()]))

df_train['message']
df_test['message'] = df_test['message'].apply(lambda x: " ".join([Word(word).

lemmatize() for word in x.split()]))

df_test['message']
def word_cloud(df,class_no,class_name):

  """

  This function generates word cloud visualizations across different classes.



  Parameters: 

    df (obj): Data frame.

    class_no (int): Class number

    class_name (obj): Class name



   Returns:

    word cloud visual

  """

  # create list of words per class

  sentiment_class = ' '.join([text for text in df['message'][df['sentiment'] == class_no]])

  #initialize wordcolud

  from wordcloud import WordCloud

  wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110,

                        background_color="white").generate(sentiment_class)



  plt.figure(figsize=(10, 7))

  plt.imshow(wordcloud, interpolation="bilinear")

  plt.title('WordCloud for' + " " + class_name)

  plt.axis('off')

  return plt.show()

word_cloud(df_train,1,'Pro Tweets')
word_cloud(df_train,-1,'Anti Tweets')
word_cloud(df_train,0,'Neutral Tweets')
word_cloud(df_train,1,'News Articles')
def common_words(df, class_no, class_name):

  """

  This is a function to extract top 20 comon words per class.

  

    Parameters: 

    df (obj): Data frame.

    class_no (int): Class number

    class_name (obj): Class name

  

    Returns: 

    Bar plot for the 20 most used words in the tweets.

    """

  # create new dataframe with top 20 common words

  name =[text for text in df['message'][df['sentiment'] == class_no]]

  series=pd.Series(' '.join(name).split()).value_counts()[:20]

  new_df=pd.DataFrame(data=series, columns=['count']).reset_index()



  #plot barplot

  plt.figure(figsize=(10, 7))

  ax=sns.barplot(x=new_df['count'],y=new_df['index'],data=new_df)

  plt.xlabel('value counts')

  plt.ylabel('common words')

  plt.title('Top 20 Common words for'+ ' '+class_name)

  return plt.show()
common_words(df_train,1, 'Pro Tweets')
common_words(df_train,-1, 'Anti Tweets')
common_words(df_train,0, 'Neutral Tweets')
# function to collect hashtags

def hashtag_extract(data):

  """

  Function to extact hashtags.



  Parameter(s):

    data (obj): a dataframe object

  

  Returns:

  List of hashtags

  """

  hashtags = []

    # Loop over the words in the tweet

  for i in data:

    ht = re.findall(r"#(\w+)", i)

    hashtags.append(ht)

  return hashtags
# extracting hashtags from pro climate change tweets



HT_pro = hashtag_extract(df_train['message'][df_train['sentiment'] == 1])



# extracting hashtags from anti climate change tweets

HT_anti = hashtag_extract(df_train['message'][df_train['sentiment'] == -1])



#extracting hashtags from neutral tweets

HT_neutral = hashtag_extract(df_train['message'][df_train['sentiment'] == 0])

# unnesting list

HT_pro = sum(HT_pro,[])

HT_anti = sum(HT_anti,[])

HT_neutral= sum(HT_neutral,[])
def common_tags(class_list, name):

  """

  Function to plot top 10 common hashtags.



  Returns:

  Bar plot of common hashtags.

  """

  a = nltk.FreqDist(class_list)

  d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})

# selecting top 10 most frequent hashtags    

  d = d.nlargest(columns="Count", n = 10)

  plt.figure(figsize=(10,12))

  ax=sns.barplot(data=d, x='Count',y = 'Hashtag')

  plt.xlabel('counts')

  plt.ylabel('Hashtags')

  plt.title('Top 10 Common Hashtags for'+ ' '+ name)

  return plt.show()
common_tags(HT_pro, 'Pro Climate Change Tweets')
common_tags(HT_anti, 'Anti Climate Change Tweets')
common_tags(HT_neutral, 'Neutral Climate Change Tweets')
# Removing hashtags

df_train.message = df_train.message.apply(lambda x: re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",x))
def map_labels(df):

  """

  This function finds maps labels onto new column



  Parameters:

    df (obj) : a dataframe



  Returns:

  A dataframe object.

  """

  # tokenize tweets

  df['split_tokens']= df['message'].apply(lambda x: x.split())

  # create a labels column on the dataframe

  Bdict = {-1: 'anti tweet', 1: 'pro tweet', 0:'neutral tweet', 2:'news article'}

  df['labels'] = df['sentiment'].map(Bdict)

  return df
# Find politics related tweets, using political hashtags

political_tags = ['maga','trump','parisagreement','imvotingbecause', 'cop']

df_pol=df_train[map_labels(df_train).split_tokens.apply(lambda x:any(set(x).intersection(political_tags)))]



# Find social related tweets, using social related hashtags

social_tags = ['beforetheflood','opchemtrails','qanda','amreading', 'actonclimate','husband']

df_soc=df_train[map_labels(df_train).split_tokens.apply(lambda x:any(set(x).intersection(social_tags)))]
def plot_pie(df,sentiment_name):



  """

  Function to plot pie chart, show distribution of classes between 

  political and social related tweets.



  Parameters:

    df (obj): a dataframe

    sentiment_name : Political or Social relation

     

  Returns:

  A pie chart plot, and a value counts table

  """

  # Define labels

  labels = 'pro tweet', 'news article', 'neutral tweet', 'anti tweet'

  sizes= df['labels'].value_counts()

  

  # Plot pie chart

  plt.figure(figsize=(18,16))

  fig1, (ax1, ax2) = plt.subplots(1,2)

  ax1.pie(sizes, labels=labels, autopct='%1.1f%%', pctdistance= 1.5, labeldistance=1.8,

        shadow=False, startangle=90)

  plt.title('Tweet Classes with a'+' '+ sentiment_name)

  

  # Plot Value counts table

  table = pd.DataFrame(df['labels'].value_counts())

  table.reset_index(inplace=True)

  table.columns=['Class Name','Count']





  cell_text = []

  for row in range(len(table)):

    cell_text.append(table.iloc[row])



  ax2.table(cellText=cell_text, colLabels=table.columns)

  plt.axis('off')

  return plt.show()
plot_pie(df_pol, 'Political View')
plot_pie(df_soc, 'Social View')
#function to build a library of words from the cleaned tweets

def build_corpus(data):

    corpus = []

    for column in ['message']:

        for sentence in df_train[column].iteritems():

            word_list = sentence[1].split(" ")

            corpus.append(word_list)

            

    return corpus



corpus = build_corpus(df_train)
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=100, workers=4)

model.wv['beforetheflood']
#function to create TSNE model, and plot the word vectors

def tsne_plot(model):

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
tsne_plot(model)
words= model.most_similar('beforetheflood')
df = pd.DataFrame(words, columns =['word', 'Similarity_score']) 

  

print(df)
def get_top_n_words(corpus, n=None):

  """

  Function to get top n words.



  Parameters:

    corpus (obj): a library of word vectors

  

  Returns:

  list of frequent words

  """

  # Vectorize words and count frequency

  vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

  bag_of_words = vec.transform(corpus)

  sum_words = bag_of_words.sum(axis=0) 

  words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

  words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

  return words_freq[:n]

# create dataframe

frequent_words = get_top_n_words(df_train['message'][df_train['sentiment'] == 1], 20)

for word, freq in frequent_words:

  df2 = pd.DataFrame(frequent_words, columns = ['text' , 'count'])

  grouped= df2.groupby('text').sum()['count'].sort_values(ascending=False)

  df3=pd.DataFrame(data=grouped, columns=['count']).reset_index()

#plot barplot

plt.figure(figsize=(10, 7))

ax=sns.barplot(x=df3['count'],y=df3['text'],data=df3)

plt.xlabel('Word count')

plt.ylabel('Word Pairs')

plt.title('Top 20 Common word Pairs for Pro Climate Change Tweets')

plt.show()
# Create dataframe

frequent_words = get_top_n_words(df_train['message'][df_train['sentiment'] == -1], 20)

for word, freq in frequent_words:

  df2 = pd.DataFrame(frequent_words, columns = ['text' , 'count'])

  grouped=df2.groupby('text').sum()['count'].sort_values(ascending=False)

  df4=pd.DataFrame(data=grouped, columns=['count']).reset_index()
plt.figure(figsize=(10, 7))

ax=sns.barplot(x=df4['count'],y=df4['text'],data=df4)

plt.xlabel('Word count')

plt.ylabel('Word Pairs')

plt.title('Top 20 Common word Pairs for Anti Climate Change Tweets')

plt.show()
X=df_train['message']

y= df_train['sentiment']

unseen_data = df_test['message']
X
y
unseen_data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size =0.2, random_state=42)

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', 

                             min_df=1, 

                             max_df=0.9, 

                             ngram_range=(1, 2))),

                     ('clf',LogisticRegression()),

])

# Feed the training data through the pipeline

text_clf.fit(X_train, y_train)  
predictions = text_clf.predict(X_test)
#from unseen data

y_pred = text_clf.predict(unseen_data)
from sklearn import metrics 

print(metrics.confusion_matrix(y_test,predictions))
#Print a classification report

print(metrics.classification_report(y_test,predictions))
#print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC



text_clf = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', LinearSVC()),

])



# Feed the training data through the pipeline

text_clf.fit(X_train, y_train)  
predictions = text_clf.predict(X_test)
#from unseen data

y_pred = text_clf.predict(unseen_data)
from sklearn import metrics 

print(metrics.confusion_matrix(y_test,predictions))
#Print a classification report

print(metrics.classification_report(y_test,predictions))
#print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm

text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', 

                             min_df=1, 

                             max_df=0.9, 

                             ngram_range=(1, 2))),

                     ('clf', svm.SVC(decision_function_shape='ovo')),

])

# Feed the training data through the pipeline

text_clf.fit(X_train, y_train)  
predictions = text_clf.predict(X_test)
#from unseen data

y_pred = text_clf.predict(unseen_data)
from sklearn import metrics 

print(metrics.confusion_matrix(y_test,predictions))
#Print a classification report

print(metrics.classification_report(y_test,predictions))
#print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', 

                             min_df=1, 

                             max_df=0.9, 

                             ngram_range=(1, 2))),

                     ('clf', SVC(kernel='rbf')),

])

# Feed the training data through the pipeline

text_clf.fit(X_train, y_train) 
predictions = text_clf.predict(X_test)
#from unseen data

y_pred = text_clf.predict(unseen_data)
from sklearn import metrics 

print(metrics.confusion_matrix(y_test,predictions))
#Print a classification report

print(metrics.classification_report(y_test,predictions))
#print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
#Separate minority and majority classes

df_majority = df_train[(df_train.sentiment==1) |

                      (df_train.sentiment==0) | 

                      (df_train.sentiment ==2)]

df_minority = df_train[df_train.sentiment == -1]



#Upsample minority class

df_minority_upsampled= resample(df_minority,replace= True,

                            n_samples= 4000, random_state =42) #sample with replacement



#Combine majority class with upsampled minority class

df_upsampled = pd.concat ([df_majority,

                          df_minority_upsampled])

#Display new class counts

df_upsampled.sentiment.value_counts()
# message  Distribution ove the classes

dist_class = df_upsampled['sentiment'].value_counts()

labels = ['1', '2','0','-1']



fig, (ax1 )= plt.subplots(1, figsize=(12,6))



sns.barplot(x=dist_class.index, y=dist_class, ax=ax1).set_title("Tweet message distribution over the sentiments")
#Independent Feature of the train dataframe

X=df_upsampled['message']

#Dependent feature of the train dataframe

y= df_upsampled['sentiment']

#Independent feature of the test dataframe

unseen_data = df_test['message']
#Splitting the train dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', 

                             min_df=1, 

                             max_df=0.9, 

                             ngram_range=(1, 2))),

                     ('clf',LogisticRegression()),

])

# Feed the training data through the pipeline

text_clf.fit(X_train, y_train)  
predictions = text_clf.predict(X_test)
#from unseen data

y_pred = text_clf.predict(unseen_data)
from sklearn import metrics 

print(metrics.confusion_matrix(y_test,predictions))
#Print a classification report

print(metrics.classification_report(y_test,predictions))
#print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
from sklearn.svm import LinearSVC



text_clf = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', LinearSVC()),

])



# Feed the training data through the pipeline

text_clf.fit(X_train, y_train)  
predictions = text_clf.predict(X_test)
#from unseen data

y_pred = text_clf.predict(unseen_data)
from sklearn import metrics 

print(metrics.confusion_matrix(y_test,predictions))
#Print a classification report

print(metrics.classification_report(y_test,predictions))
#print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
from sklearn import svm

text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', 

                             min_df=1, 

                             max_df=0.9, 

                             ngram_range=(1, 2))),

                     ('clf', svm.SVC(decision_function_shape='ovo')),

])

# Feed the training data through the pipeline

text_clf.fit(X_train, y_train)  
predictions = text_clf.predict(X_test)
#from unseen data

y_pred = text_clf.predict(unseen_data)
from sklearn import metrics 

print(metrics.confusion_matrix(y_test,predictions))
#Print a classification report

print(metrics.classification_report(y_test,predictions))
#print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
A=df_test.set_index('tweetid')   

B= A.index
Final_Submission ={'tweetid': B,'sentiment': np.round(y_pred,0)}
Submission =pd.DataFrame(data=Final_Submission)
Submission =Submission[['tweetid','sentiment']]
Submission.set_index('tweetid')
Submission.to_csv('ClimateChange.csv', index=False)
from sklearn.svm import SVC

text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', 

                             min_df=1, 

                             max_df=0.9, 

                             ngram_range=(1, 2))),

                     ('clf', SVC(kernel='rbf')),

])

# Feed the training data through the pipeline

text_clf.fit(X_train, y_train) 
predictions = text_clf.predict(X_test)
#from unseen data

y_pred = text_clf.predict(unseen_data)
from sklearn import metrics 

print(metrics.confusion_matrix(y_test,predictions))
#Print a classification report

print(metrics.classification_report(y_test,predictions))
#print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
#experiment.display()