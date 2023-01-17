!pip install --upgrade pip
!pip install wordcloud
!pip install --upgrade scikit-learn
!pip install --upgrade pandas
!pip install --upgrade scikit-learn
#Make all the necessary imports for modules used in the notebook
# coding: utf-8
import re
import time
import pickle
import nltk, warnings
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag.perceptron import PerceptronTagger
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, jaccard_similarity_score, hamming_loss, make_scorer
import pandas as pd
from bs4 import BeautifulSoup
import itertools
import os
import numpy as np
import calendar
import math
import matplotlib as mpl
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import Counter
import plotly.graph_objs as go
import matplotlib.cm as cm
from io import BytesIO
# We load the data set
#df = pd.read_csv(BytesIO(csv_as_bytes))
df = pd.read_csv("../input/QueryResults2018.csv")
print(df.shape)
df.head()
df=df.drop_duplicates()
df.dropna(inplace=True)
df.head()
def separate_code(text):
    pointer=text.find('<code>')
    result=''
    while pointer!=-1:
        ender=text.find(u'</code>',pointer)
        result=result+text[pointer+6:ender]
        pointer=text.find('<code>',ender)
    return result

  
def remove_code(text):
    pointer=text.find('<code>')
    while pointer!=-1:
        ender=text.find(u'</code>')
        text=text.replace(text[pointer:ender+7],' ')
        pointer=text.find('<code>')
    return text
def remove_html(text):
    return BeautifulSoup(text, 'lxml').get_text()


def letters_only(text):
    text=text.lower()
    text=re.sub("c\+\+","cplusplus", text)
    text=re.sub("c#","csharp", text)
    text=re.sub("\.net","dotnet", text)
    text=re.sub("d3\.js","d3js", text)
    text=re.sub("[^a-zA-Z]"," ", text)
    return text

  
def tokenize_body(text):
    text=word_tokenize(text)
    return text
  
  
lm = WordNetLemmatizer()

def wordnet_tag(tag):
        # Convert POS default tags to wordnet lemmatizer tags
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            # Default pos in lemmatization is Noun
            return wordnet.NOUN

tagger=PerceptronTagger()          
def postag_body(text):
    text=tagger.tag(text)
    return text
  
  
def lemm(text):
    for i,word in enumerate(text):
        text[i]=lm.lemmatize(word[0],pos=wordnet_tag(word[1]))
    return text
  
  
ps = PorterStemmer()

def stem(text):
    for word in text:
        word=ps.stem(word)
    return text
  
  
default_stopwords = set(stopwords.words('english'))
# The custom Stopwords list is a custom list built and curated manually after running a count vectorizer on the body a first time
custom_stopwords = pickle.load( open( "../input/custom_stopwords.p", "rb" ) )
stpwrds= default_stopwords.union(custom_stopwords)


def remove_stopwords(text):
    return [ w for w in text if not w in stpwrds]
  
  
def code_strip(text):
    text=text.strip(u'\n')
    text=text.lower()
    text=re.sub("[^a-zA-Z]"," ", text)
    return text
  
  
def tag_clean(text):
    text=re.sub("<","", text)
    text=re.sub(">"," ", text)
    return text
  
def body_join(text):
    text=' '.join(text)
    return text  
df['Code']=df['Body'].apply(separate_code).apply(code_strip)
df['Body']=df['Body'].apply(remove_code).apply(remove_html).apply(letters_only)
df['Body']=df['Body'].apply(tokenize_body)
df['Body']=df['Body'].apply(postag_body)
df['Body']=df['Body'].apply(lemm)
df['Body']=df['Body'].apply(stem) 
df['Body']=df['Body'].apply(remove_stopwords).apply(body_join)
df['Title']=df['Title'].apply(letters_only)
df['Title']=df['Title'].apply(tokenize_body)
df['Title']=df['Title'].apply(postag_body)
df['Title']=df['Title'].apply(lemm)
df['Title']=df['Title'].apply(stem)
df['Title']=df['Title'].apply(remove_stopwords).apply(body_join)
df['Tags']=df['Tags'].apply(tag_clean)
title_vectorizer= CountVectorizer()
title_CV=title_vectorizer.fit_transform(df['Title'])
title_feature_names=title_vectorizer.get_feature_names()
#no_dummytags = 100

#lda_title = LatentDirichletAllocation(n_components=no_dummytags, max_iter=5, learning_method='online', learning_offset=50., n_jobs=4,random_state=0).fit(title_CV)
#we directly load the model since lda is slow
lda_title = pickle.load( open( "../input/lda_title.p", "rb" ) )
wordcloud= WordCloud(mode="RGBA", background_color=None, max_words=50)
fig = plt.figure(figsize=(30, 30))
fig.subplots_adjust(hspace=0.01, wspace=0.1)
for k,topic in enumerate([6,10,21,23,24,28,39,51,58]):    
    freq={}
    for i,j in enumerate(lda_title.components_[topic]):
        freq[title_feature_names[i]]=j

    wordcloud.generate_from_frequencies(freq)
    sp=331+k
    plt.subplot(sp)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
# Sort the values in the TFIDF matrix in descending order
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
  
# Extract the top n words from each topic  
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results

# Build a dictionary of topics' main features
        
def list_topics(model, feature_names, no_top_words):
    topic_dic={}
    for idx, topic in enumerate(model.components_):
        topic_dic[idx]=" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    return topic_dic
title_vectorizer= TfidfVectorizer(sublinear_tf=True, min_df=0.001)

title_tfidf= title_vectorizer.fit_transform(df['Title'])

feature_names=title_vectorizer.get_feature_names()

title_topics= list_topics(lda_title, title_feature_names, 150)

title_tags=set()

for i in range(len(title_topics)):
  
    tf_idf_vector=title_vectorizer.transform([title_topics[i]])

    sorted_items=sort_coo(tf_idf_vector.tocoo())

    title_tags=title_tags.union(set(k for k in extract_topn_from_vector(feature_names,sorted_items,20)))

tag_vectorizer= CountVectorizer()
tag_CV=tag_vectorizer.fit_transform(df['Tags'])
tag_names=tag_vectorizer.get_feature_names()
    
    
print("\nNumber of Extracted Tags:")
print(len(title_tags))

print("\nTags found in both the extracted tags and the tag column of the dataset:")

cross_tags= [tag for tag in tag_names if tag in title_tags]
print(len(cross_tags))
print(sorted(cross_tags))
df = pd.read_csv("../input/training_set.csv")
df=df.drop_duplicates()
df.dropna(inplace=True)
print(df.shape)
df.head().style
tag_vectorizer= CountVectorizer()
tag_CV=tag_vectorizer.fit_transform(df['Tags'])
tag_names=tag_vectorizer.get_feature_names()
# Sum up the counts of each vocabulary word
tag_CV=tag_CV.toarray()
dist = np.sum(tag_CV, axis=0)
sorted_counts=dist.argsort()
tags={}
for i in sorted_counts:
  tags[tag_names[i]]=dist[i]
tags_df=pd.DataFrame(columns=['Tag', 'Count'])
tags_df['Tag']=list(tags.keys())
tags_df['Count']=list(tags.values())
tags_df.sort_values(by=['Count'],ascending=False,inplace=True)
tags_df.shape

tags_df[0:20].plot.bar(x='Tag',y='Count',rot=60,figsize=(15,10))
print("\n------------------Top 20 tags:--------------------------")
tag_set=set(tags_df.Tag[tags_df['Count']>200])

def select_tags(text):
  text=text.split()
  return [t for t in text if t in tag_set]

df['main_tags']=df['Tags'].copy(deep=True)

df['main_tags']=df['main_tags'].apply(select_tags)

df['main_tags']=df['main_tags'].apply(body_join)

Y=df['main_tags'].str.get_dummies(sep=' ')
def wrong_label_counter(y, y_pred):
  try:
    y=y.values
  except:
    pass
  diff = y - y_pred
  diff[diff==1]=0
  diff=np.abs(diff)
  diff=np.sum(diff,axis=1)
  size=y.shape[0]*y.shape[1]-np.count_nonzero(y)
  return 100*np.sum(diff)/size

def missed_label_counter(y, y_pred):
  try:
    y=y.values
  except:
    pass
  diff = y - y_pred
  diff[diff==-1]=0
  diff=np.abs(diff)
  diff=np.sum(diff,axis=1)
  size=np.count_nonzero(y)
  return 100*np.sum(diff)/size
df['full_text']=df['Title']+' '+df['Body']+' '+df['Code']
X_train, X_test, y_train, y_test= train_test_split(df, Y,test_size=0.2)
# Create a list of the tag indexes in descending order of frequency for the classifier chains
tag_order_dict={}
for index,tag in enumerate(list(Y.columns)):
    tag_order_dict[tag]=index
tags_ordered_indexes=[]
for tag in tags_df['Tag']:
    try:
        tags_ordered_indexes.append(tag_order_dict[tag])
    except:
        pass
title_vectorizer= CountVectorizer()

x_train=title_vectorizer.fit_transform(X_train['Title'])
x_test=title_vectorizer.transform(X_test['Title'])

# initialize classifier chains multi-label classifier
cmb_title = ClassifierChain(ComplementNB(), order=tags_ordered_indexes)
mnb_title = ClassifierChain(MultinomialNB(), order=tags_ordered_indexes)

# Training model on train data
cmb_title.fit(x_train, y_train)
mnb_title.fit(x_train, y_train)

predictions_cmb_title= cmb_title.predict(x_test)
predictions_mnb_title= mnb_title.predict(x_test)  

precision_scores_titles=pd.DataFrame(columns=['Model'])
precision_scores_titles['Model']=['MultinomialNB Titles','ComplementNB Titles']

for col, metric in zip(['Accuracy','Hamming Loss','Wrong Labels','Missed Labels'], [accuracy_score, hamming_loss, wrong_label_counter, missed_label_counter]):
  precision_scores_titles[col]=[metric(y_test, predictions_mnb_title), metric(y_test, predictions_cmb_title) ]
  
precision_scores_titles.style
body_vectorizer= CountVectorizer()

x_train=body_vectorizer.fit_transform(X_train['Body'])
x_test=body_vectorizer.transform(X_test['Body'])

# initialize classifier chains multi-label classifier
cmb_body = ClassifierChain(ComplementNB(),order=tags_ordered_indexes)
mnb_body = ClassifierChain(MultinomialNB(),order=tags_ordered_indexes)

# Training model on train data
cmb_body.fit(x_train, y_train)
mnb_body.fit(x_train, y_train)

predictions_cmb_body= cmb_body.predict(x_test)
predictions_mnb_body= mnb_body.predict(x_test)  

precision_scores_body=pd.DataFrame(columns=['Model'])
precision_scores_body['Model']=['MultinomialNB Body','ComplementNB Body']

for col, metric in zip(['Accuracy','Hamming Loss','Wrong Labels','Missed Labels'], [accuracy_score, hamming_loss, wrong_label_counter, missed_label_counter]):
  precision_scores_body[col]=[metric(y_test, predictions_mnb_body), metric(y_test, predictions_cmb_body) ]
  
precision_scores_body.style
code_vectorizer= CountVectorizer()
code_vectorizer.fit(X_train['Code'])
x_train=code_vectorizer.transform(X_train['Code'])
x_test=code_vectorizer.transform(X_test['Code'])

# initialize classifier chains multi-label classifier
cmb_code = ClassifierChain(ComplementNB(),order=tags_ordered_indexes)
mnb_code = ClassifierChain(MultinomialNB(),order=tags_ordered_indexes)

# Training model on train data
cmb_code.fit(x_train, y_train)
mnb_code.fit(x_train, y_train)

predictions_cmb_code= cmb_code.predict(x_test)
predictions_mnb_code= mnb_code.predict(x_test)  

precision_scores_code=pd.DataFrame(columns=['Model'])
precision_scores_code['Model']=['MultinomialNB Code','ComplementNB Code']

for col, metric in zip(['Accuracy','Hamming Loss','Wrong Labels','Missed Labels'], [accuracy_score, hamming_loss, wrong_label_counter, missed_label_counter]):
  precision_scores_code[col]=[metric(y_test, predictions_mnb_code), metric(y_test, predictions_cmb_code) ]
  
precision_scores_code.style
X_train['full_text']=X_train['Title']+' '+X_train['Body']+' '+X_train['Code']
X_test['full_text']=X_test['Title']+' '+X_test['Body']+' '+X_test['Code']

full_text_vectorizer= CountVectorizer()

x_train=full_text_vectorizer.fit_transform(X_train['full_text'])
x_test=full_text_vectorizer.transform(X_test['full_text'])

# initialize classifier chains multi-label classifier
cmb_full_text = ClassifierChain(ComplementNB(), order=tags_ordered_indexes)
mnb_full_text = ClassifierChain(MultinomialNB(), order=tags_ordered_indexes)

# Training model on train data
cmb_full_text.fit(x_train, y_train)
mnb_full_text.fit(x_train, y_train)

predictions_cmb_full_text= cmb_full_text.predict(x_test)
predictions_mnb_full_text= mnb_full_text.predict(x_test)  

precision_scores_full_text=pd.DataFrame(columns=['Model'])
precision_scores_full_text['Model']=['MultinomialNB Full Text','ComplementNB Full Text']

for col, metric in zip(['Accuracy','Hamming Loss','Wrong Labels','Missed Labels'], [accuracy_score, hamming_loss, wrong_label_counter, missed_label_counter]):
  precision_scores_full_text[col]=[metric(y_test, predictions_mnb_full_text), metric(y_test, predictions_cmb_full_text) ]
  
precision_scores_full_text.style
y_pred_combined_parts=predictions_mnb_title+predictions_mnb_body+predictions_mnb_code+predictions_cmb_title+predictions_cmb_body+predictions_cmb_code
y_pred_combined_parts[y_pred_combined_parts>=1]=1

y_pred_combined_full=predictions_cmb_full_text+predictions_mnb_full_text
y_pred_combined_full[y_pred_combined_full>=1]=1

precision_scores_vote=pd.DataFrame(columns=['Model'])
precision_scores_vote['Model']=['Combined Partial Models','Combined Full Text Models']

for col, metric in zip(['Accuracy','Hamming Loss','Wrong Labels','Missed Labels'], [accuracy_score, hamming_loss, wrong_label_counter, missed_label_counter]):
  precision_scores_vote[col]=[metric(y_test, y_pred_combined_parts),metric(y_test, y_pred_combined_full)]
  
precision_scores_vote.style
final_scores=precision_scores_titles.append(precision_scores_body,ignore_index=True).append(precision_scores_code,ignore_index=True).append(precision_scores_full_text,ignore_index=True).append(precision_scores_vote,ignore_index=True)
final_scores.style
from sklearn.metrics import recall_score
sample_recall=recall_score(y_test, y_pred_combined_parts, average= None)
sample_scores=pd.DataFrame()
sample_scores['Label']=list(Y.columns)
sample_scores['Score']=sample_recall
sample_scores.sort_values(by='Score',ascending=False,inplace=True)
print("Recall Scores for each label in descending order")
sample_scores.style
from sklearn.metrics import precision_score
sample_precision=precision_score(y_test, y_pred_combined_parts, average= None)
sample_scores=pd.DataFrame()
sample_scores['Label']=list(Y.columns)
sample_scores['Score']=sample_precision
sample_scores.sort_values(by='Score',ascending=False,inplace=True)
print("Precision Scores for each label in descending order")
sample_scores.style