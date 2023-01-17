!pip install comet_ml
import comet_ml
from comet_ml import Experiment
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
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
#import seaborn as sns
import string
import re
import spacy

#The Natural Language Toolkit library
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import brown 
from nltk import bigrams, trigrams

#Machine learning library
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#library for oversampling 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

!pip install -U spacy
!python -m spacy download en_core_web_md
##### Setting the API key (saved as environment variable)
experiment = Experiment(api_key='e03Z32Ilv0BBSwULP9xqfZwkO',
                       project_name = "general",workspace="lee-roy")
lemmatizer = WordNetLemmatizer() 
nltk.download('wordnet')
nltk.download('punkt')
#nlp = spacy.load('en_core_web_md')
tokenizer = RegexpTokenizer(r'\w+')
nltk.download('averaged_perceptron_tagger')
train_df = pd.read_csv('../input/climate-change-belief-analysis/train.csv')
test_df = pd.read_csv('../input/climate-change-belief-analysis/test.csv')
def removing_stopwords(post):
    """
    This function gets all the words in the tweets tokenizes, removes stopwords
    and lemmatizes all the words in a sentence
    
    """
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(post) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    
    allwords = [lemmatizer.lemmatize(w) for w in filtered_sentence]
    return allwords
def nouns(post):
    """
    This function gets all the nouns in tweets 
    """
    text = word_tokenize(post)
    nouns = set()
    for word, pos in nltk.pos_tag(text): 
        if pos in ['NN']:
            nouns.add(word)
    return nouns
def verbs(post):
    """
    This function gets all the verbs in tweets 
    """
    text = word_tokenize(post)
    nouns = set()
    for word, pos in nltk.pos_tag(text): 
        if pos in ['VB']: 
            nouns.add(word)
    return nouns
def tweets_from_tokens(post):
    """
    This function creates clean messages from lists
    
    """
    str1 = " "  
    sentence = str1.join([w for w in post])
    return sentence
def remove_punctuation_numbers(post):
    """
    This function removes all puntuation marks and numbers
    """
    punc_numbers = string.punctuation + '0123456789'
    return ''.join([l for l in post if l not in punc_numbers])
def remove_urls(data):
    """
    This function removes url links
    
    """
    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    subs_url = r''
    data['message'] = data['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
    return data

def remove_pattern(input_txt, pattern):
    """
    This function removes specified patterns
    
    """
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt
def length_of_message(data):
    """
    This function gets the length of a tweet
    
    """
    data['length'] = data['message'].apply(lambda x:len(x))
    return data
def make_bigrams(post):
    """
    Tokenise and get bi-grams from the tweets
    
    """
    text = word_tokenize(post)
    return list(bigrams(text))
def make_trigrams(post):
    """
    Tokenise and get bi-grams from the tweets
    
    """
    text = word_tokenize(post)
    return list(trigrams(text))
def metric_evaluation(y_test,predictions):
    cfn_m = confusion_matrix(y_test,predictions)
    c_r = classification_report(y_test,predictions)
    accuracy = metrics.accuracy_score(y_test,predictions)
    print(cfn_m)
    print(c_r)
    print(accuracy)
def scored(y_test,predictions):
    a = accuracy_score(y_test,predictions)
    p = precision_score(y_test,predictions,average='macro')
    r = recall_score(y_test,predictions,average='macro')
    f = f1_score(y_test,predictions,average='macro')
    return a,p,r,f
train_df.isnull().sum()
train = train_df.copy()
train['unclean'] =  train['message'].str.lower()
train = remove_urls(train)
train['message'] = np.vectorize(remove_pattern)(train['message'], "@[\w]*")
train['message'] = train['message'].apply(remove_punctuation_numbers)
train['message'] = train['message'].str.lower()
train['POS_nouns']  = train['message'].apply(nouns)
train['POS_verbs']  = train['message'].apply(verbs)
train['clean_tokens']  = train['message'].apply(removing_stopwords)
train['message']  = train['clean_tokens'].apply(tweets_from_tokens)
def convert_bigrams(post):
    """
    This function converts bigrams from tuples
    """
    res = ['_'.join(tups) for tups in post]
    return res
train['bigrams_text_list']  = train['message'].apply(make_bigrams).apply(convert_bigrams)
train['trigrams_text_list']  = train['message'].apply(make_trigrams).apply(convert_bigrams)
train['bigrams_text'] = train['bigrams_text_list'].apply(tweets_from_tokens)
train['trigrams_text'] = train['trigrams_text_list'].apply(tweets_from_tokens)
train
test = test_df.copy()
test['unclean'] =  test['message'].str.lower()
test = remove_urls(test)
test['message'] = np.vectorize(remove_pattern)(test['message'], "@[\w]*")
test['message'] = test['message'].apply(remove_punctuation_numbers)
test['message'] = test['message'].str.lower()
test['clean_tokens']  = test['message'].apply(removing_stopwords)
test['message']  = test['clean_tokens'].apply(tweets_from_tokens)
test.head()
train['sentiment'].unique()
counts =  list(train['sentiment'].value_counts())
labels = train['sentiment'].unique()
heights = counts
plt.bar(labels,heights,color='tab:red')
plt.xticks(labels)
plt.ylabel("# of observations")
plt.xlabel("Sentiment")
plt.show()
#plot a word cloud of the most common words
all_words = ' '.join([text for text in train['message']])

def plotwordclouds(text):
    wordcloud = WordCloud(width=1000, height=700, random_state=21,background_color="White",
                          colormap="Reds", max_font_size=110).generate(text)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

plotwordclouds(all_words)
def plot_most_frequent(text,title):
    """
    This function plots a bar plot of the words  
    """
    mostcommon_small = FreqDist(text).most_common(25)
    x, y = zip(*mostcommon_small)
    plt.figure(figsize=(50,30))
    plt.margins(0.02)
    plt.bar(x, y,color='tab:red')
    plt.xlabel('Words', fontsize=50)
    plt.ylabel('Frequency of Words', fontsize=50)
    plt.yticks(fontsize=40)
    plt.xticks(rotation=90, fontsize=40)
    plt.title(title, fontsize=60)
    plt.show() 
allwords = []
for wordlist in train['clean_tokens']:
    allwords += wordlist
    
plot_most_frequent(allwords,'Frequency of 25 Most Common Words')
allnouns = []
for wordlist in train['POS_nouns']:
    allnouns += wordlist
    
plot_most_frequent(allnouns,'Frequency of 25 Most Common nouns')
allverbs = []
for wordlist in train['POS_verbs']:
    allverbs += wordlist
plot_most_frequent(allverbs,'Frequency of 25 Most Common verbs')
all_bigrams = ' '.join([str(text) for text in train['bigrams_text']])
plotwordclouds(all_bigrams)
allwords = []
for wordlist in train['bigrams_text_list']:
    allwords += wordlist
    
plot_most_frequent(allwords,'Frequency of 25 Most Common bi-grams')
all_bigrams = ' '.join([str(text) for text in train['trigrams_text']])
plotwordclouds(all_bigrams)
allwords = []
for wordlist in train['trigrams_text_list']:
    allwords += wordlist
    
plot_most_frequent(allwords,'Frequency of 25 Most Common Words')
for i in range(5):
    print(train[train['sentiment']==-1]['message'].iloc[i] + '\n')
all_words = ' '.join([text for text in train[train['sentiment']==-1]['message']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==-1]['clean_tokens']:
    allwords += wordlist
    train[train['sentiment']==-1]['clean_tokens']

plot_most_frequent(allwords,'Frequency of 25 Most Common words')
allnouns = []
for wordlist in train[train['sentiment']==-1]['POS_nouns']:
    allnouns += wordlist

plot_most_frequent(allnouns,'Frequency of 25 Most Common nouns')
allverbs = []
for wordlist in train[train['sentiment']==-1]['POS_verbs']:
    allverbs += wordlist

plot_most_frequent(allverbs,'Frequency of 25 Most Common verbs')
all_words = ' '.join([text for text in train[train['sentiment']==-1]['bigrams_text']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==-1]['bigrams_text_list']:
    allwords += wordlist

plot_most_frequent(allwords,'Frequency of 25 Most Common words')
all_words = ' '.join([text for text in train[train['sentiment']==-1]['trigrams_text']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==-1]['trigrams_text_list']:
    allwords += wordlist

plot_most_frequent(allwords,'Frequency of 25 Most Common words')
for i in range(5):
    print(train[train['sentiment']==0]['message'].iloc[i] + '\n')
all_words = ' '.join([text for text in train[train['sentiment']==0]['message']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==0]['clean_tokens']:
    allwords += wordlist
    train[train['sentiment']==-1]['clean_tokens']

plot_most_frequent(allwords,'Frequency of 20 Most Common words')
allnouns = []
for wordlist in train[train['sentiment']==0]['POS_nouns']:
    allnouns += wordlist

plot_most_frequent(allnouns,'Frequency of 20 Most Common nouns')
allverbs = []
for wordlist in train[train['sentiment']==0]['POS_verbs']:
    allverbs += wordlist

plot_most_frequent(allverbs,'Frequency of 25 Most Common verbs')
all_words = ' '.join([text for text in train[train['sentiment']==0]['bigrams_text']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==0]['bigrams_text_list']:
    allwords += wordlist

plot_most_frequent(allwords,'Frequency of 25 Most Common words')
all_words = ' '.join([text for text in train[train['sentiment']==0]['trigrams_text']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==0]['trigrams_text_list']:
    allwords += wordlist

plot_most_frequent(allwords,'Frequency of 25 Most Common words')
for i in range(5):
    print(train[train['sentiment']==1]['message'].iloc[i] + '\n')
    
all_words = ' '.join([text for text in train[train['sentiment']==1]['message']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==1]['clean_tokens']:
    allwords += wordlist
    train[train['sentiment']==1]['clean_tokens']

plot_most_frequent(allwords,'Frequency of 25 Most Common words')
allnouns = []
for wordlist in train[train['sentiment']==1]['POS_nouns']:
    allnouns += wordlist

plot_most_frequent(allnouns,'Frequency of 25 Most Common nouns')
allverbs = []
for wordlist in train[train['sentiment']==1]['POS_verbs']:
    allverbs += wordlist

plot_most_frequent(allverbs,'Frequency of 25 Most Common verbs')
all_words = ' '.join([text for text in train[train['sentiment']==1]['bigrams_text']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==1]['bigrams_text_list']:
    allwords += wordlist

plot_most_frequent(allwords,'Frequency of 25 Most Common words')
all_words = ' '.join([text for text in train[train['sentiment']==1]['trigrams_text']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==1]['trigrams_text_list']:
    allwords += wordlist

plot_most_frequent(allwords,'Frequency of 25 Most Common words')
for i in range(5):
    print(train[train['sentiment']==2]['message'].iloc[i] + '\n')
all_words = ' '.join([text for text in train[train['sentiment']==2]['message']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==2]['clean_tokens']:
    allwords += wordlist


plot_most_frequent(allwords,'Frequency of 25 Most Common words')
allnouns = []
for wordlist in train[train['sentiment']==2]['POS_nouns']:
    allnouns += wordlist

plot_most_frequent(allnouns,'Frequency of 25 Most Common nouns')
allverbs = []
for wordlist in train[train['sentiment']==2]['POS_verbs']:
    allverbs += wordlist

plot_most_frequent(allverbs,'Frequency of 25 Most Common verbs')
all_words = ' '.join([text for text in train[train['sentiment']==2]['bigrams_text']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==2]['bigrams_text_list']:
    allwords += wordlist

plot_most_frequent(allwords,'Frequency of 25 Most Common words')
all_words = ' '.join([text for text in train[train['sentiment']==2]['trigrams_text']])
plotwordclouds(all_words)
allwords = []
for wordlist in train[train['sentiment']==2]['trigrams_text_list']:
    allwords += wordlist

plot_most_frequent(allwords,'Frequency of 25 Most Common words')
X = train['message']
y = train['sentiment']
test_x = test['message']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_up = train['unclean']
y_up = train['unclean']
test_up_x = test['unclean']
X_train_up, X_test_up, y_train_up, y_test_up=train_test_split(X_up,y_up,test_size=0.2,random_state=42)
vectorizer = TfidfVectorizer()
tfidf_vect = vectorizer.fit(X_train)
xtrain_tfidf = tfidf_vect.transform(X_train)
xtest_tfidf =  tfidf_vect.transform(test_x)
vectorizer_up = TfidfVectorizer()
tfidf_vect_up = vectorizer_up.fit(X_train_up)
xtrain_tfidf_up = tfidf_vect_up.transform(X_train_up)
xtest_tfidf_up =  tfidf_vect_up.transform(test_up_x)
clf = LinearSVC()
clf.fit(xtrain_tfidf,y_train)
scores = cross_val_score(clf, xtrain_tfidf, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
text_clf.fit(X_train,y_train)
predictions_p = text_clf.predict(X_test)
metric_evaluation(y_test,predictions_p)
accuracy,precision,recall,f1 = scored(y_test,predictions_p)
predictions = clf.predict(xtest_tfidf)
predictions_df = pd.DataFrame(data=predictions, index=test['tweetid'],
                      columns=['sentiment'])
predictions_df
predictions_df.to_csv('predictions_df.csv')
clf_up = LinearSVC()
clf_up.fit(xtrain_tfidf_up,y_train_up)
scores = cross_val_score(clf, xtrain_tfidf_up, y_train_up, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf_lr = LogisticRegression(max_iter=4000)
clf_lr.fit(xtrain_tfidf,y_train)
scores = cross_val_score(clf_lr, xtrain_tfidf, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predictions_2 = clf_lr.predict(xtest_tfidf_up)
predictions_df_2 = pd.DataFrame(data=predictions_2, index=test['tweetid'],
                      columns=['sentiment'])
predictions_df_2
#predictions_df_2.to_csv('predictions_df_2.csv')
clf_RF = RandomForestClassifier(criterion = 'entropy',random_state=0)
clf_RF.fit(xtrain_tfidf_up,y_train)
scores = cross_val_score(clf_RF, xtrain_tfidf_up, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predictions_3 = clf_RF.predict(xtest_tfidf_up)
predictions_df_3 = pd.DataFrame(data=predictions_3, index=test['tweetid'],
                      columns=['sentiment'])
predictions_df_3
sm = SMOTE(random_state=1234)
X_res, y_res = sm.fit_resample(xtrain_tfidf_up, y_train)
counts =  list(y_res.value_counts())
labels = y_res.unique()
heights = counts
plt.bar(labels,heights,color='tab:red')
plt.xticks(labels)
plt.ylabel("# of observations")
plt.xlabel("Sentiment")
plt.show()
clf1 = LinearSVC()
clf1.fit(X_res, y_res)
scores = cross_val_score(clf1, X_res, y_res)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predictions_4 = clf1.predict(xtest_tfidf_up)
predictions_df4 = pd.DataFrame(data=predictions_4, index=test['tweetid'],
                      columns=['sentiment'])
predictions_df4.to_csv(r'predictions_df4.csv')
rus = RandomUnderSampler(random_state=123)
X_res_u, y_res_u = rus.fit_resample(xtrain_tfidf_up, y_train)
counts =  list(y_res_u.value_counts())
labels = y_res_u.unique()
heights = counts
plt.bar(labels,heights,color='tab:red')
plt.xticks(labels)
plt.ylabel("# of observations")
plt.xlabel("Sentiment")
plt.show()
clf2 = LinearSVC()
clf2.fit(X_res_u, y_res_u)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf2, X_res_u, y_res_u, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predictions_5 = clf2.predict(xtest_tfidf_up)
predictions_df5 = pd.DataFrame(data=predictions_5, index=test['tweetid'],
                      columns=['sentiment'])
predictions_df5
#predictions_df5.to_csv(r'predictions_df5.csv')
# creating a dictionary with all the metrics
dua = [True,False]
c = [0.001,0.01,0.1,1,10]
param_grid = {'dual':dua,
             'C':c}
# using gridsearch to choose the best parameters
grid_SVM = GridSearchCV(LinearSVC(),param_grid,cv=10)
grid_SVM.fit(xtrain_tfidf,y_train)
# getting a list of the best parameters
grid_SVM.best_params_
params = {"model_type": "clf",
         "stratify": True}
metrics = {'accuracy': scores.mean(),
          'precision':precision,
          'recall':recall,
          'f1':f1}
experiment.log_parameters(params)
experiment.log_metrics(metrics)
experiment.end()
experiment.display()