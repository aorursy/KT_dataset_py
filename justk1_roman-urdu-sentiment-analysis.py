import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading required python packages and libraries

import nltk

import pandas as pd

import re

from sklearn.feature_extraction.text import TfidfVectorizer

import string

import pickle

import random

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from wordcloud import WordCloud,STOPWORDS



stopwords = nltk.corpus.stopwords.words('english')

ps = nltk.PorterStemmer()
# Load the csv file using Pandas and print first 5 lines

data = pd.read_csv("../input/urduromansentiment/Roman Urdu DataSet.csv",header=None)

data.head()
# Adding column names

data.columns =['body_text','sentiment','unknown']
# Print unique values in column 1 and 2

print ('Unique values of the sentiments are', data.sentiment.unique())

print ('Unique values of the unknonwn column are', data.unknown.unique())
# 'Neative' sentiment will be most likely Negative, so it is replaced accordingly. 

data[data['sentiment']=='Neative']='Negative'
# Verify we replaced all the 'Neative'

print ('Unique values of the sentiments are', data.sentiment.unique())
# Checking Null values in the data

data.isna().sum()
# Dropping the text body row which has a null value

data = data.dropna(subset=['body_text'])
# Last column can be dropped as it does not contain any useful information. Here axis=1, means column. 

data = data.drop('unknown', axis=1)
data.head()
data.describe()
print ('Number of sentiments in each category are as below')

print (data.sentiment.value_counts())



print ('\nPerecentage sentiments in each category are as below')

print (data.sentiment.value_counts()/data.shape[0]*100)



data.sentiment.value_counts().plot(kind='bar')
# Dropping neutral sentiment sentences. 

data = data[data.sentiment != 'Neutral']
data = data.reset_index(drop=True)
data.head()
data.sentiment.value_counts().plot(kind='bar')
data.describe()
text_wordcloud = " ".join(word.lower() for word in data.body_text)

print ('There are total {} words in text provided.'.format(len(text_wordcloud)))
def freq(str): 

  

    # Break the string into list of words  

    str = str.split()          

    str2 = [] 

  

    # Loop till string values present in list str 

    for i in str:              

  

        # Checking for the duplicacy 

        if i not in str2: 

  

            # Append value in str2 

            str2.append(i)  

              

    for i in range(0, len(str2)): 

        if(str.count(str2[i])>100): 

            print('Frequency of word,', str2[i],':', str.count(str2[i]))

            

freq(text_wordcloud)
UrduStopWordList = [line.rstrip('\n') for line in open('../input/urdustopwords/stopwords.txt')]



print (UrduStopWordList)
stopwords_with_urdu = set(STOPWORDS)

stopwords_with_urdu.update(UrduStopWordList)





wordcloud = WordCloud(stopwords=stopwords_with_urdu,

                      background_color='white',

                      width=3000,

                      height=2500

                     ).generate(text_wordcloud)

plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis('off')

plt.show()
neg_text_wordcloud = " ".join(word.lower() for word in data[data['sentiment']=='Negative']['body_text'])

print ('There are total {} words in sentences with negative sentiments.'.format(len(neg_text_wordcloud)))
# Plotting Plotting words in setences with negative sentiment

wordcloud = WordCloud(stopwords=stopwords_with_urdu,

                      background_color='white',

                      width=3000,

                      height=2500

                     ).generate(neg_text_wordcloud)

plt.figure(1,figsize=(12, 12))

plt.title('Negative Sentiment Words',fontsize = 20)

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis('off')

plt.show()
pos_text_wordcloud = " ".join(word.lower() for word in data[data['sentiment']=='Positive']['body_text'])

print ('There are total {} words in text with positive sentements.'.format(len(pos_text_wordcloud)))
# Plotting words in positive sentiment sentences



wordcloud = WordCloud(stopwords=stopwords_with_urdu,

                      background_color='white',

                      width=3000,

                      height=2500

                     ).generate(pos_text_wordcloud)

plt.figure(1,figsize=(12, 12))

plt.title('Positive Sentiment Words',fontsize = 20)

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis('off')

plt.show()
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))



punct = string.punctuation



def count_punct(text):

    count = sum([1 for char in text if char in punct])

    return count



data['punct_count'] = data['body_text'].apply(lambda x: count_punct(x))



data.head()
#Counting emojis in sentences

emoji = ['😝','😂','😃','😄','😅','😒','😉','😍','😑','😥','🤣','💃','👫','😛','😉','♥️','😍','🙈','👊','🤪','😘','🤭','💔']



def count_emoji(text):

    count = sum([1 for char in text if char in emoji])

    return count



data['emoji_count'] = data['body_text'].apply(lambda x: count_emoji(x))
bins = np.linspace(0, 200, 40)



plt.hist(data['body_len'], bins)

plt.title('Distribution of word count in body text')



plt.show()
bins = np.linspace(0, 200, 40)

plt.hist(data[data['sentiment']=='Negative']['body_len'], bins, density = True, alpha =0.4, label = 'Negative')

plt.hist(data[data['sentiment']=='Positive']['body_len'], bins, density = True, alpha =0.4, label = 'Positive')

plt.legend(loc='upper right')

plt.title('Distribution of word count in body text by each sentiment')

plt.show()
bins = np.linspace(0, 20, 10)



plt.hist(data['punct_count'],bins)

plt.title('Distribution of punctuation count in sentences')



plt.show()
bins = np.linspace(0, 20, 10)

plt.hist(data[data['sentiment']=='Negative']['punct_count'], bins, density = True, alpha =0.4, label = 'Negative')

plt.hist(data[data['sentiment']=='Positive']['punct_count'], bins, density = True, alpha =0.4, label = 'Positive')

plt.legend(loc='upper right')

plt.title('Distribution of punctuation count in sentences')

plt.show()
plt.hist(data['emoji_count'])

plt.title('Distribution of emoji count in body text')

plt.show()
bins = np.linspace(0, 20, 10)

plt.hist(data[data['sentiment']=='Negative']['emoji_count'], density = True, alpha =0.4, label = 'Negative')

plt.hist(data[data['sentiment']=='Positive']['emoji_count'], density = True, alpha =0.4, label = 'Positive')

plt.legend(loc='upper right')

plt.title('Distribution of emoji count in sentences')

plt.show()
data.head()
from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(data[['body_text', 'body_len', 'punct_count','emoji_count']],\

                                                    data['sentiment'], test_size=0.2,random_state=42,\

                                                    stratify=data['sentiment'])
y_train.describe()
def clean_text(text):

    #Change each character to lowercase and avoid any punctuation. Finally join word back. 

    text = "".join([char.lower() for char in text if char not in string.punctuation])

    

    # Use non word characters to split the sentence

    tokens = re.split('\W+', text)



    # Remove the stop words - commonly used words such as I, we, you, am, is etc in Urdu language 

    # that do not contribute to sentiment. 

    text = [word for word in tokens if word not in stopwords_with_urdu]

    return text



data_clean = data['body_text'].apply(lambda x: clean_text(x))

pd.set_option('max_colwidth', 800)

data_clean.head(10)
# Function clean_text used to clean the sentiment data before vectorizing to remove stop words and punctuations. 

# min_df is the minimum numbers of documents a word must be present in to be kept.

# norm is set to l2, to ensure all our feature vectors have a euclidian norm of 1 with norm='l2', but not set

# ngram_range is set to (1, 2) to consider both unigrams and bigrams  



tfidf_vect = TfidfVectorizer(analyzer=clean_text,ngram_range=(1, 2)) 

tfidf_vect_fit = tfidf_vect.fit(X_train['body_text'])



tfidf_train = tfidf_vect_fit.transform(X_train['body_text'])

tfidf_test = tfidf_vect_fit.transform(X_test['body_text'])
X_train_vect = pd.concat([X_train[['body_len', 'punct_count','emoji_count']].reset_index(drop=True), 

           pd.DataFrame(tfidf_train.toarray())], axis=1)

X_test_vect = pd.concat([X_test[['body_len', 'punct_count','emoji_count']].reset_index(drop=True), 

           pd.DataFrame(tfidf_test.toarray())], axis=1)



X_train_vect.head()
X_train_vect.shape
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import precision_recall_fscore_support as score, roc_auc_score

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from mlxtend.plotting import plot_confusion_matrix



import time
# Grid search on Random Forest Classifier

def train_RF(n_est,depth):

    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, n_jobs=-1,class_weight="balanced")

    rf_model = rf.fit(X_train_vect, y_train)

    y_pred = rf_model.predict(X_test_vect)

    precision, recall, fscore, train_support = score(y_test, y_pred, labels='Positive', average='macro')

    print('Est: {}, Depth: {}, Precision: {} / Recall: {} / F1 score {}/ Accuracy: {}'.format(n_est, depth,\

                                                            round(precision, 3), round(recall, 3), round(fscore, 3),\

                                                            round((y_pred==y_test).sum()/len(y_pred), 3)))    

for n_est in [10,50,150,250]:

    for depth in [10,20,30,50,None]:

        train_RF(n_est,depth)
# n_jobs = -1 for building parallel 150 decision trees. 

# Max_depth = None means it will build decision tree until minminzation of loss

rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1,class_weight="balanced",random_state=42)



rf_model = rf.fit(X_train_vect, y_train)

y_pred_rf = rf_model.predict(X_test_vect)



acc = accuracy_score(y_test, y_pred_rf)

precision, recall, fscore, train_support = score(y_test, y_pred_rf, labels='Positive', average='macro')





print('Precision: {} / Recall: {} / F1 score: {} /Accuracy: {}'.format(round(precision, 3), round(recall, 3), round(fscore, 3),\

                                                                       round(acc,3)))    

print ('Accuracy of Random Forest Model is: {}'.format(round(acc,3)))
print ('Labels of the classes are as below. These are required when generate classification report')

print (rf.classes_)
print ('Classification Report for Random Forest Classifier:\n',classification_report(y_test, y_pred_rf,digits=3))

#print ('\nConfussion matrix for Random Forest Classifier:\n'),confusion_matrix(y_test,  y_pred_rf,)
cm = confusion_matrix(y_test,y_pred_rf)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Reds)

plt.xticks(range(2), rf.classes_, fontsize=20)

plt.yticks(range(2), rf.classes_, fontsize=20)

plt.show()
k = random.randint(0,data.shape[0])

message = data['body_text'][k]

message
data['sentiment'][k]
predict_data = [message]

df = pd.DataFrame(predict_data,columns=['body_text'])

df['body_len'] = df['body_text'].apply(lambda x: len(x) - x.count(" "))

df['punct_count'] = df['body_text'].apply(lambda x: count_punct(x))

df['emoji_count'] = data['body_text'].apply(lambda x: count_emoji(x))
df
tfidf_predict = tfidf_vect_fit.transform(df['body_text'])

X_predict_vect = pd.concat([df[['body_len', 'punct_count','emoji_count']].reset_index(drop=True), 

           pd.DataFrame(tfidf_predict.toarray())], axis=1)
X_predict_vect
rf_model.predict(X_predict_vect)