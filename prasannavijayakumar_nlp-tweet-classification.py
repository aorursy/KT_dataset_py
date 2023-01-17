# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import nltk

import string

from nltk.tokenize import regexp_tokenize, sent_tokenize, word_tokenize, TweetTokenizer

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set()



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/nlp-getting-started/train.csv')

train.head()
#Understand the dataset

train.shape
#summary of the dataset

train.info()
# Find the number of positive and negative classes

train.target.value_counts()



train.target.value_counts() / len(train)
plt.figure(figsize=(8,6))

sns.countplot(x='target', data=train)

plt.title('No.of Disaster Vs Non-Disaster Tweets')

plt.xlabel('Target', fontsize=11)

plt.show()
# importing all necessery modules 

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt 



stopwords = set(STOPWORDS)



tweet_words= ''



# iterate through the csv file 

for val in train.text: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    tweet_words += " ".join(tokens)+" "
plt.figure(figsize=(10,6))

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(tweet_words) 

plt.imshow(wordcloud) 

plt.axis("off") 

  

plt.show() 
#Removal of URLS, HTML link



def remove_urls(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    return url_pattern.sub(r'', text)



def remove_html(text):

    html_pattern = re.compile('<.*?>')

    return html_pattern.sub(r'', text)
EMOTICONS = {

    u":‑\)":"Happy face or smiley",

    u":\)":"Happy face or smiley",

    u":-\]":"Happy face or smiley",

    u":\]":"Happy face or smiley",

    u":-3":"Happy face smiley",

    u":3":"Happy face smiley",

    u":->":"Happy face smiley",

    u":>":"Happy face smiley",

    u"8-\)":"Happy face smiley",

    u":o\)":"Happy face smiley",

    u":-\}":"Happy face smiley",

    u":\}":"Happy face smiley",

    u":-\)":"Happy face smiley",

    u":c\)":"Happy face smiley",

    u":\^\)":"Happy face smiley",

    u"=\]":"Happy face smiley",

    u"=\)":"Happy face smiley",

    u":‑D":"Laughing, big grin or laugh with glasses",

    u":D":"Laughing, big grin or laugh with glasses",

    u"8‑D":"Laughing, big grin or laugh with glasses",

    u"8D":"Laughing, big grin or laugh with glasses",

    u"X‑D":"Laughing, big grin or laugh with glasses",

    u"XD":"Laughing, big grin or laugh with glasses",

    u"=D":"Laughing, big grin or laugh with glasses",

    u"=3":"Laughing, big grin or laugh with glasses",

    u"B\^D":"Laughing, big grin or laugh with glasses",

    u":-\)\)":"Very happy",

    u":‑\(":"Frown, sad, andry or pouting",

    u":-\(":"Frown, sad, andry or pouting",

    u":\(":"Frown, sad, andry or pouting",

    u":‑c":"Frown, sad, andry or pouting",

    u":c":"Frown, sad, andry or pouting",

    u":‑<":"Frown, sad, andry or pouting",

    u":<":"Frown, sad, andry or pouting",

    u":‑\[":"Frown, sad, andry or pouting",

    u":\[":"Frown, sad, andry or pouting",

    u":-\|\|":"Frown, sad, andry or pouting",

    u">:\[":"Frown, sad, andry or pouting",

    u":\{":"Frown, sad, andry or pouting",

    u":@":"Frown, sad, andry or pouting",

    u">:\(":"Frown, sad, andry or pouting",

    u":'‑\(":"Crying",

    u":'\(":"Crying",

    u":'‑\)":"Tears of happiness",

    u":'\)":"Tears of happiness",

    u"D‑':":"Horror",

    u"D:<":"Disgust",

    u"D:":"Sadness",

    u"D8":"Great dismay",

    u"D;":"Great dismay",

    u"D=":"Great dismay",

    u"DX":"Great dismay",

    u":‑O":"Surprise",

    u":O":"Surprise",

    u":‑o":"Surprise",

    u":o":"Surprise",

    u":-0":"Shock",

    u"8‑0":"Yawn",

    u">:O":"Yawn",

    u":-\*":"Kiss",

    u":\*":"Kiss",

    u":X":"Kiss",

    u";‑\)":"Wink or smirk",

    u";\)":"Wink or smirk",

    u"\*-\)":"Wink or smirk",

    u"\*\)":"Wink or smirk",

    u";‑\]":"Wink or smirk",

    u";\]":"Wink or smirk",

    u";\^\)":"Wink or smirk",

    u":‑,":"Wink or smirk",

    u";D":"Wink or smirk",

    u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":‑\|":"Straight face",

    u":\|":"Straight face",

    u":$":"Embarrassed or blushing",

    u":‑x":"Sealed lips or wearing braces or tongue-tied",

    u":x":"Sealed lips or wearing braces or tongue-tied",

    u":‑#":"Sealed lips or wearing braces or tongue-tied",

    u":#":"Sealed lips or wearing braces or tongue-tied",

    u":‑&":"Sealed lips or wearing braces or tongue-tied",

    u":&":"Sealed lips or wearing braces or tongue-tied",

    u"O:‑\)":"Angel, saint or innocent",

    u"O:\)":"Angel, saint or innocent",

    u"0:‑3":"Angel, saint or innocent",

    u"0:3":"Angel, saint or innocent",

    u"0:‑\)":"Angel, saint or innocent",

    u"0:\)":"Angel, saint or innocent",

    u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"0;\^\)":"Angel, saint or innocent",

    u">:‑\)":"Evil or devilish",

    u">:\)":"Evil or devilish",

    u"\}:‑\)":"Evil or devilish",

    u"\}:\)":"Evil or devilish",

    u"3:‑\)":"Evil or devilish",

    u"3:\)":"Evil or devilish",

    u">;\)":"Evil or devilish",

    u"\|;‑\)":"Cool",

    u"\|‑O":"Bored",

    u":‑J":"Tongue-in-cheek",

    u"#‑\)":"Party all night",

    u"%‑\)":"Drunk or confused",

    u"%\)":"Drunk or confused",

    u":-###..":"Being sick",

    u":###..":"Being sick",

    u"<:‑\|":"Dump",

    u"\(>_<\)":"Troubled",

    u"\(>_<\)>":"Troubled",

    u"\(';'\)":"Baby",

    u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(-_-\)zzz":"Sleeping",

    u"\(\^_-\)":"Wink",

    u"\(\(\+_\+\)\)":"Confused",

    u"\(\+o\+\)":"Confused",

    u"\(o\|o\)":"Ultraman",

    u"\^_\^":"Joyful",

    u"\(\^_\^\)/":"Joyful",

    u"\(\^O\^\)／":"Joyful",

    u"\(\^o\^\)／":"Joyful",

    u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",

    u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",

    u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",

    u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",

    u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",

    u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",

    u"\('_'\)":"Sad or Crying",

    u"\(/_;\)":"Sad or Crying",

    u"\(T_T\) \(;_;\)":"Sad or Crying",

    u"\(;_;":"Sad of Crying",

    u"\(;_:\)":"Sad or Crying",

    u"\(;O;\)":"Sad or Crying",

    u"\(:_;\)":"Sad or Crying",

    u"\(ToT\)":"Sad or Crying",

    u";_;":"Sad or Crying",

    u";-;":"Sad or Crying",

    u";n;":"Sad or Crying",

    u";;":"Sad or Crying",

    u"Q\.Q":"Sad or Crying",

    u"T\.T":"Sad or Crying",

    u"QQ":"Sad or Crying",

    u"Q_Q":"Sad or Crying",

    u"\(-\.-\)":"Shame",

    u"\(-_-\)":"Shame",

    u"\(一一\)":"Shame",

    u"\(；一_一\)":"Shame",

    u"\(=_=\)":"Tired",

    u"\(=\^\·\^=\)":"cat",

    u"\(=\^\·\·\^=\)":"cat",

    u"=_\^=	":"cat",

    u"\(\.\.\)":"Looking down",

    u"\(\._\.\)":"Looking down",

    u"\^m\^":"Giggling with hand covering mouth",

    u"\(\・\・?":"Confusion",

    u"\(?_?\)":"Confusion",

    u">\^_\^<":"Normal Laugh",

    u"<\^!\^>":"Normal Laugh",

    u"\^/\^":"Normal Laugh",

    u"\（\*\^_\^\*）" :"Normal Laugh",

    u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",

    u"\(^\^\)":"Normal Laugh",

    u"\(\^\.\^\)":"Normal Laugh",

    u"\(\^_\^\.\)":"Normal Laugh",

    u"\(\^_\^\)":"Normal Laugh",

    u"\(\^\^\)":"Normal Laugh",

    u"\(\^J\^\)":"Normal Laugh",

    u"\(\*\^\.\^\*\)":"Normal Laugh",

    u"\(\^—\^\）":"Normal Laugh",

    u"\(#\^\.\^#\)":"Normal Laugh",

    u"\（\^—\^\）":"Waving",

    u"\(;_;\)/~~~":"Waving",

    u"\(\^\.\^\)/~~~":"Waving",

    u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",

    u"\(T_T\)/~~~":"Waving",

    u"\(ToT\)/~~~":"Waving",

    u"\(\*\^0\^\*\)":"Excited",

    u"\(\*_\*\)":"Amazed",

    u"\(\*_\*;":"Amazed",

    u"\(\+_\+\) \(@_@\)":"Amazed",

    u"\(\*\^\^\)v":"Laughing,Cheerful",

    u"\(\^_\^\)v":"Laughing,Cheerful",

    u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",

    u'\(-"-\)':"Worried",

    u"\(ーー;\)":"Worried",

    u"\(\^0_0\^\)":"Eyeglasses",

    u"\(\＾ｖ\＾\)":"Happy",

    u"\(\＾ｕ\＾\)":"Happy",

    u"\(\^\)o\(\^\)":"Happy",

    u"\(\^O\^\)":"Happy",

    u"\(\^o\^\)":"Happy",

    u"\)\^o\^\(":"Happy",

    u":O o_O":"Surprised",

    u"o_0":"Surprised",

    u"o\.O":"Surpised",

    u"\(o\.o\)":"Surprised",

    u"oO":"Surprised",

    u"\(\*￣m￣\)":"Dissatisfied",

    u"\(‘A`\)":"Snubbed or Deflated"

}
def remove_emoji(string):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', string)



def remove_emoticons(text):

    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')

    return emoticon_pattern.sub(r'', text)



PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):

    """custom function to remove the punctuation"""

    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
#HTML removal

train['text_clean'] = train['text'].apply(lambda x:remove_html(x))



#removing url tags

train['text_clean'] = train['text'].apply(lambda x:remove_urls(x))



#removing emoticons

train['text_clean'] = train['text'].apply(lambda x:remove_emoticons(x))



#removing emojis

train['text_clean'] = train['text'].apply(lambda x:remove_emoji(x))



#removing punctuation

train['text_clean'] = train['text'].apply(lambda x:remove_punctuation(x))
def find_hashtags(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'



def process_text(df):

        

    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x))

    

    return df

    

train = process_text(train)
tweet_words= ''



# iterate through the csv file 

for val in train.text_clean: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    tweet_words += " ".join(tokens)+" "



    

plt.figure(figsize=(10,6))

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(tweet_words) 

plt.imshow(wordcloud) 

plt.axis("off") 

  

plt.show()
disastertweet_words= ''



# iterate through the csv file 

for val in train[train["target"]==1].text_clean: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    disastertweet_words += " ".join(tokens)+" "

    

plt.figure(figsize=(10,6))

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(disastertweet_words) 

plt.imshow(wordcloud) 

plt.title("Word Cloud of tweets if real disaster")

plt.axis("off") 

  

plt.show() 
nodisastertweet_words= ''



# iterate through the csv file 

for val in train[train["target"]==0].text_clean: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    nodisastertweet_words += " ".join(tokens)+" "

    

plt.figure(figsize=(10,6))

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(nodisastertweet_words) 

plt.imshow(wordcloud) 

plt.title("Word Cloud of tweets if no disaster")

plt.axis("off") 

  

plt.show() 
# Import the necessary scikit learn modules

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Initialize a CountVectorizer object: count_vectorizer

#tfidf_vectorizer_text = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), min_df=5)

tfidf_vectorizer_text_clean = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), min_df=5)

count_vectorizer_hashtags = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), min_df=5)





# Transform the training data using only the 'text' column values: count_train 

#tfidf_text = tfidf_vectorizer_text.fit_transform(train.text)

tfidf_text_clean = tfidf_vectorizer_text_clean.fit_transform(train.text_clean)

count_hashtags = count_vectorizer_hashtags.fit_transform(train.hashtags)



#train_text = pd.DataFrame(tfidf_text.toarray(), columns=tfidf_vectorizer_text.get_feature_names())

train_text_clean = pd.DataFrame(tfidf_text_clean.toarray(), columns=tfidf_vectorizer_text_clean.get_feature_names())

train_hashtags = pd.DataFrame(count_hashtags.toarray(), columns=count_vectorizer_hashtags.get_feature_names())



print(train_text_clean.shape, train_hashtags.shape)
# Joining the dataframes together



#train = train.join(train_text, rsuffix='_count_text')

train = train.join(train_text_clean, rsuffix='_count_text_clean')

train = train.join(train_hashtags, rsuffix='_count_hashtags')



print (train.shape)
features_to_drop = ['id', 'keyword','location','text','text_clean', 'hashtags','target_count_text_clean']



final_df = train.drop(columns = features_to_drop, axis=1)

final_df.shape
y= final_df['target']

X= final_df.drop('target', axis=1)
# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99, stratify=y)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
logreg= LogisticRegression(solver='liblinear', penalty='l2')



logreg.fit(X_train, y_train)



y_predicted= logreg.predict(X_test)



# Print accuracy score and confusion matrix on test set

print('Accuracy on the test set: ', accuracy_score(y_test, y_predicted))

print(confusion_matrix(y_test, y_predicted)/len(y_test))
#accuracy of training and test set

print("Training set accuracy is:", logreg.score(X_train, y_train))

print("Test set accuracy is:", logreg.score(X_test, y_test))
print(classification_report(y_test,y_predicted))
# Cross validation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import make_scorer



scorer= make_scorer(accuracy_score)



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=123)

cv_score = cross_val_score(logreg, X_train, y_train, cv=cv, scoring=scorer)

print('Cross validation accuracy score: %.3f' %np.mean(cv_score))
## Feature selection

from sklearn.feature_selection import RFECV



steps = 20

n_features = len(X_train.columns)

X_range = np.arange(n_features - (int(n_features/steps)) * steps, n_features+1, steps)



rfecv = RFECV(estimator=logreg, step=steps, cv=cv, scoring=scorer)



rfecv.fit(X_train, y_train)
print ('Optimal no. of features: %d' % np.insert(X_range, 0, 1)[np.argmax(rfecv.grid_scores_)])
selected_features = X_train.columns[rfecv.ranking_ == 1]

X_train2 = X_train[selected_features]

X_test2 = X_test[selected_features]
print(classification_report(y_test,y_predicted))
logreg.fit(X_train2, y_train)

cv2 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=99)

cv_score2 = cross_val_score(logreg, X_train2, y_train, cv=cv2, scoring=scorer)

print('Cross validation accuracy score: %.3f' %np.mean(cv_score2))
from sklearn.model_selection import GridSearchCV



grid={"C":np.logspace(-2,2,5), "penalty":["l1","l2"]}

searcher_cv = GridSearchCV(logreg, grid, cv=cv2, scoring = scorer)

searcher_cv.fit(X_train2, y_train)



print("Best parameter: ", searcher_cv.best_params_)

print("accuracy score: %.3f" %searcher_cv.best_score_)
#accuracy of training and test set

print("Training set accuracy is:", searcher_cv.score(X_train2, y_train))

print("Test set accuracy is:", searcher_cv.score(X_test2, y_test))
# Import the necessary modules

from sklearn.naive_bayes import MultinomialNB



# Instantiate a Multinomial Naive Bayes classifier: nb_classifier

nb_classifier = MultinomialNB()



# Fit the classifier to the training data

nb_classifier.fit(X_train2, y_train)
# Create the predicted tags: pred

pred = nb_classifier.predict(X_test2)



# Calculate the accuracy score: score

score = accuracy_score(y_test, pred)

print(score)
#accuracy of training and test set

print("Training set accuracy is:", nb_classifier.score(X_train2, y_train))

print("Test set accuracy is:", nb_classifier.score(X_test2, y_test))
#import and read test dataset

test = pd.read_csv('../input/nlp-getting-started/test.csv')

test.head()
#convert the text column to string

test['text']= test['text'].astype('str')



test['text_clean'] = test['text'].apply(lambda x:remove_html(x))

test['text_clean'] = test['text'].apply(lambda x:remove_urls(x))

test['text_clean'] = test['text'].apply(lambda x:remove_emoticons(x))

test['text_clean'] = test['text'].apply(lambda x:remove_emoji(x))

test['text_clean'] = test['text'].apply(lambda x:remove_punctuation(x))



test = process_text(test)
#tfidf_text2 = tfidf_vectorizer_text.transform(test.text)

tfidf_text2_clean = tfidf_vectorizer_text_clean.transform(test.text_clean)

count_hashtags2 = count_vectorizer_hashtags.transform(test.hashtags)



#test_text = pd.DataFrame(tfidf_text2.toarray(), columns=tfidf_vectorizer_text.get_feature_names())

test_text_clean = pd.DataFrame(tfidf_text2_clean.toarray(), columns=tfidf_vectorizer_text_clean.get_feature_names())

test_hashtags = pd.DataFrame(count_hashtags2.toarray(), columns=count_vectorizer_hashtags.get_feature_names())



print(test_text_clean.shape, test_hashtags.shape)
# Joining the dataframes together



#test = test.join(test_text, rsuffix='_count_text')

test = test.join(test_text_clean, rsuffix='_count_text_clean')

test = test.join(test_hashtags, rsuffix='_count_hashtags')



print(test.shape)
features_to_drop = ['id', 'keyword','location','text','text_clean', 'hashtags']



test_df = test.drop(columns = features_to_drop, axis=1)



#select optimal features 

final_df= test_df[selected_features]

final_df.shape
#predict the target label for test set

test_predictions = searcher_cv.predict(final_df)

test_predictions
submission = pd.DataFrame()

submission['id'] = test['id']

submission['target'] = test_predictions



submission.to_csv("submission.csv", index=False)



submission.tail()