# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import necessary nltk libraries

import re

import nltk

import string

from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords
#import the dataset

df= pd.read_csv(r'/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')

df.head()
df= df.drop(['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'], axis=1)
#Set column names

df.columns= ['target', 'text']
#preview first 5 rows of the dataset again

df.head()
print(df.shape)



df.info()
df['text']= df['text'].astype('str')
#Explore the target classes

df['target'].value_counts()
# importing all necessery modules 

from wordcloud import WordCloud, STOPWORDS 



stopwords = set(STOPWORDS)



token_words= ''



# iterate through the csv file 

for val in df.text: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    token_words += " ".join(tokens)+" "
plt.figure(figsize=(12,8))

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(token_words) 

plt.imshow(wordcloud) 

plt.axis("off") 

  

plt.show()
#initial CLean up.

#lowercasing letters

df["text"] = df["text"].str.lower()
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
def remove_emoticons(text):

    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')

    return emoticon_pattern.sub(r'', text)



PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):

    """custom function to remove the punctuation"""

    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))



from nltk.corpus import stopwords

", ".join(stopwords.words('english'))



STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):

    """custom function to remove the stopwords"""

    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
#removing emoticons

df['text'] = df['text'].apply(lambda x:remove_emoticons(x))

#removing punctuation

df['text_wo_punct'] = df['text'].apply(lambda x:remove_punctuation(x))



#removing stop words from text without punctuations

df["text_clean"] = df["text_wo_punct"].apply(lambda x: remove_stopwords(x))



df.head()
#WordCloud for Ham messages



ham_words= ''



# iterate through the csv file 

for val in df[df["target"]== 'ham'].text_clean: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    ham_words += " ".join(tokens)+" "

    

plt.figure(figsize=(10,6))

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = STOPWORDS, 

                min_font_size = 10).generate(ham_words) 

plt.imshow(wordcloud) 

plt.title("Word Cloud of Ham messages")

plt.axis("off") 

  

plt.show() 
#WordCloud for Spam messages



spam_words= ''



# iterate through the csv file 

for val in df[df["target"]== 'spam'].text_clean: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    spam_words += " ".join(tokens)+" "

    

plt.figure(figsize=(10,6))

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = STOPWORDS, 

                min_font_size = 10).generate(spam_words) 

plt.imshow(wordcloud) 

plt.title("Word Cloud of Spam messages")

plt.axis("off") 

  

plt.show() 
#Create the series to store independent and dependent variable

X= df['text_clean']

y= df.target
# Import the scikit necessary modules

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#create training and test set



X_train, X_test, y_train, y_test= train_test_split(X, y, stratify=y, random_state=99)
# Initialize a CountVectorizer object: count_vectorizer

count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)



# Transform the training data using only the 'text' column values: count_train 

count_train = count_vectorizer.fit_transform(X_train)



# Transform the test data using only the 'text' column values: count_test 

count_test = count_vectorizer.transform(X_test)
#instantiate Multinomial Model

nb= MultinomialNB()



#fit the training set to model

nb.fit(count_train, y_train)
#predict the label for Test set

y_pred= nb.predict(count_test)

y_pred
#accuracy of training and test set

print('Training set accuracy: ', nb.score(count_train, y_train))

print('Test set accuracy: ', nb.score(count_test, y_test))
# Print accuracy score and confusion matrix on test set

print('Accuracy on the test set: ', accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred)/len(y_test))
#Print Classification report

cm= classification_report(y_test, y_pred)

print(cm)
# Cross validation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import make_scorer



scorer= make_scorer(accuracy_score)



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=99)

cv_score = cross_val_score(nb, count_train, y_train, cv=cv, scoring=scorer)

print('Cross validation accuracy score: %.3f' %np.mean(cv_score))
from sklearn.model_selection import GridSearchCV



grid={"alpha":np.logspace(-2,2,5)}

searcher_cv = GridSearchCV(nb, grid, cv=cv, scoring = scorer)

searcher_cv.fit(count_train, y_train)



print("Best parameter: ", searcher_cv.best_params_)

print("accuracy score: %.3f" %searcher_cv.best_score_)
#accuracy of training and test set

print("Training set accuracy is:", searcher_cv.score(count_train, y_train))

print("Test set accuracy is:", searcher_cv.score(count_test, y_test))
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC



logreg= LogisticRegression(solver='liblinear')

dt= DecisionTreeClassifier()

rf= RandomForestClassifier(n_estimators=100)

gbr= GradientBoostingClassifier()

svm= SVC(kernel='linear')
classifiers = [('Logistic Regression', logreg),('Decision Tree Classifier', dt), ('RandomForest Classifier', rf), ('Gradient Boost', gbr), ('SVC', svm)]



# Iterate over the pre-defined list of regressors

for classifier_name, classifier in classifiers:   

    # Fit clf to the training set

    classifier.fit(count_train, y_train)    

    y_pred = classifier.predict(count_test) 

    

    training_set_score = classifier.score(count_train, y_train)

    test_set_score = classifier.score(count_test, y_test)

    

    

    print('{:s} : {:.3f}'.format(classifier_name, training_set_score))

    print('{:s} : {:.3f}'.format(classifier_name, test_set_score)) 