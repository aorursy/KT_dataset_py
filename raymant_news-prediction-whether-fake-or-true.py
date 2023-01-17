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
t_news = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

f_news = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
import matplotlib.pyplot as plt

import seaborn as sns

import spacy

from spacy.lang.en.stop_words import STOP_WORDS
t_news['category'] = 'true'

f_news['category'] = 'fake'
f_news = f_news.sample(t_news.shape[0])
news = f_news.append(t_news, ignore_index = True)
news['Word_Count'] = news['text'].apply(lambda x: len(str(x).split()))
news['Char_Count'] = news['text'].apply(lambda x: len(x))
news['text'] = news['text'].apply(lambda x: ' '.join(x.split()))
import re
news['punct_count'] = news['text'].apply(lambda x: len(re.findall('[^a-z A-Z 0-9-]+', x)))
news['hashtags_count'] = news['text'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))

news['mention_count'] = news['text'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))
news['numerics_count'] = news['text'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))
news['UPPER_CASE_COUNT'] = news['text'].apply(lambda x: len([t for t in  x.split() if t.isupper() and len(x)>3]))
contractions = {

"aight": "alright",

"ain't": "am not",

"amn't": "am not",

"aren't": "are not",

"can't": "can not",

"cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"daren't": "dare not",

"daren't": "dared not",

"daresn't": "dare not",

"dasn't": "dare not",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"don't": "does not",

"d'ye": "do you",

"d'ye": "did you",

"e'er": "ever",

"everybody's": "everybody is",

"everyone's": "everyone is",

"finna": "fixing to",

"finna": "going to",

"g'day": "good day",

"gimme": "give me",

"giv'n": "given",

"gonna": "going to",

"gon't": "go not",

"gotta": "got to",

"hadn't": "had not",

"had've": "had have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had",

"he'd": "he would",

"he'dn't've'd": "he would not have had",

"he'll": "he shall",

"he'll": "he will",

"he's": "he has",

"he's": "he is",

"he've": "he have",

"how'd": "how did",

"how'd": "how would",

"howdy": "how do you do",

"howdy": "how do you fare",

"how'll": "how will",

"how're": "how are",

"I'll": "I shall",

"I'll": "I will",

"I'm": "I am",

"I'm'a": "I am about to",

"I'm'o": "I am going to",

"innit": "is it not",

"I've": "I have",

"isn't": "is not",

"it'd": "it would",

"it'll": "it shall",

"it'll": "it will",

"it's": "it has",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"may've": "may have",

"methinks": "me thinks",

"mightn't": "might not",

"might've": "might have",

"mustn't": "must not",

"mustn't've": "must not have",

"must've": "must have",

"needn't": "need not",

"ne'er": "never",

"o'clock": "of the clock",

"o'er": "over",

"ol'": "old",

"oughtn't": "ought not",

"'s": "is, has, does, or us",

"shalln't": "shall not",

"shan't": "shall not",

"she'd": "she had",

"she'd": "she would",

"she'll": "she shall",

"she'll": "she will",

"she's": "she has",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"somebody's": "somebody has",

"somebody's": "somebody is",

"someone's": "someone has",

"someone's": "someone is",

"something's": "something has",

"something's": "something is",

"so're": "so are",

"that'll": "that shall",

"that'll": "that will",

"that're": "that are",

"that's": "that has",

"that's": "that is",

"that'd": "that would",

"that'd": "that had",

"there'd": "there had",

"there'd": "there would",

"there'll": "there shall",

"there'll": "there will",

"there're": "there are",

"there's": "there has",

"there's": "there is",

"these're": "these are",

"these've": "these have",

"they'd": "they had",

"they'd": "they would",

"they'll": "they shall",

"they'll": "they will",

"they're": "they are",

"they're": "they were",

"they've": "they have",

"this's": "this has",

"this's": "this is",

"those're": "those are",

"those've": "those have",

"'tis": "it is",

"to've": "to have",

"'twas": "it was",

"wanna": "want to",

"wasn't": "was not",

"we'd": "we had",

"we'd": "we would",

"we'd": "we did",

"we'll": "we shall",

"we'll": "we will",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'd": "what did",

"what'll": "what shall",

"what'll": "what will",

"what're": "what are",

"what're": "what were",

"what's": "what has",

"what's": "what is",

"what's": "what does",

"what've": "what have",

"when's": "when has",

"when's": "when is",

"where'd": "where did",

"where'll": "where shall",

"where'll": "where will",

"where're": "where are",

"where's": "where has",

"where's": "where is",

"where's": "where does",

"where've": "where have",

"which'd": "which had",

"which'd": "which would",

"which'll": "which shall",

"which'll": "which will",

"which're": "which are",

"which's": "which has",

"which's": "which is",

"which've": "which have",

"who'd": "who would",

"who'd": "who had",

"who'd": "who did",

"who'd've": "who would have",

"who'll": "who shall",

"who'll": "who will",

"who're": "who are",

"who's": "who has",

"who's": "who is",

"who's": "who does",

"who've": "who have",

"why'd": "why did",

"why're": "why are",

"why's": "why has",

"why's": "why is",

"why's": "why does",

"won't": "will not",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd've": "you all would have",

"y'all'dn't've'd": "you all would not have had",

"y'all're": "you all are",

"you'd": "you had",

"you'd": "you would",

"you'll": "you shall",

"you'll": "you will",

"you're": "you are",

"you're": "you are",

"you've": "you have",

" u ": "you",

" ur ": "your",

" n ": "and"

}
def cont_to_exp(x):

    if type(x) is str:

        for key in contractions:

            value = contractions[key]

            x = x.replace(key,value)

        return x

    else:

        return x
news['text'] = news['text'].apply(lambda x: cont_to_exp(x))
news['Emails'] = news['text'].apply(lambda x: re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',x))
news['text'] = news['text'].apply(lambda x: re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '',x))
news['URL_Flags'] = news['text'].apply(lambda x: len(re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))
news['text'] = news['text'].apply(lambda x: re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x))
news['text'] = news['text'].apply(lambda x: re.sub('[^a-z A-Z 0-9-]+', '', x))
from wordcloud import WordCloud
text = ' '.join(news['text'])
text = text.split()
x = ' '.join(text[:20000])
wc = WordCloud(width = 2000, height = 1000).generate(x)

plt.imshow(wc)

plt.axis('off')

plt.show()
news['date'] = news['date'].str.replace('Jul', 'July')
news['date'] = news['date'].str.replace('Sep', 'September')
news['date'] = news['date'].str.replace('Oct', 'October')
news['date'] = news['date'].str.replace('Aug', 'August')
news['date'] = news['date'].str.replace('Augustust', 'August')
news['date'] = news['date'].str.replace('Dec', 'December')
news['date'] = news['date'].str.replace('Nov', 'November')
news['date'] = news['date'].str.replace('Decemberember', 'December')
news['date'] = news['date'].str.replace('Septembertember', 'September')
news['date'] = news['date'].str.replace('Jun', 'June')
news['date'] = news['date'].str.replace('Junee', 'June')
news['date'] = news['date'].str.replace('Feb', 'February')
news['date'] = news['date'].str.replace('Februaryruary', 'February')
news['date'] = news['date'].str.replace('Mar', 'March')
news['date'] = news['date'].str.replace('Marchch', 'March')
news['date'] = news['date'].str.replace('Apr', 'April')
news['date'] = news['date'].str.replace('Aprilil', 'April')
news['date'] = news['date'].str.replace('Julyy', 'July')
news['date'] = news['date'].str.replace('Jan', 'January')
news['date'] = news['date'].str.replace('Januaryuary', 'January')
news['date'] = news['date'].str.replace('Novemberember', 'November')
news['date'] = news['date'].str.replace('Octoberober', 'October')
i = news[(news.date == '14-February-18')].index
news = news.drop(i)
j = news[(news.date == '15-February-18')].index
news = news.drop(j)
k = news[(news.date == '16-February-18')].index
news = news.drop(k)
l = news[(news.date == '17-February-18')].index
news = news.drop(l)
m = news[(news.date == '18-February-18')].index
news = news.drop(m)
n = news[(news.date == '19-February-18')].index
news = news.drop(n)
o = news[(news.date == 'https://100percentfedup.com/video-hillary-asked-about-trump-i-just-want-to-eat-some-pie/')].index
news = news.drop(o)
p = news[(news.date == 'https://100percentfedup.com/12-yr-old-black-conservative-whose-video-to-obama-went-viral-do-you-really-love-america-receives-death-threats-from-left/')].index
news = news.drop(p)
q = news[(news.date == 'https://fedup.wpengine.com/wp-content/uploads/2015/04/hillarystreetart.jpg')].index
news = news.drop(q)
r = news[(news.date == 'https://fedup.wpengine.com/wp-content/uploads/2015/04/entitled.jpg')].index
news = news.drop(r)
s = news[(news.date == 'MSNBC HOST Rudely Assumes Steel Worker Would Never Let His Son Follow in His Footsteps…He Couldn’t Be More Wrong [Video]')].index
news = news.drop(s)
t = news[(news.date == 'https://100percentfedup.com/served-roy-moore-vietnamletter-veteran-sets-record-straight-honorable-decent-respectable-patriotic-commander-soldier/')].index
news = news.drop(t)
news['date'] = pd.to_datetime(news['date'])
news['Day'] = news['date'].dt.day

news['Month'] = news['date'].dt.month

news['Year'] = news['date'].dt.year
plt.hist(news[news['category']=='fake']['Word_Count'], bins=100, alpha=0.7)

plt.hist(news[news['category']=='true']['Word_Count'], bins=100, alpha=0.7)

plt.show()
plt.hist(news[news['category']=='fake']['punct_count'], bins=100, alpha=0.7)

plt.hist(news[news['category']=='true']['punct_count'], bins=100, alpha=0.7)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.pipeline import Pipeline



from sklearn.feature_extraction.text import TfidfVectorizer 
X = news['text']

y = news['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle = True, 

                                                    stratify = news['category'])
vectorizer = TfidfVectorizer()
X_train1 = vectorizer.fit_transform(X_train)
X_train1.shape
clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',RandomForestClassifier(n_estimators=100, n_jobs=-1))])
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
accuracy_score(y_test,y_pred)
clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',SVC(C = 1000, gamma = 'auto'))])
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
accuracy_score(y_test,y_pred)