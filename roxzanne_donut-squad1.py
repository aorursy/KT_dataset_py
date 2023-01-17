# Basic imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re

# NLP imports(nltk)
import nltk
#nltk.download('punkt')
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
from nltk.util import ngrams
from nltk.corpus import stopwords

# to automate the NLP extraction...
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# Cross_val_score is the new class for today...
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

# different regression models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# additional models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import GradientBoostingClassifier

# Neural Network!!
from sklearn.neural_network import MLPClassifier

# machine learning basics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
#####

import nltk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
import string
import re
import scipy.stats as stats

##data preprocessing
#nltk.download('punkt')
from nltk.tokenize import word_tokenize, TreebankWordTokenizer

from nltk.corpus import stopwords

from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer

#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

#nltk.download('stopwords')
from nltk.corpus import stopwords

from textblob import TextBlob

# to automate the NLP extraction and create bag of words
from sklearn.feature_extraction.text import CountVectorizer

#cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

# different regression models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# additional models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import GradientBoostingClassifier

# Neural Network!!
from sklearn.neural_network import MLPClassifier

#select model
from sklearn.metrics import confusion_matrix, classification_report

from collections import Counter
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
train_data = pd.read_csv('../input/train.csv')
train_data.info()
train_data
train_data['president'] = train_data['president'].replace('deKlerk',0)
train_data['president'] = train_data['president'].replace('Mandela',1)
train_data['president'] = train_data['president'].replace('Mbeki',2)
train_data['president'] = train_data['president'].replace('Motlanthe',3)
train_data['president'] = train_data['president'].replace('Zuma',4)
train_data['president'] = train_data['president'].replace('Ramaphosa',5)
faxis = train_data.copy()
faxis.head()
#Bar graph that shows how many post for each personality type there is 

count_person = faxis.groupby('president').count().sort_values('text', ascending=False)

f, ax = plt.subplots(figsize=(10, 7))
count_person['text'].plot(kind='bar', edgecolor='black',
                           linewidth=1.5, width=0.6, color = 'c')
ax.set_xlabel('President')
ax.set_ylabel('Number of posts')
ax.set_title('Number of posts per President');
plt.show()
train_donut = []
for i, row in faxis.iterrows():
    for text in row['text'].split('.'):
        train_donut.append([row['president'], text])
train_donut = pd.DataFrame(train_donut, columns=['president', 'text'])

train_donut
train_donut['text'] = train_donut['text'].str.lower()
train_donut.head()
def remove_punctuation(text):
    punc = ''.join([l for l in text if l not in string.punctuation])
    return punc

train_donut['text'] = train_donut['text'].apply(remove_punctuation)
train_donut['text'] = train_donut['text'].apply(lambda x: x.replace('”',''))
train_donut['text'] = train_donut['text'].apply(lambda x: x.replace('“',''))
train_donut['text'] = train_donut['text'].apply(lambda x: x.replace('‘',''))
train_donut['text'] = train_donut['text'].apply(lambda x: x.replace('ê','e'))
train_donut['text'] = train_donut['text'].apply(lambda x: x.replace('�',''))
train_donut
train_donut['text'].replace(' ', np.nan, inplace=True)
train_donut['text'].replace('', np.nan, inplace=True)
train_donut.dropna(subset=['text'], inplace=True)
train_donut

tokeniser = TreebankWordTokenizer()
train_donut['tokens'] = train_donut['text'].apply(tokeniser.tokenize)
train_donut.head()
lemmatizer = WordNetLemmatizer()
def donut_lemma(words, lemmatizer):
    lemma = [lemmatizer.lemmatize(word) for word in words]
    return lemma
#apply above function to the mbti data
train_donut['lemma'] = train_donut['text'].apply(donut_lemma, args=(lemmatizer,  ))
train_donut.head(12)
#All presidents
from wordcloud import WordCloud
cloud = WordCloud(width=1440, height=1080, max_words = 100).generate(' '.join(train_donut['text']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
#DeKlerk

deklerk = train_donut.loc[train_donut['president'] ==0]
cloud = WordCloud(width=1440, height=1080,max_words=100).generate(' '.join(deklerk['text']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
#Mandela

mandela = train_donut.loc[train_donut['president'] ==1]
cloud = WordCloud(width=1440, height=1080,max_words=100).generate(' '.join(mandela['text']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
#Mbeki

mbeki = train_donut.loc[train_donut['president'] ==2]
cloud = WordCloud(width=1440, height=1080,max_words=100).generate(' '.join(mbeki['text']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
#Mothlanthe

mothlanthe = train_donut.loc[train_donut['president'] ==3]
cloud = WordCloud(width=1440, height=1080,max_words=100).generate(' '.join(mothlanthe['text']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
#Zuma

zuma = train_donut.loc[train_donut['president'] ==4]
cloud = WordCloud(width=1440, height=1080,max_words=100).generate(' '.join(zuma['text']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
#Ramaphose

ramaphose = train_donut.loc[train_donut['president'] ==5]
cloud = WordCloud(width=1440, height=1080,max_words=100).generate(' '.join(ramaphose['text']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
import nltk
nltk.download('stopwords')

stop = stopwords.words('english')

vect = CountVectorizer(stop_words=stop,  
                             min_df=2, 
                             max_df=0.5, 
                             ngram_range=(1, 2))



vect.fit(train_donut['text'])
X = vect.fit_transform(train_donut['text'])
X = X.toarray()
Y = train_donut['president']
# Train-test split, using type variable as target and posts variable as predictor
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.001, random_state=42)
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)
# Fit and score a LR
lrc = LogisticRegression(random_state = 0)
lrc.fit(X_train, y_train)
print("TRAINING SET")
print("Accuracy: ", lrc.score(X_train, y_train))
print("Confusion Matrix:")
print(confusion_matrix(y_train, lrc.predict(X_train)))
print("Logistic Regression Classification Report:")
print(classification_report(y_train, lrc.predict(X_train)))
print("")


print("TEST SET")
print("Accuracy: ", lrc.score(X_test, y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, lrc.predict(X_test)))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, lrc.predict(X_test)))

## read the test data
pres_speech = pd.read_csv('../input/test.csv')
#Make text Lower
pres_speech['text'] = pres_speech['text'].str.lower()


#Remove punctuation
def remove_punctuation(text):
    punc = ''.join([l for l in text if l not in string.punctuation])
    return punc

pres_speech['text'] = pres_speech['text'].apply(remove_punctuation)


#Remove special Characters
pres_speech['text'] = pres_speech['text'].apply(lambda x: x.replace('”',''))
pres_speech['text'] = pres_speech['text'].apply(lambda x: x.replace('“',''))
pres_speech['text'] = pres_speech['text'].apply(lambda x: x.replace('‘',''))
pres_speech['text'] = pres_speech['text'].apply(lambda x: x.replace('ê','e'))
pres_speech['text'] = pres_speech['text'].apply(lambda x: x.replace('�',''))


#Delete Blanks
pres_speech['text'].replace(' ', np.nan, inplace=True)
pres_speech['text'].replace('', np.nan, inplace=True)
pres_speech.dropna(subset=['text'], inplace=True)

pres_speech

X_Test = vect.transform(pres_speech['text'])
X_Test = X_Test.toarray()
Y = pres_speech['sentence']
Xt_Train = vect.transform(train_donut['text'])
Xt_Train = Xt_Train.toarray()
Y_train = train_donut['president']
Pred_Train = lrc.predict(Xt_Train)
Train_Submit = pd.DataFrame(
{
    'President': Y_train,
    'Pred_president': Pred_Train,
})


Train_Submit.head(10)
Pred_Pres = lrc.predict(X_Test)

Test_Submit = pd.DataFrame(
{
    'sentence': Y,
    'president': Pred_Pres,
})

Test_Submit = Test_Submit.reset_index(drop=True)
Test_Submit
Test_Submit.to_csv('test_data_submission_donut_squad.csv',index=False)

