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
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
pd.options.display.max_columns = 1999
pd.options.display.max_rows = 999
#Read the dataset of CSV file
df_train = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding = 'latin-1')
df_train.head()
df_train.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis = 1,inplace = True)
df_train.columns = ['label', 'message']
df_train.head(4)
x = {'ham' :0 , 'spam' :1}
df_train['label'] = df_train['label'].map(x)
df_train.head(3)
df_train['label'].value_counts().plot.bar()
df_train['label'].value_counts()
df_train.describe()
df_train.shape
df_train.groupby('label').describe()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (5,5))
sns.countplot(x = df_train['label'])
df_train.head()
df_train.head()
#Change all character to lower case
df_train['message'][0]
df_train['message'][0].split()
df_train.head()
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import spacy
sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words
import re
def punc(text):
    text = text.lower()
    text = re.sub(r'[^a-z A-Z 0-9-]+','', text)#Remove special/digit character and punctuation
    text = re.sub(r'@[A-Za-z0-9]+','',text) #Remove URL
    text = text.strip(' ')
    text = text.strip('. .')
    text = text.replace('.',' ')
    text = text.replace('-',' ')
    text = text.replace("’", "'").replace("′", "'").replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")
    text = text.split(" ")
    tokens_filtered= [word for word in text if not word in all_stopwords]
    lemm = WordNetLemmatizer()
    lemm_text = [lemm.lemmatize(i) for i in tokens_filtered]
    return (" ").join(tokens_filtered)
df_train['message']=df_train['message'].apply(lambda x:punc(x))
df_train.head(2)
df_train['message'] = df_train['message'].apply(lambda x:''.join(c for c in x if not c.isnumeric()))
df_train.head(3)

df_train['message'][2]
df_train['message'][44]
#contraction to Expansion
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
"daresn't": "dare not",
"dasn't": "dare not",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"d'ye": "do you",
"e'er": "ever",
"everybody's": "everybody is",
"everyone's": "everyone is",
"finna": "fixing to",
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
"he'dn't've'd": "he would not have had",
"he'll": "he will",
"he's": "he is",
"he've": "he have",
"how'd": "how would",
"howdy": "how do you do",
"how'll": "how will",
"how're": "how are",
"I'll": "I will",
"I'm": "I am",
"I'm'a": "I am about to",
"I'm'o": "I am going to",
"innit": "is it not",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
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
"'s": "is",
"shalln't": "shall not",
"shan't": "shall not",
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
df_train['message'] = df_train['message'].apply(lambda x: cont_to_exp(x))
df_train.head(3)
df_train.shape
df_train[df_train['label']==0].shape
df_train[df_train['label']==1].shape
x_df_train=df_train['message']
y_df_train=df_train['label']
from sklearn.feature_extraction.text import TfidfVectorizer
TfidfVect = TfidfVectorizer(max_features=2500)
x_vector  =TfidfVect.fit_transform(x_df_train)
x_vector.todense()[1],x_vector.data
x_vector.toarray()
# #cat = x_vector.select_dtypes(include=[np.number])
# numeric_train = x_vector.select_dtypes(include=[np.number])
# numeric_train
from sklearn.model_selection import train_test_split 

# split into 70:30 ration 
X_train, X_test, y_train, y_test = train_test_split(x_vector, y_df_train, test_size = 0.3, random_state = 0) 

# describes info about train and test set 
print("Number transactions X_train dataset: ", X_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions X_test dataset: ", X_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape) 

y_train.value_counts(),y_test.value_counts()
x_features_test = pd.DataFrame(x_vector.toarray())
x_features_test.shape
from imblearn.over_sampling import SMOTE
smote = SMOTE()
xtrain_smote, ytrain_smote = smote.fit_sample(x_vector,y_df_train)
xtrain_smote
xtrain_smote.shape,ytrain_smote.shape
xtrain_smote.dtype,ytrain_smote.dtype,
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(xtrain_smote, ytrain_smote)

y_pred=spam_detect_model.predict(X_test)
y_pred
from sklearn.metrics import classification_report
class_report = classification_report(y_pred,y_test)
class_report
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_pred,y_test)
conf
sns.heatmap(conf,annot = True)