# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
datadir = '../input/'
train = pd.read_csv(datadir+'train.csv')
train.head()
datadir = '../input/'
test = pd.read_csv(datadir+'test.csv')
test.head()
train.tail()
train.shape
train['comment_text'][0]
train.columns[3:]
#these are the 6 classes which we need to decide to which class a message belongs to
train.head()
train['comment_text'][6]
#it is really bad,isnt it?
train.comment_text.str.len().describe()
#here the minimum length of a message is 6 as you can see the below table
train['seventh'] = 1 - train[train.columns[2:]].max(axis =1)
train.head()
#we will see are there any null values 
train.isnull().any()

print('you can find null values in train set like this also')
print(train.isnull().sum())
print('for test set null values are')
print(test.isnull().sum())
# Here is the total number of samples belongs to each class
x = train.iloc[:,2:].sum()
print('total number of comment:',len(train),'\n','samples belongs to each class','\n',x)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,5))
sns.barplot(x.index,x.values)
plt.xticks(rotation=90)
plt.title('class distribution')
plt.show()
y = train.corr()
plt.figure(figsize=(8,8))
sns.heatmap(y,annot=True,center=True,square=True)
plt.title('heatmap showing correlation between classes')
plt.show()
#Here i intentionally included seventh class which we created
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from scipy.sparse import hstack
merge=pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])
df=merge.reset_index(drop=True)
df.head()
print('train shape',train.shape,'\n','test shape',test.shape,'\n','df shape',df.shape,'\n')

import re
import string
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   
# start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
import warnings
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

%matplotlib inline
#Sentense count in each comment:
    #  '\n' can be used to count the number of sentences in each comment
df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
#Word count in each comment:
df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))
#Unique word count
df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))
#Letter count
df['count_letters']=df["comment_text"].apply(lambda x: len(str(x)))
#punctuation count
df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
#upper case words count
df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
#title case words count
df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
#Number of stopwords
df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
#Average length of the words

df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df['ip'] = df['comment_text'].apply(lambda x: re.findall("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", str(x)))
df['count_ip']=df["ip"].apply(lambda x: len(x))

