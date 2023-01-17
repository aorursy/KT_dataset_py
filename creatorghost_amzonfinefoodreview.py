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
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
import sqlite3
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import nltk
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.cm as cm
import numpy as np
import collections
%matplotlib inline
from tqdm import tqdm
import os
from sklearn.metrics import plot_confusion_matrix
# Conecting to our database
con=sqlite3.connect('/kaggle/input/amazon-fine-food-reviews/database.sqlite')
data=pd.read_sql_query("""
SELECT * 
From reviews
where Score !=3
""",con)
data.head(4)
data['total']=data['Summary']+' '+data['Text']
data.total.head(4)
rcParams["figure.figsize"] = 5,5
data["Score"].value_counts()[:4].plot(kind="bar")
## Writting function to convert score into Positive or Negative
## Positive -> 1 and Negative -> 0
def convert(x):
  if x >3:
    return 1
  return 0
score=data.Score
new=score.map(convert)
#new column after geting score converted in Positive or Negative
# Updating our score  column with  0 & 1 \\ the new colum that we have ceated
data.Score=new
rcParams["figure.figsize"] = 5,5
data["Score"].value_counts().plot(kind="pie")
display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND UserId="AR5J8UI46CURR"
ORDER BY ProductID
""", con)
display.head(5)
sort_data=data.sort_values(by=['ProductId'],axis=0,ascending =True)
clean=data.drop_duplicates(subset={'ProductId','UserId','ProfileName','Score','Time','Summary'},keep='first',inplace=False)
print("Before Cleaning: ",data.shape)
print("After Cleaning: ",clean.shape)
print("Data Lost :",data.shape[0]-clean.shape[0])
display=pd.read_sql_query('''
Select * 
from Reviews
where Score !=3 and HelpfulnessDenominator < HelpfulnessNumerator ''',con)
display.head()
# Removing data that violate this
final=clean[clean.HelpfulnessDenominator>=clean.HelpfulnessNumerator]
text=final.total
final.isna().sum()
import random
i=0
while i <5:
    
    ran=random.randint(1,1000)
    print("Found at: ",ran)
    print(text.values[ran])
    print('-'*100)
    i+=1
## function to remove URl
def removeUrl(text):
  pr=re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
  return pr

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
from nltk.stem import SnowballStemmer
snow=nltk.stem.SnowballStemmer('english')
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
def removeStopWord(word):
  token=word.split(" ")   ## coverting string to token (list of word) \\ like ["this","is","token"]
  removestop=[snow.stem(x) for x in token if x not in stopwords]   ##removing stopwords and also doing Stemming
  removed=" ".join(removestop)  ##joing back the list into sentence
  return removed
from tqdm import tqdm
preprocessed_reviews = []
for line in tqdm(final.total.values):
  line= BeautifulSoup(line, 'lxml').get_text() ## Remove Html Tags
  line=removeUrl(line) #removing url
  line=decontracted(line)    #Coverting word like { are't -> are not}
  line = re.sub(r'[0-9]+', '', line)   ## To Remove Numbers from the string
  line=line.lower()   ## Converting every word to lower case
  line = re.sub(r'[^a-z0-9\s]', '', line)   ## To clean all special Charaters
  line=removeStopWord(line)    ## Removing Stop Words And doing Steaming
  preprocessed_reviews.append(line.strip()) ## ading cleaned word into a list after removing spaces {By using strip()}

#tfidf
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(max_features=5000,ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(preprocessed_reviews)
tfidf = transformer.fit_transform(counts)
y=np.array(final['Score'])


#split in samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf,y,test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5)

logreg.fit(X_train, y_train)
print('Accuracy of Lasso classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Lasso classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
predictions = logreg.predict(X_test)
print('AUC: ', roc_auc_score(y_test, predictions))
cm= confusion_matrix(y_test,predictions)

conf_mat = confusion_matrix(y_test, predictions)
class_label = ["negative", "positive"]
df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
sns.heatmap(df, annot = True,fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
all_features = count_vectorizer.get_feature_names()
weight=logreg.coef_
pos_indx=np.argsort(weight)[:,::-1]

neg_indx=np.argsort(weight)
positive=[]
negative=[]
for i in list(pos_indx[0][0:45]):
    positive.append(all_features[i])
for i in list(neg_indx[0][:45]):
    negative.append(all_features[i])
def plotCloud(word):
  stwords = STOPWORDS
  word=' '.join(word)  ## Used When List of words are passed . Remove if ugoing string
  wordcloud = WordCloud(stopwords=stwords,width = 3000,height = 2000, background_color="black", max_words=10).generate(word)
  rcParams['figure.figsize'] = 20, 50
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.show()
print("Positive Words")
plotCloud(positive)
print("negative Words")
plotCloud(negative)
