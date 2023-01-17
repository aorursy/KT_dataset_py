#Data Manipulation Library:
import pandas as pd

#Importing Data Manipulation Library:
import pandas as pd

#Importing Scientific computing library:
import numpy as np

#Importing Plotting libraries:
import matplotlib.pyplot as plt
import seaborn as sns

#Importing Machine Learning Libraries:
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#Importing Text Analysis Libraries:
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
!pip install textblob 
from nltk.corpus import stopwords
import nltk
nltk.download()
from nltk.sentiment.vader import SentimentIntensityAnalyzer as vader
print('Libraries Imported')
data2=pd.read_csv("https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/DPQMQH/17352493abb-cf8c4a43d6c3?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27india-news-headlines.csv&response-content-type=text%2Fcsv&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20201025T140932Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3599&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20201025%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e573a0dbc02a5078107335d8c103375e43b9f3fa6514b5543d56561e69b0fd9c")
data2.head()
data2.tail()
data2.columns
data2=data2.rename(columns={'publish_date':'Date'})
data2.head()
data2['Date']=pd.to_datetime(data2['Date'],format='%Y%m%d')
data2=data2.drop('headline_category',axis=1)
data2.head(5)
data2=data2[data2['Date']>='2015-10-19']
data2.sort_values(by='Date')
data2['headline_text']=data2.groupby(['Date']).transform(lambda x: ' '.join(x))
data2=data2.drop_duplicates()
data2.reset_index()

print('Dimension of dataset:{}'.format(data2.shape),'\n',70*'-')
print('Number of duplicated values:{}'.format(data2.duplicated().sum()),'\n',70*'-')
print('rows contain null values:\n{}'.format(data2.isnull().sum()),'\n',70*'-')
print('Schema of the dataset:\n')
print(data2.info(),'\n',70*'-')
df2=data2
df2['word_count'] = df2['headline_text'].apply(lambda x: len(str(x).split(" ")))
df2[['headline_text','word_count']].head()
df2['char_count'] = df2['headline_text'].str.len()
df2[['headline_text','char_count']].head()
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

df2['avg_word'] = df2['headline_text'].apply(lambda x: avg_word(x))
df2[['headline_text','avg_word']].head()
stop = stopwords.words('english')

df2['stopwords'] = df2['headline_text'].apply(lambda x: len([x for x in x.split() if x in stop]))
df2[['headline_text','stopwords']].head()
df2['hastags'] = df2['headline_text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
df2[['headline_text','hastags']].head()
df2['numerics'] = df2['headline_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df2[['headline_text','numerics']].head()
df2['upper'] = df2['headline_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
df2[['headline_text','upper']].head()
df2['headline_text'] = df2['headline_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df2['headline_text'].head()
df2['headline_text'] = df2['headline_text'].str.replace('[^\w\s]','')
df2['headline_text'].head()
stop = stopwords.words('english')
df2['headline_text'] = df2['headline_text'].apply(
    lambda x: " ".join(x for x in x.split() if x not in stop))
df2['headline_text'].head()
freq = pd.Series(' '.join(df2['headline_text']).split()).value_counts()[:10]
freq
freq = list(freq.index)
df2['headline_text'] = df2['headline_text'].apply(
    lambda x: " ".join(x for x in x.split() if x not in freq))
df2['headline_text'].head()
freq = pd.Series(' '.join(df2['headline_text']).split()).value_counts()[-10:]
freq
freq = list(freq.index)
df2['headline_text'] = df2['headline_text'].apply(
    lambda x: " ".join(x for x in x.split() if x not in freq))
df2['headline_text'].head()
df2.shape
Data=df2

TextBlob(str(Data['headline_text'])).words
st = PorterStemmer()
Data['headline_text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
Data['sentiment'] = Data['headline_text'].apply(lambda x: TextBlob(x).sentiment[0] )
Data[['headline_text','sentiment']].head()
Data1=Data[['Date','headline_text','sentiment']].reset_index()
Data1=Data1.drop('index',axis=1)
Data1.head()
x1=Data1[Data1['sentiment']>=0.5]
x2=Data1[Data1['sentiment']<0.5]
Data1['sentiment']=Data1['sentiment'].astype(float)
Data1.sentiment[Data1.sentiment>0]=1
Data1.sentiment[Data1.sentiment<0]=-1;
Data1.head(20)
plt.figure(figsize=(10,5))
ax=sns.countplot(Data1['sentiment'],palette='Set3')
ax.set_xticklabels(['Negative','Neutral','Positive']);