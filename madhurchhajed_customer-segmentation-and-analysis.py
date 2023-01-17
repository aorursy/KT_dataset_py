import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt 
from sklearn.cluster import KMeans
data = pd.read_csv('/kaggle/input/ecommerce-data/data.csv', encoding = 'unicode_escape')

data.head()
data.describe()
data.info()
data.isnull().sum()
pd.to_datetime(data.InvoiceDate.max())-pd.to_datetime(data.InvoiceDate.min()) 
data.duplicated().sum()
data.drop_duplicates(inplace = True)
data.duplicated().sum()
data = data.loc[(data.CustomerID.isnull()==False) & (data.Description.isnull()==False)].copy()
data.isnull().sum()
data.info()
data.min()
data.max()
data.nunique()
data['Quantity'][data['Quantity'] < 0].nunique()
data = data[data['Quantity'] > 0]
data.min()
data.Quantity.describe()
data.head()
data.describe()
data['Sales'] = data['Quantity'] * data['UnitPrice']
data[:5]
data[data['InvoiceNo'].str.startswith('c')]
data['Sales'].describe()
print('Duplicate invoice = ',data['InvoiceNo'].duplicated().sum())
print('Unique invoce = ',data['InvoiceNo'].nunique())
print('Unique Values :- ')
print('Country : ',data['Country'].nunique())
print('Quantity : ',data['Quantity'].nunique())
print('Items : ',data['Description'].nunique())
print("Most Occured :- ")
print('Country = ', data['Country'].mode()[0])
print('Description = ', data['Description'].mode()[0])
data.groupby(['Country']).sum().head()
data['InvoiceDate'] = pd.to_datetime(data.InvoiceDate, format='%m/%d/%Y %H:%M')
data.insert(loc=4, column='Day', value=data.InvoiceDate.dt.day)
data.insert( loc = 5,column='Month', value=data.InvoiceDate.dt.month)
data.insert( loc = 6,column='Year', value=data.InvoiceDate.dt.year)
data.insert( loc = 7,column='WeekDay', value=data.InvoiceDate.dt.weekday)
data.insert( loc = 8,column='Hour', value=data.InvoiceDate.dt.hour)
data.insert( loc = 9,column='Minute', value=data.InvoiceDate.dt.minute)
data.insert( loc = 10,column='Date', value=data.InvoiceDate.dt.date)

data.head()
sns.catplot(data=data, x= 'Month', kind = 'count')
plt.title('month vs orders')
sns.catplot(data=data, x= 'Month', y='Sales', kind = 'bar')
plt.title('Month wise Sales ')
sns.catplot(data=data, x= 'WeekDay', y='Sales', kind = 'bar')
plt.title('Sales vs WeekDay ')
# Monday = 0 to Sunday = 6
data['InvoiceNo'].value_counts().head(10)
data['CustomerID'].value_counts().head(10)
data['StockCode'].value_counts().head()
plt.figure(figsize=(15,8))
#sns.countplot(data['Country'])
sns.countplot(data[data['Country'] != 'United Kingdom']['Country'] , order = data[data['Country'] != 'United Kingdom']['Country'].value_counts().index)

plt.xticks(rotation=90)
plt.title('Order Count Abroad (Outside UK) ')
descrip_count =  data.Description.value_counts().sort_values(ascending=False).iloc[0:15]
plt.figure(figsize=(15,8))
sns.barplot(y = descrip_count.values, x=descrip_count.index )
plt.xticks(rotation=90)
plt.title('Top 10 Products ')
sns.catplot(data=data, x = 'Hour', kind = 'count')
plt.title('Order count wrt Hour')
data['InvoiceDate'].max()
now = dt.date(2011,12,9) 
new_df = data.groupby(by='CustomerID', as_index=False)['Date'].max()
new_df.columns = ['CustomerID', 'LastPurchaseDate']
new_df[:5]
new_df['Recency'] =  new_df['LastPurchaseDate'].apply(lambda x : (now-x).days)
new_df.drop('LastPurchaseDate',axis = 1, inplace = True)
new_df[:5]
new_df2 = data.groupby(by = 'CustomerID', as_index=False)['InvoiceNo'].count()
new_df2.columns = ['CustomerID','Frequency']
new_df2[:4]
new_df3 = data.groupby(by='CustomerID',as_index=False).agg({'Sales': 'sum'})
new_df3.columns = ['CustomerID','Monetary']
new_df3[:4]
temp = new_df.merge(new_df2, on = 'CustomerID')
rfm_df = temp.merge(new_df3, on = 'CustomerID')
rfm_df.set_index('CustomerID',inplace = True)
rfm_df.head()
rfm_df['R_quartile'] = pd.qcut(rfm_df['Recency'], 4, ['1','2','3','4'])
rfm_df['F_quartile'] = pd.qcut(rfm_df['Frequency'], 4, ['4','3','2','1'])
rfm_df['M_quartile'] = pd.qcut(rfm_df['Monetary'], 4, ['4','3','2','1'])
rfm_df.head()
rfm_df['RFM_Score'] = rfm_df.R_quartile.astype(str)+ rfm_df.F_quartile.astype(str) + rfm_df.M_quartile.astype(str)
rfm_df.head()
rfm_df[rfm_df['RFM_Score']==str(111)].head()
rfm_df[rfm_df['F_quartile']=='1'].head()
rfm_df[rfm_df['M_quartile']=='1'].head()
rfm_df[rfm_df['RFM_Score']==str(444)].head()
rfm_df[rfm_df['RFM_Score']==str(111)].shape
temp2 = rfm_df[rfm_df['RFM_Score']==str(111)]
temp2.head()
temp3 = pd.DataFrame()
temp2.reset_index(level=0, inplace=True)
temp2.head()
print(data.shape)
print(temp2.shape)
temp3 =  pd.merge(temp2,data.drop_duplicates(),on='CustomerID',how='right')
temp3.shape
temp3['CustomerID'].nunique()
temp2['CustomerID'].nunique()
data['CustomerID'].nunique()
temp3.dropna(inplace=True)
temp3['CustomerID'].nunique()
temp3.shape
temp3.head()
#Fetch wordcount for each Description
temp3['word_count'] = temp3['Description'].apply(lambda x: len(str(x).split(" ")))
temp3[['Description','word_count']].head()
temp4 = temp3[['Description','word_count']]
temp4.head()
temp4.word_count.describe()
#Identify common words
freq = pd.Series(' '.join(temp4['Description']).split()).value_counts()[:20]
freq
#Identify uncommon words
freq1 =  pd.Series(' '.join(temp4 ['Description']).split()).value_counts()[-20:]
freq1

import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
stop_words = set(stopwords.words("english"))
new_words = ['RED','PINK', 'BLUE', 'OF', 'BROWN',"BLACK"]
stop_words = stop_words.union(new_words)

for i in new_words:
  if i in stop_words:
    print(i)

corpus = []
for i in range(0, 164373):
    #Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', temp4['Description'][i])
    
    #Convert to lowercase
    text = text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    ##Convert to list from string
    text = text.split()
    
    ##Stemming
    ps=PorterStemmer()
    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if word not in  
            stop_words] 
    text = " ".join(text)
    corpus.append(text)
corpus[:10]
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
wordcloud = WordCloud(    #background_color='white',
                          stopwords=stop_words,
                          max_words=200,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(corpus))
plt.figure(figsize=(25,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Word Cloud for Best Customer\'s Products')
from sklearn.feature_extraction.text import CountVectorizer
import re
cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
X=cv.fit_transform(corpus)
list(cv.vocabulary_.keys())[:20]
#Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]
top_words = get_top_n_words(corpus, n=20)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]
top_df[:20]

sns.catplot(data=top_df,x='Word',y='Freq',kind='bar')
plt.xticks(rotation = 60)
