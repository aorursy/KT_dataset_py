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
speeches = pd.read_csv("../input/speeches-modi/PM_Modi_speeches.csv")
speeches.head()
speeches.describe()
speeches.drop(["url","lang","words"], axis = 1, inplace = True)
speeches.head()
import matplotlib.pyplot as plt
%matplotlib inline  
import nltk
import textblob
import wordcloud
import seaborn as sns
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
def wc(data,bgcolor,title):
    plt.figure(figsize = (100,100))
    wc = WordCloud(background_color = bgcolor, max_words = 1000,  max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import re

top_N = 100
#convert list of list into text
#a=''.join(str(r) for v in speeches['title'] for r in v)

a = speeches['title'].str.lower().str.cat(sep=' ')

# removes punctuation,numbers and returns list of words
b = re.sub('[^A-Za-z]+', ' ', a)

#remove all the stopwords from the text
stop_words = get_stop_words('en')        
nltk_words = stopwords.words('english')   
stop_words.extend(nltk_words)

word_tokens = word_tokenize(b)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

# Remove characters which have length less than 2  
without_single_chr = [word for word in filtered_sentence if len(word) > 2]

# Remove numbers
cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]        

# Calculate frequency distribution
word_dist = nltk.FreqDist(cleaned_data_title)
rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Word",y="Frequency", data=rslt.head(10))
wc(cleaned_data_title,'black','Common Words' )
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import re

top_N = 100
#convert list of list into text
#a=''.join(str(r) for v in speeches['text'] for r in v)

desc_lower = speeches['text'].str.lower().str.cat(sep=' ')

# removes punctuation,numbers and returns list of words
desc_remove_pun = re.sub('[^A-Za-z]+', ' ', desc_lower)

#remove all the stopwords from the text
stop_words = get_stop_words('en')         
nltk_words = stopwords.words('english')   
stop_words.extend(nltk_words)

word_tokens_desc = word_tokenize(desc_remove_pun)
filtered_sentence_desc = [w_desc for w_desc in word_tokens_desc if not w_desc in stop_words]
filtered_sentence_desc = []
for w_desc in word_tokens_desc:
    if w_desc not in stop_words:
        filtered_sentence_desc.append(w_desc)

# Remove characters which have length less than 2  
without_single_chr_desc = [word_desc for word_desc in filtered_sentence_desc if len(word_desc) > 2]

# Remove numbers
cleaned_data_desc = [word_desc for word_desc in without_single_chr_desc if not word_desc.isnumeric()]        

# Calculate frequency distribution
word_dist_desc = nltk.FreqDist(cleaned_data_desc)
rslt_desc = pd.DataFrame(word_dist_desc.most_common(top_N),
                    columns=['Word', 'Frequency'])

#print(rslt_desc)
#plt.style.use('ggplot')
#rslt.plot.bar(rot=0)


plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Word", y="Frequency", data=rslt_desc.head(10))
wc(cleaned_data_desc,'black','Frequent Words' )
from textblob import TextBlob

bloblist_desc = []
speeches_text = speeches['text'].astype(str)
for row in speeches_text:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    speeches_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])
 
def f(speeches_polarity_desc):
    if speeches_polarity_desc['sentiment'] > 0:
        val = "Positive"
    elif speeches_polarity_desc['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

speeches_polarity_desc['Sentiment_Type'] = speeches_polarity_desc.apply(f, axis=1)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=speeches_polarity_desc)
speeches_50 = speeches.head(50)
speeches_50.head()
from PIL import Image
mask = np.array(Image.open("../input/india-png/india.png"))
def wordcloud(tweets, title):
    stopwords = set(STOPWORDS)
    stopwords.add("will")
    stopwords.add("now")
    wordcloud = WordCloud(width=512, height=512, background_color="white", max_font_size=18, min_font_size=4,
                          max_words=200, stopwords=stopwords, contour_color = 'firebrick', colormap='Dark2',
                          random_state=2018, mask=mask).generate(" ".join([i for i in speeches_50['text']]))
    plt.figure(title, figsize=(10, 10), facecolor='white', edgecolor='blue')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0)
    plt.show()
wordcloud(speeches_50['text'], "India")
import pandas as pd  

list = speeches_50['text']
  
 
series = pd.Series(list) 
  

economy_Count = series.str.count("economy")
Economy_Count = series.str.count("Economy")
economic_Count = series.str.count("economic")
Economic_Count = series.str.count("Economic")
GDP_Count = series.str.count("GDP")
Total_Count = economy_Count + Economy_Count + economic_Count + Economic_Count + GDP_Count 
economy_Count.aggregate(sum)
Economy_Count.aggregate(sum)
economic_Count.aggregate(sum)
Economic_Count.aggregate(sum)
GDP_Count.aggregate(sum)
Total_Count.aggregate(sum)
economy_Count = pd.concat([speeches_50['date'], economy_Count], axis = 1)
economy_Count.head()
economy_Count = economy_Count.set_index("date")
economy_Count.head(5)
# Create figure and plot space
fig, ax = plt.subplots(figsize=(15, 10))

# Add x-axis and y-axis
ax.bar(economy_Count.index.values,
        economy_Count['text'],
        color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Frequency",
       title="word count of economy")


# Rotate tick marks on x-axis
plt.setp(ax.get_xticklabels(), rotation=50)

plt.show()
Economy_Count = pd.concat([speeches_50['date'], Economy_Count], axis = 1)
Economy_Count = Economy_Count.set_index("date")
Economy_Count.head(5)

# Create figure and plot space
fig, ax = plt.subplots(figsize=(15, 10))

# Add x-axis and y-axis
ax.bar(Economy_Count.index.values,
        Economy_Count['text'],
        color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Frequency",
       title="word count of Economy")


# Rotate tick marks on x-axis
plt.setp(ax.get_xticklabels(), rotation=50)

plt.show()
economic_Count = pd.concat([speeches_50['date'], economic_Count], axis = 1)
economic_Count = economic_Count.set_index("date")
economic_Count.head(5)

# Create figure and plot space
fig, ax = plt.subplots(figsize=(15, 10))

# Add x-axis and y-axis
ax.bar(economic_Count.index.values,
        economic_Count['text'],
        color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Frequency",
       title="word count of economic")


# Rotate tick marks on x-axis
plt.setp(ax.get_xticklabels(), rotation=50)

plt.show()
Economic_Count = pd.concat([speeches_50['date'], Economic_Count], axis = 1)
Economic_Count = Economic_Count.set_index("date")
Economic_Count.head(5)

# Create figure and plot space
fig, ax = plt.subplots(figsize=(15, 10))

# Add x-axis and y-axis
ax.bar(Economic_Count.index.values,
        Economic_Count['text'],
        color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Frequency",
       title="word count of Economic")


# Rotate tick marks on x-axis
plt.setp(ax.get_xticklabels(), rotation=50)

plt.show()
Total_Count = pd.concat([speeches_50['date'], Total_Count], axis = 1)
Total_Count = Total_Count.set_index("date")
Total_Count.head(5)

# Create figure and plot space
fig, ax = plt.subplots(figsize=(15, 10))

# Add x-axis and y-axis
ax.bar(Total_Count.index.values,
        Total_Count['text'],
        color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Frequency",
       title="word count of Economic")


# Rotate tick marks on x-axis
plt.setp(ax.get_xticklabels(), rotation=50)

plt.show()