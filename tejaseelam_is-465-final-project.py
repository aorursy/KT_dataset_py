# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import os
import glob
import matplotlib as mpl
# import natural language tool kit
import nltk as nlp
# Just making the plots look better
mpl.style.use('ggplot')
mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['font.size'] = 12

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
filename = "/kaggle/input/cbc-news-coronavirus-articles-march-26/news.csv"
df = pd.read_csv(filename)
df.head()
# import natural language tool kit
import nltk as nlp

# counting vocabulary of words
text = df.text[0]
splitted = text.split(" ")
print("number of words: ",len(splitted))

# counting unique vocabulary of words
text = df.text[0]
print("number of unique words: ",len(set(splitted)))

# print first five unique words
print("first 5 unique words: ",list(set(splitted))[:5])

# frequency of words 
dist = nlp.FreqDist(splitted)
print("frequency of words: ",dist)

# look at keys in dist
print("words in text: ",dist.keys())

# count how many time a particalar value occurs. Lets look at "box"
print("the word box is occured how many times:",dist["box"])
#find frequency of the words in the "text" column

#stop = stopwords.words('english')
stop = set(['the', 'to', 'of'])
#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#stopwords.words('english')
#stop = set(stopwords.words('english'))

for x in df.text:
    if x not in stop:
        pd.Series(np.concatenate([ x.split()])).value_counts()
        #pd.Series(np.concatenate([ x.split() for x in df.text])).value_counts()[:10]
text[text == "Corona"].value_counts()
#describes numeric values of column by providing stats
df['publish_date'].describe()
filter = data["Age"]=="Twenty five"
 
# printing only filtered columns
data.where(filter).dropna()
df.iloc[0]
type(_)
#returns items with Coronavirus in the title
#df[df['title'].str.count('^[Coronavirus].*')>0]
df[df['title'].str.count('Coronavirus')>0]
#find times that USA is mentioned in the title
df[df['title'].str.count('USA')>0]  #0 times

df[df['title'].str.count('United States')>0]  #0 times

df[df['title'].str.count('U.S.')>0]  #68 times


#find times that USA is mentioned in the text
df[df['text'].str.count('USA')>0]  #8 times

df[df['text'].str.count('United States')>0]  #337 times

df[df['text'].str.count('U.S.')>0]  #427 times

df[df['title'].str.count('Trump')>0]
df[df['text'].str.count('Trump')>0]
# number of articles published in January 2020
df[df['publish_date'].str.count('^2020-01.*')>0]  #184
df[df['publish_date'].str.count('^2020-02.*')>0]  #398
df[df['publish_date'].str.count('^2020-03.*')>0]  #2952
# counting vocabulary of words
text = df.text[0]
splitted = text.split(" ")
print("number of words: ",len(splitted))

# counting unique vocabulary of words
text = df.text[0]
print("number of unique words: ",len(set(splitted)))

# print first five unique words
print("first 5 unique words: ",list(set(splitted))[:5])

# frequency of words 
dist = nlp.FreqDist(splitted)
print("frequency of words: ",dist)

# look at keys in dist
print("words in text: ",dist.keys())

# count how many time a particalar value occurs. Lets look at "box"
print("the word box is occured how many times:",dist["box"])
for index, row in df.iterrows():
    text = df.text[index]
    splitted = text.split(" ")
    print("number of words: ",len(splitted))
    # counting unique vocabulary of words

    print("number of unique words: ",len(set(splitted)))

    # print first five unique words
    print("first 5 unique words: ",list(set(splitted))[:5])
    
    # frequency of words 
    dist = nlp.FreqDist(splitted)
    print("frequency of words: ",dist)

    # look at keys in dist
    #print("words in text: ",dist.keys())

    # count how many time a particalar value occurs. Lets look at "box"
    print("the word Coronavirus is occured how many times:",dist["COVID-19"])
groups = df.groupby('publish_date')
counts = groups.size()
import matplotlib.pyplot as plt
%matplotlib inline
counts.plot(kind='bar')
plt.xlabel('Date')
plt.ylabel('Number of Papers per Group')
plt.title('Machine Learning Publications since 1987')

# Remove the columns
df = df.drop(columns=['Unnamed: 0', 'authors', 'publish_date', 'url'])
# Print out the first rows of df
df.head()
# Load the regular expression library
import re

# Remove punctuation
df['df_text_processed'] = df['text'].map(lambda x: re.sub('[,\.!?]', '', x))

# Convert the titles to lowercase
df['df_text_processed'] = df['df_text_processed'].map(lambda x: x.lower())

from nltk.corpus import stopwords
stop = stopwords.words('english')
df['df_text_processed'] = df['df_text_processed'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#Print out the first rows of df
df['df_text_processed'].head()
# Import the wordcloud library
from wordcloud import WordCloud
from nltk.corpus import stopwords

# Join the different processed titles together.
long_string = ','.join(list(df['df_text_processed'].values))

stopwords = {'said', 'say','would', 'also', 'says'}
#stopwords.words('english')
wordcloud =WordCloud(stopwords=stopwords)

# Create a WordCloud object
wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
wordcloud.to_image()
# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='Most Common Words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(df['df_text_processed'])
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 10
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)
%%time
from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis
LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(number_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == True:
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)

        
pyLDAvis.display(LDAvis_prepared)