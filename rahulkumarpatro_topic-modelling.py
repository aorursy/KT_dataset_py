# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/trump-tweets/trumptweets.csv")
df
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer 
clean=[]
for i in range(0, 15000):
    review = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"', ' ', df['content'][i])
    review = review.lower()
    review = review.split()
    lm= WordNetLemmatizer() 
    review = [lm.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    clean.append(review)
df['content'][0]
clean[0]
df_new=pd.DataFrame(df['content'][0:15000])
df_new
df_new['tweets']=clean
df_new
from nltk.tokenize import word_tokenize
df_new['tokens']=df_new['tweets'].apply(word_tokenize)
df_new
df_new['tokens'][90]
from sklearn.feature_extraction.text import CountVectorizer
df_new['tokens'][90]
vect = CountVectorizer().fit(df_new['tokens'][90])
bag_of_words = vect.transform(df_new['tokens'][90])
sum_words = bag_of_words.sum(axis=0) 
sum_words
def most_freq_words(s, n=None):
    vect = CountVectorizer().fit(s)
    bag_of_words = vect.transform(s)
    sum_words = bag_of_words.sum(axis=0) 
    freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
    freq =sorted(freq, key = lambda x: x[1], reverse=True)
    return freq[:n]
most_freq_words([ word for tweet in df_new.tokens for word in tweet],20)
def least_freq_words(s, n=None):
    vect = CountVectorizer().fit(s)
    bag_of_words = vect.transform(s)
    sum_words = bag_of_words.sum(axis=0) 
    freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
    freq =sorted(freq, key = lambda x: x[1], reverse=False)
    return freq[:n]
least_freq_words([ word for tweet in df_new.tokens for word in tweet],20)
df_new['tokens'][1]
df_new
vectorizer = CountVectorizer(min_df=0)# Here "min_df" in the parameter refers to the minimum document frequency and the vectorizer will simply drop all words that occur less than that value set (either integer or in fraction form)
sentence_transform = vectorizer.fit_transform(df_new['tweets'])
sentence_transform
sentence_transform.shape
print("\nThe vectorized array looks like:\n {}".format(sentence_transform.toarray()))
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=8, max_iter=5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)
#n_components are the number of topics you want to classify.
lda.fit(sentence_transform)
words=[]
in_arr = np.array([ 2, 0,  1, 5, 4, 1, 9]) 
print ("Input unsorted array : ", in_arr)  
  
out_arr = np.argsort(in_arr) 
print ("Output sorted array indices : ", out_arr) 
print("Output sorted array in Ascending Order: ", in_arr[out_arr]) 

out_arr_new=np.argsort(in_arr)[::-1]
print("Output Sorted Array in Descending Order",in_arr[out_arr_new])
out_arr_new=np.argsort(in_arr)[::-1][:4]
print("Output Sorted Array in Descending Order",in_arr[out_arr_new])
out_arr_new=np.argsort(in_arr)[:-4-1:-1]
print("Output Sorted Array in Descending Order",in_arr[out_arr_new])
def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[::-1][:n_top_words]])
        print(message)
        words.append(message)
        print("="*170)
n_top_words = 40
print("\nTopics in LDA model: ")
tf_feature_names = vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
first_topic = lda.components_[0]
second_topic = lda.components_[1]
third_topic = lda.components_[2]
first_topic
first_topic.shape
second_topic.shape
words[0]
from wordcloud import WordCloud 
for i in range(0,8):
    wordcloud = WordCloud(max_font_size=40, max_words=40).generate(words[i])

# Display the generated image:
    plt.figure(figsize=(10,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()