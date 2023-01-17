# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
papers = pd.read_csv("../input/papers.csv")
papers.head(5)
papers_copy = papers.copy()
papers_copy.drop(columns = ['id','event_type','pdf_name'],inplace=True)
papers_copy.head(3)
# We can now go on and prepare the daraset for modelling
groups = papers_copy.groupby("year")
counts = groups.size()
counts.plot(kind='bar',figsize=(9,7));
plt.title("ML Publications per year");
plt.xlabel("Years..!!");
#Preprocessing the Text Data
#1. Remove any punctutations
#2. Perform LowerCasing
#3. Print title before and after modifications
import re
print("Before MODIFICATION:")
print(papers_copy['title'].head())
papers_copy['title_processed'] = papers_copy['title'].map(lambda x: re.sub('[,\.!?]','',x))
papers_copy['title_processed'] = papers_copy['title_processed'].map(lambda x:x.lower())
print("After MODIFICATION")
print(papers_copy['title_processed'].head())
import wordcloud
long_string = "".join(papers_copy.title_processed)
wordcloud = wordcloud.WordCloud()
wordcloud.generate(long_string)
wordcloud.to_image()

#Word cloud tends out to be a very helpful tool to visualize the mmost used words in the particular feature.
from sklearn.feature_extraction.text import CountVectorizer
def plot_10_most_common_words(count_data,count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts +=t.toarray()[0]
    count_dict = (zip(words,total_counts))
    count_dict= sorted(count_dict,key=lambda x:x[1],reverse = True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    
    plt.bar(x_pos,counts,align='center')
    plt.xticks(x_pos,words,rotation=90)
    plt.ylabel('counts')
    plt.xlabel('words')
    plt.title('10 Most Common Words..!!')
    plt.show()
    
#Initialize the count vectorizer with the English Stop Words:
count_vectorizer = CountVectorizer(stop_words = 'english')
#Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(papers_copy['title_processed'])
#Visualize the most 10-most common words
plot_10_most_common_words(count_data,count_vectorizer)
warnings.simplefilter("ignore",DeprecationWarning)
from sklearn.decomposition import LatentDirichletAllocation as LDA
#Load the LDA model from sklearn
def print_topics(model,count_vectorizer,n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:  "% topic_idx)
        print(" ".join([words[i] for i in topic.argsort()[:- n_top_words -1: -1]]))
#Tweak the two parameters below:
number_topics =10
number_words=6
#Create and fit the LDA Model
lda = LDA(n_components = number_topics)
lda.fit(count_data)
#Print the topics found by the LDA Model:
print("Topics Found: ")
print_topics(lda,count_vectorizer,number_words)
