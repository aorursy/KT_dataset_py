# Importing modules

import pandas as pd



# Read datasets/papers.csv into papers

papers = pd.read_csv("../input/nips-papers/papers.csv")



# Print out the first rows of papers

papers.head()
# Remove the columns

papers.drop(['id', 'event_type', 'pdf_name'], axis=1, inplace=True)



# Print out the first rows of papers

papers.head()
# Group the papers by year

groups = papers.groupby('year')



# Determine the size of each group

counts = groups.size()



# Visualise the counts as a bar plot

import matplotlib.pyplot

%matplotlib inline

counts.plot(kind='bar')
# Load the regular expression library

import re



# Print the titles of the first rows 

print(papers['title'].head())



# The following line 

papers['title_processed'] = papers['title'].map(lambda x: re.sub('[,\.!?]', '', x))



# Convert the titles to lowercase

papers['title_processed'] = papers['title_processed'].map(lambda x: x.lower())



# Print the processed titles of the first rows 

papers['title_processed'].head()
# Import the wordcloud library

import wordcloud



# Join the different processed titles together.

long_string = ' '.join(papers['title_processed'])



# Create a WordCloud object, generate a wordcloud and visualise it

wordcloud = wordcloud.WordCloud()



# Generate a word cloud

wordcloud.generate(long_string)



# Visualize the word cloud

wordcloud.to_image()
# Load the library with the CountVectorizer method

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np



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



    plt.bar(x_pos, counts,align='center')

    plt.xticks(x_pos, words, rotation=90) 

    plt.xlabel('words')

    plt.ylabel('counts')

    plt.title('10 most common words')

    plt.show()



# Initialise the count vectorizer with the English stopwords

count_vectorizer = CountVectorizer(stop_words='english')



# Fit and transform the processed titles

count_data = count_vectorizer.fit_transform(papers['title_processed'])



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

        

# Tweak the two parameters below (use int values below 15)

number_topics = 10

number_words = 10



# Create and fit the LDA model

lda = LDA(n_components=number_topics)

lda.fit(count_data)



# Print the topics found by the LDA model

print("Topics found via LDA:")

print_topics(lda, count_vectorizer, number_words)
# The historical data indicates that:

more_papers_published_in_2018 = True