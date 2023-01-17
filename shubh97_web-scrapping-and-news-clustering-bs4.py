"""

We need to download the following packages to be used 

in our news scrapping as well as in our 

Language Processing task :-



1. BeautifulSoup4 (bs4) - An awesome web scrapping 

  and DOM Data extraction library.



"""

!pip3 install bs4
"""

Importing the following packages - 



1. requests - Requests is an inbuilt python package 

  for making a call to a web URL. It allows HTTP/1.1 

  requests to be carried out in an easy manner.



2. bs4 - An awesome web scrapping 

  and DOM Data extraction library.



3. nltk - For processing text, NLTK provides us with 

  lots of great functionality built into it.



4. sklearn - It is a popular library having a 

  collection of a numerous Machine Learning Algorithms 

  implemented into it that are just ready to use.



5. collections - It is an inbuilt python library that

  contains a great collection of special container

  datatypes.



6. textwrap - It is also an inbuilt package for 

  wrapping up the long text so that it doesn't 

  go out of screen width. (For those who hate 

  horizontal scrolling :p).

"""



import requests

from bs4 import BeautifulSoup



import nltk



# Downloading the stopwords and punkt Tokenizer to be used later

nltk.download('stopwords')

nltk.download('punkt')



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans



import collections

import textwrap
"""

Building a word tokenizer to extract all the 

tokens in the given sentence/text.

"""



def word_tokenizer(text):



        # NLTK provides the way to break text into tokens.

        # Example - Tokenizing the text "Man beats coronavirus"

        # results in ["Man", "beats", "coronavirus"].

        # It is more than just a space separator.



        tokenizer = nltk.word_tokenize(text)



        # While analyzing the english words, since we know that

        # grow, grew, growing, grown and many such words are possible

        # that mean the same. So we stem the word to its minimal form

        # in order to get the least length token which is same in all 

        # such word stemming. Eg - stem('grows') => 'grow' ideally



        porter_stemmer = nltk.stem.PorterStemmer()



        # Stopwords is a list of those words which are very commonly used

        # in the english library and there is not much use of them in

        # clustering. Eg - The articles do not make much sense and so

        # they are included in stopwords



        stopwords = nltk.corpus.stopwords.words('english')

        tokens = []



        for token in tokenizer:

              if token not in stopwords:

                    # Stemming the token, if not present in stopwords

                    # and appending it in tokens list.

                    stemmed_token = porter_stemmer.stem(token)

                    tokens.append(stemmed_token)



        # Returning the list of all the tokens

        return tokens
"""

Based on features, we will cluster the sentences whose features are 

most similar to one another.

"""



def cluster_sentences(sentences, nb_of_clusters=10):

        # A TF-IDF vectorizer is a bag of words model. It works 

        # on the basis of occurence of a term in a document and

        # the number of documents containing the same term. It 

        # converts a collection of raw text 

        # to a matrix of Term frequency and Inverse 

        # Document Frequency and features.



        tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,

                                        lowercase=True)



        # Fitting all the sentences/documents in TF-IDF vectorizer

        # for text to numerical feature matrix.

        

        news_tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        

        # Initializing KMeans clustering to segregate the documents

        # on the basis of TF-IDF scores

        

        kmeans = KMeans(n_clusters=nb_of_clusters)

        cost = []



        # Fitting the TF-IDF matrix in the feature space to split out the

        # clusters on the basis of pattern between text.

        

        kmeans.fit(news_tfidf_matrix)



        # Initializing a dictionary/map object for storing all the

        # news clusters.



        clusters = collections.defaultdict(list)

        

        # Since, we got a news index and its corresponding cluster label,

        # we will use it to create the dictionary for clustering the news

        # by their labels and indices.



        for index, label in enumerate(kmeans.labels_):

                clusters[label].append(index)



        # Returning the clusters



        return clusters
"""

In precious step, we have created the clusters. Now its time 

to get back news articles using the clusters mapping.

"""



def create_cluster(news_articles, num_clusters = 10):

    # Creating a list of all the headlines by using the keys of

    # news articles mapping object.



    headlines = list(news_articles.keys())



    # Creating a list of all the news bodies by using the values of

    # news articles mapping object.



    news_body = list(news_articles.values())



    # Calling the clustering function we wrote earlier for 

    # getting the cluster labels



    clusters = cluster_sentences(headlines, num_clusters)



    # Creating a map for storing the news articles in order of the

    # cluster they are assigned to.

    clustering_results = collections.defaultdict(list)



    news_articles_mapping = {}



    # Creating a map for storing the news articles in order of the

    # cluster they are assigned to.

    for cluster_index in range(num_clusters):

            for index, headline in enumerate(clusters[cluster_index]):

                  # Appending all the similar grouped articles 

                  # in a single clustered map object.

                  clustering_results["Cluster "+ 

                                     str(cluster_index+1)].append(headlines[headline])

                  news_articles_mapping[headlines[headline]] = news_body[headline]

    return (clustering_results, news_articles_mapping)
# Declaring the url on which we are going to apply scrapping.



url_to_crawl = 'https://inshorts.com/en/read/'



# Since, we will be dealing with HTML content, we will be using HTML parser

parser = 'html.parser'



# Getting the crawled webpage.

response = requests.get(url_to_crawl)



# Converting response to BeatifulSoup object for DOM manipulations

# and extracting text from it

bs4obj = BeautifulSoup(response.text, parser)
# Printing the bs4 data



print(bs4obj.prettify()[:1000])
# Now since we have the whole page HTML,

# we can extract the any text we would wish to like.



all_news_map = {}



# Observing the pattern of the web page, we can see that 

# all news in the webpage are covered under their div tags 

# which contain news-card class.



all_news_cards = bs4obj.findAll('div', {"class": "news-card"})



# We loop through all news object in order to find the 

# news heading and its corresponding news body.



for newsObj in all_news_cards:

    # Finding span tag under newsObj with attribute itemprop 

    # set to description and extracting the value of its content.

    

    news_headline = newsObj.find('span', {"itemprop": "description"})['content']

    

    # Finding the div under newsObj having attribute itemprop 

    # set to articleBody and extracting its inner text.

    

    news_body = newsObj.find('div', {"itemprop": "articleBody"}).text

    

    # Adding a key with news heading and it body as its value.



    all_news_map[news_headline] = news_body
# Its time to get the news articles clustered. So 

# calling the create_cluster function we created 

# earlier for this. For passing on, we have news map 

# object and the maximum number of clusters, we wish

# the algorithm to output. 



clustering_result, news_article_mapping = create_cluster(all_news_map, 

                                                         num_clusters=10)
# Creating a text wrapper object to display the cluster results

# and prevents it from overflowing.



wrapper = textwrap.TextWrapper(width=100, initial_indent='\t\t', subsequent_indent='\t\t')



# Looping over all the items in the clustering_result



for cluster, news_headlines in clustering_result.items():

    print(cluster)

    for headline in news_headlines:

      print('\t', headline)

      for text in wrapper.wrap(news_article_mapping[headline]):

        print(text)

      print('\n')