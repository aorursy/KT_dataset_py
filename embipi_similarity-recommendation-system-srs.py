## Basics libraries

import numpy as np

import pandas as pd

# pd.set_option('display.max_colwidth', -1)



import warnings

warnings.filterwarnings('ignore')



## Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")



import pyLDAvis

import pyLDAvis.gensim



import networkx as nx



## Web Scrapping libraries

from requests import get

from bs4 import BeautifulSoup



import time

from time import sleep

from IPython.core.display import clear_output

from random import randint

from warnings import warn



## Data Cleaning libraries

from sklearn.base import BaseEstimator, TransformerMixin

import re

import string



from spacy.lang.en.stop_words import STOP_WORDS



import nltk

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk import pos_tag



import gensim

from gensim.parsing.preprocessing import STOPWORDS

from gensim.corpora import Dictionary



## Machine Learning libraries

from gensim.models import LdaModel, CoherenceModel

from scipy.spatial import distance
## Copying the url of the most popular series from IMDb

url = "https://www.imdb.com/chart/tvmeter"



## Making a request to the web site with and alert in case of failure

response = get(url)

if response.status_code != 200:

    warn('Request for series information: Status code: {0}'.format(response.status_code))  



## Parsing the html and filtering the content

html_soup = BeautifulSoup(response.text, 'html.parser')

movie_container = html_soup.find_all("td", class_=("titleColumn","ratingColumn imdbRating"))



## Generating empty list for information filling

urls = []

names = []

years_begins = []

popular_rankings = []

imdb_ratings = []

popularities_trends = []



for i in range(0,len(movie_container)):

    if i%2==0:

        print("Loop de la serie número", i)

        clear_output(wait=True)

        

        ## Url title

        url_titulo = str(movie_container[i].a["href"][7:-1])

        urls.append(url_titulo)

        

        ## Name 

        name = str(movie_container[i].a.text).lower()

        names.append(name)

        

        ## Release year

        if len(movie_container[i].span.text[1:-1]) == 4:

            year_begin = str(movie_container[i].span.text[1:-1])

            years_begins.append(year_begin)

        else:

            year_begin = np.nan

            years_begins.append(year_begin)

            

        ## Popularity ranking

        popular_ranking = int(movie_container[i].div.text.split("\n")[0])

        popular_rankings.append(popular_ranking)

        

        ## Popularity trend

        neg = str(movie_container[i].find("span", class_="global-sprite titlemeter down")) 

        pos = str(movie_container[i].find("span", class_="global-sprite titlemeter up")) 

        tendency = neg + pos



        if "down" in tendency:

            temp="-"

        elif "up" in tendency: 

            temp="+"

        else:

            temp=""

        if "no change" in movie_container[i].div.text:

            trend = movie_container[i].div.text.replace("no change", "0").split("(")[1].split(")")[0]

        else:

            trend = movie_container[i].div.text.replace(",", "").split("\n\n")[1].split(")")[0]

            

        popularity_trend = int(temp + trend)

        popularities_trends.append(popularity_trend)

                              

    if i%2!=0 and len(str(movie_container[i].text)) != 1:

        print("Loop de la serie número", i)

        clear_output(wait=True)



        ## IMDb rating

        imdb_rating = float(str(movie_container[i].text[1:4]).replace(",","."))

        imdb_ratings.append(imdb_rating)

           

    elif i%2!=0 and len(str(movie_container[i].text)) == 1:

        print("Loop de la serie número", i)

        clear_output(wait=True)



        ## Filling with NaN

        imdb_rating = np.nan

        imdb_ratings.append(imdb_rating)



print("Bucle finalizado")
## Creating a dataframe with the scrapped data

data_series = pd.DataFrame({'urls' : urls,

                            'names' : names,

                            'years_begins' : years_begins,

                            'popular_rankings' : popular_rankings,

                            'popularities_trends' : popularities_trends,

                            'imdb_ratings' : imdb_ratings}, )

print(data_series.info())



data_series.sample(5)
## Generating empty list for information filling

users_names = []

users_ratings = []

comments_dates = []

comments_titles = []

users_texts = []

urls_comments = []



requests=0

start_time_title = time.time()



## Making a loop

for url_title in urls:

    url = "https://www.imdb.com/title/"+url_title+"/reviews"

    

    ## Making a request to the web site with and alert in case of failure

    response = get(url)

    if response.status_code != 200:

        warn('Request for new serie: {0}; Status code: {1}'.format(requests, response.status_code))    

    

    ## Stopping the loop for a few seconds and controling the frequency of requests

    requests += 1

    sleep(randint(1,3))

    elapsed_time = time.time() - start_time_title

    print('Request for new serie: {0}; Frequency: {1:.4} requests/s'.format(requests, requests/elapsed_time))

    clear_output(wait=True)    

    

    ## Parsing the html and filtering the content

    html_soup = BeautifulSoup(response.text, 'html.parser')

    comment_container = html_soup.find_all("div", class_=("header", "review-container", "load-more-data"))



    ## Obtanining the total number of comments

    number_comments = int(str(comment_container[0].div.text).replace(",","").replace("Reviews",""))

   

    ## Initializing the parameters of the loop

    n1=1

    n2=0

    requests_ajax=0

    start_time_comment = time.time()

    length = len(comment_container)-2

    flag = True

    

    ## Obtaining the URL-AJAX

    data_ajaxurl = comment_container[length]["data-ajaxurl"]



    ## Creagint a nested loop

    while (n2 < number_comments and flag):        

        for i in range(n1, length):

            n2+=1

            ## User rating

            if comment_container[i].find("span", class_="rating-other-user-rating") is not None:

                user_rating = int(comment_container[i].find("span", class_="rating-other-user-rating").span.text)

                users_ratings.append(user_rating)



                ## User name

                user_name = comment_container[i].find("span", class_="display-name-link").text

                users_names.append(user_name)



                ## Comment date

                comment_date = comment_container[i].find("span", class_="review-date").text

                comments_dates.append(comment_date)



                ## Comment title

                comment_title = comment_container[i].find(class_="title").text[1:-1]

                comments_titles.append(comment_title)



                ## User comment

                user_text = comment_container[i].find(class_="text show-more__control").text

                users_texts.append(user_text)



                ## URL title

                urls_comments.append(url_title)

        

        ## URL-AJAX key

        if "data-key" in str(comment_container[length]):

            data_key= comment_container[length]["data-key"]

            

            ## Generating a new URL with more comments

            new_url = "http://www.imdb.com/{0}?paginationKey={1}".format(data_ajaxurl, data_key)

        

            ## Making a request to the web site with and alert in case of failure

            response = get(new_url)

            if response.status_code != 200:

                warn('Request for more comments: {0}.{1}; Status code: {2}'.format(requests, requests_ajax, response.status_code))    

        

            ## Stopping the loop for a few seconds and controling the frequency of requests

            requests_ajax += 1

            sleep(randint(1,3))

            elapsed_time = time.time() - start_time_comment

            print('Request for comments: {0}.{1}; Frequency: {2:.4} requests/s'.format(requests, requests_ajax, requests_ajax/elapsed_time))

            clear_output(wait=True)



            ## Parsing the html and filtering the content

            html_soup = BeautifulSoup(response.text, 'html.parser')

            comment_container = html_soup.find_all("div", class_=("review-container", "load-more-data"))



            ## Updating the parameters of the loop

            length = len(comment_container)-1

            n1=0   



        else:

            flag = False



print("Bucle finalizado", "\n")
## Creating a dataframe with the scrapped data

data_comments = pd.DataFrame({'urls' : urls_comments,

                              'users_names' : users_names,

                              'users_ratings' : users_ratings,

                              'comments_dates' : comments_dates,

                              'comments_titles' : comments_titles,

                              'users_texts' : users_texts})



## Obatining some information from the variables and showing a sample

print(data_comments.info())

data_comments.sample(5)
## Saving the data sets

path = "../input/"



print("Tamaño del set de series:", data_series.shape)

print("Tamaño del set de comentarios:", data_comments.shape)



data_series.to_csv(path + "data_series.csv", header=True, sep=";", index=False)

data_comments.to_csv(path + "data_comments.csv", header=True, sep="|", index=False)



print("\nSets de datos guardados")
## Loading the data sets

path = "../input/"



data_series = pd.read_csv(path + "data_series.csv", header=0, sep=";", dtype={"years_begins":str})

data_comments = pd.read_csv(path + "data_comments.csv", header=0, sep="|", dtype={"users_ratings":float})



print("Sets de datos cargados")
## Joining the data from the series with the comments of the users by the url name

data = data_series.merge(data_comments, how='left', on="urls")



## Obtaining the number of comments per serie

comments_per_serie = data.groupby(["names"])["users_texts"].agg("count").reset_index()

comments_per_serie.columns = ["names","total_comments"]



## Joining the preview data set with the comments count by the serie name

data = data.merge(comments_per_serie, how='left', on="names")



## Transforming the type of the data

data.comments_dates = pd.to_datetime(data.comments_dates)

data.total_comments = data.total_comments.astype("int64")

data["short_dates"] = data.comments_dates.astype(str).apply(lambda x: x[:][0:4])



## Obatining some information from the variables and showing a sample

print(data.info())

data.sample(3)
## Obtaining the sum of the NaN values

miss_values = data.isnull().sum()



## Calculating the percentaje of NaN values and rounding the result

miss_values_percent = (miss_values*100/len(data)).round(2)



## Joining both results in the same table

miss_values_table = pd.concat([miss_values,miss_values_percent], axis=1)



## Renaming the columns and filtering the rows with non-zero values for a proper visualization

miss_values_table = miss_values_table.rename(columns={0:"Total de NaN", 1:"% de NaN"})

miss_values_table[miss_values_table.loc[:,"Total de NaN"] != 0]
## Dropping rows with NaN values

data = data.dropna(axis=0)



## Counting for possible duplicates and erasing them

duplicated_rows = data.duplicated().sum()   

if (duplicated_rows > 0):

    data = data.drop_duplicates().reset_index(drop=True)

    print("Número de filas duplicadas eliminadas:", duplicated_rows)

else:

    print("No se han encontrado filas duplicadas")
## Obtaining a descriptive analisis of the cuantitative variables and rounding the results

data.describe().round(2)
## Generating a correlation table to observ the interaction between variables

corr= data.corr().round(2)



## Visualizing these correlations with a color map

fig, ax = plt.subplots(figsize=(15,7))

ax=sns.heatmap(corr, 

               ax=ax,           

               cmap="coolwarm", 

               annot=True, 

               fmt='.2f',       

               annot_kws={"size": 14},

               linewidths=3)



ax.set_title("Mapa de calor de correlaciones entre variables", fontsize=18, fontweight="bold")



plt.show()
## Generating a pairplot graph to observ the correlation of the numeric variables

pp = sns.pairplot((data),

              kind="scatter",

              diag_kind="kde",

              height=2.5,

              markers="o",

              vars=["popular_rankings", "popularities_trends", "imdb_ratings", "total_comments", "users_ratings",])



fig = pp.fig 

fig.subplots_adjust(top=0.95, wspace=0.05)

fig.suptitle("Pairplot de correlaciones entre variables", fontsize=18, fontweight="bold")



plt.show()
## Generating a frequency graph

fig, ax = plt.subplots(figsize=(15,5))

ax=sns.distplot(data.years_begins.astype(int),

             bins=max(data.years_begins.astype(int))-min(data.years_begins.astype(int))+1,

             hist=True,

             hist_kws={"color": "g", "alpha": 0.5},

             kde=True,          

             kde_kws={"color": "k", "lw": 2})



ax.set_title("Ditribución de años de estreno de series", fontsize=18, fontweight="bold")

ax.set_xlabel(" ")

ax.set_ylabel("Frecuencia de años de estreno", fontsize=14)



## Generating a boxplot graph

fig, ax = plt.subplots(figsize=(15,2.5))

ax=sns.boxplot(data=data.years_begins.astype(int),

               color="g",

               width=0.5,

               linewidth=2,

               orient="h")



ax.set_xlabel("Año de estreno", fontsize=14)

ax.set_ylabel(" ", fontsize=14)



plt.show()
## Generating a frequency graph

fig, ax = plt.subplots(figsize=(15,5))

ax=sns.distplot(data.short_dates.astype(int),

             bins=max(data.short_dates.astype(int))-min(data.short_dates.astype(int))+1,

             hist=True,

             hist_kws={"color": "r", "alpha": 0.5},

             kde=True,          

             kde_kws={"color": "k", "lw": 2})



ax.set_title("Ditribución de comentarios a lo largo del tiempo", fontsize=18, fontweight="bold")

ax.set_xlabel(" ")

ax.set_ylabel("Frecuencia de comentarios", fontsize=14)



## Generating a boxplot graph

fig, ax = plt.subplots(figsize=(15,2.5))

ax=sns.boxplot(data=data.short_dates.astype(int),

               color="r",

               width=0.5,

               linewidth=2,

               orient="h")



ax.set_xlabel("Año del comentario", fontsize=14)

ax.set_ylabel(" ", fontsize=14)



plt.show()
## Filtering the rows with comments

comments_per_serie= comments_per_serie[comments_per_serie.total_comments > 0]



## Sorting the data by the number of comments

top10 = comments_per_serie.sort_values(by=["total_comments"], ascending=False).reset_index(drop=True).head(10)

bottom10 = comments_per_serie.sort_values(by=["total_comments"], ascending=False).reset_index(drop=True).tail(10)



## Generating a subplot with some parameters

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,15))

explode= [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]



## Generating a pie chart with the top10 series with more comments

ax1.set_title("Series más comentadas", fontsize=18, fontweight="bold")

ax1.pie(top10["total_comments"],

       labels=top10.names,

       explode=explode,

       autopct="%.2f%%",

       startangle=90)



## Generating a pie chart with the top10 series with less comments

ax2.set_title("Series menos comentadas", fontsize=18, fontweight="bold")

ax2.pie(bottom10["total_comments"],

       labels=bottom10.names,

       explode=explode,

       autopct="%.2f%%",

       startangle=90)



plt.show()
## Generating a list with the top10 series and obtaining their stadistics

names_list = top10.names.tolist()

top10_data = data_series[data_series["names"].isin(names_list)]

print(top10_data.describe().round(2), "\n")



## Visualizing the data distribution of the numerical variables

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(18,5))



ax1.violinplot(top10_data.imdb_ratings,vert=False, showmeans=True)

ax1.set_title("IMDb Ratings", fontsize=18, fontweight="bold")



ax2.violinplot(top10_data.popularities_trends, vert=False, showmeans=True)

ax2.set_title("Popularities Trends", fontsize=18, fontweight="bold")



ax3.violinplot(top10_data.popular_rankings, vert=False, showmeans=True)

ax3.set_title("Popular Rankings", fontsize=18, fontweight="bold")



plt.show()
## Generating a list with the bottom10 series and obtaining their stadistics

names_list = bottom10.names.tolist()

bottom10_data = data_series[data_series["names"].isin(names_list)]

print(bottom10_data.describe().round(2), "\n")



## Visualizing the data distribution of the numerical variables

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(18,5))



ax1.violinplot(bottom10_data.imdb_ratings,vert=False, showmeans=True)

ax1.set_title("IMDb Ratings", fontsize=18, fontweight="bold")



ax2.violinplot(bottom10_data.popularities_trends, vert=False, showmeans=True)

ax2.set_title("Popularities Trends", fontsize=18, fontweight="bold")



ax3.violinplot(bottom10_data.popular_rankings, vert=False, showmeans=True)

ax3.set_title("Popular Rankings", fontsize=18, fontweight="bold")



plt.show()
## Selecting the variables with natural language

text = data.loc[:,("names", "comments_titles", "users_texts")]



## Joining the comments title with the comments texts of each user

text["all_text"] = np.NAN

text.all_text = text.comments_titles + ". " + text.users_texts



## Generating a string like a document with each text of each serie

series_texts = []

n=0

for i in text.names.unique():

    text_union = text[text.names==i].all_text.tolist()

    text_union = " ".join(text_union)

    series_texts.append(i + ". " + text_union)

    

## Creating a dataframe with all the natural language joined on one variable

data_NLP = pd.DataFrame({"names":text.names.unique().tolist(),

                         "series_texts":series_texts})



print("Tamaño del DataFrame:", data_NLP.shape)
## Generating a object class to clean the data

warnings.filterwarnings('ignore')



class CleanText(BaseEstimator, TransformerMixin):

   

    def remove_urls(self, input_text):

        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

    

    def remove_emoji(self, input_text):

        return input_text.encode('ascii', 'ignore').decode('ascii')

  

    def remove_punctuation(self, input_text):

        punct = string.punctuation

        trantab = str.maketrans(punct, len(punct)*' ') 

        return input_text.translate(trantab)



    def remove_digits(self, input_text):

        return re.sub(r'\d+', '', input_text)



    def to_lower(self, input_text):

        return input_text.lower()

    

    def remove_stopwords(self, input_text):

        words = input_text.split() 

        clean_words = [word for word in words if (word not in STOPWORDS and word not in STOP_WORDS) and len(word) > 3] 

        return " ".join(clean_words) 

      

    def lemmatize_all(self, input_text):

        wn = WordNetLemmatizer()

        words = input_text.split() 

        clean_words = []

        for word, tag in pos_tag(words):

            if tag.startswith("NN"):

                clean_words.append(wn.lemmatize(word, pos='n'))

            elif tag.startswith('VB'):

                clean_words.append(wn.lemmatize(word, pos='v'))

            elif tag.startswith('JJ'):

                clean_words.append(wn.lemmatize(word, pos='a'))

            elif tag.startswith('ADV'):

                clean_words.append(wn.lemmatize(word, pos='r'))               

            else:

                clean_words.append(word)

        return " ".join(clean_words)

    

    def fit(self, X, y=None, **fit_params):

        return self

    

    def transform(self, X, **transform_params):

        clean_X = X.apply(self.remove_urls).apply(self.remove_emoji).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.lemmatize_all)

        return clean_X
## Generating a CleanText object to clean the text

ct = CleanText()

data_NLP.series_texts = ct.fit_transform(data_NLP.series_texts)



## Tokenining the text of each serie

tokenize = [word_tokenize(text) for text in data_NLP.series_texts]



## Generating a dictionary with the tokens

dictionary = Dictionary(tokenize)

print("Tamaño del diccionario original:", len(dictionary))



## Filtering the extreme values of the dictionary to keep the more representative tokens

dictionary.filter_extremes(no_below=10, no_above=0.90)

print("Tamaño del diccionario filtrado:", len(dictionary))



## Generating a frequencies bag of words

bow_corpus = [dictionary.doc2bow(doc) for doc in tokenize]
## Optimizing the coherence of the model

warnings.filterwarnings('ignore')



coherence_list = []

model_list = []

topics_range = range(2,6,1)

for num_topics in topics_range:

    lda_model = LdaModel(corpus=bow_corpus,

                         id2word=dictionary,

                         num_topics=num_topics,

                         chunksize=5,

                         passes=20,

                         alpha="auto",

                         eta="auto",

                         per_word_topics=True)

    model_list.append(lda_model)

    

    coherence_model = CoherenceModel(model=lda_model, texts=tokenize, dictionary=dictionary, coherence='c_v')

    coherence_list.append(coherence_model.get_coherence())

    

fig, ax = plt.subplots(figsize=(15,5))



ax = sns.lineplot(x=topics_range, y=coherence_list, color="darkorange", linewidth=2.5)



ax.set_title("Número optimo de Topics", fontsize=18)

ax.set_xlabel("Número de Topics", fontsize=14)

ax.set_ylabel("Coherencia", fontsize=14)



plt.show()
## Obtaining the more representative words of each topic by chossing the model with the best coherence

lda_model_opt = model_list[np.argmax(coherence_list)]



for idx, topic in lda_model_opt.print_topics(-1):

    print('\nTopic: {} \nWord: {}'.format(idx, topic))
warnings.filterwarnings('ignore')



## Visualizing the obtained topics

vis = pyLDAvis.gensim.prepare(lda_model_opt, bow_corpus, dictionary, mds='mmds')

pyLDAvis.display(vis)
## Obtaining the topic of each document

doc_topic = []

for i in range(0, len(bow_corpus)):

    topic = pd.DataFrame(lda_model_opt[bow_corpus[i]][0]).sort_values(by=1, ascending=False).values[0][0].astype(int)

    doc_topic.append(topic)



## Obtaning the number of the document

num_doc = ["Doc " + str(i+1) for i in range(len(bow_corpus))]



## Generating a dataframe with the data

data_doc_topic = pd.DataFrame({"Documento":num_doc, "Topics":doc_topic})



## Obtaining a count of the number of documents per topic 

doc_topic_count = data_doc_topic.groupby("Topics")["Documento"].aggregate("count")

print(pd.DataFrame(doc_topic_count))



## Creating a threshold to avoid a huge standard desviation of the topics per document

threshold = 10

out_thrs = doc_topic_count[doc_topic_count < threshold].index.tolist()



print("\nEstablecemos un umbral de {threshold} documentos que afecta a los topics: {out_thrs}".format(threshold=threshold, out_thrs=out_thrs))



## Grouping the series under the threshold with the topic that has the less series

index_list = []

for j in out_thrs:

    ind = data_doc_topic[data_doc_topic["Topics"] == j].index.tolist()

    [index_list.append(k) for k in ind]

    

    new_topic_list = doc_topic_count.sort_values(ascending=True).index.tolist()

    for l in new_topic_list:

        if l not in out_thrs:

            new_topic = l

            break

        else:

            continue

    

    print("El nuevo topic asignado es:", new_topic)

    data_doc_topic.loc[index_list, "Topics"] = new_topic

    

## Obtaining again a count of the number of documents per topic 

doc_topic_count = data_doc_topic.groupby("Topics")["Documento"].aggregate("count")

        

## Visualizating the distribution of topics along all the series  

fig, ax = plt.subplots(figsize=(15,5))

ax = plt.bar(doc_topic_count.index.astype(str), doc_topic_count, linewidth=2, width=1, edgecolor="black", color=sns.color_palette("bright"))



plt.title("Distribución de documentos por topic", fontsize=18, fontweight="bold")

plt.xlabel("Topic", fontsize=14)

plt.ylabel("Número de documentos", fontsize=14)



plt.show()
## Joining the data of the series with the obtained topic for each serie

data_series_topics = pd.concat([data_series, data_doc_topic.Topics.astype(str)], axis=1)



## Obtaining the sum of the NaN values

miss_values = data_series_topics.isnull().sum()



## Obtaining the percentaje of NaN values and rounding the result

miss_values_percent = (miss_values*100/len(data_series_topics)).round(2)



## Joining both results on the same table

miss_values_table = pd.concat([miss_values,miss_values_percent], axis=1)



## Renaming the columns and filtering those rows with comments for a proper visualization

miss_values_table = miss_values_table.rename(columns={0:"Total de NaN", 1:"% de NaN"})

miss_values_table[miss_values_table.loc[:,"Total de NaN"] != 0]
## Dropping all rows with NaN values

data_series_topics = data_series_topics.dropna(axis=0)



## Counting for posible duplicates and erasing them

duplicated_rows = data_series_topics.duplicated().sum()   

if (duplicated_rows > 0):

    data_series_topics = data_series_topics.drop_duplicates().reset_index(drop=True)

    print("Número de filas duplicadas eliminadas:", duplicated_rows)

else:

    print("No se han encontrado filas duplicadas")



## Joining the data of the series with the comments by the url name

data_total = data_series_topics.loc[:,["urls","Topics"]].merge(data, how='left', on="urls")
## Saving the data sets

path = "../input/"



print("Tamaño del set de series con topics:", data_series_topics.shape)

print("Tamaño del set total:", data_total.shape)



data_series_topics.to_csv(path + "data_series_topics.csv", header=True, sep=";", index=False)

data_total.to_csv(path + "data_total.csv", header=True, sep="|", index=False)



print("\nSets de datos guardados")
## Loading the data sets

path = "../input/"



data_series_topics = pd.read_csv(path + "data_series_topics.csv", header=0, sep=";", dtype={"years_begins":str})

data_total = pd.read_csv(path + "data_total.csv", header=0, sep="|", dtype={"user_ratings": float})



print("Sets de datos cargados")
## Creating a function to obtain the interaction item-user matrices by topics

def get_interaction_matrix(data, topic):

    if topic=="all":

        interaction_matrix = data.pivot_table(index=["users_names"], columns=["names"], values="users_ratings")

    else:

        interaction_matrix = data[data["Topics"]==topic].pivot_table(index=["users_names"], columns=["names"], values="users_ratings")

    

    return interaction_matrix
## Obtaining the interaction matrix without filtering the topic

interaction_matrix = get_interaction_matrix(data_total, "all")

print("Tamaño de la matriz de interacciones:", interaction_matrix.shape)



## Calculating how sparse the interaction matrix is

suma = []

[suma.append(i) for i in (interaction_matrix > 0).sum()]

sparsicity = (sum(suma)/interaction_matrix.size)*100



print("Sparsicity de la matriz de interacciones: {:.5}%".format(sparsicity))
## ## Creating a function to obtain the similarity item-item matrices by topics

def get_similarity_matrix(interaction_matrix):

    

    ## Generating a matrix of zeros

    cos_distance = np.zeros((interaction_matrix.shape[1], interaction_matrix.shape[1]))



    ## Itering over the interaction matrix to obtaing the cosine distance dismissing NaN values

    for i in range(0, interaction_matrix.shape[1]):

        u = interaction_matrix[interaction_matrix.columns[i]]

        for j in range(0, interaction_matrix.shape[1]):

            v = interaction_matrix[interaction_matrix.columns[j]]

            ind = []

            for k in range(0, len(u)):

                if (np.isnan(u[k]) or np.isnan(v[k])) == False:

                    ind.append(k)

                else:

                    continue

            cos_distance[i,j] = 1-distance.cosine(u[ind], v[ind]).round(4)



    similarity_matrix = pd.DataFrame(cos_distance, columns=interaction_matrix.columns, index= interaction_matrix.columns)

    similarity_matrix = similarity_matrix.replace(np.NaN, "0").astype("float64")

    

    return similarity_matrix
## Obtaining the similarity matrix without filtering by topics

similarity_matrix = get_similarity_matrix(interaction_matrix)

print("Tamaño de la matriz de similitudes:", similarity_matrix.shape)



## Calculating how sparse the similarity matrix is

suma = []

[suma.append(i) for i in (similarity_matrix > 0).sum()]

sparsicity = (sum(suma)/similarity_matrix.size)*100



print("Sparsicity de la matriz de similitudes: {:.5}%".format(sparsicity))
## Obtaining a list with the unique topics

topic_list = data_total.Topics.unique().astype("int64").tolist()



## Appliying the preview functions we can obtaing the simlarity and interaction matrices per topic

for i in topic_list:

    globals()['interaction_matrix_'+ str(i)] = get_interaction_matrix(data_total, i)

    globals()['similarity_matrix_'+ str(i)] = get_similarity_matrix(eval('interaction_matrix_%d'% (i)))
## Generating a graph visualization with the differents recommendations

plt.figure(figsize=(15,20))



## Indicating the path to save each graph

path = "../input/"



n=0

for i in topic_list:

    n+=1

    ## Generating a graph using the dataframe of similarities

    G = nx.from_pandas_adjacency(eval('similarity_matrix_%d'% (i)))



    ## Selecting a design to visualize the graph

    pos = nx.circular_layout(G)



    ## Creating a dictionary with the names of each serie to use labels on the edges of each node

    labels = eval('similarity_matrix_%d'% (i)).columns.values

    G = nx.relabel_nodes(G, dict(zip(range(len(labels)), labels)))



    ## Applying the maximum spanning tree algorithm

    G = nx.maximum_spanning_tree(G, algorithm='kruskal', weight="weight", ignore_nan=False)



    ## Obtaining the similarities of each edge

    edge_labels = nx.get_edge_attributes(G, "weight")



    values = []

    [values.append(round(j,4)) for j in list(edge_labels.values())]

    

    edge_labels = dict(zip(edge_labels.keys(), values))



    ## Showing the final graph    

    plt.subplot(int(np.ceil(len(topic_list)/2)*100+20+n))

    nx.draw(G,

            pos,

            with_labels=True,

            font_size=15,

            font_weight='bold',

            node_color='skyblue',

            node_size=800,

            edge_color='grey',

            linewidths=2)



    nx.draw_networkx_edge_labels(G,

                                 pos,

                                 edge_labels=edge_labels,

                                 font_color='darkblue',

                                 font_size=13,

                                 font_weight='bold')

    

    ## Saving the obtaing graph

    nx.write_edgelist(G, path=path+"grafo_topic_"+str(i), delimiter=";")
## Generating a function to obtain the final recommendation that the user will see

def Recomendation(serie):

    

    data_series_topics = pd.read_csv(path + "data_series_topics.csv", header=0, sep=";", dtype={"years_begins":str})



    topic = int(data_series_topics[data_series_topics.names==serie].Topics)



    G = nx.read_edgelist(path=path+"grafo_topic_"+str(topic), delimiter=";")



    EG = nx.ego_graph(G, n=serie, radius=2, center=True, undirected=True, distance="weight")

    ego_node=nx.ego_graph(G, n=serie, radius=0)



    fig, ax = plt.subplots(figsize=(15,10))

    pos = nx.spring_layout(EG)



    ax = nx.draw(EG,

                 pos,

                 with_labels=True,

                 font_size=15,

                 font_weight='bold',

                 node_color='lime',

                 node_size=800,

                 edge_color='grey',

                 linewidths=2)



    ax = nx.draw_networkx_nodes(ego_node, pos=pos, radius=0, node_size=900, node_color='red')

    

    plt.show()
## Using the function to show a example

Recomendation("perdidos")