import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from IPython.display import display, Markdown, Latex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import pickle
import joblib
from sklearn.externals import joblib

import os
import math

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from sklearn.cluster import KMeans
articles = pd.read_csv('../input/clean-and-steamed-corpus/articles_clean.csv' ,header='infer')
articles.head()
plt.figure(figsize=(10,8))
ax = sns.distplot(articles.length.apply(math.log))
ax.set(title='Histogram of the articles number of characters log transformed.',xlabel='Log of Article length', ylabel='Proportions')
plt.show()

print('There are in total {} articles containing a text body with their lengths normally distributed around the average length of: {} words.'.format(len(articles.length),round(np.mean(articles.length))))
refdoc = "Household Infrastructure Containment Ecological Historical Perception internalization financial support aide bailout policy policies politics social economic socioeconomic cultural technology Environment Confinement Stay-at-home Shelter-in-place Inequalities Iniquities Disparities Communication Mass communication Information Language barrier Propaganda Poverty Homelessness homeless Beliefs Religion religious Behaviors Education Educational achievement Access Health insurance Public transportation Food security insecurity stamp Housing instability Internet access Conspiracy theory Vulnerable populations Low-income Middle-income Social status Socioeconomic status Social aide Government aide Government subsidies Remote villages Churches Schools School closure Social distancing Physical distancing Utilities Abuse  Domectic violence Neighborhood Social safety Social determinants health Traditional medicine Traditions traditional Legal Law Prisons prisoner Underdeveloped countries Developed countries Health systems Race Ethnicity urban rural areas cities villages"
refdoc = refdoc.lower()
shortword = re.compile(r'\W*\b\w{1,5}\b')
subs=""

articles['article'] = articles['article'].map(lambda x: re.sub(shortword, subs, x))
articles['length'] = articles['article'].map(lambda x: len(x))
refdf = pd.Series({'article':refdoc,'length':len(refdoc),'lang':'en'})

refdf=pd.DataFrame(refdf)
articles = pd.concat([refdf.T, articles], ignore_index=True)
articles.head()
#### Tokenising | Loading for speed of processing

tokenized_articles = articles['article'].apply(lambda x: x.split())

#### Removing English stopwords
stopwords = text.ENGLISH_STOP_WORDS

tokenized_articles = tokenized_articles.apply(lambda x: [i for i in x if i not in stopwords])

#### Stemming will help us reduce considerably the dimension of our idf by removing words repetitions due to grammar variations such as plural, conjugated words and so on.

stemmer = SnowballStemmer("english")

## Reading from prestemmed dataset
stemmed_articles = tokenized_articles.apply(lambda x: [stemmer.stem(token) for token in x])
#stemmed_articles = pd.read_csv('../input/moredata/',header='infer')

#### Detokenization

#### detokenization
detokenized_articles = []
for i in range(len(stemmed_articles)):
    if i in stemmed_articles:
        t = ' '.join(stemmed_articles[i])
        detokenized_articles.append(t)
        

#### Loading the detokenized words for faster processing.
with open('tokenized_articles.data', 'wb') as filehandle:
    # store the data as binary data stream
    tokenized_articles=pickle.load(filehandle)


def preprocessor(tokens):
    r = re.sub( '(\d)+', '', tokens.lower() )
    return r

ngram_range=(1,3)

vectorizer = TfidfVectorizer(stop_words=stopwords,max_features=10000, max_df = 0.5, use_idf = True, ngram_range=ngram_range)
  
### Vectorizing abstract series
cv = CountVectorizer( preprocessor=preprocessor, stop_words = stopwords,lowercase = True, ngram_range = ngram_range,max_features = 100 )

### Creating the bag of words | Reading instead of running
bow_articles = vectorizer.fit_transform(detokenized_articles)

### Getting the list of features from the bag of words
terms = vectorizer.get_feature_names()
bow_articles = joblib.load('../input/nparrays/bow_articles.pkl')
print('We end-up with a bag of word in form of a matrix of {} unique words within {} documents in rows, ready to be clustered.'.format(bow_articles.shape[1], bow_articles.shape[0]))
##### Using K-means Algorythm, we predefine 10 as the number of clusters for grouping documents.
num_clusters = 10

km = KMeans(n_clusters=num_clusters, n_jobs=10)
km.fit(bow_articles)
## Loading pre-trained model
km = joblib.load('../input/models/km_compressed.pkl')

## Getting prediction
y_kmeans = km.predict(bow_articles)

### Getting the clusters
clusters = km.labels_.tolist()
clusters_articles = pd.Series(clusters).value_counts().sort_index()

clusters_info = pd.DataFrame({'ClusterId':list(clusters_articles.index),'Members':clusters_articles})

print('Discovered clusters Ids and membership counts:\n\n {}'.format(clusters_info))

plt.figure(figsize=(10,8))
ax = sns.barplot(x='ClusterId', y='Members', data=clusters_info)
#ax.set(xlim=(0,13000))
ax.set(xlabel='Clusters', ylabel='Membership')

plt.title('Histogram number of articles per cluster.', fontsize=20)

for index, row in clusters_info.iterrows():
    ax.text(row.ClusterId,row.Members, round(row.Members,0), color='black', ha="center")
    
plt.show()
### Plotting the cluster
centers = km.cluster_centers_

x=centers[:,0]
y=centers[:,1]

x=centers[:,0]
y=centers[:,1]

color=clusters_info['ClusterId']
size=clusters_info['Members']

plt.figure(figsize=(15, 10), dpi=80)

plt.scatter(np.log(x), np.log(y),  c=color, s=size, alpha=0.5, cmap = 'hsv')

plt.title('Clusters size and reference plot', fontsize=20)

# zip joins x and y coordinates in pairs
i=0
for a,b in zip(x,y):

    label = "Id {}: {}".format(clusters_info['ClusterId'][i],clusters_info['Members'][i])

    plt.annotate(label, # this is the text
                 (np.log(a),np.log(b)), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    i=i+1

plt.show()

### Getting clusters Ids associated to the documents Ids
df_clust = pd.DataFrame({'DocumentId':np.arange(0,len(y_kmeans),1),'ClusterId':y_kmeans})

print('Our bait document id is 0 hence will be displayed on the fisrt row of the below output with its cluster id')
df_clust.head()
indices = df_clust[df_clust['ClusterId']==0].index
cluster0_corpus = [stemmed_articles[i] for i in indices if i!=0]

cluster0_corpus = pd.Series(cluster0_corpus)

cluster0_corpus.index=indices[1:]

### Verification of original dataframe indices for tracking articles.
cluster0_corpus.index

#### detokenization
detokenized_corpus0 = []

for i in range(len(cluster0_corpus)):
    if i in cluster0_corpus:
        t = ' '.join(cluster0_corpus[i])
        detokenized_corpus0.append(t)
ngram_range=(2,2)

### Vectorizing abstract series
cv = CountVectorizer( preprocessor=preprocessor, stop_words = stopwords,lowercase = True, ngram_range = ngram_range, max_features = 500 )

### Creating the bag of words
bow_c0 = cv.fit_transform( detokenized_corpus0 ).todense()

### Getting Cluster0 list of features
features_c0 = cv.get_feature_names()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

########################################################################
### Take the TF_IDF and converts it into a WordCloud object to be plotted
def WordCloud_covid(tfidf): 
    x = tfidf
    x[ 'word' ] = x.index
    dt_c0 = pd.Series( x.frequency, index = x.word.values ).to_dict()

    ##stopwords = set(STOPWORDS)


    ## plotting the word cloud
    wordcloud_c0 = WordCloud( 
        stopwords = stopwords_med,
        background_color = 'white',
        width = 5000,
        height = 3000,
        random_state = 40 ).generate_from_frequencies( dt_c0 )
    return(wordcloud_c0)

########################################################################

def WordCloudplot(wordcloud, size=(18,10)):
    plt.figure(figsize=size,facecolor = 'white', edgecolor='blue')
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show();

### Functions for TFIDF and WordCloud

### This function generates the tfidf from the Bag of Words and the Features

def tfidf_covid19(bow_c0, features_c0):
    ### Creating the DF-ITF
    ## Building the documents term matrix in a dataframe
    df_dtm_c0 = pd.DataFrame( data = bow_c0, columns = features_c0 )

    ###################################################################
    ##
    ## Display the top words and their frequency.
    ## Find column sums and create a DataFrame.
    ##
    x = df_dtm_c0.sum( axis = 0 )
    df_tf_c0 = pd.DataFrame( { 'frequency':x } )
    ##
    ## Display the top five records.
    ##
    topn = df_tf_c0.nlargest( columns = 'frequency', n = 10 )

    # topn.style.set_caption( 'Top Ten Words' )
    ##
    ##
    ##
    ## Calculate the tf-idf weights.
    ##
    transformer = TfidfTransformer( norm = 'l2' )
    tfidf_mat_c0 = transformer.fit_transform( bow_c0 )
    data = np.round( tfidf_mat_c0.todense(), 3 )
    df_tfidf_c0 = pd.DataFrame( data = data, columns = features_c0 )
    ##
    ## Transforming the df_tfidf into a DataFrame
    ##
    x1 = df_tfidf_c0.sum( axis = 0 )
    df_tfidf_c0 = pd.DataFrame( { 'frequency':x1 } )
    
    return(df_tfidf_c0)


### Function for Exploratory Data Analysis on TF-IDF

def TfIdf_Stats(df_tfidf_c0):
    ### Descriptive statistics on TFIDF 
    
    global tfidf_mean,tfidf_median, tfidf_sd, conf_int95, conf_int64, iqr_low, iqr_high

    ### Printing the histogram of the word TFIDF
    display(Markdown('#### <font color=\'blue\'>Line plot of the TF-IDF of the corpus'))
    ## print('Line plot of the TF-IDF of the corpus \n\n')

    df_tfidf_c0.plot(rot=45, figsize=(12,6))
    # Add title and axis names
    plt.title('Term Frequency plot by n-grams', fontsize=14, color='blue')
    plt.xlabel('Expressions')
    plt.ylabel('TF-IDF')
    plt.show()

    ### Printing the histogram of the word TFIDF
    display(Markdown('#### <p><font color=\'blue\'>Histogram plot of the TF-IDF of n-grams in the corpus'))
    ## print('Histogram plot of the TF-IDF of n-grams in the corpus \n\n')

    df_tfidf_c0['frequency'].hist(bins=50, figsize=(15,5))
    # Add title and axis names
    plt.title('Histogram of Term Frequency', fontsize=20, color='blue')
    plt.xlabel('TF-IDF')
    plt.ylabel('Terms count')
    plt.show()

    ## investigating the Mean and confidence interval and median
    display(Markdown('#### <p><font color=\'blue\'>Investigating the Mean and confidence interval and median.'))
    tfidf_mean = np.log(df_tfidf_c0['frequency']).mean()
    ##
    tfidf_sd = np.log(df_tfidf_c0['frequency']).std()

    tfidf_median = np.log(df_tfidf_c0['frequency']).median()

    print('\n\n Words log(TFIDF) have a Mean of {:.2f}, Median of {:.2f} and Standard Deviation of {:.2f}:'.format(tfidf_mean, tfidf_median, tfidf_sd)+'\n\n')

    ## 95% confidence interval to the mean log(tfidf)
    conf_int95 = [round(tfidf_mean-2*tfidf_sd,2), round(tfidf_mean+2*tfidf_sd,2)]

    ## 64% confidence interval to the mean log(tfidf)
    conf_int64 = [round(tfidf_mean-tfidf_sd,2), round(tfidf_mean+tfidf_sd,2)]

    print('95% of the Word have their log(TFIDF) between [{}-{}]:'.format(conf_int95[0], conf_int95[1])+'\n\n')

    print('68% of the Word have their log(TFIDF) between [{}-{}]:'.format(conf_int64[0], conf_int64[1])+'\n\n')
    #
    ## investigating the Median and IQR (Robust Statistics)
    #
    display(Markdown('#### <p><font color=\'blue\'>Investigating the Median and IQR (Robust Statistics)'))

    iqr_low = np.log(df_tfidf_c0['frequency']).quantile(0.25)

    iqr_high = np.log(df_tfidf_c0['frequency']).quantile(0.75)

    print('IQR range of word log(TFIDF) is [{}-{}]:'.format(iqr_low, iqr_high)+'\n\n')

    ### Transforming and plotting the log transformed TFIDF
    display(Markdown('#### <p><font color=\'blue\'>Transforming and plotting the log transformed TFIDF'))
    
    np.log(df_tfidf_c0['frequency']).hist(bins=50, figsize=(15,5), alpha=0.7)
    
    # Add title and axis names
    plt.title('Log of Term Frequency plot by bigram', fontsize=14, color='blue')
    plt.axvline(x=conf_int95[0], color='r', linestyle='dashed', linewidth=1)
    plt.axvline(x=conf_int95[1], color='r', linestyle='dashed', linewidth=1)
    plt.axvline(x=conf_int64[0], color='g', linestyle='dashed', linewidth=1)
    plt.axvline(x=conf_int64[1], color='g', linestyle='dashed', linewidth=1)
    plt.xlabel('log(TF-IDF)')
    plt.ylabel('Terms count')
    plt.show()

    ### Plotting the different sections of the line plot of TFIDF
    display(Markdown('#### <p><font color=\'blue\'>Plotting the different sections of the line plot of TFIDF'))

    ### lower 25% of the TFIDF
    print('Plotting the lower 25% of TFIDF \n\n')
    df_tfidf_c0['frequency'][np.log(df_tfidf_c0['frequency'])<iqr_low].plot(rot=45, figsize=(15,10))
    plt.show()

    print('Plotting the TFIDF range between [25% 75%]\n\n')
    df_tfidf_c0['frequency'][(np.log(df_tfidf_c0['frequency'])<=iqr_low) | (np.log(df_tfidf_c0['frequency'])>=iqr_high)].plot(rot=45, figsize=(15,10))
    plt.show()

    print('Plotting the higher 75% of TFIDF \n\n')

    df_tfidf_c0['frequency'][np.log(df_tfidf_c0['frequency'])>iqr_high].plot(rot=45, figsize=(15,10));
    plt.show()
    
#####################################################
## Building the Bag of Words
#####################################################
def bow_gen(articles):
    
    ### Creating the bag of words
    bow_articles4 = vectorizer.fit_transform(articles)
    
    return(bow_articles4)

#####################################################
## Cleaning corpora of articles
#####################################################
def CleanText(articles):
    
    #### Tokenising
    
    tokenized_cluster4 = articles['article'].apply(lambda x: x.split())

    tokenized_cluster4 = tokenized_cluster4.apply(lambda x: [i for i in x if i not in stopwords])

    ### Stemming
    stemmed_cluster4 = tokenized_cluster4.apply(lambda x: [stemmer.stem(token) for token in x])


    ### Detokenizing
    #### detokenization
    detokenized_cluster4 = []
    for i in range(len(stemmed_cluster4)):
        if i in stemmed_cluster4:
            t = ' '.join(stemmed_cluster3[i])
            detokenized_cluster3.append(t)
    return(detokenized_cluster3)


#####################################################
## This function is used to plot a Kmean Cluster around centroids
## Plotting the cluster
## in this case model = km3 and prediction = y_means3
#####################################################
def ClusterPlot(model, prediction):
    
    clusters_v3 = model.labels_.tolist()

    clusters_v3_articles = pd.Series(clusters_v3).value_counts().sort_index()

    clusters_v3_articles

    clusters_v3_info = pd.DataFrame({'ClusterId':list(clusters_v3_articles.index),'Members':clusters_v3_articles})

    clusters_v3_info


    ### Plotting the cluster
    centers_v3 = model.cluster_centers_

    x3=centers_v3[:,0]
    y3=centers_v3[:,1]

    clusters_v3_info = pd.Series(prediction).value_counts().sort_index()
    clusters_v3_info=pd.DataFrame({'ClusterId':clusters_v3_info.index, 'Members':clusters_v3_info})
    clusters_v3_info


    ### Plotting the cluster
    centers_v3 = model.cluster_centers_

    plt.figure(figsize=(15, 10), dpi=80)

    plt.scatter(np.log(x3), np.log(y3),  c=color, s=s, alpha=0.5, cmap = 'hsv')

    plt.title('Clusters size and reference plot', fontsize=20)

    # zip joins x and y coordinates in pairs
    i=0
    for a3,b3 in zip(x3,y3):

        label = "Id {}: {}".format(clusters_v3_info['ClusterId'][i],clusters_v3_info['Members'][i])

        plt.annotate(label, # this is the text
                     (np.log(a3),np.log(b3)), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center
        i=i+1

    plt.show();
### Generating and sorting the TFIDF
df_tfidf_c0 = tfidf_covid19(bow_c0=bow_c0, features_c0=features_c0)

df_tfidf_c0 = df_tfidf_c0.sort_values('frequency')
###
df_tfidf_c0 = joblib.load('../input/data002/df_tfidf_c0.pkl')
morewords ='disease, diseases, disorder, symptom, symptoms, drug, drugs, problems, problem,prob, probs, med, meds,pill,  pills,  medicine,  medicines,  medication,  medications,  treatment,  treatments,  caps,  capsules,  capsule, tablet,  tablets,  tabs,  doctor,  dr,  dr.,  doc,  physician,  physicians,  test,  tests,  testing,  specialist, specialists, side-effect, side-effects, pharmaceutical, pharmaceuticals, pharma, diagnosis, diagnose, diagnosed, exam, challenge,  device,  condition,  conditions,  suffer,  suffering  ,suffered,  feel,  feeling,  prescription,  prescribe, prescribed, over-the-counter, ot'

print('Content of our reference document:\n\n {}'.format(morewords))

stemmer = SnowballStemmer("english")

morewords = morewords.replace(' ', '').split(',')

morewords = [stemmer.stem(token) for token in morewords]
morewords = set(morewords)

stopwords = text.ENGLISH_STOP_WORDS
stopwords_med = set(stopwords).union(morewords)
TfIdf_Stats(df_tfidf_c0)
#wordcloud2 = WordCloud_covid(tfidf=df_tfidf_c0[np.log(df_tfidf_c0['frequency'])>conf_int64[1]])
## Testing IQR approach
wordcloud2 = WordCloud_covid(tfidf=df_tfidf_c0[np.log(df_tfidf_c0['frequency'])>iqr_high])

display(Markdown('#### <p><font color=\'blue\'>Wordcloud of the upper 75% important words'))

WordCloudplot(wordcloud2)
## wordcloud3 = WordCloud_covid(tfidf=df_tfidf_c0[(np.log(df_tfidf_c0['frequency'])>=conf_int64[0])&(np.log(df_tfidf_c0['frequency'])<=conf_int64[1])])

wordcloud3 = WordCloud_covid(tfidf=df_tfidf_c0[(np.log(df_tfidf_c0['frequency'])>=iqr_low)&(np.log(df_tfidf_c0['frequency'])<=iqr_high)])

WordCloudplot(wordcloud3)
## wordcloud4 = WordCloud_covid(tfidf=df_tfidf_c0[(np.log(df_tfidf_c0['frequency'])<conf_int64[0])])
wordcloud4 = WordCloud_covid(tfidf=df_tfidf_c0[(np.log(df_tfidf_c0['frequency'])<iqr_low)])

## Generating the wordcloud
display(Markdown('#### <p><font color=\'blue\'>Wordcloud of the 25% less words in our list'))

WordCloudplot(wordcloud4)
original_text = pd.read_csv("original_text.csv")

articles_cluster0 = original_text.loc[indices,]
### Loading original data and stored intermediary results

original_text = joblib.load('../input/originaldata/original_raw_data.pkl')

articles_cluster0 = joblib.load('../input/data001/articles_cluster0.pkl')


### Creating Search pattern

pat10=r'social distancing'
pat11=r'effectiveness'

### Rewritting the columns name
original_text.columns=['article', 'length', 'article1']
articles_cluster0.columns=['article']
original_text.head()
orig_match = original_text['article'][original_text['article'].apply(lambda x: re.search(pat10,str(x))).notnull()]
orig_count= orig_match[orig_match.apply(lambda x: re.search(pat11,str(x))).notnull()].count()

print("Number of articles with the keyword: social distancing effectiveness in the original pool of articles was: {}".format(orig_count))
new_match = articles_cluster0['article'][articles_cluster0['article'].apply(lambda x: re.search(pat10,str(x))).notnull()]
social_dist = new_match[new_match.apply(lambda x: re.search(pat11,str(x))).notnull()]
new_count=new_match[new_match.apply(lambda x: re.search(pat11,str(x))).notnull()].count()

print("Number of articles with the keyword: social distancing effectiveness in sub-Cluster of our reference document is: {}".format(new_count))
original_filtered = original_text.loc[social_dist.index]['article1']
original_filtered
import json

from pandas.io.json import json_normalize  

### Detection of the articles language
!pip install langdetect
from langdetect import detect

### Summarizer package
!pip install bert-extractive-summarizer
# pip install summarizer
from summarizer import Summarizer
!pip install langdetect
model = Summarizer()
results = [model(x, max_length=160) for x in original_filtered[:5]]
[print('Article {} summary:\n\n {}'.format(results.index(i),i.capitalize())) for i in results]
