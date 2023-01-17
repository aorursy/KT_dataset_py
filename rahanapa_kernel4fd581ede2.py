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
import nltk

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

import spacy

from tqdm import tqdm



import random

from spacy.util import compounding

from spacy.util import minibatch

import os



df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

df_test=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

df_submission=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

df.head()
# Checking Null Values

df.isnull().sum()
#drop null values

dfnew=df.dropna()

dfnew
dfnew.describe()
dfnew['cleaned_tweet']=dfnew['text'].replace(r'\'|\"|\,|\.|\?|\+|\-|\/|\=|\(|\)|\n|"','',regex=True)

dfnew['cleaned_tweet']=dfnew['cleaned_tweet'].replace(" "," ")





words_remove = ["ax","i","you","edu","s","t","m","subject","can","lines","re","what", "there","all","we",

                "one","the","a","an","of","or","in","for","by","on","but","is","in","a","not","with","as",

                "was","if","they","are","this","and","it","have","has","from","at","my","be","by","not","that","to",

                "from","com","org","like","likes","so","said","from","what","told","over","more","other",

                "have","last","with","this","that","such","when","been","says","will","also","where","why",

                "would","today", "in", "on", "you", "r", "d", "u", "hw","wat", "oly", "s", "b", "ht", 

                "rt", "p","the","th", "n", "was"]
dfnew
def cleantext(dfnew,words_to_remove=words_remove):

    # remove emoticons form the tweets

    dfnew['cleaned_tweet'] = dfnew['cleaned_tweet'].replace(r'<ed>','', regex = True)

    dfnew['cleaned_tweet'] = dfnew['cleaned_tweet'].replace(r'\B<U+.*>|<U+.*>\B|<U+.*>','', regex = True)

    

    # convert tweets to lowercase

    dfnew['cleaned_tweet'] = dfnew['cleaned_tweet'].str.lower()

    

    #remove user mentions

    dfnew['cleaned_tweet'] = dfnew['cleaned_tweet'].replace(r'^(@\w+)',"", regex=True)

    

    #remove 'rt' in the beginning

    dfnew['cleaned_tweet'] = dfnew['cleaned_tweet'].replace(r'^(rt @)',"", regex=True)

    

    #remove_symbols

    dfnew['cleaned_tweet'] = dfnew['cleaned_tweet'].replace(r'[^a-zA-Z0-9]', " ", regex=True)



    #remove punctuations 

    dfnew['cleaned_tweet'] = dfnew['cleaned_tweet'].replace(r'[[]!"#$%\'()\*+,-./:;<=>?^_`{|}]+',"", regex = True)



    #remove_URL(x):

    dfnew['cleaned_tweet'] = dfnew['cleaned_tweet'].replace(r'https.*$', "", regex = True)



    #remove 'amp' in the text

    dfnew['cleaned_tweet'] = dfnew['cleaned_tweet'].replace(r'amp',"", regex = True)

    

    #remove words of length 1 or 2 

    dfnew['cleaned_tweet'] = dfnew['cleaned_tweet'].replace(r'\b[a-zA-Z]{1,2}\b','', regex=True)



    #remove extra spaces in the tweet

    dfnew['cleaned_tweet'] = dfnew['cleaned_tweet'].replace(r'^\s+|\s+$'," ", regex=True)

    

    #remove stopwords and words_to_remove

    stop_words = set(stopwords.words('english'))

    mystopwords = [stop_words, "via", words_to_remove]

    

    dfnew['fully_cleaned_tweet'] = dfnew['cleaned_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in mystopwords]))

    



    return dfnew

    

#get the processed tweets

dfclean = cleantext(dfnew)
dfclean.head()
dfclean['tokenized_tweet'] = dfclean['fully_cleaned_tweet'].apply(word_tokenize)

dfclean.head()
#if a word has a digit, remove that word

dfclean['tokenized_tweet'] = dfclean['tokenized_tweet'].apply(lambda x: [y for y in x if not any(c.isdigit() for c in y)])

dfclean.head()
dfclean['no_words_ST']=dfclean['selected_text'].apply(lambda x:len(str(x).split()))

dfclean['no_words_text']=dfclean['text'].apply(lambda x:len(str(x).split()))

dfclean['diff_words']=dfclean['no_words_text'] - dfclean['no_words_ST']

dfclean.head()
# Set values for various parameters

num_features = 100    # Word vector dimensionality                      

min_word_count = 1   # Minimum word count                        

num_threads = 4       # Number of threads to run in parallel

context = 10          # Context window size   
# Initialize and train the model (this will take some time)

from gensim.models import word2vec

print("Training model...")

model = word2vec.Word2Vec(dfclean['tokenized_tweet'], workers=num_threads, \

            size=num_features, min_count = min_word_count, \

            window = context)

# If you don't plan to train the model any further, calling 

# init_sims will make the model much more memory-efficient.

model.init_sims(replace=True)
# Find vector corresponding to each tweet

import numpy as np

vocab = list(model.wv.vocab)

def sentence_vector(sentence, model):

    nwords = 0

    featureV = np.zeros(100, dtype="float32")

    for word in sentence:

        if word not in vocab:

            continue

        featureV = np.add(featureV, model[word])

        nwords = nwords + 1

    if nwords > 0: 

        featureV = np.divide(featureV, nwords)

    return featureV



tweet_vector = dfclean['tokenized_tweet'].apply(lambda x: sentence_vector(x, model))  



tweet_vector = tweet_vector.apply(pd.Series)
#Tweet vector should vary from 0 to 1 (normalize the vector)

for x in range(len(tweet_vector)):

    x_min = tweet_vector.iloc[x].min()

    x_max = tweet_vector.iloc[x].max()

    X  = tweet_vector.iloc[x]

    i = 0

    if (x_max - x_min) == 0:

        for y in X:

            tweet_vector.iloc[x][i] = (1/len(tweet_vector.iloc[x]))

            i = i + 1

    else:

        for y in X:

            tweet_vector.iloc[x][i] = ((y - x_min)/(x_max - x_min))

            i = i + 1
tweet_vector
#Cluster the narratives = opinions + expressions

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, silhouette_samples



range_n_clusters = [4, 5, 6, 7, 8, 9, 10, 11]

X = tweet_vector

n_best_clusters = 0

silhouette_best = 0

for n_clusters in range_n_clusters:

    

    # Initialize the clusterer with n_clusters value and a random generator

    # seed of 10 for reproducibility.

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)

    cluster_labels = clusterer.fit_predict(X)



    # The silhouette_score gives the average value for all the samples.

    # This gives a perspective into the density and separation of the formed

    # clusters

    silhouette_avg = silhouette_score(X, cluster_labels)

                                      #, sample_size = 5000)

    print("For clusters =", n_clusters,

          "The average silhouette_score is :", silhouette_avg)

    

    if silhouette_avg > silhouette_best:

        silhouette_best = silhouette_avg

        n_best_clusters = n_clusters
n_best_clusters
clusterS = KMeans(n_clusters= n_best_clusters , random_state=10)

cluster_labels = clusterS.fit_predict(X)
#Array of tweets, the corresponding cluster number, sentiment

finaldf = pd.DataFrame({'cl_num': cluster_labels,'fully_cleaned_tweet': dfclean['fully_cleaned_tweet'], 'cleaned_tweet': dfclean['cleaned_tweet'], 'tweet': dfclean['text'],'sentiment': dfclean['sentiment']})

finaldf = finaldf.sort_values(by=['cl_num'])

finaldf.head()
dfclean['cl_num'] = cluster_labels
dfOrdered = pd.DataFrame(dfclean)



#Compute how many times a tweet has been 'retweeted' - that is, how many rows in dfOrdered are identical

dfOrdered['tokenized_tweet'] = dfOrdered['tokenized_tweet'].apply(tuple)

dfUnique = dfOrdered.groupby(['text', 'selected_text','cleaned_tweet', 'fully_cleaned_tweet', 'sentiment','tokenized_tweet', 'cl_num']).size().reset_index(name="freq")

dfUnique = dfUnique.sort_values(by=['cl_num'])
dfUnique['tokenized_tweet'] = dfUnique['tokenized_tweet'].apply(list)

dfOrdered['tokenized_tweet'] = dfOrdered['tokenized_tweet'].apply(list)
#Discard the clusters with poor silhoutte score

# Compute the silhouette scores for each sample

sample_silhouette_values = silhouette_samples(X, cluster_labels)



poor_cluster_indices = []

avg_cluster_sil_score = []



for i in range(n_best_clusters):

# Aggregate the silhouette scores for samples belonging to

# cluster i, and sort them

        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        avgscore = (np.mean(ith_cluster_silhouette_values))   #average silhouette score for each cluster

        avg_cluster_sil_score = np.append(avg_cluster_sil_score, avgscore)

        print('Cluster',i, ':', avgscore)

        if avgscore < 0.2:

            poor_cluster_indices = np.append(poor_cluster_indices, i)

            

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
#remove those rows where cluster value match poor_cluster_indices 

avg_cluster_sil_score_final = []

cluster_name = np.unique(dfOrdered['cl_num'])



if (len(poor_cluster_indices)!=0):

    n_final_clusters = n_best_clusters - len(poor_cluster_indices)

    for i in poor_cluster_indices:

        dfUnique = dfUnique[dfUnique['cl_num'] != i]

    for j in cluster_name:

        if j not in poor_cluster_indices:    

            avg_cluster_sil_score_final = np.append(avg_cluster_sil_score_final, avg_cluster_sil_score[j])

            

    cluster_name = np.unique(dfUnique['cl_num'])
dfUnique['cl_num'] = abs(dfUnique['cl_num'])

dfUnique = dfUnique.sort_values(by=['cl_num'])
dfUnique.head()
def save_model(output_dir, nlp, new_model):

    ''' This Function Saves model to 

    given output directory'''

    

    output_dir = f'./working/{output_dir}'

    if output_dir is not None:        

        if not os.path.exists(output_dir):

            os.makedirs(output_dir)

        nlp.meta["name"] = new_model

        nlp.to_disk(output_dir)

        print("Saved model :", output_dir)
#Returns Model output path

def model_out_path(sentiment):

    out_path = None

    if sentiment == 'positive':

        out_path = 'models/model_positive'

    elif sentiment == 'negative':

        out_path = 'models/model_negative'

    return out_path
#Load the model, set up the pipeline and train the entity recognizer.

    

def train(train_data, output_dir, n_iter=20, model=None):

  

    

    if model is not None:

        nlp = spacy.load(output_dir)  # load existing spaCy model

        print("Loaded model '%s'" % model)

    else:

        nlp = spacy.blank("en")  # create blank Language class

        print("Created blank 'Named-Entity' model")

    

    # create the built-in pipeline components and add them to the pipeline

    

    if "ner" not in nlp.pipe_names:

        ner = nlp.create_pipe("ner")

        nlp.add_pipe(ner, last=True)

    # otherwise, get it so we can add labels

    else:

        ner = nlp.get_pipe("ner")

    

    # add labels

    for _, annotations in train_data:

        for entity in annotations.get("entities"):

            ner.add_label(entity[2])



    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):  

       

        if model is None:

            nlp.begin_training()

        else:

            nlp.resume_training()





        for itn in tqdm(range(n_iter)):

            random.shuffle(train_data)

            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.0001))    

            losses = {}

            for batch in batches:

                texts, annotations = zip(*batch)

                nlp.update(texts,annotations,drop=0.5,losses=losses)

            print("Losses", losses)

    save_model(output_dir, nlp, 'str-ner')
# Training models for positive and negative tweets

sentiment = 'positive'



#train_data = train_data(sentiment)

train_data = []

for index, row in dfUnique.iterrows():

    if row.sentiment == sentiment:

        selected_text = row.selected_text

        text = row.text

        start = text.find(selected_text)

        end = start + len(selected_text)

        train_data.append((text, {"entities": [[start, end, 'selected_text']]}))

model_path = model_out_path(sentiment)



train(train_data, model_path, n_iter=5, model=None)
sentiment = 'negative'



#train_data = train_data(sentiment)

train_data = []

for index, row in dfUnique.iterrows():

    if row.sentiment == sentiment:

        selected_text = row.selected_text

        text = row.text

        start = text.find(selected_text)

        end = start + len(selected_text)

        train_data.append((text, {"entities": [[start, end, 'selected_text']]}))

model_path = model_out_path(sentiment)



train(train_data, model_path, n_iter=5, model=None)
# predicting with trained model



def entity_predict(text, model):

    doc = model(text)

    ent_array = []

    for ent in doc.ents:

        start = text.find(ent.text)

        end = start + len(ent.text)

        new_int = [start, end, ent.label_]

        if new_int not in ent_array:

            ent_array.append([start, end, ent.label_])

    select_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text

    return select_text
select_texts = []

base_path = './working/models/'



if base_path is not None:

    print("Loading: ", base_path)

    model_positive = spacy.load(base_path + 'model_positive')

    model_negative = spacy.load(base_path + 'model_negative')

        

    for index, row in df_test.iterrows():

        text = row.text

        output_str = ""

        if row.sentiment == 'neutral' or len(text.split()) <= 2:

            select_texts.append(text)

        elif row.sentiment == 'positive':

            select_texts.append(entity_predict(text, model_positive))

        else:

            select_texts.append(entity_predict(text, model_negative))

        

df_test['selected_text'] = select_texts
df_submission['selected_text'] = df_test['selected_text']

df_submission.to_csv("./working/submission.csv", index=False)

display(df_submission.head(10))