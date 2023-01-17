#Standard data analysis tools: numpy and pandas

import numpy as np 

import pandas as pd 



#nltk, re and string for text pre-processing

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

import re

import string



#Keyword is generated using TF-IDF provided by sklearn

from sklearn.feature_extraction.text import TfidfVectorizer



#Models loading and unloading using pickle

import pickle



#Majority of the task is done using gensim

from gensim.corpora.dictionary import Dictionary

from gensim.parsing.preprocessing import remove_stopwords

from gensim.models.phrases import Phrases, Phraser

from gensim.models import Word2Vec



#Speed up modelling time using multiprocessing

import multiprocessing



#sklearn tools for Kmeans clustering, model evaluation and visualisation of clusters 

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import CountVectorizer



#For visualisation, we use matplotlib for plotting and wordcloud for visualisation of keywords

from matplotlib import pyplot as plt

from wordcloud import WordCloud



#Use Mode for finding most frequent cluster

from statistics import mode



#Manually enter questions into a list

#The questions are found in https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=561

question_list = ["What do we know about vaccines and therapeutics?", \

                 "What has been published concerning research and development and evaluation efforts of vaccines and therapeutics?" \

                 "Effectiveness of drugs being developed and tried to treat COVID-19 patients.", \

                 "Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocycline that may exert effects on viral replication.", \

                 "Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.", \

                 "Exploration of use of best animal models and their predictive value for a human vaccine.",\

                 "Capabilities to discover a therapeutic for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.",\

                 "Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.", \

                 "Efforts targeted at a universal coronavirus vaccine.", \

                 "Efforts to develop animal models and standardize challenge studies", \

                 "Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers", \

                 "Approaches to evaluate risk for enhanced disease after vaccination", \

                 "Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models in conjunction with therapeutics"]



#Put list into a dataframe for easier viewing and manipulation

df_task = pd.DataFrame(question_list, columns = ['tasks'])

df_task.head()

#Define function to clean text

def text_cleaner(text):

    #Convert to lower case

    text = str(text).lower()

    #Remove punctuations

    text = re.sub('[%s]' % re.escape(string.punctuation), " ", text)

    return text



df_task['tasks'] = df_task.tasks.apply(lambda x:text_cleaner(x))

df_task.head()                         
#Create TFIDF model note that min_df was tuned to give optimal output

vectorizer = TfidfVectorizer(min_df=0.08, tokenizer = lambda x: x.split(), sublinear_tf=False, stop_words = "english")

tasks_keywords = vectorizer.fit_transform(df_task.tasks)



#Print keywords in the order of importantce

print(sorted([(k,v) for k, v in vectorizer.vocabulary_.items()], key = lambda x: x[1], reverse = True))

#Grab all keywords in the TF-IDF vectorizer

new_dict = vectorizer.vocabulary_

#manually remove useless words and put into a new keyword_list

stop_words = ["use", "tried", "studies", "know", "need", "concerning", "alongside"]

for word in stop_words:

    new_dict.pop(word, None)

keyword_list = list(new_dict.keys())



#Do the same processing as in the previous workbook that was used to form the topic topic titles

#This include the replacement of various keywords, removal of numbers and lemmatization

keyword_list = [x.replace("severe acute respiratory syndrome", "sars") for x in keyword_list]

keyword_list = [re.sub('viral|viruses', 'virus', x) for x in keyword_list]

keyword_list = [re.sub('[0-9]', '', x) for x in keyword_list]

wordnet_lemmatizer = WordNetLemmatizer()

lemma = WordNetLemmatizer()

keyword_list = [lemma.lemmatize(x) for x in keyword_list] 



print(keyword_list)
#Load the LDA topic model

topic_model = pickle.load(open('/kaggle/input/cord-19-challenge-titles-topic-modelling/title_model', "rb"))

word_key = pickle.load(open('/kaggle/input/cord-19-challenge-titles-topic-modelling/dictionary' , "rb"))



#Load the dataframe with the title, abstract, and the corresponding title models

df_metadata = pd.read_csv('/kaggle/input/cord-19-challenge-titles-topic-modelling/topic_data.csv')



df_metadata.head()



df_metadata.Tokens[0]
#Format the keyword list into a form acceptable by the gensim lda model

corpus_dict = Dictionary([keyword_list])

corpus = [corpus_dict.doc2bow(words) for words in [keyword_list]]

#predict the topic probabilities using the model

vector = topic_model[corpus]

topic_list = []

for topic in vector:

      print(topic)

     

topic_list = [tup[0] for tup in vector[0]]
for topic in topic_list:

      print(topic_model.show_topic(topic))
word2id = word_key.token2id 

new_topic_list = []

#Initial test shows that two important keywords, naproxen and minocycline are not in the topic keywords

#Since these are antiinflammatory and antibiotics respectively, I have decided to manually add these keywords into the keyword_list

keyword_list.append('antiinflammatory')

keyword_list.append('antibiotic')

for word in keyword_list:

    try: 

        word_id = word2id[word]

        topics = topic_model.get_term_topics(word_id, minimum_probability=None)

        for topic in topics:

            new_topic_list.append(topic[0])

    except:

        print(word + " not in topic words")



new_topic_list = list(set(new_topic_list))



#Seperate out the three topics into seperate columns for easier processing

df_metadata['topic1'] = df_metadata['topics'].apply(lambda x:int(x.split(",")[0][1:]))

df_metadata['topic2'] = df_metadata['topics'].apply(lambda x:int(x.split(",")[1]))

df_metadata['topic3'] = df_metadata['topics'].apply(lambda x:int(x.split(",")[2][:-1]))

#with the relatively large topic list, I will increase the requirement that all three top topics needs to be in the topic list for an article 

#to be selected

df_metadata['Select'] = df_metadata['topic1'].isin(new_topic_list) & df_metadata['topic2'].isin(new_topic_list) & df_metadata['topic3'].isin(new_topic_list) 

df_selected = df_metadata[df_metadata['Select'] == True]



print(df_selected.shape)



df_selected.head()
#Function to clean up abstract

def abstract_cleaner(text):

    #standard preprocessing - lower case and remove punctuation

    text = str(text).lower()

    text = re.sub('[%s]' % re.escape(string.punctuation), "", text)

    #remove other punctuation formats that appears in the abstract

    text = re.sub("’", "", text)

    text = re.sub('“', "", text)

    text = re.sub('”', "", text)

    #remove the word abstract and other stopwords

    text = re.sub("abstract", "", text)

    text = remove_stopwords(text)

    #lemmatize and join back into a sentence

    text = " ".join([lemma.lemmatize(x) for x in word_tokenize(text)])

    

    return text



#Clean abstract

df_selected['abstract'] = df_selected['abstract'].apply(lambda x: abstract_cleaner(x))

df_selected.head()
#Check for bi-grams - first split sentence into tokens

words = [abstract.split() for abstract in df_selected['abstract']]

#Check for phrases, with a phrase needing to appear over 30 times to be counted as a phrase

phrases = Phrases(words, min_count=30, progress_per=10000)

#Form the bigram model

bigram = Phraser(phrases)

#Tokenise the sentences, using both words and bigrams. Tokenised_sentence is the word tokens that we will use to form word vectors

tokenised_sentences = bigram[words]

print(list(tokenised_sentences[0:5]))



#Make use of multiprocessing

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

#Create a word vector model. The number of dimensions chosen for word vector is 300

w2v_model = Word2Vec(window=2,

                     size=300,

                     sample=6e-5, 

                     alpha=0.03, 

                     min_alpha=0.0007, 

                     negative=20,

                     workers=cores-1)

#Build vocab for the model

w2v_model.build_vocab(tokenised_sentences)

#Train model using the tokenised sentences

w2v_model.train(tokenised_sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)



print("Model trained")

print("Top words that are most similar to COVID19:", "\n", w2v_model.wv.most_similar(positive = ["covid19"]))

print("\n")

print("Oseltamivir (Tamiflu) to influenza as what to COVID19:\n", 

      w2v_model.wv.most_similar(positive=["oseltamivir", "influenza"], negative=["covid19"], topn=20))

#Turn abstract into a single vector by averaging word vectors of all words in the abstract

def abstract2vec(abstract):

    vector_list = []

    for word in abstract:

        #seperate out cases where the word is in the word vector space, and words that are not

        if word in w2v_model.wv.vocab:

            vector_list.append(w2v_model[word])

        else:

            vector_list.append(np.zeros(300))

    #In case there are empty abstracts, to avoid error

    if (len(vector_list) == 0):

        return np.zeros(300)

    else:

        return sum(vector_list)/len(vector_list)



#Store tokens into dataframe and turn it into vectors

df_selected['sentences'] = tokenised_sentences

df_selected['avg_vector'] = df_selected['sentences'].apply(lambda x: abstract2vec(x))

df_selected.head()
# Turn data in an array as input to sklearn packages

X = np.array(df_selected.avg_vector.to_list())



#Perform standard scaling and kmeans strongly affected by scales

sc = StandardScaler()

X_scaled = sc.fit_transform(X)



#Form kmeans model with cluster size from 2 - 100, and record the inertia, which is a measure of the average distance of each point 

#in the cluster to the cluster centroid 

sum_square = []

for i in range(2,100,5):

    km_model = KMeans(init='k-means++', n_clusters=i, n_init=10)

    cluster = km_model.fit_predict(X_scaled)

    sum_square.append(km_model.inertia_)





x = range(2,100,5)

plt.figure(figsize=(20,10))

plt.plot(x,sum_square)

plt.scatter(x,sum_square)

plt.title('Sum of square as a function of cluster size')

plt.xlabel('Number of clusters')

plt.ylabel('Sum of square distance from centroid')

plt.show()





#Sweep from 10 to 30 (range around the elbow) and look for the record the silhouette score

silhouette = []

model_list = []

cluster_list = []

for i in range(10,30,1):

    km_model = KMeans(init='k-means++', n_clusters=i, n_init=10, random_state = 1075)

    cluster = km_model.fit_predict(X_scaled)

    model_list.append(km_model)

    cluster_list.append(cluster)

    silhouette.append(silhouette_score(X_scaled, cluster))





#Plot to observe the maximum silhouette score across this range

x = range(10,30,1)

plt.figure(figsize=(20,10))

plt.plot(x,silhouette)

plt.scatter(x,silhouette)

plt.title('Silhouette score as a function of number of clusters')

plt.xlabel('Number of clusters')

plt.ylabel('Silhouette Score')

plt.show()

#store in dataframe

optimal_model = model_list[np.argmax(silhouette)]

df_selected['cluster'] = cluster_list[np.argmax(silhouette)]

df_selected.head()
#Create Principal components

pca = PCA(n_components=50)

X_reduced = pca.fit_transform(X)



#Create t-SNE components and stored in dataframe

X_embedded = TSNE(n_components=2, perplexity = 40).fit_transform(X_reduced)

df_selected['TSNE1'] = X_embedded[:,0]

df_selected['TSNE2'] = X_embedded[:,1]



#plot cluster label against TSNE1 and TSNE2 using different colour for each cluster

color = ['b','g','r','c','m','y','yellow','orange','pink','purple', 'deepskyblue', 'lime', 'aqua', 'grey', 'gold', 'yellowgreen', 'black']

plt.figure(figsize=(20,10))

plt.title("Clusters of abstract visualised with t-SNE")

plt.xlabel("PC1")

plt.ylabel("PC2")

for i in range(17):

    plt.scatter(df_selected[df_selected['cluster'] == i].TSNE1, df_selected[df_selected['cluster'] == i].TSNE2, color = color[i])

plt.show()
#Take the questions from the question list and clean using the same function

q_cleaned = [abstract_cleaner(x) for x in question_list]

#Create tokens from the bigram phraser

q_words = [q.split() for q in q_cleaned]

q_tokens = bigram[q_words]

#Turn tokens into a single summary word vector

q_vectors = [abstract2vec(x) for x in q_tokens]

#Predict cluster based on the summary word_vector

question_cluster = optimal_model.predict(q_vectors)

print(question_cluster)
#Make tokens back into a string

df_selected['sentence_str'] = df_selected['sentences'].apply(lambda x: " ".join(x))

#Perform Count Vectorize to obtain words that appeared most frequently in these abstracts

main_cluster = mode(question_cluster)

cv = CountVectorizer()

text_cv = cv.fit_transform(df_selected[df_selected['cluster'] == main_cluster].sentence_str)

dtm = pd.DataFrame(text_cv.toarray(), columns = cv.get_feature_names())

top_words = dtm.sum(axis = 0).sort_values(ascending = False)

topword_string = ",".join([x for x in top_words[0:40].index])

print("Main cluster: Top 50 words - " + topword_string + "\n")

#Create word cloud using the topwords

wordcloud1 = WordCloud(width = 800, height = 600, 

                background_color ='white', 

                min_font_size = 10).generate(topword_string)



  

# plot the WordCloud image        





plt.figure(figsize=(10,6))

plt.imshow(wordcloud1) 

plt.title("Word Cloud for Main Cluster")

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 


df_cluster = df_selected[df_selected['cluster'] == main_cluster]

print("Number of articles in main cluster: ", df_cluster.shape[0])

#Have a look at first 10 articles from cluster 5:

print(list(df_cluster['title'][0:10]))
#Save data for later use

df_cluster.to_csv("Cluster.csv", index = False, header=True)
