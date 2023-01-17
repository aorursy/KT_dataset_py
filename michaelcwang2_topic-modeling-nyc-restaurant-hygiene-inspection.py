#import pandas
import pandas as pd
pd.__version__
metadata=pd.read_csv('../input/DOHMH_New_York_City_Restaurant_Inspection_Results0620.csv', engine='python')
metadata.head(3)
metadata['VIOLATION DESCRIPTION'].head(5)
#Replace NaN with an empty string
metadata['VIOLATION DESCRIPTION'] = metadata['VIOLATION DESCRIPTION'].fillna('')
metadata['VIOLATION DESCRIPTION'].head(10)
#Tokenization and clean up by gensim's simple preprocess
import gensim
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(metadata['VIOLATION DESCRIPTION']))
print(data_words)
print(data_words[:1])
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# Run in terminal: python3 -m spacy download en
import re, nltk, spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
# Do lemmatization keeping only Noun, Adj, Verb, Adverb
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[:2])
#scikit_learn.__version__
#import TfIdfVectorizer and CountVectorizer from Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# NMF is able to use tf-idf
tfidf_vectorizer=TfidfVectorizer(stop_words='english')
tfidf=tfidf_vectorizer.fit_transform(data_lemmatized)
tfidf.shape
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer=CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum read occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{2,}',  # num chars > 2
                            )
tf=tf_vectorizer.fit_transform(data_lemmatized)
tf.shape
tf_feature_names = tf_vectorizer.get_feature_names()
# Materialize the sparse data
data_dense = tf.todense()
# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")
from sklearn.decomposition import NMF, LatentDirichletAllocation
no_topics = 5
# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
# Run LDA
# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=no_topics,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='batch',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(tf)
print(lda_model)  # Model attributes
# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(tf))
# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(tf))
# Define Search Param
search_params = {'n_components': [5, 10, 15, 20], 'learning_decay': [.5, .7, .9]}
# Init the Model
lda = LatentDirichletAllocation()
# Init Grid Search Class
from sklearn.model_selection import GridSearchCV
model = GridSearchCV(lda, param_grid=search_params)
# Do the Grid Search
model.fit(tf)
# Best Model
best_lda_model = model.best_estimator_
# Model Parameters
print("Best Model's Params: ", model.best_params_)
# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)
# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(tf))
# Get Log Likelyhoods from Grid Search Output
gscore=model.fit(tf).cv_results_
type(gscore)
print(gscore)
#n_components = [5, 10, 15, 20]
#log_likelyhoods_5 = [round(gscore.mean_test_score) for mean_test_score in gscore if gscore.params['learning_decay']==0.5]
#log_likelyhoods_7 = [round(model.scorer.mean_validation_score) for model.scorer in model.cv_results_ if model.scorer.parameters['learning_decay']==0.7]
#log_likelyhoods_9 = [round(model.scorer.mean_validation_score) for model.scorer in model.cv_results_ if model.scorer.parameters['learning_decay']==0.9]
#import matplotlib as plt
#%matplotlib inline
# Show graph
#plt.figure(figsize=(12, 8))
#plt.plot(n_components, log_likelyhoods_5, label='0.5')
#plt.plot(n_components, log_likelyhoods_7, label='0.7')
#plt.plot(n_components, log_likelyhoods_9, label='0.9')
#plt.title("Choosing Optimal LDA Model")
#plt.xlabel("Num Topics")
#plt.ylabel("Log Likelyhood Scores")
#plt.legend(title='Learning decay', loc='best')
#plt.show()
# Create Document - Topic Matrix
lda_output = best_lda_model.transform(tf)
# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
# index names
docnames = ["Doc" + str(i) for i in range(len(data_lemmatized))]
# Make the pandas dataframe
import numpy as np
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic
# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)
def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)
# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
df_document_topics
df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
#panel = pyLDAvis.sklearn.prepare(best_lda_model, tf, tf_vectorizer) #no other mds function like tsne used.
# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(best_lda_model.components_/best_lda_model.components_.sum(axis=1)[:,np.newaxis])
# Assign Column and Index
df_topic_keywords.columns = tf_vectorizer.get_feature_names()
df_topic_keywords.index = topicnames
# View
df_topic_keywords.head(15)
# Show top n keywords for each topic
def show_lda_topics(lda_model=lda_model, n_words=20):
    keywords = np.array(df_topic_keywords.columns)
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
topic_keywords = show_lda_topics(lda_model=best_lda_model, n_words=15)
# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords
#lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
# Topic-Keyword matrix
#tf_topic_keywords=pd.DataFrame(lda.components_/lda.components_.sum(axis=1)[:,np.newaxis])
no_top_words = 8
# Assign Columns and Index
#tf_topic_keywords.columns=tf_feature_names
#tf_topic_keywords.index=np.arange(0,no_topics)
#print(tf_topic_keywords.head())
# display topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
display_topics(nmf, tfidf_feature_names, no_top_words)
# display lda through weighting
# def display_topics(feature_names, no_of_words):
   # for topic_idx, topic in enumerate(tf_topic_keywords):
   #     print ("Topic %d:" % (topic_idx))
   #     print (" ".join([feature_names[i]
   #                     for i in topic.argsort()[:-no_of_words - 1:-1]]))
# type(tf_feature_names)
# tf_feature_array=np.asarray(tf_feature_names)
#display_topics(lda, tf_feature_names, no_top_words)
#doc_topic_dist = lda.transform(tf)
#print(doc_topic_dist)
# lda_perplexity=LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).perplexity(tf)
# lda_score=LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).score(tf)
# print("and lda score= "lda_score)
# Importing Gensim
#import matplotlib as plt
#%matplotlib inline