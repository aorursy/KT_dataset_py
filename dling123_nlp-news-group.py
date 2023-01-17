import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
import seaborn as sns
import re
from nltk.corpus import stopwords
import string
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics    
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import timeit
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from nltk.corpus import gutenberg, stopwords
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
dataset_full = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
df_full = pd.DataFrame()
df_full['text'] = dataset_full.data
df_full['source'] = dataset_full.target
label=[]
for i in df_full['source']:
    label.append(dataset_full.target_names[i])
df_full['label']=label

dataset_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
df_train = pd.DataFrame()
df_train['text'] = dataset_train.data
df_train['source'] = dataset_train.target
label=[]
for i in df_train['source']:
    label.append(dataset_train.target_names[i])
df_train['label']=label

dataset_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
df_test = pd.DataFrame()
df_test['text'] = dataset_test.data
df_test['source'] = dataset_test.target
label=[]
for i in df_test['source']:
    label.append(dataset_test.target_names[i])
df_test['label']=label

stopWords = set(stopwords.words('english'))

def textcleaner_stem(text):
    ''' Takes in raw unformatted text and strips punctuation, removes whitespace,
    strips numbers, tokenizes and stems.
    Returns string of processed text to be used into CountVectorizer
    '''
    # Lowercase and strip everything except words
    cleaner = re.sub(r"[^a-zA-Z ]+", ' ', text.lower())
    # Tokenize
    cleaner = word_tokenize(cleaner)
    ps = PorterStemmer()
    clean = []
    for w in cleaner:
        # filter out stopwords
        if w not in stopWords:
            # filter out short words
            if len(w)>2:
                # Stem 
                clean.append(ps.stem(w))
    return ' '.join(clean)

df_full['clean_text_stem'] = df_full.text.apply(lambda x: textcleaner_stem(x))
df_train['clean_text_stem'] = df_train.text.apply(lambda x: textcleaner_stem(x))
df_test['clean_text_stem'] = df_test.text.apply(lambda x: textcleaner_stem(x))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

stopWords = set(stopwords.words('english'))

def textcleaner_lemmas(text):
    ''' Takes in raw unformatted text and strips punctuation, removes whitespace,
    strips numbers, tokenizes and stems.
    Returns string of processed text to be used into CountVectorizer
    '''
    # Lowercase and strip everything except words
    cleaner = re.sub(r"[^a-zA-Z ]+", ' ', text.lower())
    # Tokenize
    cleaner = word_tokenize(cleaner)
    ps = PorterStemmer()
    clean = []
    for w in cleaner:
        # filter out stopwords
        if w not in stopWords:
            # filter out short words
            if len(w)>2:
                # lemmatizer 
                clean.append(lemmatizer.lemmatize(w))
    return ' '.join(clean)
df_full['clean_text_lemma'] = df_full.text.apply(lambda x: textcleaner_lemmas(x))
df_train['clean_text_lemma'] = df_train.text.apply(lambda x: textcleaner_lemmas(x))
df_test['clean_text_lemma'] = df_test.text.apply(lambda x: textcleaner_lemmas(x))
vectorizer = TfidfVectorizer(min_df=6, strip_accents='ascii', analyzer='word', lowercase=True,
                             ngram_range=(1,2))

x_train_stem = vectorizer.fit_transform(df_train['clean_text_stem'])
y_train_stem = df_train['source']
x_test_stem = vectorizer.transform(df_test['clean_text_stem'])
y_test_stem = df_test['source']
features_train = vectorizer.get_feature_names()
len(features_train)
from sklearn.naive_bayes import MultinomialNB

# Start timing
start = timeit.default_timer()

#Initialize and fit
nb = MultinomialNB()
nb.fit(x_train_stem, y_train_stem)

# Apply to testing data
y_pred_stem = nb.predict(x_test_stem)

# Stop timing
stop = timeit.default_timer()
nb_time = stop-start
print("Run time: %0.3f" % (nb_time))

# Showing model performance
print("Accuracy is: %0.3f" % nb.score(x_test_stem, y_test_stem))
print(metrics.classification_report(y_test_stem, y_pred_stem, target_names=dataset_test.target_names))
vectorizer = TfidfVectorizer(min_df=6, strip_accents='ascii', analyzer='word', lowercase=True,
                             ngram_range=(1,2))

x_train_lemma = vectorizer.fit_transform(df_train['clean_text_lemma'])
y_train_lemma = df_train['source']
x_test_lemma = vectorizer.transform(df_test['clean_text_lemma'])
y_test_lemma = df_test['source']
features_train = vectorizer.get_feature_names()
len(features_train)
from sklearn.naive_bayes import MultinomialNB

# Start timing
start = timeit.default_timer()

#Initialize and fit
nb = MultinomialNB()
nb.fit(x_train_lemma, y_train_lemma)

# Apply to testing data
y_pred_lemma = nb.predict(x_test_lemma)

# Stop timing
stop = timeit.default_timer()
nb_time = stop-start
print("Run time: %0.3f" % (nb_time))

# Showing model performance
print("Accuracy is: %0.3f" % nb.score(x_test_lemma, y_test_lemma))
print(metrics.classification_report(y_test_lemma, y_pred_lemma, target_names=dataset_test.target_names))
for n in range(1,4):
    vectorizer = TfidfVectorizer(min_df=6, strip_accents='ascii', analyzer='word', lowercase=True,
                                 ngram_range=(1,n))

    x_train_lemma = vectorizer.fit_transform(df_train['clean_text_lemma'])
    y_train_lemma = df_train['source']
    x_test_lemma = vectorizer.transform(df_test['clean_text_lemma'])
    y_test_lemma = df_test['source']
    features_train = vectorizer.get_feature_names()
    print(len(features_train))
    
# Start timing
    start = timeit.default_timer()

#Initialize and fit
    nb = MultinomialNB()
    nb.fit(x_train_lemma, y_train_lemma)

# Apply to testing data
    y_pred_lemma = nb.predict(x_test_lemma)

# Stop timing
    stop = timeit.default_timer()
    nb_time = stop-start
    print("Run time: %0.3f" % (nb_time))

# Showing model performance
    print("Accuracy is: %0.3f" % nb.score(x_test_lemma, y_test_lemma))
X_train =  x_train_lemma
Y_train = y_train_lemma
X_test = x_test_lemma 
Y_test = y_test_lemma
from sklearn.svm import SVC

# Start timing
start = timeit.default_timer()

# Create instance and fit
sv = SVC(kernel='linear')
sv.fit(X_train, Y_train)

# Apply to testing data
y_pred = sv.predict(X_test)

# Stop timing
stop = timeit.default_timer()
sv_time = stop - start
print("Run time:%0.3f" %sv_time)

# Showing model performance
cross = pd.crosstab(y_pred, Y_test)
print("Accuracy is: %0.3f" % sv.score(X_test, Y_test))
print(metrics.classification_report(Y_test, y_pred, target_names=dataset_test.target_names))
from sklearn.linear_model import LogisticRegression

# Start timing
start = timeit.default_timer()

lr = LogisticRegression()
lr.fit(X_train, Y_train)

y_pred = lr.predict(X_test)

# Stop timing
stop = timeit.default_timer()
lr_time = stop - start
print("Run time:%0.3f" %sv_time)

# Showing model performance
cross = pd.crosstab(y_pred, Y_test)
print("Accuracy is: %0.3f" % lr.score(X_test, Y_test))
print(metrics.classification_report(Y_test, y_pred, target_names=dataset_test.target_names));
from sklearn import ensemble

# Start timing
start = timeit.default_timer()

gbc = ensemble.GradientBoostingClassifier()
gbc.fit(X_train, Y_train)

pred = gbc.predict(X_test)

# Stop timing
stop = timeit.default_timer()
gbc_time = stop - start
print("Run time:%0.3f" %sv_time)

# Showing model performance
cross = pd.crosstab(y_pred, Y_test)
print("Accuracy is: %0.3f" % gbc.score(X_test, Y_test))
print(metrics.classification_report(Y_test, y_pred, target_names=dataset_test.target_names))

from sklearn import ensemble

# Start timing
start = timeit.default_timer()

rfc = ensemble.RandomForestClassifier()
rfc.fit(X_train, Y_train)

y_pred = rfc.predict(X_test)

# Stop timing
stop = timeit.default_timer()
rfc_time = stop - start
print("Run time:%0.3f" %rfc_time)

# Showing model performance
cross = pd.crosstab(y_pred, Y_test)
print("Accuracy is: %0.3f" % rfc.score(X_test, Y_test))
print("Accuracy is: %0.3f" % rfc.score(X_train, Y_train))
print(metrics.classification_report(Y_test, y_pred, target_names=dataset_test.target_names));
accuracy = [0.671,0.658,0.678,0.594,0.534]

labels = ['Naive Bayes', 'SVC', 'Logistic regression', 'Gradient boosting','Random forest']
x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars
fig, ax = plt.subplots(1, 1,  figsize=(10, 4))
g1 = ax.bar(x, accuracy, width, label='Model accuracy')

ax.set_ylabel('Model accuracy')
ax.set_title('Comparison for different ML algorithms')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=50)
ax.set_ylim(0, 1)
ax.legend();
def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))
        
a = show_top10(nb, vectorizer, dataset.target_names)
# Number of topics.
ntopics=20

# Linking words to topics
def word_topic(tfidf, solution, wordlist):
    
    # Loading scores for each word on each topic/component.
    words_by_topic=tfidf.T * solution

    # Linking the loadings to the words in an easy-to-read way.
    components=pd.DataFrame(words_by_topic,index=wordlist)

    return components
    
# Extracts the top N words and their loadings for each topic.
def top_words(components, n_top_words):
    n_topics = range(components.shape[1])
    index= np.repeat(n_topics, n_top_words, axis=0)
    topwords=pd.Series(index=index)
    for column in range(components.shape[1]):
        # Sort the column so that highest loadings are at the top.
        sortedwords=components.iloc[:,column].sort_values(ascending=False)
        # Choose the N highest loadings.
        chosen=sortedwords[:n_top_words]
        # Combine loading and index into a string.
        chosenlist=chosen.index +" "+round(chosen,2).map(str) 
        topwords.loc[column]=[x for x in chosenlist]
    return(topwords)

# Number of words to look at for each topic.
n_top_words = 10
terms = vectorizer.get_feature_names()

svd= TruncatedSVD(ntopics)
lsa = make_pipeline(svd, Normalizer(copy=False))
news_group_lsa = lsa.fit_transform(x_train_lemma)

components_lsa = word_topic(x_train_lemma, news_group_lsa, terms)

topwords=pd.DataFrame()
topwords['LSA']=top_words(components_lsa, n_top_words) 
from sklearn.decomposition import LatentDirichletAllocation as LDA

lda = LDA(n_components=ntopics, 
          doc_topic_prior=None, # Prior = 1/n_documents
          topic_word_prior=1/ntopics,
          learning_decay=0.7, # Convergence rate.
          learning_offset=10.0, # Causes earlier iterations to have less influence on the learning
          max_iter=10, # when to stop even if the model is not converging (to prevent running forever)
          evaluate_every=-1, # Do not evaluate perplexity, as it slows training time.
          mean_change_tol=0.001, # Stop updating the document topic distribution in the E-step when mean change is < tol
          max_doc_update_iter=100, # When to stop updating the document topic distribution in the E-step even if tol is not reached
          n_jobs=-1, # Use all available CPUs to speed up processing time.
          verbose=0, # amount of output to give while iterating
          random_state=0
         )

news_group_lda = lda.fit_transform(x_train_lemma) 

components_lda = word_topic(x_train_lemma, news_group_lda, terms)

topwords['LDA']=top_words(components_lda, n_top_words)
from sklearn.decomposition import NMF

nmf = NMF(alpha=0.0, 
          init='nndsvdar', # how starting value are calculated
          l1_ratio=0.0, # Sets whether regularization is L2 (0), L1 (1), or a combination (values between 0 and 1)
          max_iter=200, # when to stop even if the model is not converging (to prevent running forever)
          n_components=ntopics, 
          random_state=0, 
          solver='cd', # Use Coordinate Descent to solve
          tol=0.0001, # model will stop if tfidf-WH <= tol
          verbose=0 # amount of output to give while iterating
         )
news_group_nmf = nmf.fit_transform(x_train_lemma) 

components_nmf = word_topic(x_train_lemma, news_group_nmf, terms)

topwords['NNMF']=top_words(components_nmf, n_top_words)
for topic in range(ntopics):
    print('Topic {}:'.format(topic))
    print(topwords.loc[topic])
    
# The words to look at.
targetwords=['god','file','game','motor']

# Storing the loadings.
wordloadings=pd.DataFrame(columns=targetwords)

# For each word, extracting and string the loadings for each method.
for word in targetwords:
    loadings=components_lsa.loc[word].append(
        components_lda.loc[word]).append(
            components_nmf.loc[word])
    wordloadings[word]=loadings

# Labeling the data by method and providing an ordering variable for graphing purposes. 
wordloadings['method']=np.repeat(['LSA','LDA','NNMF'], 20, axis=0)
wordloadings['loading']=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]*3

sns.set(style="darkgrid")

for word in targetwords:
    sns.barplot(x="method", y=word, hue="loading", data=wordloadings)
    plt.title(word)
    plt.ylabel("")
#     plt.tight_layout()
    plt.show()
# kmean
x_norm = normalize(news_group_lsa)

kmeans = KMeans(n_clusters=20, random_state=123)
y_predict = kmeans.fit_predict(x_norm)

# Check the solution against the data.
print('Comparing k-means clusters against news groups:')
print(pd.crosstab(Y_train, y_predict))
# GaussianMixtureModel
from sklearn.mixture import GaussianMixture

gmm_cluster = GaussianMixture(n_components=20, random_state=123)
clusters = gmm_cluster.fit_predict(news_group_lsa)

# Check the solution against the data.
print('Comparing k-means clusters against news groups:')
print(pd.crosstab(Y_train, clusters))
for news_group in ['rec.autos','rec.sport.baseball','talk.politics.mideast']:

    auto = df_train[df_train['label']== news_group]['text'][:50]
    auto_str = ' '.join(auto)


# Importing the text the lazy way.
    text = auto_str

# We want to use the standard english-language parser.
    parser = spacy.load('en')
    
# Parsing Gatsby.
    text = parser(text)

# Dividing the text into sentences and storing them as a list of strings.
    sentences=[]
    for span in text.sents:
    # go from the start to the end of each span, returning each token in the sentence
    # combine each token using join()
        sent = ''.join(text[i].string for i in range(span.start, span.end)).strip()
        sentences.append(sent)

# Creating the tf-idf matrix.
    counter = TfidfVectorizer(lowercase=False, 
                              stop_words=None,
                              ngram_range=(1, 1), 
                              analyzer=u'word', 
                              max_df=.5, 
                              min_df=1,
                              max_features=None, 
                              vocabulary=None, 
                              binary=False)

#Applying the vectorizer
    data_counts=counter.fit_transform(sentences)
    
# Calculating similarity
    similarity = data_counts * data_counts.T

# Identifying the sentence with the highest rank.
    nx_graph = nx.from_scipy_sparse_matrix(similarity)
    ranks=nx.pagerank(nx_graph, alpha=.85, tol=.00000001)

    ranked = sorted(((ranks[i],s) for i,s in enumerate(sentences)),
                reverse=True)
    print('Topic: {}'.format(news_group))
    print(ranked[0])
    print()
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
news_lines=list()
lines = df_train['clean_text_lemma'].values.tolist()

for line in lines:
    tokens = word_tokenize(line)
    
    tokens = [w.lower() for w in tokens]
    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    words = [word for word in stripped if word.isalpha()]
    
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    news_lines.append(words)
import gensim
# from gensim.models import Word2Vec

model = gensim.models.Word2Vec(sentences=news_lines, window=5, workers=4, min_count=5)
# model=Word2Vec(sentences=review_lines)

words = list(model.wv.vocab)
print('vocabulary size: %d' % len(words))
vocab = model.wv.vocab.keys()

print(model.wv.most_similar(positive=['hard', 'drive', 'floppy'], negative=['god']))

# Similarity is calculated using the cosine, so again 1 is total
# similarity and 0 is no similarity.
print(model.wv.similarity('drive', 'floppy'))
print(model.wv.similarity('drive', 'think'))

# One of these things is not like the other...
print(model.doesnt_match("hard disk drive think".split()))
print(model.wv.most_similar(negative=["floppy"]))
print()
print(model.wv.most_similar(positive=["floppy"]))
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 100), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 19 to 10 dimensions with PCA
    reduc = PCA(n_components=10).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
tsnescatterplot(model, 'floppy', ['dog', 'bird', 'good', 'make', 'bob', 'mel', 'apple', 'duff'])
tsnescatterplot(model, 'floppy', [i[0] for i in model.wv.most_similar(negative=["floppy"])]);

