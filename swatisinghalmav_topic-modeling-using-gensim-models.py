import warnings
warnings.filterwarnings("ignore")
import pandas as pd
orig_data = pd.read_excel("../input/blockchain_data.xlsx")
orig_data.head()
data = orig_data.loc[:,['AB','DE','TI']].copy()
data.head()
data.fillna("", inplace=True)
corpus = [(data['AB'][i] + " " + data['DE'][i] + " " + data['TI'][i]) for i in range(data.shape[0])]
print(corpus[2])
len(corpus) #941 awesome!!!
import string
trash = string.punctuation + string.digits
table = str.maketrans(trash, " " * len(trash))
corpus_cleaned = [corpus[i].translate(table) for i in range(len(corpus))]
print(corpus_cleaned[2])
from nltk import word_tokenize
corpus_words = [word_tokenize(corpus_cleaned[i].lower()) 
                for i in range(len(corpus_cleaned))] #List of list of words
print(corpus_words[2])
from nltk import WordNetLemmatizer
lemma = WordNetLemmatizer()
corpus_lemmatized = [[lemma.lemmatize(w) for w in corpus_words[i]] 
                     for i in range(len(corpus_words))] #List of list of lemmatized words
print(corpus_lemmatized[2])
from nltk.corpus import stopwords
stops = stopwords.words("english")
corpus_prepped = [" ".join([w for w in corpus_words[i] if w not in stops]) 
                  for i in range(len(corpus_words))] #list of bag of words for each article
print(corpus_prepped[2])
corpus_all = " ".join([corpus_prepped[i] for i in range(len(corpus_prepped))])
print(corpus_all[0:99]) #bag of words
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud(background_color='white').generate(corpus_all)
plt.figure(figsize = (10,5))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words = 'english')
count_dtm = vectorizer.fit_transform(corpus_prepped)
count_dtm.shape
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=25, learning_method="batch",
                                max_iter=500, random_state=1)
lda_topics = lda.fit_transform(count_dtm)
lda_topics.shape, lda.components_.shape
import numpy as np
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
words = np.array(vectorizer.get_feature_names())

#since word 'blockchain' will be there in each topic, lets see top 3 words
for i, word in enumerate(words[sorting[:, :3]]):
    print("Topic{:>2}: ".format(i) + " ".join(word))
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 2)
tfidf_dtm = vectorizer.fit_transform(corpus_prepped)
tfidf_dtm.shape
from sklearn.decomposition import NMF
nmf = NMF(n_components=25, random_state=1)
nmf_topics = nmf.fit_transform(tfidf_dtm)
nmf_topics.shape, nmf.components_.shape
import numpy as np
sorting = np.argsort(nmf.components_, axis=1)[:, ::-1]
words = np.array(vectorizer.get_feature_names())

#since there is penalty for high frequency words in tfidf, 'blockchain' may not be in top 3 words
for i, word in enumerate(words[sorting[:, :3]]):
    print("Topic{:>2}: ".format(i) + " ".join(word))
#Since we know the articles are all on 'Blockchain', let's remove this word
words_all = [w for w in corpus_all.split() if w != 'blockchain'] 

from nltk import FreqDist
plt.figure(figsize = (20,5))
plt.xlabel('off')
FreqDist(words_all).plot(100)
plt.show()
#list of list of prepped words
corpus_prepped_words = [corpus_prepped[i].split() for i in range(len(corpus_prepped))]
print(corpus_prepped_words[2])
import gensim
# Building the bigram and trigram models
bigram = gensim.models.Phrases(corpus_prepped_words, min_count=5, threshold=10) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[corpus_prepped_words], threshold=10)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

corpus_trigrammed = [trigram_mod[bigram_mod[corpus_prepped_words[i]]] for i in range(len(corpus_prepped_words))]
print(corpus_trigrammed[2])
import gensim.corpora as corpora

# Creating Dictionary
id2word = corpora.Dictionary(corpus_trigrammed)

# Term Document Frequency
texts = [id2word.doc2bow(text) for text in corpus_trigrammed]

# View
print(texts[2])
#better view
print([(id2word[id], freq) for id, freq in texts[2]])
warnings.filterwarnings("ignore")
from gensim.models import ldamodel, CoherenceModel
lda_model = ldamodel.LdaModel(corpus=texts, id2word=id2word, num_topics=25, 
                              random_state=1, update_every=1, chunksize=100, 
                              passes=10, alpha='auto', per_word_topics=True)
# For computing Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=corpus_trigrammed, 
                                     dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence() #higher the coherence score, better the model.
print('\nPerplexity: ', lda_model.log_perplexity(texts))  # lower the perplexity, better the model.

print('\nCoherence Score: ', coherence_lda)
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, texts, id2word)
vis
warnings.filterwarnings("ignore")
def evaluation(dictionary, txt, crps, limit, start=2, step=1):
    coherence_values = []
    perplexity_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = ldamodel.LdaModel(random_state=1, corpus=txt, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        perplexity_values.append(model.log_perplexity(txt))
        coherencemodel = CoherenceModel(model=model, texts=crps, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values, perplexity_values

model_list, coherence_values, perplexity_values = evaluation(dictionary=id2word, txt=texts, 
                                                             crps=corpus_trigrammed, limit=40)

x = range(2, 40)
plt.plot(x, coherence_values, label='Coherence')
plt.xlabel("Number of Topics")
plt.legend(loc='best')
plt.show()
plt.plot(x, perplexity_values, label='Perplexity')
plt.xlabel("Number of Topics")
plt.legend(loc='best')
plt.show()
warnings.filterwarnings("ignore")
lda_model = ldamodel.LdaModel(corpus=texts, id2word=id2word, num_topics=7, 
                              random_state=1, update_every=1, chunksize=100, 
                              passes=10, alpha='auto', per_word_topics=True)
doc_lda = lda_model[texts]
coherence_model_lda = CoherenceModel(model=lda_model, texts=corpus_trigrammed, 
                                     dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, texts, id2word)
vis