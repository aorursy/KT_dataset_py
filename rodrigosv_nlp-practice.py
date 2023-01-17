import nltk
myfile = open("../input/bible.txt", encoding="utf8")

txt = myfile.read()

txt[:300]
txt_tkn = txt.split('\n\n')

txt_tkn = [tkn for tkn in txt_tkn if tkn != '']

txt_tokens = []

for verse in txt_tkn:

    if '\n' in verse:

        verse = verse.replace('\n', ' ')

    txt_tokens.append(verse)

txt_tokens[:20]
tokenizer_spa=nltk.data.load('tokenizers/punkt/english.pickle')



tokenizer_spa.tokenize(txt)[:20]
from nltk.tokenize import WordPunctTokenizer # TreebankWordTokenizer

tokenizer=WordPunctTokenizer() # TreebankWordTokenizer

txt_wd = tokenizer.tokenize(txt)

txt_wd[:20]
from nltk.tokenize import RegexpTokenizer

tokenizer=RegexpTokenizer("[\w]+")

txt_words = tokenizer.tokenize(txt)

txt_words[:20]
from nltk.corpus import stopwords

stops=set(stopwords.words('english'))

stops
# Getting out stopwords and change to lower case

text_out_stopw = [word.lower() for word in txt_words if word not in stops]

text_out_stopw[:20]
freq = nltk.FreqDist(text_out_stopw)

freq
sorted_by_value = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)

for key,val in sorted_by_value:

    print (str(key) + ':' + str(val))

    if val < 1000: break
from nltk import ngrams
n = 3

n_grams = ngrams(text_out_stopw, n)

n_grams_list = [grams for grams in n_grams]
from collections import defaultdict



n_grams_freq = defaultdict(int)



for curr in n_grams_list:

    n_grams_freq[curr] += 1



n_grams_sorted_by_value = sorted(n_grams_freq.items(), key=lambda kv: kv[1], reverse=True)



for key,val in n_grams_sorted_by_value:

    print (str(key) + ':' + str(val))

    if val < 100: break
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')
stems = []

for word in text_out_stopw:

    word = stemmer.stem(word)

    if word != "":

        stems.append(word)

stems[:20]
stems_freq = nltk.FreqDist(stems)

stems_sorted_by_value = sorted(stems_freq.items(), key=lambda kv: kv[1], reverse=True)

for key,val in stems_sorted_by_value:

    print (str(key) + ':' + str(val))

    if val < 1000: break
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

bag_of_words = vectorizer.fit_transform(txt_tokens).todense()

print('Bag shape =',bag_of_words.shape, 'Bag type =', type(bag_of_words))
list(vectorizer.vocabulary_)[:50]
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vectorizer = TfidfVectorizer(stop_words=stops)

tfidf_matrix = tfidf_vectorizer.fit_transform(txt_tokens) 

print(list(tfidf_vectorizer.vocabulary_)[:50])
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD



lda_model = LatentDirichletAllocation(n_components=15,               # Number of topics

                                      max_iter=10,               # Max learning iterations

                                      learning_method='online',   

                                      random_state=100,          # Random state

                                      batch_size=128,            # n docs in each learning iter

                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't

                                     )

lda_output = lda_model.fit_transform(tfidf_matrix)



print(lda_model)  # Model attributes
from sklearn.model_selection import GridSearchCV



lda_test_model = LatentDirichletAllocation()

param_grid = {'n_components': [5, 10, 15, 20], 'max_iter': [5, 10, 15, 20],

              'learning_method': ['batch', 'online'], 'random_state': [50, 100, 200]}



grid_search = GridSearchCV(lda_test_model, param_grid, cv=5)



grid_search.fit(tfidf_matrix)
def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        message = "Topic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

        

tf_feature_names = tfidf_vectorizer.get_feature_names()



n_top_words = 10



print_top_words(lda_model, tf_feature_names, n_top_words)
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

%matplotlib inline



clusters = KMeans(n_clusters=6, random_state=100).fit_predict(lda_output)



svd_model = TruncatedSVD(n_components=2)

lda_output_svd = svd_model.fit_transform(lda_output)



x = lda_output_svd[:, 0]

y = lda_output_svd[:, 1]



plt.figure(figsize=(12,12))

plt.scatter(x,y,c=clusters)

plt.ylabel('Component 2')

plt.xlabel('Component 1')

plt.title('Topic segregation',)
from textblob import TextBlob
count = 1

polarity_list = []

sub_list = []

num_tkn = []



for num, token in enumerate(txt_tokens):

        analysis = TextBlob(token)

        num_tkn.append(num)

        sub_list.append(analysis.subjectivity)

        polarity_list.append(analysis.polarity)
plt.figure(figsize=(15,15))

plt.scatter(num_tkn, polarity_list)

plt.title("Sentiment on Bible's verses")

plt.xlabel("Token")

plt.ylabel("Polarity")  
plt.figure(figsize=(15,15))

plt.scatter(num_tkn[:500], sub_list[:500])

plt.title("Sentiment on Bible's verses")

plt.xlabel("Token")

plt.ylabel("Subjectivity")  