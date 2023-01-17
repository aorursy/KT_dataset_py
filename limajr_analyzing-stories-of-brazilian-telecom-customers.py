file = "oi.tsv"
import unicodedata
import re

def removeURL(text):
    text = re.sub("http\\S+\\s*", "", text)
    return text

# 'NFKD' is the code of the normal form for the Unicode input string.
def remove_accentuation(text):
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII','ignore')
    return text.decode("utf-8")

def remove_punctuation(text):
    # re.sub(replace_expression, replace_string, target)
    new_text = re.sub(r"\.|,|;|!|\?|\(|\)|\[|\]|\$|\:|\\|\/", "", text)
    return new_text

def remove_numbers(text):
    # re.sub(replace_expression, replace_string, target)
    new_text = re.sub(r"[0-9]+", "", text)
    return new_text

# Conver a text to lower case
def lower_case(text):
    return text.lower()
# Remove stop words from a text
from nltk.corpus import stopwords
nltk_stop = set(stopwords.words('portuguese'))
for word in ["2018","claro","oi","tim","vivo","dia","e","pois","r$"]:
    nltk_stop.add(word)

def remove_stop_words(text, stopWords=nltk_stop):
    for sw in stopWords:
        text = re.sub(r'\b%s\b' % sw, "", text)
        
    return text
def pre_process(text):
    new_text = lower_case(text)
    new_text = removeURL(new_text)
    new_text = remove_stop_words(new_text)
    new_text = remove_numbers(new_text)
    new_text = remove_punctuation(new_text)
    new_text = remove_accentuation(new_text)
    return new_text
n_features = 2200 # 10 percent of tokens
n_components = 3
n_top_words = 10
print("{} topics will be modelled for {} file.".format(n_components, file))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

print("Loading and preprocessing dataset...")
dataset = pd.read_csv("../input/"+file, sep="\t", encoding="utf-8")
data_samples = dataset.iloc[:, 4]
pre_processed_samples = data_samples.apply(pre_process)
n_samples = len(pre_processed_samples)

# Use tf-idf features for LDA.
print("Extracting tf features and fitting models for LDA...")
tf_vectorizer = TfidfVectorizer(max_features=n_features)
tf = tf_vectorizer.fit_transform(pre_processed_samples)

print("\nFiting LDA model...")
lda = LatentDirichletAllocation(n_components=n_components, max_iter=10,
                                learning_method='online', learning_offset=50., random_state=0)
lda.fit(tf)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

print("Topics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

print("Fiting LSA model...")
lsa = TruncatedSVD(n_components=n_components, n_iter=40, tol=0.01)
lsa.fit(tf)

print("Topics in LSA model:")
print_top_words(lsa,tf_feature_names,n_top_words)
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def gen_word_cloud(topic,string):
    # Generate a word cloud image
    print("Wordcloud for {}".format(topic))
    wordcloud = WordCloud().generate(string)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

string = ' '.join(pre_processed_samples)
gen_word_cloud("whole dataset",string)
    
    # Loops through Topics
for n in range(n_components):
    indexes = [index for (index,doc) in enumerate(lda.transform(tf)) if doc.argmax()==n]
    string = ' '.join([pre_processed_samples[i] for i in indexes])
    topic = "Topic #"+str(n)
    gen_word_cloud(topic,string)
from collections import OrderedDict

from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity

X = tf
vectorizer = tf_vectorizer

print("DTM Matrix")
feature_names = vectorizer.get_feature_names()
# Transform the text collection from our dataset to a Ordered Dictionary, with items as <index, text>.
text_collection = OrderedDict([(index, text) for index, text in enumerate(pre_processed_samples.values)])
#print(text_collection)
corpus_index = [n for n in text_collection]
# Build a DataFrame from the Document-Term Matrix returned by the Vectorizer, to print a nice visualization
df = pd.DataFrame(X.todense(), index=corpus_index, columns=feature_names)


dist = 1 - cosine_similarity(X.T)

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 800)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=feature_names, leaf_font_size=20);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
#plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clust
!pip install polyglot
!polyglot download embeddings2.pt ner2.pt
import nltk
nltk.download('rslp')
from polyglot.text import Text

def tokenize(string):
    text = Text(string, hint_language_code='pt')
    text.language = "pt"
    return text.words
import nltk

stemmer = nltk.stem.RSLPStemmer()

def stemming(text):
    words = tokenize(text)
    stemmer_words = [stemmer.stem(item) for item in words]
    return ' '.join(stemmer_words)
from collections import Counter

string = ' '.join(pre_processed_samples)
text = Text(string)
print("Identified language \"{}\" with confidence {}\n"
      .format(text.language.name, text.language.confidence))

print("Performing Tokenization and Stemming tasks...")
stemmed_string = stemming(string)
stemmed_text = Text(stemmed_string)
most_commons = Counter(stemmed_text.words).most_common(15)
print("The 15 most common (stemmed) items found: \n")

print("{:<16}{}".format("Stemmed word", "Occurrence")+"\n"+"-"*30)
for i,j in most_commons:
    print("{:<16}{}".format(i,j))
from functools import reduce

def NER_sentiments(sent):
    sent.language = "pt"
    if sent.entities:
        try:
            positive = list(map(lambda x: x.positive_sentiment, sent.entities))
            sum_positive = reduce((lambda x, y: x + y), positive)
            negative = list(map(lambda x: x.negative_sentiment, sent.entities))
            sum_negative = reduce((lambda x, y: x + y), negative)
            sum_total = sum_positive - sum_negative    
            #print("{}\n".format(sum_total))
            return sum_total
        except:
            return 0
    else:
        return 0

#new_data = data_samples.apply(remove_accentuation).values
avaliacoes = dataset.iloc[:,6]
avaliacoes.replace({"<não há comentários do consumidor>": "-"})
avaliacoes.apply(remove_accentuation)
avaliacoes = avaliacoes.values

for row_idx, row in enumerate(avaliacoes):
    #print("row {}: {}".format(row_idx, row))
    text = Text(row, hint_language_code='pt')
    sentences = text.sentences
    if sentences:
        sentiments = list(map(lambda x: NER_sentiments(x), sentences))
        sentiment = reduce((lambda x, y: x + y), sentiments)
        if sentiment < 0:
            sentiment = "Negative"
        elif sentiment == 0:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"
        print("Sentiment in doc {}: {}".format(row_idx, sentiment))
