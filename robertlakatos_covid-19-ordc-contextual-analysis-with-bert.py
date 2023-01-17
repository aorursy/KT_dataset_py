import pandas as pd

path_data = "../input/covid19-ordc-cleaned-metadata/COVID-19-ORDC-cleaned-metadata.xlsx"
data = pd.read_excel(path_data)

print("Metadata was read. Total number of rows in file: " + str(len(data["title"])))
path_output_data = "/kaggle/working/COVID-19-ORDC-cleaned-metadata-extended.xlsx"

cleared_abstract = []
for abstract in data["abstract"]:

    if abstract[:8] == "Abstract":
        abstract = abstract[9:]

    cleared_abstract.append(abstract)
data["abstract"] = cleared_abstract

data.to_excel(path_output_data, index=False)

print("'Abstract' text removed")
import re
import nltk

stop_words = set(nltk.corpus.stopwords.words("english"))

def clean_text(text):
    # To lower
    result = text.lower()
    # Remove punctuations
    result = re.sub('[^a-zA-Z]', ' ', result)
    # remove special characters and digits
    result = re.sub("(\\d|\\W)+", " ", result)
    # convert to list from string
    result = result.split()
    # [1.] Stemming
    ps = nltk.stem.porter.PorterStemmer()
    # [2.] Lemmatisation
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    result = [lem.lemmatize(word) for word in result if not word in stop_words]

    return " ".join(result)
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(stop_words='english')

common_words = []
for abstract in data["abstract"]:
    abstract = [clear_text(abstract)]
    count_data = count_vectorizer.fit_transform(abstract)
    words = count_vectorizer.get_feature_names()
    tmp = words[::-1]
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:20]
    words = [w[0] for w in count_dict]
    common_words.append(" ".join(words))


data["common_words"] = common_words

data.to_excel(path_output_data, index=False)

print("The most common words was defined")
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

len_data = len(data["abstract"])
vectors = [""]*(len_data)

for i in range(0, len_data):
    try:
        # collect sentence and common words
        sentences = data["abstract"][i].split(". ")
        common_words = data["common_words"][i].split(" ")
        
        # tokenize with BERT [3] tokenizer
        encoded_sentences = []
        tokenized_sentences = []
        for sentence in sentences:
            sentence = sentence[:512]
            encoded_sentences.append(tokenizer.encode(sentence))
            tokenized_sentences.append(tokenizer.tokenize(sentence))

        index_sentences = []
        for sentence in tokenized_sentences:
            index_sentence = []
            for word in common_words:
                try:
                    tmp = sentence.index(word)
                    index_sentence.append(tmp+1)
                except:
                    continue
            index_sentences.append(index_sentence)
        
        # Create embedded vectors with BERT [3]
        abstract_vectors = tf.constant([[0]*768]).numpy()
        for j in range(0,len(encoded_sentences)):
            if len(index_sentences[j]) != 0:
                input_ids = tf.constant(encoded_sentences[j])[None, :]
                outputs = model(input_ids)
                sentence_vectors = outputs[0][0]
                sentence_vectors = sentence_vectors.numpy()
                len_isj = len(index_sentences[j])
                sentence_vectors_collected = [[0]*768] * len_isj
                for n in range(0,len_isj):
                    sentence_vectors_collected[n] = sentence_vectors[index_sentences[j][n]]

                abstract_vectors = np.concatenate((abstract_vectors, sentence_vectors_collected), axis=0)

        encoded_title = tokenizer.encode(data["title"][i])
        input_ids = tf.constant(encoded_title)[None, :]
        outputs = model(input_ids)
        title_vectors = outputs[0][0]
        title_vectors = title_vectors[1:len(title_vectors)-1].numpy()
        abstract_vectors = np.concatenate((abstract_vectors, title_vectors), axis=0)

        vektor = tf.reduce_mean(abstract_vectors, 0).numpy()
        vectors[i] = "\t".join(map(str, vektor))

        # chekpoint
        if (i % 1000) == 0 and i != 0:
            data["vectors"] = vectors
            data.to_excel(path_output_data, index=False)
            print("Processed data: " + str(i))
    except: 
        continue

data["vectors"] = vectors
data.to_excel(path_output_data, index=False)

print("Processed data: " + str(i))
print("The Embedded vectors was created")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Read vectors
number_of_vectors = len(data["vectors"])
vectors = np.array([[0]*768]*number_of_vectors, dtype=float)
for i in range(0, number_of_vectors):
    vector = data["vectors"][i].split("\t")
    vectors[i] = np.array(vector, dtype=float)

# Scale vectors
scaler = MinMaxScaler(feature_range=[0, 1])
vectors = scaler.fit_transform(vectors)

# Fitting the PCA [4] algorithm on data
pca = PCA().fit(vectors)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Dataset Variance')
plt.show()
pca = PCA(n_components=200)
vectors = pca.fit_transform(vectors)

print("PCA is on")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

distortions = []
maxK = 100
K = range(2, maxK, 5)
for k in K:
    kmeanModel = KMeans(n_clusters=k, verbose=0)
    kmeanModel.fit(vectors)
    distortions.append(sum(np.min(cdist(vectors, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / vectors.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
kmeanModel = KMeans(n_clusters=15,
                    verbose=0)
kmeanModel.fit(vectors)

data["clusters"] = kmeanModel.labels_
data.to_excel(path_output_data, index=False)

print("Custering completed")
from sklearn.decomposition import LatentDirichletAllocation as LDA

pd.options.mode.chained_assignment = None

number_of_documents = len(data["title"])
data["words_of_topic_in_clusters"] = [""]*number_of_documents
data_grouped = data.groupby(['clusters'])
for key in data_grouped.groups:
    group = data_grouped.groups[key]
    texts = []
    for index in group:
        text = clear_text(data["title"][index] + " " + data["abstract"][index])
        texts.append(text)
    
    # Create document matrix [7]
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(texts)    
    words = count_vectorizer.get_feature_names()

    number_topics = 1
    number_words = 10
    lda = LDA(n_components=number_topics, verbose=0)
    lda.fit(count_data)

    lda_words = " ".join([words[i] for i in lda.components_[0].argsort()[:-number_words - 1:-1]])
    for index in group:
        data["words_of_topic_in_clusters"][index] = lda_words

data.to_excel(path_output_data, index=False)
print("LDA completed")
number_of_documents = len(data["title"])
f_vecs = open("/kaggle/working/vecs.tsv","w",encoding="utf8")
f_meta = open("/kaggle/working/meta.tsv","w",encoding="utf8")
f_meta.write("title\tclusters\twords_of_topic\n")
for i in range(0, number_of_documents):
    f_vecs.write(data["vectors"][i] + "\n")
    f_meta.write(data["title"][i] + "\t" + str(data["clusters"][i]) + "\t" + data["words_of_topic_in_clusters"][i] + "\n")
f_vecs.close()
f_meta.close()

print(".tsv files creating completed")