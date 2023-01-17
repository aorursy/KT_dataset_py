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
import pandas as pd

import nltk, re, multiprocessing

nltk.download("stopwords")

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from keras.preprocessing.text import Tokenizer

import spacy

from zipfile import ZipFile

import os

from wordcloud import WordCloud

from gensim.models.phrases import Phrases, Phraser

from gensim.models.fasttext import FastText

from gensim.models import Word2Vec

from sklearn.manifold import TSNE
data = pd.read_csv("/kaggle/input/medium-articles/articles.csv")

data.head()
data.shape
data["author"].nunique()
cloud=WordCloud(colormap="Reds",width=600,height=400).generate(str(data["text"]))

fig=plt.figure(figsize=(12,12))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("Word Cloud on uncleaned text", size = 20)
pd.set_option('display.max_colwidth', None)

data["text"].head()
def cleaned_data_3(text):

    cleaned_txt = re.sub("AI","Artificial Intelligence", text)

    return cleaned_txt

data["text"] = data["text"].apply(cleaned_data_3)
pd.set_option('display.max_colwidth', None)

data["title"].head()
def cleaned_text(text_data):

    clean = text_data.lower()

    clean = re.sub("\n"," ",clean)

    clean = re.sub("http\S+"," ",clean)

    clean = re.sub("www\S+"," ",clean)

    #clean = re.sub(r"[,-.:;]"," ", clean)

    clean=re.sub("[^a-z]"," ",clean)

    clean=clean.lstrip()

    clean=re.sub("\s{2,}"," ",clean)

    return clean

data["cleaned_data"] = data["text"].apply(cleaned_text)
data["cleaned_data"].head()
nlp = spacy.load("en_core_web_sm")

doc = nlp(str(data["cleaned_data"]))

for token in doc: 

  print(token, token.pos_)

  #print(token, token.lemma_)
list = [token.pos_ for token in doc]

list = pd.Series(list)

plt.style.use("dark_background")

list.value_counts().plot(figsize = (12,6), kind = "bar", color = "r")

plt.title("Frequency of POS tagger", size = 22)

plt.xlabel("POS tagger", size = 18)
data['Number_of_words'] = data['cleaned_data'].apply(lambda x:len(str(x).split()))
plt.style.use('dark_background')

plt.figure(figsize=(12,6))

sns.distplot(data['Number_of_words'],kde = False,color="springgreen", bins = 100)

plt.title("Frequency distribution of number of words from each text", size = 20)

plt.xlabel("Number of words", size = 18)
len(data[data["Number_of_words"]>5000])
cloud=WordCloud(colormap="spring",width=600,height=400).generate(str(data["cleaned_data"]))

fig=plt.figure(figsize=(12,12))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("Word Cloud on cleaned text", size = 22)
stop=stopwords.words('english')

stop.extend(["make","get","also","use","using","used","even","though","could","would","us","much","uses","makes","part"])

data["stopwords_rem"]=data["cleaned_data"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data["stopwords_rem"].head()
def cleaned_text_2(text):

    #cleaned = text_data.lower()

    cleaned = re.sub("networks","network",text)

    cleaned = re.sub("weeks","week",cleaned)

    cleaned = re.sub("boxes","box",cleaned)

    cleaned = re.sub("algorithms","algorithm",cleaned)

    cleaned = re.sub("functions","function",cleaned)

    cleaned = re.sub("nets","network",cleaned)

    cleaned = re.sub("proposals","proposal",cleaned)

    cleaned = re.sub("imitative ai","imitative artificial intelligence",cleaned)

    cleaned = re.sub("deep rl","deep reinforcement learning",cleaned)

    cleaned = re.sub("word vec","word2vec",cleaned)#crartificial

    cleaned = re.sub("crartificial intelligenceg","artificial intelligence",cleaned)

    cleaned = re.sub("nlp","natural language processing",cleaned)

    cleaned = re.sub("openartificial","open artificial",cleaned)

    cleaned = re.sub("intelligences","intelligence",cleaned)

    return cleaned

data["stopwords_rem"] = data["stopwords_rem"].apply(cleaned_text_2)
data["stopwords_rem"] = data["stopwords_rem"].apply(lambda x: ' '.join([word for word in x.split() if len(word)>1]))
plt.style.use('ggplot')

plt.figure(figsize=(14,6))

freq=pd.Series(" ".join(data["stopwords_rem"]).split()).value_counts()[:30]

freq.plot(kind="bar", color = "orangered")

plt.title("30 most frequent words",size=20)
data['Number_of_words_after_stpwrd'] = data['stopwords_rem'].apply(lambda x:len(str(x).split()))
plt.style.use('dark_background')

plt.figure(figsize=(12,6))

sns.distplot(data['Number_of_words_after_stpwrd'],kde = False,color="yellow", bins = 100)

plt.title("Histogram showing number of words from each doc after stop removal", size = 18)

plt.xlabel("Number of words", size = 18)
tokens = data["stopwords_rem"].apply(lambda x: nltk.word_tokenize(x))
#phrases = Phrases(tokens, min_count = 20, threshold = 50, delimiter=b'_')

#phrases = Phrases(tokens, min_count = 25, threshold = 40, delimiter=b'_')

phrases = Phrases(tokens, min_count = 27, threshold = 41.5, delimiter=b'_')

bigram = Phraser(phrases)
bigram.phrasegrams
#trigram = Phrases(bigram[tokens], min_count=22, delimiter=b' ', threshold = 50)

#trigram = Phrases(bigram[tokens], min_count = 15, threshold = 25, delimiter=b'_')

trigram = Phrases(bigram[tokens], min_count = 18, threshold = 26, delimiter=b'_')
trigram_final = Phraser(trigram)

trigram_final.phrasegrams
!wget http://nlp.stanford.edu/data/glove.6B.zip
with ZipFile('glove.6B.zip', 'r') as zip: 

    # printing all the contents of the zip file 

    zip.printdir()

    zip.extractall()
t = Tokenizer()

t.fit_on_texts(data["stopwords_rem"])

encoded_docs = t.texts_to_sequences(data["stopwords_rem"])
print(encoded_docs)
embeddings_dict = {}

with open("glove.6B.100d.txt", 'r') as f:

    for line in f:

        values = line.split()

        word = values[0]

        vector = np.asarray(values[1:], "float32")

        embeddings_dict[word] = vector
embeddings_dict["deep"]
embeddings_dict["random"]
embeddings_dict["basic"]
vocab_size = len(t.word_index) + 1

print(vocab_size)
t.word_index
embedding_matrix = np.zeros((vocab_size, 100))

for word, i in t.word_index.items():

    embedding_vector = embeddings_dict.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
embedding_matrix[2]   # Here 2 is the word index, and from above we can check 2 is for the word "data"
embeddings_dict["data"]
def cosine_similarity(A, B):

 

    dot = np.dot(A,B)

    norma = np.sqrt(np.dot(A,A))

    normb = np.sqrt(np.dot(B,B))

    cos = dot / (norma*normb)

    return cos
cosine_similarity(embedding_matrix[1],embedding_matrix[4])
cosine_similarity(embeddings_dict["loss"],embeddings_dict["function"])
multiprocessing.cpu_count()
tokens_3 = [word for word in trigram_final[tokens]] 
fastext_mdl = FastText(tokens_3,

                      window = 5,

                      size = 100,

                      alpha = 0.01,

                      min_alpha = 0.0005,

                      workers = multiprocessing.cpu_count(),

                      seed = 42)
len(fastext_mdl.wv.vocab)
print(fastext_mdl.wv['artificial_intelligence'])
print(fastext_mdl.wv.similarity(w1='artificial_intelligence', w2='machine_learning'))
def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity = 30, n_components=2, init='pca', n_iter=2000, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(15, 13)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        #plt.annotate(labels[i],

                     #xy=(x[i], y[i]),

                     #xytext=(5, 2),

                     #textcoords='offset points',

                     #ha='right',

                     #va='bottom')

    plt.show()
sns.set_style('whitegrid')

#plt.style.use("ggplot")

tsne_plot(fastext_mdl)