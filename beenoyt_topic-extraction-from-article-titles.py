import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





import spacy

spacy.load('en')

import nltk

#nltk.download("wordnet") #(To be done once only)

from spacy.lang.en import English

parser = English()

import nltk

from nltk.corpus import wordnet as wn

from nltk.stem.wordnet import WordNetLemmatizer

from gensim import corpora

import gensim
df = pd.read_csv("/kaggle/input/financial-times-brexit-articles-database/financial_times_brexit_database.csv")

df.reference_time = pd.to_datetime(df.reference_time)
df.url = df.url.apply(lambda x: "https://www.ft.com/content/"+x)
with open("urls.txt","w") as f:

    f.write("\n".join(df.url.values))
documents = df.title.dropna().values
print(f"{len(documents)} kept from a total of {len(df.title)}")
stop_words_en=gensim.parsing.preprocessing.STOPWORDS

stop_words_en = stop_words_en.union(["brexit"])











def tokenize(text):

	"""

    Tokenizing texts

	"""

	lda_tokens = []

	tokens = parser(text)

	for token in tokens:

		if token.orth_.isspace():

			continue

		else:

			lda_tokens.append(token.lower_)

	return lda_tokens



def get_lemma(word):

	"""

	lemmatization

	"""

	lemma = wn.morphy(word)

	if lemma is None:

		return word

	else:

		return lemma





def prepare_text_for_lda(text):

	"""

	complete preparation of the text

	"""

	tokens = tokenize(text.replace("-"," "))

	tokens = [token for token in tokens if len(token) > 4]

	tokens = [token for token in tokens if token not in stop_words_en]

	tokens = [get_lemma(token) for token in tokens]

	return tokens



def texttokens(texts):

	"""

    Prepare all texts for nlp

	"""

	text_data = []

	for t in texts:

		tokens = prepare_text_for_lda(t)

		text_data.append(tokens)

	return text_data

















def LSI_topicExtraction(texts, n_topics):

	"""

	topic extraction with LSI

	"""





	print("Tokenization...")

	text_data=texttokens(texts)

	print("Dictionarisation...")

	dictionary = corpora.Dictionary(text_data)

	print("Corpusisation...")

	corpus = [dictionary.doc2bow(text) for text in text_data]

    



	#print(corpus)

	print("modelization...")

	lsimodel = gensim.models.LsiModel(corpus, id2word=dictionary,num_topics=n_topics)



	return lsimodel, corpus



def LDA_topicExtraction(texts, n_topics):

	"""

	topic extraction with LDA

	"""





	print("Tokenization...")

	text_data=texttokens(texts)

	print("Dictionarisation...")

	dictionary = corpora.Dictionary(text_data)

	print("Corpusisation...")

	corpus = [dictionary.doc2bow(text) for text in text_data]

    



	#print(corpus)

	print("modelization...")

	ldamodel = gensim.models.LdaModel(corpus, id2word=dictionary,num_topics=n_topics)



	return ldamodel, corpus





def format_topic(topic):

	"""

	Formatage des topics renvoyés par la librairie gensim pour les afficher dans l'interface web

	"""

	t = {}

	t["id"] = topic[0]

	a = topic[1].split(" + ")

	t["words"] = {}

	for i,m in enumerate(a):

		k = m.split("*")

		if i == 0:

			max_weight = float(k[0])

		t["words"][k[1].replace('"','')]=float(k[0])/max_weight



	return t

lda_model,corpus = LDA_topicExtraction(documents,10)
topics = [format_topic(t) for t in lda_model.print_topics()]
for t in topics:

    plt.figure(figsize=(15,7))

    sns.barplot(list(t["words"].keys()),list(t["words"].values()))

    plt.title(f"Most important words of topic n°{t['id']}")
def get_name(u):

    return "-".join([str(u.reference_time.year),str(u.reference_time.month),str(u.reference_time.day)]) + "  " + str(u.title)

df["name"] = df.apply(lambda x: get_name(x),axis=1)
df.sort_values("comment_count",ascending=False)[["name","comment_count"]]