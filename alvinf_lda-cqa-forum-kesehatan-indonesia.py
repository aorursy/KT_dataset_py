# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import library pandas dan inisialisasikan menjadi pd

import pandas as pd



# import nltk untuk toknisasi

import nltk.data

nltk.download('punkt')



# Gensim

import gensim

import gensim.corpora as corpora

from gensim.corpora import Dictionary

from gensim.models import CoherenceModel



# Plotting tools

!pip install pyldavis

import pyLDAvis

import pyLDAvis.gensim

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#baca data mentah yang sudah diambil saat scrapping

doc = pd.read_csv('../input/data-questionanswer-forum-kedokteran-indonesia/hasilScrapFinal.csv', encoding='utf-8')



#rapikan data

doc = doc[doc['Judul'].notnull()]

doc = doc[doc['Deskripsi'].notnull()]

doc = doc[['Judul','Deskripsi','link']]

doc
# membaca files wordBank

# wordBank kata-kata trbanyak yang sudah dipilih yang rlvan dngan kdoktran

vocab = pd.read_csv('../input/../input/word-library-kesehatan-indonesia/wordlibFinalFinal.csv', encoding='utf-8')

vocab = vocab.set_index('Kata').T.to_dict()

vocab
# buat tokenisasi dari kata-kata yang ada di judul dan deskripsi

doc['tokenized'] = doc.apply(lambda row: nltk.word_tokenize(row['Judul']+' '+row['Deskripsi']), axis=1)

doc
# fungsi stopword, memilih kata-kata yang ada di vocab

def stop(item):

    out = []

    for it in item:

        dop = str(it)

        if dop in vocab:

            out.append(dop)

    return out



# apply fungsi ke data yang sudah ditokenisasi

doc['tokenized'] = doc['tokenized'].apply(stop)

doc
# mengubah data yang berbentuk dataframe ke list

data = doc.tokenized.values.tolist()

data
# membuat Dictionary

id2word = corpora.Dictionary(data)



# Term Document Frequency (TF) dalam bentuk id

corpus = [id2word.doc2bow(text) for text in data]



# View corpus, tapi bentuknya kata

[[(id2word[id], freq) for id, freq in cp] for cp in corpus]
data = doc.tokenized.tolist()



# Term Document Frequency (TF) dalam bentuk id

corpus = [id2word.doc2bow(text) for text in data]



# View corpus, tapi bentuknya kata

[[(id2word[id], freq) for id, freq in cp] for cp in corpus]
# Fungsi coherence, mencari jumlah topik optimal

def compute_coherence_values(dictionary, corpus, texts, limit, start, step):

  coherence_values = []

  model_list = []

  for num_topics in range(start, limit, step):

      model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)

      model_list.append(model)

      coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

      coherence_values.append(coherencemodel.get_coherence())

  return model_list, coherence_values
limit=300; start=30; step=4;

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data, start=start, limit=limit, step=step)
x = range(start, limit, step)

plt.plot(x, coherence_values)

plt.xlabel("Num Topics")

plt.ylabel("Coherence score")

plt.legend(("coherence_values"), loc='best')

plt.show()
# list coherence value

i=0

for m, cv in zip(x, coherence_values):

    print(i, "Num Topics =", m, " has Coherence Value of", round(cv, 4))

    i+=1
# secara graf terlihat yang paling tinggi 30, tapi dari pengamatan, topiknya lebih dari itu, jadi diambil tertinggi terakhir

# ambil 70 topik, posisi 10 di array

optimal_model = model_list[0]

topic_num = 30

lda_model = optimal_model

model_topics = optimal_model.show_topics(formatted=False)

optimal_model.print_topics(num_words=5)
pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(optimal_model, corpus, id2word)

vis
def format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data):

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row in enumerate(ldamodel[corpus]):

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic

                wp = ldamodel.show_topic(topic_num)

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']



    # Add original text to the end of the output

    contents = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)





df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)



# Format

df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']



# Show

df_dominant_topic
pdfull = pd.merge(doc,df_dominant_topic,left_index=True,right_index=True)

pdfull = pdfull[["Judul", "Deskripsi", "link", "Dominant_Topic", "Topic_Perc_Contrib"]]

pdfull
pertopic = [None] * topic_num

topic_keywords = pertopic

for i in range(topic_num):

    wp = optimal_model.show_topic(i)

    topic_keywords[i] = ", ".join([word for word, prop in wp])

    pertopic[i] = (pdfull[pdfull['Dominant_Topic'] == i])

    pertopic[i] = pertopic[i].sort_values(by ='Topic_Perc_Contrib', ascending = False)

    pertopic[i] = pertopic[i][pertopic[i]['Topic_Perc_Contrib']>=0.5]

pertopic[1]
text = 'lambung asam'

text = re.sub('[^A-Za-z]+', ' ', text)

text = text.lower()

token = text.split()

out = []

for it in token:

    if it in vocab:

        out.append(it)

print(out)

out = id2word.doc2bow(out)



for index, score in sorted(optimal_model[out], key=lambda tup: -1*tup[1]):

    print("\nScore: {}\t \nTopic Num: {}\t \nTopic: {}".format(score,index, optimal_model.print_topic(index, 10)))

    break

    

print("Pertanyaan terkait: ")

pertopic[index]