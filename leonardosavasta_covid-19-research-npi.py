# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from collections import Counter

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

import os

import json

import re

import gc
def find_paper(paper_metadata):

    

    # Function takes row from data in all_sources_metadata.csv in pandas dataframe 

    # format and finds file in input data files. Returns dictionary from json format

    

    if paper_metadata['has_full_text'] != True:

    

        return None

    

    directory = '/kaggle/input/CORD-19-research-challenge'

    license = paper_metadata['license']

    paper_id = paper_metadata['sha'].split(';')[0]

    

    folder = paper_metadata['full_text_file']

        

    path = os.path.join(directory, folder, folder, str(paper_id + '.json'))

    

    with open(path, 'r') as infile:

        data = json.load(infile)

        

    return data

    
# Load csv file into pandas dataframe



papers_metadata = pd.read_csv(

    '/kaggle/input/CORD-19-research-challenge/metadata.csv'

)

papers_metadata
def most_used_words(texts, stopwords_input = []):

    

    # Function identifies most used words in a text skipping stopwords

    

    stopwords = ["a", "these", "have", "which", "that", "is", "by", "was", "we", "were", "be", "not", "has", "this", "are", "on", "an", "my","to","at","for","it","the","with","from","would","there","or","if","it","but","of","in","as","and",'NaN','dtype']



    stopwords.append(stopwords_input)

    

    sentences = list()

    temp = list()



    for i, text in enumerate(texts):



        if type(text) != str:



            continue



        temp = text.lower().split()

        

        temp = filter(lambda w: w not in stopwords, temp)



        sentences += temp



    count_sentences_text = Counter(sentences)

    count_sentences_text = sorted(count_sentences_text.items(), key = lambda i: i[1], reverse=True)

    

    return count_sentences_text




tmp = most_used_words(papers_metadata['abstract'])[:10]



x, y = map(list, zip(*tmp))

fig, axs = plt.subplots(1, 1)

plt.xticks(rotation='vertical')

axs.set_title("Most used words in paper abstracts")

axs.bar(x, y)
def multiple_most_used_words(texts, num, stopwords_input = []):

    

    # Paper identifies the most used num of words in paper skipping over stopwords

    

    stopwords = ["a", "that", "is", "by", "was", "we", "were", "be", "not", "has", "this", "are", "on", "an", "my","to","at","for","it","the","with","from","would","there","or","if","it","but","of","in","as","and",'NaN','dtype']



    stopwords.append(stopwords_input)

    

    sentences = list()

    temp = list()



    for i, text in enumerate(texts):



        if type(text) != str:



            continue



        temp = text.lower().split()

        

        temp = list(filter(lambda w: w not in stopwords, temp))



        temp = [' '.join(temp[i: i + num]) for i in range(0, len(temp), 1)]



        sentences += temp



    count_sentences_text = Counter(sentences)

    count_sentences_text = sorted(count_sentences_text.items(), key = lambda i: i[1], reverse=True)

    

    return count_sentences_text

tmp = multiple_most_used_words(papers_metadata['abstract'], 3)[:10]



x, y = map(list, zip(*tmp))

fig, axs = plt.subplots(1, 1)

plt.xticks(rotation='vertical')

axs.set_title("Most used words in paper abstracts")

axs.barh(x, y)
papers_metadata['nonpharmaceutical'] = papers_metadata['abstract'].str.find('non-pharmaceutical')

papers_metadata['nonpharmaceutical'] += papers_metadata['abstract'].str.find('nonpharmaceutical')

len(papers_metadata['nonpharmaceutical'].where(papers_metadata['nonpharmaceutical'] > -1).dropna())
papers_metadata['economic'] = papers_metadata['abstract'].str.find('economic impact')

len(papers_metadata['economic'].where(papers_metadata['economic'] > -1).dropna())
papers_metadata['intervention'] = papers_metadata['abstract'].str.find('intervention')

len(papers_metadata['intervention'].where(papers_metadata['intervention'] > -1).dropna())
papers_metadata['infrastructure'] = papers_metadata['abstract'].str.find('infrastructure')

len(papers_metadata['infrastructure'].where(papers_metadata['infrastructure'] > -1).dropna())
papers_metadata['compliance'] = papers_metadata['abstract'].str.find('compliance')

len(papers_metadata['compliance'].where(papers_metadata['compliance'] > -1).dropna())
papers_metadata['policy'] = papers_metadata['abstract'].str.find('policy')

len(papers_metadata['policy'].where(papers_metadata['policy'] > -1).dropna())
papers_metadata['financial'] = papers_metadata['abstract'].str.find('financial')

len(papers_metadata['financial'].where(papers_metadata['financial'] > -1).dropna())
papers_metadata['government'] = papers_metadata['abstract'].str.find('government')

len(papers_metadata['government'].where(papers_metadata['government'] > -1).dropna())
papers_metadata['scale up'] = papers_metadata['abstract'].str.find('scale up')

len(papers_metadata['scale up'].where(papers_metadata['scale up'] > -1).dropna())
papers_metadata['coordinated'] = papers_metadata['abstract'].str.find('coordinated')

len(papers_metadata['coordinated'].where(papers_metadata['coordinated'] > -1).dropna())
def join_all_text(papers):

    

    # Function joins body text from the research papers using the specified JSON format



    papers_text = list()

    

    i = 0

    

    for paper in papers:

        

        i += 1

        

        if paper == None:

            

            continue

        

        papers_text.append(' '.join(i['text'] for i in paper['body_text']))

        

    return papers_text
papers = papers_metadata.loc[papers_metadata['nonpharmaceutical'] > -1]

full_text = []



for i, paper in papers.iterrows():

    

    full_text.append(find_paper(paper))
tmp = multiple_most_used_words(join_all_text(full_text), 3)[:10]



x, y = map(list, zip(*tmp))

fig, axs = plt.subplots(1, 1)

plt.xticks(rotation='vertical')

axs.set_title("Most used three words in non-pharmaceutical paper abstracts")

axs.barh(x, y)



tmp.clear()

full_text.clear()
!pip install pytextrank



import spacy

import pytextrank

from math import sqrt

from operator import itemgetter





def getSummary(text, max_sentences, max_phrases, stopwords=[], verbose=False):

    

    # Based on https://github.com/DerwenAI/pytextrank/blob/master/explain_summ.ipynb

    

    # Function uses the Text Ranking algorithm to find the most relevant sentences in a text

    

    unit_vector = []



    # Initialize NLP to english language

    nlp = spacy.load("en_core_web_sm")

    textRank = pytextrank.TextRank()

    nlp.add_pipe(textRank.PipelineComponent, name="textrank", last=True)



    # Load text to process

    doc = nlp(text)



    # Get start and end indexes of sentences in text

    sentence_bounds = [[s.start, s.end, set([])] for s in doc.sents]

    

    # Classify max_phrases of relevant phrases to the sentence it belongs to



    phrase_id = 0



    for p in doc._.phrases:



        unit_vector.append(p.rank)

        isContinue = False

        

        # Do not count phrases with stopwords

        

        for word in stopwords:

            if word in p.text:

                isContinue = True

                

        if isContinue:

            continue

                



        for chunk in p.chunks:



            for sent_start, sent_end, sent_vector in sentence_bounds:



                if chunk.start >= sent_start and chunk.start <= sent_end:



                    sent_vector.add(phrase_id)

                    break

                    

        if(verbose):

            print(p.rank, p)



        phrase_id += 1



        if phrase_id == max_phrases:



                break

            

    

    # Normalize unit_vector values to sum 1

    sum_ranks = sum(unit_vector)

    unit_vector = [rank/sum_ranks for rank in unit_vector]



    sent_rank = {}

    sent_id = 0

    

    # Calculate euclidean distance for each sentence based on number of relevant phrases



    for sent_start, sent_end, sent_vector in sentence_bounds:



        sq_sum = 0.0



        for phrase_id in range(len(unit_vector)):



            if phrase_id not in sent_vector:



                sq_sum += unit_vector[phrase_id]**2.0



        sent_rank[sent_id] = sqrt(sq_sum)

        sent_id += 1

        

    # Get sentence text



    sent_text = {}

    sent_id = 0



    for sent in doc.sents:



        sent_text[sent_id] = sent.text

        sent_id += 1



    num_sent = 0

    summary_sentences = []

    

    # Print most relevant sentences according to model



    for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):



        summary_sentences.append((sent_id, sent_text[sent_id]))



        num_sent += 1



        if num_sent == max_sentences:



            break

    

    return sorted(summary_sentences, key=itemgetter(0))
tmp = papers_metadata.loc[papers_metadata['nonpharmaceutical'] > -1]

tmp = tmp.loc[tmp['intervention'] > -1]

tmp = tmp['abstract']

len(tmp)
abstract_summaries = []

abstract_indexes = []



for i, abstract in tmp.iteritems():



    for index, sentence in getSummary(abstract, 2, 4):

        

        if sentence in abstract_summaries:

            

            continue

        

        abstract_summaries.append(sentence)

        abstract_indexes.append(i)

        

abstract_summaries
print(len(abstract_summaries))

papers_summary = ' '.join(abstract_summaries)
selected_summaries = getSummary(papers_summary, 5, 8)

selected_summaries
selected_summary_indexes = []



for i, summary in selected_summaries:

    

    index = abstract_summaries.index(summary)

    selected_summary_indexes.append(abstract_indexes[index])
for i in selected_summary_indexes:

    

    print(papers_metadata['title'].iloc[i])
for i in list(set(selected_summary_indexes)):

    

    full_text = find_paper(papers_metadata.iloc[i])

    

    print(papers_metadata['title'].iloc[i], '\n')



    tmp_summary = getSummary(*join_all_text([full_text]), 10, 20, stopwords=['et al.'], verbose=False)

    

    for i, text in tmp_summary:

        print(text)

        

    print('\n\n')
tmp = papers_metadata.loc[papers_metadata['economic'] > -1]

tmp = tmp.loc[tmp['policy'] > -1]

tmp = tmp['abstract']



abstract_summaries = []

abstract_indexes = []



for i, abstract in tmp.iteritems():



    for index, sentence in getSummary(abstract, 2, 4):

        

        sentence = sentence.replace('Abstract ', '')

        

        if sentence in abstract_summaries:

            

            continue

        

        abstract_summaries.append(sentence)

        abstract_indexes.append(i)

        

    

        

papers_summary = ' '.join(abstract_summaries)

selected_summaries = getSummary(papers_summary, 5, 5)



selected_summary_indexes = []



for i, summary in selected_summaries:

    

    index = abstract_summaries.index(summary)

    selected_summary_indexes.append(abstract_indexes[index])

    

for i in list(set(selected_summary_indexes)):

    

    full_text = find_paper(papers_metadata.iloc[i])

    

    print(papers_metadata['title'].iloc[i], '\n')

    

    if full_text == None:

        

        continue



    tmp_summary = getSummary(*join_all_text([full_text]), max_sentences=10, max_phrases=10, stopwords=['et al.'], verbose=False)

    

    list_index = selected_summary_indexes.index(i)

    print(selected_summaries[list_index], '\n_____________')

    

    for i, text in tmp_summary:

        print(text)

        

    print('\n\n')
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
sentiment_data = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1")

sentiment_data
# Assign meaningful column names



sentiment_data.columns = ['Class', 'ID', 'Date', 'QUERY', 'Username', 'Text']

sentiment_data
# Disregard neutral classification



sentiment_data = sentiment_data.drop(sentiment_data.loc[sentiment_data['Class'] == 2].index, axis=0)
# Reduce size of data 10000 - 20000 large sample seems to be large enough



sentiment_data = sentiment_data.sample(frac=0.01).reset_index(drop=True)
# Obtain data and labels from dataframe



classifications = sentiment_data['Class'].tolist()

sentiment_x = sentiment_data['Text'].tolist()
# Tokenize abstracts



sentiment_x_tokens = [text.split() for text in sentiment_x]
# Encode data in one hot encoding



onehot_enc_data = MultiLabelBinarizer()

onehot_enc_data.fit(sentiment_x_tokens)
X_train, X_test, y_train, y_test = train_test_split(sentiment_x_tokens, classifications, test_size=0.1, random_state=None, shuffle=True)

validation = int(len(X_train)/9)

X_valid, y_valid = X_train[:validation], y_train[:validation]

X_train, y_train = X_train[validation:], y_train[validation:]
def labelBinaryEncoder(labels):

    

    # Function encodes labels in binary

    

    binary_labels = []

    

    for label in labels:

        

        if label == 0:            

            binary_labels.append([1,0])

            

        elif label == 4:            

            binary_labels.append([0,1])

            

        else:

            raise

            

    return binary_labels
def get_batch(X, y, batch_size):

    

    # Function gets batch of data

  

    for batch_pos in range(0,len(X),batch_size):

      

        yield X[batch_pos:batch_pos+batch_size], y[batch_pos:batch_pos+batch_size] 
tf.reset_default_graph()



# Create input layer



vocabulary_len = len(onehot_enc_data.classes_)

inputs_ = tf.placeholder(dtype=tf.float32, shape=[None, vocabulary_len], name='inputs')

targets_ = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='targets')



# Creating neural network



hidden1 = tf.layers.dense(inputs_, 200, activation=tf.nn.relu)

logits = tf.layers.dense(hidden1, 2, activation=None)

output = tf.nn.sigmoid(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets_))

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)



correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(targets_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
epochs_num = 10

batch = 2000



session = tf.Session()



session.run(tf.global_variables_initializer())



for epoch in range(epochs_num):

    

    for X_batch, y_batch in get_batch(onehot_enc_data.transform(X_train), labelBinaryEncoder(y_train), batch):

        

        loss_value, _ = session.run([loss, optimizer], feed_dict={

            inputs_: X_batch,

            targets_: y_batch

        })

        

        print("Epoch:", epoch, "Loss:", loss_value)

        

    acc = session.run(accuracy, feed_dict={

        inputs_ : onehot_enc_data.transform(X_valid),

        targets_ : labelBinaryEncoder(y_valid)

    })

    

    print("Epoch:", epoch, "Accuracy:", acc)

    

test_acc = session.run(accuracy, feed_dict={

    inputs_: onehot_enc_data.transform(X_test),

    targets_: labelBinaryEncoder(y_test)

})



print("Test Accuracy", test_acc)
tmp = papers_metadata.loc[papers_metadata['nonpharmaceutical'] > -1]

tmp = tmp['nonpharmaceutical'].dropna()

abstracts = []



for i, _ in tmp.iteritems():



    abstract = papers_metadata['abstract'].iloc[i]



    if type(abstract) != str:



        continue



    abstracts.append(abstract.split())



results = session.run(output, feed_dict={

    inputs_: onehot_enc_data.transform(abstracts)

})



#for i, result in enumerate(results):



    #print("Sentence:", ' '.join(abstracts[i]), "Results:", result)
negative_sent = sum([1 if np.where(i == np.amax(i))[0] == 0 else 0 for i in results])

positive_sent = sum([1 if np.where(i == np.amax(i))[0] == 1 else 0 for i in results])
objects = ('Negative', 'Positive')

y_pos = np.arange(len(objects))

values = [negative_sent, positive_sent]



plt.bar(y_pos, values, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Quantity')

plt.title('School closure sentiment')



plt.show()
tmp = papers_metadata.loc[papers_metadata['economic'] > -1]

tmp = tmp['economic'].dropna()

abstracts = []



for i, _ in tmp.iteritems():



    abstract = papers_metadata['abstract'].iloc[i]



    if type(abstract) != str:



        continue



    abstracts.append(abstract.split())



results = session.run(output, feed_dict={

    inputs_: onehot_enc_data.transform(abstracts)

})



#for i, result in enumerate(results):



    #print("Sentence:", ' '.join(abstracts[i]), "Results:", result)

    
negative_sent = sum([1 if np.where(i == np.amax(i))[0] == 0 else 0 for i in results])

positive_sent = sum([1 if np.where(i == np.amax(i))[0] == 1 else 0 for i in results])



objects = ('Negative', 'Positive')

y_pos = np.arange(len(objects))

values = [negative_sent, positive_sent]



plt.bar(y_pos, values, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Quantity')

plt.title('School closure sentiment')



plt.show()