import numpy as np

import json

import os

from tqdm import tqdm

data_dir = '/kaggle/input/CORD-19-research-challenge'

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

import nltk

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()



import tensorflow_hub as hub

import matplotlib.pyplot as plt

module_url = "https://tfhub.dev/google/universal-sentence-encoder/1?tf-hub-format=compressed"

embed = hub.Module(module_url)

def get_json_data(folder):

    text  =  ""

    #title = ""

    data = []

    for txt in os.listdir(folder):

        if not txt.startswith('.') and txt in ['biorxiv_medrxiv','comm_use_subset','custom_license','noncomm_use_subset']:

            for filename in tqdm(os.listdir(f"{folder}/{txt}/{txt}")):

                if not filename.startswith('.'):

                    json_data =  json.load(open(f"{folder}/{txt}/{txt}/{filename}",'rb'))

                    for t in json_data['body_text']:            

                        text += t['text']+'\n\n'



    return text

txt_data = get_json_data(data_dir)
print('sentences containing vaccines or therapeutics ...')

doc = ""

for sentnece in txt_data.split('\n'):

    if('vaccines' in sentnece) or ('therapeutics' in sentnece):

        #word_tokens = word_tokenize(sentnece)

        doc +=sentnece 



print('Removing stopwords ...')



stop_words = set(stopwords.words('english'))

text = ""

for i in doc.split(' '):

    if i not in stop_words:

        text += ' ' + i.lower()

Corpus = []

print('Focusing on short sentences for visualization  ...')

for i in text.split(','):

    if ('vaccines' in i) or ('therapeutics' in i): 

        if len(i.split(' ')) < 15:

            Corpus.append(i)
messages2 = Corpus[:10]

similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))

similarity_message_encodings = embed(similarity_input_placeholder)

with tf.compat.v1.Session()  as session:

    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    message_embeddings_ = session.run(similarity_message_encodings, feed_dict={similarity_input_placeholder: messages2})



    corr = np.inner(message_embeddings_, message_embeddings_)

    print(corr)

    def heatmap(x_labels, y_labels, values):

        fig, ax = plt.subplots()

        im = ax.imshow(values)

        # We want to show all ticks...

        ax.set_xticks(np.arange(len(x_labels)))

        ax.set_yticks(np.arange(len(y_labels)))

        # ... and label them with the respective list entries

        ax.set_xticklabels(x_labels)

        ax.set_yticklabels(y_labels)

        # Rotate the tick labels and set their alignment.

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10,

             rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.

        for i in range(len(y_labels)):

            for j in range(len(x_labels)):

                text = ax.text(j, i, "%.2f"%values[i, j],

                               ha="center", va="center", color="w", fontsize=6)



        fig.tight_layout()

        plt.show()

    heatmap(messages2, messages2, corr)
