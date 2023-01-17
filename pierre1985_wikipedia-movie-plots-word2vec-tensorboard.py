import os
import sys
import nltk
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from gensim.models import word2vec
from nltk.corpus import stopwords 
from tensorflow.contrib.tensorboard.plugins import projector

tqdm.pandas()
trainFile = "../input/wiki_movie_plots_deduped.csv"

pwd = os.getcwd()
os.chdir(os.path.dirname(trainFile))
df = pd.read_csv(os.path.basename(trainFile))
os.chdir(pwd)
print("Nombre de lignes : {0}".format(len(df)))
dfAnalyze = df.copy()
dfAnalyze.head()
hist = dfAnalyze.plot.hist()
columns = ['Release Year', 'Director','Cast', 'Genre', 'Wiki Page', 'Plot']
dfPie = dfAnalyze.drop(columns, axis=1)

dfPie = dfPie.groupby(['Origin/Ethnicity']).count().rename(columns= {'Title':'count'})
pie = dfPie.plot.pie(subplots=True, figsize=(7, 7))
df = df.head(2000)
print("Nombre de lignes sélectionnées : {0}".format(len(df)))
def tokenize_without_stop_words(text):
    sentence = nltk.word_tokenize(text.lower())
    sentence = [w for w in sentence if not w in stopwords.words('english')]
    sentence = [w for w in sentence if len(w) > 2 ]
    return sentence

sentences = df.progress_apply(lambda row: tokenize_without_stop_words(row['Plot']), axis=1)
sentences = sentences.tolist()
model = word2vec.Word2Vec(sentences, min_count=50, workers=4)
print(model)

# model = word2vec.Word2Vec(sentences, min_count=750, workers=4)
# Result Full Dataset : Word2Vec(vocab=1686, size=100, alpha=0.025)
#for entry in sorted(model.wv.vocab):
#    print(entry)
print(model.wv.most_similar(['suicide'], topn=5))
# Result Full Dataset : [('murder', 0.54288649559021), ('failed', 0.47224748134613037), ('rape', 0.4682120680809021), 
# ('murders', 0.4570953845977783), ('death', 0.3709148168563843)]
print(model.wv.most_similar(['war'], topn=5))
# Result Full Dataset : [('u.s.', 0.597042441368103), ('army', 0.5714285969734192), ('china', 0.55967777967453), 
# ('union', 0.558294951915741), ('japan', 0.5436923503875732)]
def transform_word2vec_to_tensor(model, output_path):
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), 100))

    with open(os.path.join(output_path,meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replaced by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # Correction : purge les anciens nodes/graphes
    tf.reset_default_graph()
    
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable = False, name = 'w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path,'w2x_metadata.ckpt'))

directory = './output/'
if not os.path.exists(directory):
    os.makedirs(directory)

# Pour la visualisation Kaggle, j'ai mis en commentaire transform_word2vec_to_tensor(model, directory). Il faut donc dé-commenter la ligne ci-dessous.
#transform_word2vec_to_tensor(model, directory)