import tensorflow as tf

#from tensorflow.contrib.tensorboard.plugins import projector



%load_ext tensorboard

tensorboard_callback = tf.keras.callbacks.TensorBoard("projections")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import string

import re

import nltk

import warnings

warnings.filterwarnings('ignore')

dataset = pd.read_csv("../input/fraud-email-dataset/fraud_email_.csv")

dataset.head()
from nltk.corpus import stopwords

import string

import re



oneSetOfStopWords = set(stopwords.words('english')+['``',"''",'...','nbsp','br','/div','div'])



def CleanText(givenText):

    reqText = givenText.lower()

    reqText = re.sub(r"=2e", "", reqText)

    reqText = re.sub(r"=2c", "", reqText)

    reqText = re.sub(r"\=", "", reqText)

    reqText = re.sub(r"news.website.http\:\/.*\/.*502503.stm.", "", reqText)

    reqText = re.sub(r"http://www.forcetacticalarmy.com","",reqText)

    reqText = re.sub(r"\'s", " ", reqText)

    reqText = re.sub(r"\'", " ", reqText)

    reqText = re.sub(r":", " ", reqText)

    reqText = re.sub(r"_", " ", reqText)

    reqText = re.sub(r"-", " ", reqText)

    reqText = re.sub(r"\'ve", " have ", reqText)

    reqText = re.sub(r"can't", "can not ", reqText)

    reqText = re.sub(r"n't", " not ", reqText)

    reqText = re.sub(r"i'm", "i am ", reqText)

    reqText = re.sub(r"\'re", " are ", reqText)

    reqText = re.sub(r"\'d", " would ", reqText)

    reqText = re.sub(r"\d", "", reqText)

    reqText = re.sub(r"\b[a-zA-Z]\b","", reqText)

    reqText = re.sub(r"[\,|\.|\&|\;|<|>]","", reqText)

    reqText = re.sub(r"\S*@\S*", " ", reqText)

    reqText = reqText.replace('_','')

    sentenceWords = []

    requiredWords = nltk.word_tokenize(reqText)

    for word in requiredWords:

        if word not in oneSetOfStopWords and word not in string.punctuation:

            sentenceWords.append(word)

    reqText = " ".join(sentenceWords)     

    return reqText

print (dataset.shape)

dataset = dataset[dataset['Text'].notnull()]

print (dataset.shape)
%%time

newDataset = dataset[dataset['Text'].notnull()][:1000]

newDataset['cleaned_text'] = newDataset.Text.apply(lambda x: CleanText(x))

newDataset.head()
sentences = newDataset['cleaned_text'].values

labels = newDataset['Class'].values

reqSentences = [row.split(" ") for row in sentences]
import gensim

model = gensim.models.Word2Vec(

    reqSentences,

    size=150,

    window=5,

    min_count=1,

    workers=10,

    iter=10)
model.save('word2vec.model')

#model = gensim.models.Word2Vec.load_word2vec_format("word2vec.model", binary=True)
vocabSize = len(model.wv.vocab) - 1

print (vocabSize)

print (model.layer1_size)

tempArray =np.zeros((vocabSize,model.layer1_size))
!rm -rf projections

!mkdir projections
with open ("projections/metadata.tsv" , "w+") as fh:

    for i,word in enumerate(model.wv.index2word[:vocabSize]):

        tempArray[i] = model.wv[word]

        fh.write(word+"\n")

        

    
from tensorboard.plugins import projector

tf.compat.v1.disable_eager_execution()

session=tf.compat.v1.InteractiveSession()

embedding = tf.Variable(tempArray, trainable=False, name="embedding")

session.run(tf.compat.v1.global_variables_initializer())

saver = tf.compat.v1.train.Saver()

writer = tf.compat.v1.summary.FileWriter("projections", session.graph)

config = projector.ProjectorConfig()

embed = config.embeddings.add()



embed.tensor_name = "embedding"

embed.metadata_path="projections/metadata.tsv"



projector.visualize_embeddings(writer, config)

saver.save(session, "projections/model.ckpt", global_step=vocabSize)
%tensorboard --logdir='projections'
from tensorflow.examples.tutorials.mnist import input_data