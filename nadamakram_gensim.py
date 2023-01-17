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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]
#tag each document
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
tagged_data
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")
from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("I love chatbots".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
from gensim.similarities import Similarity
index_tmpfile = get_tmpfile("index")
batch_of_documents = common_corpus[:]  # only as example
index = Similarity(index_tmpfile, common_corpus, num_features=len(common_dictionary))  # build the index
# the batch is simply an iterable of documents, aka gensim corpus:
for similarities in index[batch_of_documents]:
     pass
batch_of_documents
import os
import gensim
# Set file names for train and test data
test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
lee_train_file = os.path.join(test_data_dir, 'lee_background.cor')
lee_test_file = os.path.join(test_data_dir, 'lee.cor')
lee_train_file
import smart_open

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))
print(train_corpus[:2])
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
print(vector)
#questions on task
riskFactorsQuery = "What do we know about potential risks factors?\
what is the effect of Smoking, pre-existing pulmonary disease?\
Do co-existing respiratory/viral infections make the virus more transmissible or virulent and other comorbidities?\
What is the effect on Neonates and pregnant women?\
What are the Socio-economic and behavioral factors on COVID-19?\
What is the economic impact of the virus?"

incubationPeriodQuery="What is known about incubation?\
How long is the incubation period in patients for covid,sars and mers virus in days?\
Range of incubation periods for the disease in humans ?\
How the incubation period varies across age, health status?\
How long individuals are contagious, even after recovery?"

TransmissionDynamicsQuery = "What are Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors?"
SeverityofDiseaseQuery = "Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups"

questions = [riskFactorsQuery, incubationPeriodQuery]
   
    
#tokens = word_tokenize(questions)
#tokens = [word_tokenize(i) for i in questions]
#for listt in tokens:
 #   print(listt)
questionVector = model.infer_vector(word_tokenize(riskFactorsQuery))
print(questionVector)
similar_doc = model.docvecs.most_similar([questionVector],topn=60) #The docvecs property of the Doc2Vec model holds all trained vectors for the 'document tags' seen during training
similar_doc