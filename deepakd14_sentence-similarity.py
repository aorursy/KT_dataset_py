%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

import re
import pickle
from nltk.stem import PorterStemmer
import math

question = "what is machine learning"

actual_answer = "Machine learning is an application of artificial intelligence (AI) that provides\
systems the ability to automatically learn and improve from experience without being explicitly programmed. \
Machine learning focuses on the development of computer programs that can access data and use it learn for \
themselves. \ The process of learning begins with observations or data, such as examples, \
direct experience, or instruction, in order to look for patterns in data and make better \
decisions in the future based on the examples that we provide. The primary aim is to allow the computers\
learn automatically without human intervention or assistance and adjust actions accordingly."


## Below are the answers written by the students

highly_accurate_answer = "Machine learning (ML) is the study of computer algorithms that\
improve automatically through experience. It is seen as a subset of artificial intelligence.\
Machine learning algorithms build a mathematical model based on sample data, known as training data, \
in order to make predictions or decisions without being explicitly programmed to do so.\
Machine learning algorithms are used in a wide variety of applications, such as email filtering and \
computer vision, where it is difficult or infeasible to develop conventional algorithms to perform the \
needed tasks."


medium_accurate_answer = "At a very high level, machine learning is the process of teaching a computer system how to \
make accurate predictions when fed data. We can do a lot of things using this technique. It learns by itself. It is invented \
by father of the computer and since then it is very widely used. We can use it in many other feilds like medicine\
education etc. It will define the new mordern age tech. In future this is going to be one of the major catalyst in every feild be it\
be defence, education, communication."

very_poor_answer = "Adolf Hitler, byname Der Führer (German: “The Leader”), (born April 20, 1889, Braunau am Inn,\
Austria—died April 30, 1945, Berlin, Germany),\
leader of the Nazi Party (from 1920/21) and chancellor (Kanzler) and Führer of Germany (1933–45). \
He was chancellor from January 30, 1933, and, after President Paul von Hindenburg’s death, assumed \
the twin titles of Führer and chancellor (August 2, 1934)."
## preprocessed Data Variables

pre_question      = ""
pre_actual_answer = ""

## Below are the preprocessed answers written by the students

pre_highly_accurate_answer = ""
pre_medium_accurate_answer = ""
pre_very_poor_answer       = ""
# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]
## Defining the utility functions

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',str(text))


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


def remove_punctuation(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def final_preprocess(text):
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\n', ' ')
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = ' '.join(e for e in text.split() if e.lower() not in stopwords)
    text = text.lower()
    ps = PorterStemmer()
    text = ps.stem(text)
    return text
    
# Removing the URL

pre_question               = remove_URL(question)
pre_actual_answer          = remove_URL(actual_answer)
pre_highly_accurate_answer = remove_URL(highly_accurate_answer)
pre_medium_accurate_answer = remove_URL(medium_accurate_answer)
pre_very_poor_answer       = remove_URL(very_poor_answer)
# Removing the Emoji

pre_question               = remove_emoji(pre_question)
pre_actual_answer          = remove_emoji(pre_actual_answer)
pre_highly_accurate_answer = remove_emoji(pre_highly_accurate_answer)
pre_medium_accurate_answer = remove_emoji(pre_medium_accurate_answer)
pre_very_poor_answer       = remove_emoji(pre_very_poor_answer)
# Removing the Html tags

pre_question               = remove_html(pre_question)
pre_actual_answer          = remove_html(pre_actual_answer)
pre_highly_accurate_answer = remove_html(pre_highly_accurate_answer)
pre_medium_accurate_answer = remove_html(pre_medium_accurate_answer)
pre_very_poor_answer       = remove_html(pre_very_poor_answer)
# Removing the puntutations

pre_question               = remove_punctuation(pre_question)
pre_actual_answer          = remove_punctuation(pre_actual_answer)
pre_highly_accurate_answer = remove_punctuation(pre_highly_accurate_answer)
pre_medium_accurate_answer = remove_punctuation(pre_medium_accurate_answer)
pre_very_poor_answer       = remove_punctuation(pre_very_poor_answer)
# Decontracting the abbrevations

pre_question               = decontracted(pre_question)
pre_actual_answer          = decontracted(pre_actual_answer)
pre_highly_accurate_answer = decontracted(pre_highly_accurate_answer)
pre_medium_accurate_answer = decontracted(pre_medium_accurate_answer)
pre_very_poor_answer       = decontracted(pre_very_poor_answer)
#Doing the extra preprocessing 

pre_question               = final_preprocess(pre_question)
pre_actual_answer          = final_preprocess(pre_actual_answer)
pre_highly_accurate_answer = final_preprocess(pre_highly_accurate_answer)
pre_medium_accurate_answer = final_preprocess(pre_medium_accurate_answer)
pre_very_poor_answer       = final_preprocess(pre_very_poor_answer)
## Making a dictionary of the words and their vector representation

embeddings_index = {}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt')
for line in f:
    values = line.split(' ')
    word = values[0] ## The first entry is the word
    coefs = np.asarray(values[1:], dtype='float32') ## These are the vectors representing the embedding for the word
    embeddings_index[word] = coefs
f.close()


print('GloVe data loaded')
glove_words =  set(embeddings_index.keys())

'''
Below is a uliity function that takes sentenes as a input and return the vector representation of the same
Method adopted is similar to average word2vec. Where i am summing up all the vector representation of the words from the glove and 
then taking the average by dividing with the number of words involved
'''

def convert_sen_to_vec(sentence):
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence
    for word in sentence.split():
        if word in glove_words:
            vector += embeddings_index[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    return vector

# Now converting the text into vectors

question_vec               = convert_sen_to_vec(pre_question)
actual_answer_vec          = convert_sen_to_vec(pre_actual_answer)
highly_accurate_answer_vec = convert_sen_to_vec(pre_highly_accurate_answer)
medium_accurate_answer_vec = convert_sen_to_vec(pre_medium_accurate_answer)
poor_ans_vec               = convert_sen_to_vec(pre_very_poor_answer)
def square_rooted(x):
    return math.sqrt(sum([a*a for a in x]))


def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return numerator/float(denominator)
# Measuring the similarity using cosine similarity

sim11 = round(cosine_similarity(actual_answer_vec, highly_accurate_answer_vec),3)
sim12 = round(cosine_similarity(actual_answer_vec, medium_accurate_answer_vec),3)
sim13 = round(cosine_similarity(actual_answer_vec, poor_ans_vec),3)

print("The cosine similarity between actual answer and accurate answer given by the student is {}".format(sim11))
print("The cosine similarity between actual answer and medium accurate answer answer given by the student is {}".format(sim12))
print("The cosine similarity between actual answer and poor answer answer given by the student is {}".format(sim13))
# We will use the official tokenization script created by the Google team
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tokenization
'''
We are preparing out data in such a way that bert model can understand it

We have to give three sequences as input to the BERT

all_tokens : It basically performs the tokenization of the input sentences
all_masks  : This is done to make every input of the same length. We choose the maximum length of the vector and pad other vectors accordingly. We padd them 
             with the help of '0' which tells tells the model not to give attension to this token
segment Ids: This is used when we are giving multiple sentences as the input. Since we are only giving one sentence as the input we set the value of the 
             segment ids as 0 for all the tokens.
'''

def bert_encode(text, tokenizer, max_len=128):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    text = tokenizer.tokenize(text)

    text = text[:max_len-2]
    input_sequence = ["[CLS]"] + text + ["[SEP]"]
    pad_len = max_len - len(input_sequence)

    tokens = tokenizer.convert_tokens_to_ids(input_sequence)
    tokens += [0] * pad_len
    pad_masks = [1] * len(input_sequence) + [0] * pad_len
    segment_ids = [0] * max_len

    all_tokens.append(tokens)
    all_masks.append(pad_masks)
    all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
%%time

# We are importing the pretrained BERT parameters using TF-Hub
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2"
bert_layer = hub.KerasLayer(module_url, trainable=False)
#Setting up the tokenizer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
#Encoding the text using the our defined bert_encode function above

ques_en = bert_encode(question, tokenizer)
answ_en = bert_encode(actual_answer, tokenizer)

correct_answ_en       = bert_encode(highly_accurate_answer, tokenizer)
medium_corect_answ_en = bert_encode(medium_accurate_answer, tokenizer)
poor_answ_en          = bert_encode(very_poor_answer, tokenizer)
#Defination of the model

max_len = 128

input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
'''
Predection happens here. After prediction we will be able to extract the [CLS] output as the sentence embedding 
model.predict returns two parameters(pooled output and sequence output). Our [CLS] output is contained in the sequence output so we are 
assiging the '_vec' notation for the second parameter and later on we will extract the information using this 
'''

pooled_output, ques_vec = model.predict([ques_en[0], ques_en[1], ques_en[2]])
pooled_output, answ_vec = model.predict([answ_en[0], answ_en[1], answ_en[2]])

pooled_output, correct_answ_vec = model.predict([correct_answ_en[0], correct_answ_en[1], correct_answ_en[2]])
pooled_output, medium_corect_answ_vec = model.predict([medium_corect_answ_en[0], medium_corect_answ_en[1], medium_corect_answ_en[2]])
pooled_output, poor_answ_vec = model.predict([poor_answ_en[0], poor_answ_en[1], poor_answ_en[2]])
## Extracting the [CLS] output from the corresponding vectors

ques_vec = ques_vec[0,0,:]
answ_vec = answ_vec[0,0,:]
correct_answ_vec = correct_answ_vec[0,0,:]
medium_corect_answ_vec = medium_corect_answ_vec[0,0,:]
poor_answ_vec = poor_answ_vec[0,0,:]
# Measuring the similarity using cosine similarity

sim21 = round(cosine_similarity(answ_vec, correct_answ_vec),3)
sim22 = round(cosine_similarity(answ_vec, medium_corect_answ_vec),3)
sim23 = round(cosine_similarity(answ_vec, poor_answ_vec),3)

print("The cosine similarity between actual answer and accurate answer given by the student is {}".format(sim21))
print("The cosine similarity between actual answer and medium accurate answer answer given by the student is {}".format(sim22))
print("The cosine similarity between actual answer and poor answer answer given by the student is {}".format(sim23))
'''
Loading the pretrained model
Paper ref - https://arxiv.org/pdf/1803.11175.pdf
'''
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#Now using the embed method to convert evert text sentences into numnerical vectors

embeddings = embed([
    question,
    actual_answer,
    highly_accurate_answer,
    medium_accurate_answer,
    very_poor_answer])
# Measuring the similarity using cosine similarity

sim31 = round(cosine_similarity(np.array(embeddings[1]), np.array(embeddings[2])),3)
sim32 = round(cosine_similarity(np.array(embeddings[1]), np.array(embeddings[3])),3)
sim33 = round(cosine_similarity(np.array(embeddings[1]), np.array(embeddings[4])),3)

print("The cosine similarity between actual answer and accurate answer given by the student is {}".format(sim31))
print("The cosine similarity between actual answer and medium accurate answer answer given by the student is {}".format(sim32))
print("The cosine similarity between actual answer and poor answer answer given by the student is {}".format(sim33))
#Credits - https://matplotlib.org/examples/api/barchart_demo.html

"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""

N = 3
best_ans = (sim11, sim12, sim13)
med_ans = (sim21, sim22, sim23)
poor_ans = (sim31, sim32, sim33)

ind = np.arange(N)  # the x locations for the groups
width = 0.25     # the width of the bars

fig, ax = plt.subplots(figsize = (10, 5))

ax.set_ylim([0,1.3])

rects1 = ax.bar(ind, best_ans, width, color='r')
rects2 = ax.bar(ind + width, med_ans, width, color='g')
rects3 = ax.bar(ind + 2*width, poor_ans, width, color='b')


# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by Different Models')
ax.set_xticks(ind + 2*width / 2)
ax.set_xticklabels(('Best Answer', 'Medium Correct Ans', 'Poor Ans'))

ax.legend((rects1[0], rects2[0], rects3[0]), ('Glove Model', 'BERT Model', 'universal-sentence-encoder Model'))


plt.show()
