# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#imported all libraries required

import numpy as np 

import pandas as pd 

import os

from tqdm import tqdm_notebook

import tensorflow as tf

import tensorflow_hub as hub

from nltk import sent_tokenize

%matplotlib inline
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
df = pd.read_csv('/kaggle/input/Text_Similarity_Dataset.csv')

df.head()
# Before dropping plot duplicates

len(df)
#cleaning text, removing unwanted keywords

import re

def clean_plot(text_list):

    clean_list = []

    for sent in text_list:

        sent = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-.:;<=>?@[\]^`{|}~"""), '',sent)

        sent = sent.replace('[]','')

        sent = re.sub('\d+',' ',sent)

        sent = sent.lower()

        clean_list.append(sent)

    return clean_list
#here instead of using word encodins we need to use sentence encoding as we need a meaning full sentence 

#for example dog bites man and man bites dog are similar in textual context but differ in semantic or logical

#So, we need to encodings which can understand differnece between the pairs of sentences and thier logic or meaning.

X=[]

with tf.Graph().as_default():

    #loading model

    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    print("embed",embed)

    messages = tf.placeholder(dtype=tf.string, shape=[None])

    output = embed(messages)

    #staring tensorflow session

    with tf.Session() as session:

        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        for text in ["text1","text2"]:

            phrases_emb_list = []

            for phrases in tqdm_notebook(df[text]):

                #breaking paragraph into enstences  

                sent_list = sent_tokenize(phrases)

                #cleaning the sentence

                clean_sent_list = clean_plot(sent_list)

                #feeding sentence in the encodings

                sent_embed = session.run(output, feed_dict={messages: clean_sent_list})

                #appending the encodings in a list

                phrases_emb_list.append(sent_embed.mean(axis=0).reshape(1,512))

            X.append(phrases_emb_list)

import pickle
#saving the encodings for future use in pickle file

pickle_out = open("X_new.pkl","wb")

pickle.dump(X, pickle_out)

pickle_out.close()
#converting the encodings in numpy array  and checking the sahpe of encodings

t1=np.array(X[0])
t2=np.array(X[1])
t1.shape
t2.shape
#converting data to numpy aaray

X_data=np.array(X)
import math
#creating cosine function 

def get_cosine(vec1, vec2):

     

     numerator = sum(vec1 * vec2 )



     sum1 = sum([vec1[x]**2 for x in range(len(vec1))])

     sum2 = sum([vec2[x]**2 for x in range(len(vec2))])

     denominator = math.sqrt(sum1) * math.sqrt(sum2)



     if not denominator:

        return 0.0

     else:

        return float(numerator) / denominator
df["similarity_score"]=df["Unique_ID"]
#getting similarity score  for dataset

for i in range(len(df)):

    cosine=get_cosine(t1[i][0],t2[i][0])

    df["similarity_score"]= cosine
#saving the pandas data frame to csv

df.to_csv('file1.csv') 
pickle_out = open("data.pkl","wb")

pickle.dump(df, pickle_out)

pickle_out.close()