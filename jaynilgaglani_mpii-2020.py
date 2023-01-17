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
import pandas as pd

# df = pd.read_csv("news_17_18_19_20.csv")

df = pd.read_csv("/kaggle/input/news_17_18_19_20_21_22.csv")

df.drop("Unnamed: 0",axis=1,inplace=True)

df.head()
from gensim.summarization import summarize

from gensim.summarization import keywords
doc = """Central Railway 

Press Release



Trains cancelled due to #COVID19 preventive measure and non-occupancy



Central Railway has cancelled trains to contain the effects of corona virus and non-occupancy of trains.  The details are as under:



1)	11007 Mumbai-Pune Deccan Express from 19.3.2020 to 31.3.2020

2)	11008 Pune-Mumbai Deccan Express from 18.3.2020 to 30.3.2020

3)	11201 LTT-Ajni Express on 23.3.2020 and 30.3.2020

4)	11202 Ajni-LTT Express on 20.3.2020 and 27.3.2020

5)	11205 LTT-Nizamabad Express on 21.3.2020 and 28.3.2020

6)	11206 Nizamabad-LTT Express on 22.3.2020 and 29.3.2020

7)	22135/22136 Nagpur-Rewa Express on 25.3.2020

8)	11401 Mumbai-Nagpur Nandigram Express from 23.3.2020 to 1.4.2020

9)	11402 Nagpur-Mumbai Nandigram Express from 22.3.2020 to 31.3.2020

10)	11417 Pune-Nagpur Express on 26.3.2020 and 2.4.2020

11)	11418 Nagpur-Pune Express on 20.3.2020 and 27.3.2020

12)	22139 Pune-Ajni Express on 21.3.2020 and 28.3.2020

13)	22140 Ajni-Pune Express on 22.3.2020 and 29.3.2020

14)	12117/12118 LTT-Manmad Express from 18.3.2020 to 31.3.2020

15)	12125 Mumbai-Pune Pragati Express from 18.3.2020 to 31.3.2020

16)	12126 Pune-Mumbai Pragati Express from 19.3.2020 to 1.4.2020

17)	22111 Bhusaval-Nagpur Express from 18.3.2020 to 29.3.2020

18)	22112 Nagpur-Bhusaval Express from 19.3.2020 to 30.3.2020

19)	11307/11308 Kalaburagi-Secunderabad Express from 18.3.2020 to 31.3.2020

20)	12262 Howrah-Mumbai Duranto Express on 24.3.2020 and 31.3.2020

21)	12261 Mumbai-Howrah Duranto Express on 25.3.2020 and 1.4.2020

22)	22221 CSMT-Nizamuddin Rajdhani Express on 20, 23, 27 and 30.3.2020

23)	22222 Nizamuddin-CSMT Rajdhani Express on 21, 24, 26 and 31.3.2020

--- ---

Date: March 17, 2020

2020/03/29

This press release is issued by Public Relations Department, Central Railway, Chhatrapati Shivaji Maharaj Terminus Mumbai"""



summary = summarize(doc)

key = keywords(doc).split("\n")

print(summary)

print(key)

# key
a = []

count = []

for i in range(len(df)):

  count = 0

  for j in key:

    count += df.Keywords.iloc[i].count(j)

  if(count>1):

    a.append(i)

result_df = pd.DataFrame(columns=df.columns)

print(a)

for i in set(a):

  result_df = result_df.append(df.iloc[i])
import tensorflow_hub as hub

import tensorflow as tf



elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

x = ["Roasted ants are a popular snack in Columbia"]



# Extract ELMo features 

messages = [doc]

message_embeddings = elmo([doc], signature="default", as_dict=True)["elmo"]



message_embeddings.shape
sentences = list(result_df.Article)



sentence_embeddings = elmo(sentences, signature="default", as_dict=True)["elmo"]

sentence_embeddings.shape
message_embeddings = None

with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())

  sess.run(tf.tables_initializer())

  message_embeddings = sess.run(message_embeddings)
sentence_embeddings = None

with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())

  sess.run(tf.tables_initializer())

  sentence_embeddings = sess.run(sentence_embeddings)
for query, query_embedding in zip(messages, message_embeddings.numpy_function()):

    # distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

    distances = cosine_distances([query_embedding], sentence_embeddings_)[0]

    results = zip(range(len(distances)), distances)

    results = sorted(results, key=lambda x: x[1])



    print("\n\n======================\n\n")

    # print("Query:", query)

    print("\nTop 3 most similar sentences in corpus:")



    final = {}

    for idx, distance in results[0:5]:

        final[sentences[idx]] = float(1-distance)

        print(sentences[idx].strip(), "\n(Cosine Score: %.4f)" % (1-distance))



scores = list(final.values())

if(round(np.mean(scores)*100,2)>70):

  print("The Article is True")

else:

  print("The Article is False")