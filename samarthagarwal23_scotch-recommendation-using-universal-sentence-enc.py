# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import re

import tensorflow as tf

import tensorflow_hub as tfhub

pd.set_option('display.max_colwidth', -1)



embed = tfhub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
reviews = pd.read_csv("../input/scotch_review.csv")

print(reviews.shape)

reviews.head(3)
# Embed whisky reviews



with tf.Session() as session:

  session.run([tf.global_variables_initializer(), tf.tables_initializer()])

  scotch_embeddings = session.run(embed(list(reviews.description)))

print(scotch_embeddings.shape)
def user_query_recommend(query, price_range, embedding_table = scotch_embeddings):

    # Embed user query

    with tf.Session() as session:

        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        embedding = session.run(embed([query]))



    # Calculate similarity with all reviews

    similarity_metrics = np.dot(embedding, embedding_table.T)

    

    # Recommend

    recommendations = reviews.copy()

    recommendations['price'] =recommendations.price.apply(lambda x: re.findall("\d+", x.replace(",","").replace(".00","").replace("$",""))[0]).astype('int')

    recommendations['sim'] = similarity_metrics.T

    recommendations = recommendations.sort_values('sim', ascending=False)

    recommendations = recommendations.loc[(recommendations.price >= price_range[0]) &

                                          (recommendations.price <= price_range[1]),

                                          ['name', 'category', 'price', 'description', 'sim']]



    return recommendations
user_query1 = "citrus and woody spice hint, along with a touch of vanilla, honey, and even popcorn"

price_range1 = [20, 60]



recommendation1 = user_query_recommend(user_query1, price_range1)

print(user_query1)

print("Price range - {}".format(price_range1))

recommendation1.head(5).T
user_query2 = "rich fruit and spice. hints of chocolate and lemon. smooth"

price_range2 = [30, 200]



recommendation2 = user_query_recommend(user_query2, price_range2)

print(user_query2)

print("Price range - {}".format(price_range2))

recommendation2.head(5).T
recommendation2.tail(5).T