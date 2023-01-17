!pip install pyspark
import os

import json

import pyspark

from pyspark import SparkContext

from pyspark.sql import SparkSession

from pyspark.sql.types import *

import pandas as pd

import os

import nltk

import re

import spacy

from spacy.lang.fr.stop_words import STOP_WORDS

import string

from pyspark.sql.functions import lit

from pyspark.sql.functions import monotonically_increasing_id 

sparkSession = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

df = pd.read_csv('../input/insurance-reviews-france/Comments.csv')
df = df.drop(['Unnamed: 0'], axis=1)

schema = StructType([

    StructField("Name", StringType(), True),

    StructField("Comment", StringType(), True),

    StructField("Month", IntegerType(), True), 

    StructField("year", StringType(), True),

])
df_sp = sparkSession.createDataFrame(df,schema =schema )

df_sp.show()
df_sp = df_sp.filter(df_sp.Comment != 'NaN')
rdd_df = df_sp.rdd.zipWithIndex()

df_sp = rdd_df.toDF()

df_sp = df_sp.withColumn('Name', df_sp['_1'].getItem("Name")).withColumn('Comment', df_sp['_1'].getItem("Comment")).withColumn('Month', df_sp['_1'].getItem("Month")).withColumn('Year', df_sp['_1'].getItem("Year")).withColumn('Index', df_sp['_2'])

df_sp = df_sp.select('Index', 'Name','Comment','Month','Year')

df_sp.show(5)
comments_rdd = df_sp.select("Comment").rdd.flatMap(lambda x: x)
comments_rdd_lower = comments_rdd.map(lambda x : x.lower())

comments_rdd_lower.collect()
def sentence_tokenization(x):

    return nltk.sent_tokenize(x)

comments_rdd_tok = comments_rdd_lower.map(sentence_tokenization)

comments_rdd_tok.collect()
def word_TokenizeFunctSentence(x):

    sentence_splitted = []

    for line in x:

        splitted = []

        for word in re.sub("\W"," ", line).split():

            splitted.append(word)

        sentence_splitted.append(splitted)

    return sentence_splitted

comments_rdd_word_tok_sentence = comments_rdd_tok.map(word_TokenizeFunctSentence)

comments_rdd_word_tok_sentence.collect()
stop_words=set(STOP_WORDS)



deselect_stop_words = ['n\'', 'ne','pas','plus','personne','aucun','ni','aucune','rien']

for w in deselect_stop_words:

    if w in stop_words:

        stop_words.remove(w)

    else:

        continue
stop_words
def removeStopWordsSentencesFunct(x):

    sentence_stop=[]

    for j in x:

        fil=[]

        for w in j:

            if not ((w in stop_words) or (len(w) == 1)):

                fil.append(w)

        sentence_stop.append(' '.join(fil))

    return sentence_stop



stopwordRDDSen = comments_rdd_word_tok_sentence.map(removeStopWordsSentencesFunct)

stopwordRDDSen.collect()
def joinTokensFunct(x):

    joinedTokens_list = []

    x = " ".join(x)

    joinedTokens_list.append(re.sub("\W"," ", x))

    return joinedTokens_list

joinedTokens = stopwordRDDSen.map(joinTokensFunct)
joinedTokens.collect()
my_words = ["sécurité","prix", "sociale" , "remboursement" , "dentaire", "aide" , "pack" , "optique" , "soins" ,

"enfant","hospitalisation" , "handicap" , "document" , "retraite" , "carte" , "médicament" , "lunettes" ,

"appareil" , "changement" , "accident" , "intervention","garantie","augmentation","implant", "pharmacie" ,"attente", "formule" ,

"maternité" , "cotisation", "cpam" , "diabète", "auditif",

"commercial", "opticien" , "euros" , "retard" , "contrat", "prestation", "dossier" , "chirurgie" , "résiliation" ]
def TopicsSentences(x):

    topics =[]        

    topic =[]



    for i in x:

        for ext in my_words:

            if (ext in i):

                topic.append(ext)

    return topic

topics = stopwordRDDSen.map(TopicsSentences)

topics.collect()
comments_after_preproc = sparkSession.createDataFrame([w for w in joinedTokens.collect()], ['comments_after_preproc'])   

rdd_df2 = comments_after_preproc.rdd.zipWithIndex()

comments_after_preproc = rdd_df2.toDF()

comments_after_preproc = comments_after_preproc.withColumn('comments_after_preproc', comments_after_preproc['_1'])

comments_after_preproc = comments_after_preproc.withColumn('Index', comments_after_preproc['_2'])

comments_after_preproc = comments_after_preproc.select('Index', 'comments_after_preproc')





Topics = sparkSession.createDataFrame(topics,schema = "array<string>")    

topics_df = Topics.rdd.zipWithIndex()

Topics = topics_df.toDF()

Topics = Topics.withColumn('Topics', Topics['_1'])

Topics = Topics.withColumn('Index', Topics['_2'])

Topics = Topics.select('Index', 'Topics')

df_spark4 = df_sp.join(comments_after_preproc, on=['Index']).join(Topics, on=['Index'])

df_spark4.show(5)