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
import nltk
nltk.download('punkt') 
nltk.download('wordnet') 
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
text="""Mike and Morris lived in the same village. While Morris owned the largest jewelry shop in the village, Mike was a poor farmer. Both had large families with many sons, daughters-in-law and grandchildren.
One fine day, Mike, tired of not being able to feed his family, decided to leave the village and move to the city where he was certain to earn enough to feed everyone. Along with his family, he left the village for the city. At night, they stopped under a large tree. There was a stream running nearby where they could freshen up themselves. He told his sons to clear the area below the tree, he told his wife to fetch water and he instructed his daughters-in-law to make up the fire and started cutting wood from the tree himself. They didn’t know that in the branches of the tree, there was a thief hiding. He watched as Mike’s family worked together and also noticed that they had nothing to cook. Mike’s wife also thought the same and asked her husband ” Everything is ready but what shall we eat?”. Mike raised his hands to heaven and said ” Don’t worry.He is watching all of this from above. He will help us.”
The thief got worried as he had seen that the family was large and worked well together. Taking advantage of the fact that they did not know he was hiding in the branches, he decided to make a quick escape. 
He climbed down safely when they were not looking and ran for his life. But, he left behind the bundle of stolen jewels and money which dropped into Mike’s lap. Mike opened it and jumped with joy when he saw the contents. The family gathered all their belongings and returned to the village. There was great excitement when they told everyone how they got rich.
Morris thought that the tree was miraculous and this was a nice and quick way to earn some money. He ordered his family to pack some clothes and they set off as if on a journey. They also stopped under the same tree and Morris started commanding everyone as Mike had done. But no one in his family was willing to obey his orders. Being a rich family, they were used to having servants all around. So, the one who went to the river to fetch water enjoyed a nice bath. The one who went to get wood for fire went off to sleep. Morris’s wife said ” Everything is ready but what shall we eat ?” Morris raised his hands and said, ” Don’t worry. He is watching all of this from above. He will help us.”
As soon as he finished saying, the thief jumped down from the tree with a knife in hand. Seeing him, everyone started running around to save their lives. The thief stole everything they had and Morris and his family had to return to the village empty handed, having lost all their valuables that they had taken with them."""
sentences = sent_tokenize(text) 
total_documents = len(sentences)
sentences
def frequency_matrix(sentences):
    freq_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        freq_matrix[sent[:10]] = freq_table

    return freq_matrix
def term_frequency_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        no_of_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / no_of_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix
def documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table
def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix
def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix
def _score_sentences(tf_idf_matrix): 
    

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        total_no_of_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / total_no_of_words_in_sentence

    return sentenceValue
def _average_score(sentenceValue):
    
    
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    average = (sumValues / len(sentenceValue))

    return average
def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " \n" + sentence
            sentence_count += 1

    return summary
#1 create the sentences by using sent_tokenize
sentences = sent_tokenize(text) 
total_documents = len(sentences)

import math
# 2 Create the Frequency matrix of the words in each sentence.
freq_matrix = frequency_matrix(sentences)
print(freq_matrix)
# 3 Calculate TermFrequency and generate a matrix
tf_matrix = term_frequency_matrix(freq_matrix)
print(tf_matrix)
# 4 creating table for documents per words
count_doc_per_words = documents_per_words(freq_matrix)
print(count_doc_per_words)
# 5 Calculate IDF and generate a matrix
idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
print(idf_matrix)
# 6 Calculate TF-IDF and generate a matrix
tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
print(tf_idf_matrix)
# 7 Important Algorithm: score the sentences
sentence_scores = _score_sentences(tf_idf_matrix)
print(sentence_scores)
# 8 Find the threshold
threshold = _average_score(sentence_scores)
print(threshold)
# 9 Important Algorithm: Generate the summary
summary = _generate_summary(sentences,sentence_scores,threshold)
print(summary)

