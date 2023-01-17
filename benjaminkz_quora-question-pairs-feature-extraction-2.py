import numpy as np

import pandas as pd

import os

import warnings

warnings.filterwarnings("ignore")

from nltk import word_tokenize
f = open("../input/glove840b300dtxt/glove.840B.300d.txt", encoding="utf-8")

embeddings_index = {}

for line in f:

    values = line.split()

    word = "".join(values[:-300])   

    coefs = np.asarray(values[-300:], dtype="float32")

    embeddings_index[word] = coefs

f.close()

print("Found {} word vectors of glove.".format(len(embeddings_index)))
train_orig = pd.read_csv("../input/quora-question-pairs-data-cleaning/train_orig.csv")

train_stop = pd.read_csv("../input/quora-question-pairs-data-cleaning/train_stop.csv")



train_orig.fillna("", inplace = True)

train_stop.fillna("", inplace = True)



train = pd.read_csv("../input/quora-question-pairs-feature-extraction-1/train.csv")

trainlabel = pd.read_csv("../input/quora-question-pairs-feature-extraction-1/trainlabel.csv")
def get_word_vector(row):

    wordlist1 = word_tokenize(row["question1"])

    wordlist2 = word_tokenize(row["question2"])

    

    rarity = 0  # 用于标记问题对是否含有非常罕见的词的特征

    

    embeddings_list1 = []

    for string in wordlist1:

        try:

            embeddings_list1.append(embeddings_index[string])

        except KeyError:

            if string in wordlist2:  # 如果两个问题包含这个词，令rarity=1，否则不进行处理

                rarity = 1

            else:

                pass          

    

    embeddings_list2 = []

    for string in wordlist2:

        try:

            embeddings_list2.append(embeddings_index[string])

        except KeyError:

            if string in wordlist1:

                rarity = 1

            else:

                pass  

    

    return pd.Series([embeddings_list1, embeddings_list2, rarity])
vector_orig = train_orig.apply(get_word_vector, axis = 1)

vector_orig.columns = ["question1", "question2", "rarity"]
def diff_word_vector(row):

    mean1 = np.mean(np.array(row["question1"]), axis = 0)

    mean2 = np.mean(np.array(row["question2"]), axis = 0)

    diff = mean1 - mean2

    L1 = np.sum(np.abs(diff))  # np.linalg.norm不能对空的数组计算1-范数和2-范数，所以只能手动计算

    L2 = np.sum(diff ** 2) ** 0.5

    norm1 = np.sum(mean1 ** 2) ** 0.5

    norm2 = np.sum(mean2 ** 2) ** 0.5

    cos = np.sum(mean1 * mean2) / (norm1 * norm2)

    return pd.Series([L1, L2, cos])
diff_vector_orig = vector_orig.apply(diff_word_vector, axis = 1)

features_vector_orig = pd.concat([diff_vector_orig, vector_orig["rarity"]], axis = 1)

features_vector_orig.columns = ["diff_word_vector_L1_orig", "diff_word_vector_L2_orig", 

                                "word_vector_cos_orig", "varity"]

train = pd.concat([train, features_vector_orig], axis = 1)



del vector_orig, diff_vector_orig, features_vector_orig
vector_stop = train_stop.apply(get_word_vector, axis = 1)

vector_stop.columns = ["question1", "question2", "rarity"]



diff_vector_stop = vector_stop.apply(diff_word_vector, axis = 1)

diff_vector_stop.columns = ["diff_word_vector_L1_stop", "diff_word_vector_L2_stop", 

                            "word_vector_cos_stop"]

train = pd.concat([train, diff_vector_stop], axis = 1)



del vector_stop, diff_vector_stop
train.to_csv("train.csv", index = False)

trainlabel.to_csv("trainlabel.csv", index = False)



del train, trainlabel, train_orig, train_stop
test_orig = pd.read_csv("../input/quora-question-pairs-data-cleaning/test_orig.csv")

test_orig.fillna("", inplace = True)



vector_orig = test_orig.apply(get_word_vector, axis = 1)

vector_orig.columns = ["question1", "question2", "rarity"]

del test_orig



diff_vector_orig = vector_orig.apply(diff_word_vector, axis = 1)

features_vector_orig = pd.concat([diff_vector_orig, vector_orig["rarity"]], axis = 1)

features_vector_orig.columns = ["diff_word_vector_L1_orig", "diff_word_vector_L2_orig", 

                                "word_vector_cos_orig", "varity"]

del vector_orig, diff_vector_orig
test_stop = pd.read_csv("../input/quora-question-pairs-data-cleaning/test_stop.csv")

test_stop.fillna("", inplace = True)



vector_stop = test_stop.apply(get_word_vector, axis = 1)

vector_stop.columns = ["question1", "question2", "rarity"]

del test_stop, embeddings_index



diff_vector_stop = vector_stop.apply(diff_word_vector, axis = 1)

diff_vector_stop.columns = ["diff_word_vector_L1_stop", "diff_word_vector_L2_stop", 

                            "word_vector_cos_stop"]

del vector_stop



test = pd.read_csv("../input/quora-question-pairs-feature-extraction-1/test.csv")

test = pd.concat([test, features_vector_orig, diff_vector_stop], axis = 1)

del features_vector_orig, diff_vector_stop
test.to_csv("test.csv", index = False)