import pandas

import functools

import math

import re

pandas.set_option('display.max_rows', 500)

pandas.set_option('display.max_columns', 500)

pandas.set_option('display.width', 1000)

pandas.set_option("max_column", None)
corpus = "I am a little confused on all of the models of the 88-89 bonnevilles. I have heard of the LE SE LSE SSE SSEI. Could someone tell me the differences are far as features or performance. I am also curious toknow what the book value is for prefereably the 89 model. And how much less than book value can you usually get them for. In other words how much are they in demand this time of year. I have heard that the mid-spring early summer is the best time to buy."

# corpus = str(input("Enter Your Coupus: "))

# print("\n\n")

# while not re.search(r'([A-Z .-/-\&/\,])\w*', corpus):

#     corpus = str(input("Enter Your Coupus: "))

print(corpus)
corpus_list = corpus.split(".")

for item in corpus_list:

    index = corpus_list.index(item)

    corpus_list[index] = corpus_list[index].strip()

    if item == "":

        corpus_list.remove("")
print(corpus_list)
# Making a dictionary where the key will be a variable named bag_Of_words_{n}

# And the value will be BAG OF WORDS made from each item in corpus_list



dictionary_of_bag_of_words = {}  # bowA



for item in corpus_list:

    dictionary_of_bag_of_words[str(corpus_list.index(item))] = item.split(" ")
print(dictionary_of_bag_of_words)

for key in dictionary_of_bag_of_words:

    print(f"Key:: {key}\t Value:  {dictionary_of_bag_of_words[key]}")
# Function for union set

# def make_UNION(set1, set2):

#     set1 = set(set1)

#     set2 = set(set2)

#     return set1.union(set2)



# Later replaced by lambda xpression

# lambda set1, set2: set(set1).union(set(set2))



all_word_set = functools.reduce(lambda set1, set2: set(set1).union(set(set2)),

                                list(dictionary_of_bag_of_words.values()))

print(all_word_set, len(all_word_set))



print(list(dictionary_of_bag_of_words.values()))
dictionary_of_word_set = {}  # wordDictA

for item in corpus_list:

    dictionary_of_word_set[str(corpus_list.index(item))] = dict.fromkeys(

        all_word_set, 0)
for key in dictionary_of_word_set:

    print(f"Key:: {key}\t Value:  {dictionary_of_word_set[key]}")
for bow_as_key in dictionary_of_bag_of_words:

    iterable = dictionary_of_bag_of_words[bow_as_key]

    for item in iterable:

        x = dictionary_of_word_set[str(bow_as_key)]

        x[str(item)] += 1
for key in dictionary_of_word_set:

    print(f"Key:: {key}\t\nValue:  {dictionary_of_word_set[key]}\n\n")
list_of_dictionary_of_word_set = list(dictionary_of_word_set.values())

# print(list_of_dictionary_of_word_set)

data_drame = pandas.DataFrame(list_of_dictionary_of_word_set)
data_drame
def computeTF(dictionary_of_word, bag_Of_words):

    tf_dictionary_to_return = {}

    length_of_doc = len(bag_Of_words)

    for word, count in dictionary_of_word.items():

        tf_dictionary_to_return[word] = count / float(length_of_doc)



    return tf_dictionary_to_return





def computerIDF(_corpus_list):

    idf_dictionary_to_return = dict.fromkeys(_corpus_list[0].keys(), 0)

    N = len(_corpus_list)

    for document_dict in _corpus_list:

        for word, val in document_dict.items():

            if val > 0:

                idf_dictionary_to_return[word] += 1



    # Final processing of idf_dictionary_to_return

    for word, val in idf_dictionary_to_return.items():

        idf_dictionary_to_return[word] = math.log(N / float(val))



    return idf_dictionary_to_return





def calculateTF_IDF(tf_dict, idf_dict):

    tfidf_to_return = {}

    for word, val in tf_dict.items():

        tfidf_to_return[word] = val * idf_dict[word]



    return tfidf_to_return
dictionary_of_word_set_for_tf = dictionary_of_word_set.copy()
for key, val in dictionary_of_word_set.items():

    x = computeTF(dictionary_of_word_set[key], dictionary_of_bag_of_words[key])

    dictionary_of_word_set_for_tf[key] = x
for key in dictionary_of_word_set_for_tf:

    print(f"Key:: {key}\t\n Value:  {dictionary_of_word_set_for_tf[key]}\n\n")
dictionary_of_word_set_list = []

for key, val in dictionary_of_word_set.items():

    dictionary_of_word_set_list.append(dictionary_of_word_set[key])
for key in dictionary_of_word_set_list:

    print(key)
idfs = computerIDF(dictionary_of_word_set_list)

tf_idf_dictionary = dict.fromkeys(dictionary_of_word_set_for_tf.keys(), 0)

for key, val in dictionary_of_word_set_for_tf.items():

    tf_idf_dictionary[key] = calculateTF_IDF(

        dictionary_of_word_set_for_tf[key], idfs)
for key in tf_idf_dictionary:

    print(f"Key:: {key}\t\n Value:  {tf_idf_dictionary[key]}\n\n")
tf_idf_list = []

for key, val in tf_idf_dictionary.items():

    tf_idf_list.append(tf_idf_dictionary[key])
data_drame_tf_idf = pandas.DataFrame(tf_idf_list)

print("TF-IDF Data Frame\n==================\n")

data_drame_tf_idf