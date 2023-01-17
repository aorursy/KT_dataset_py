import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))

import numpy as np

def read_glove_vecs(glove_file):

    with open(glove_file, 'r') as f:

        words = set()

        word_to_vec_map = {}

        

        for line in f:

            line = line.strip().split()

            curr_word = line[0]

            words.add(curr_word)

            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

            

    return words, word_to_vec_map

                                    

    

words, word_to_vec_map=read_glove_vecs("../input/glove.6B.50d.txt")

    

    
def cosine_similarity(u, v):

    distance = 0.0

    dot = np.dot(u,v)

    norm_u =np.sqrt(np.sum(u * u))

    norm_v =np.sqrt(np.sum(v * v))

    cosine_similarity = dot/(norm_u*norm_v)

    return cosine_similarity
def word_to_vector(word):

    vector=word_to_vec_map[word]

    return vector
a=cosine_similarity(word_to_vec_map["italy"]-word_to_vec_map["italian"],word_to_vec_map["france"]-word_to_vec_map["french"])

print(a)
def complete_analogy(word_a,word_b,word_c,word_to_vec_map):

    word_a, word_b, word_c=word_a.lower(),word_b.lower(),word_c.lower()

    e_a,e_b,e_c=word_to_vec_map[word_a],word_to_vec_map[word_b],word_to_vec_map[word_c]

    words = word_to_vec_map.keys()

    my_word=None

    max_cosine=-100

    for w in words:

        if w in [word_a, word_b, word_c] :

            continue

        cosine=cosine_similarity(e_b-e_a,word_to_vec_map[w]-e_c)

        if cosine > max_cosine:

            max_cosine=cosine

            my_word=w

    return my_word

            
#word1=input("Enter 1st word: ")

word1="India"

#word2=input("Enter 2st word: ")

word2="Delhi"

#word3=input("Enter 3st word: ")

word3="Pakistan"

print("If {0} -> {1}, then {2} -> {3}".format(word1,word2,word3,complete_analogy(word1,word2,word3,word_to_vec_map)))