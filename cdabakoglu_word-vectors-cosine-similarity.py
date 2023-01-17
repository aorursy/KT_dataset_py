import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
with open("../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt") as file:

    data = file.readlines()
len(data)
for i in range(len(data)):

    data[i] = data[i][:-1]
data_dict = dict()



for i in range(len(data)):

    split_data = data[i].split()

    data_dict[split_data[0]] = np.array(split_data[1:]).astype('float64')
data_dict["the"]
def cosine_similarity(a, b):

    nominator = np.dot(a, b)

    

    a_norm = np.sqrt(np.sum(a**2))

    b_norm = np.sqrt(np.sum(b**2))

    

    denominator = a_norm * b_norm

    

    cosine_similarity = nominator / denominator

    

    return cosine_similarity
table = data_dict["table"]

desk = data_dict["desk"]

football = data_dict["football"]

baseball = data_dict["baseball"]

water = data_dict["water"]

fire = data_dict["fire"]

computer = data_dict["computer"]

calculator = data_dict["calculator"]

number = data_dict["number"]

math = data_dict["math"]

boy = data_dict["boy"]

girl = data_dict["girl"]

sad = data_dict["sad"]

happy = data_dict["happy"]

good = data_dict["good"]

bad = data_dict["bad"]

turkey = data_dict["turkey"]

television = data_dict["television"]

awesome = data_dict["awesome"]

great = data_dict["great"]

coffee = data_dict["coffee"]

giraffe = data_dict["giraffe"]

cat = data_dict["cat"]

barcelona = data_dict["barcelona"]

school = data_dict["school"]

disaster = data_dict["disaster"]



print(f"Cosine similarity for pair (table, desk) = {cosine_similarity(table, desk)}")

print(f"Cosine similarity for pair (football, baseball) = {cosine_similarity(football, baseball)}")

print(f"Cosine similarity for pair (water, fire) = {cosine_similarity(water, fire)}")

print(f"Cosine similarity for pair (computer, calculator) = {cosine_similarity(computer, calculator)}")

print(f"Cosine similarity for pair (number, math) = {cosine_similarity(number, math)}")

print(f"Cosine similarity for pair (boy, girl) = {cosine_similarity(boy, girl)}")

print(f"Cosine similarity for pair (sad, happy) = {cosine_similarity(sad, happy)}")

print(f"Cosine similarity for pair (good, bad) = {cosine_similarity(good, bad)}")

print(f"Cosine similarity for pair (turkey, television) = {cosine_similarity(turkey, television)}")

print(f"Cosine similarity for pair (awesome, great) = {cosine_similarity(awesome, great)}")

print(f"Cosine similarity for pair (coffee, giraffe) = {cosine_similarity(coffee, giraffe)}")

print(f"Cosine similarity for pair (cat, barcelona) = {cosine_similarity(cat, barcelona)}")

print(f"Cosine similarity for pair (school, disaster) = {cosine_similarity(school, disaster)}")
def find_word(a, b, c, data_dict):

    a, b, c = a.lower(), b.lower(), c.lower()

    a_vector, b_vector, c_vector = data_dict[a], data_dict[b], data_dict[c]

    

    all_words = data_dict.keys()

    max_cosine_similarity = -1000

    best_match_word = None

    

    for word in all_words:

        if word in [a, b, c]:

            continue

            

        cos_sim = cosine_similarity(np.subtract(b_vector, a_vector), np.subtract(data_dict[word], c_vector))

        

        if cos_sim > max_cosine_similarity:

            max_cosine_similarity = cos_sim

            best_match_word = word

            

    return best_match_word, cos_sim
words_bag = [

    ('boy', 'girl', 'man'),

    ('bat', 'baseball', 'ball'),

    ('book', 'library', 'coffee'),

    ('orange', 'juice', 'apple'),

    ('turkey', 'turkish', 'colombia')

]



for words in words_bag:

    d, cos_sim = find_word(*words, data_dict)

    print("({}, {}) ----> ({}, {}) with {} difference".format(*words, d, cos_sim))