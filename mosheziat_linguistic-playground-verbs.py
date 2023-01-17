# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import spacy

from spacy import displacy

import operator

import collections

# Any results you write to the current directory are saved as output.
import csv

def load_dict_from_csv(path_name):

    main_dict = {}

    labels = set()

    file = open(path_name, "rU")

    reader = csv.reader(file, delimiter=',')

    headers = next(reader)

    for row in reader:

        verb = row[1]

        complement = row[2]

        num_of_appearances = int(row[3])

        #print(verb, complement, num_of_appearances)

        if verb in main_dict:

            main_dict[verb][complement] = num_of_appearances

        else:

            main_dict[verb] = {complement: num_of_appearances}

        labels.add(complement)

    return main_dict, labels
#main_dict_path = ''

main_dict_path = '../input/english-verbs-complement-distribution/verbs2m.csv'

names_dict_path = '../input/verbs-complements-semantic-distribution/verbs2m2.csv'



main_dict, labels = load_dict_from_csv(main_dict_path)

names_dict, names_labels = load_dict_from_csv(names_dict_path)

    

print(len(main_dict))

print(len(labels))



print(len(names_dict))

print(len(names_labels))
minimun_verb_appearances = 20

minimum_complement_appearances = 10



def filter_dict(dictionary):

    filtered_dict = {}

    for verb in dictionary:

        verb_appearnaces = dictionary[verb]['COUNT']

        if verb_appearnaces > minimun_verb_appearances:

            new_verb_dict = {'COUNT': verb_appearnaces}

            for complement in dictionary[verb]:

                if dictionary[verb][complement] > minimum_complement_appearances:

                    new_verb_dict[complement] = dictionary[verb][complement]

            filtered_dict[verb] = new_verb_dict

    return filtered_dict



filtered_main_dict = filter_dict(main_dict)

filtered_names_dict = filter_dict(names_dict)



print(len(main_dict))

print(len(filtered_main_dict))



print(len(names_dict))

print(len(filtered_names_dict))
def create_vectors(filtered_dict):

    verb_vectors = []

    index_to_vector_verb = {}

    verb_to_index = {}

    index = 0

    for verb in filtered_dict:

        index_to_vector_verb[index] = verb

        verb_to_index[verb] = index

        verb_values = []

        verb_count = filtered_dict[verb]['COUNT']

        for label in labels:

            if label in filtered_dict[verb]:

                verb_values.append(filtered_dict[verb][label] / verb_count)

            else:

                verb_values.append(0)

        verb_vectors.append(verb_values)

        index = index + 1

    return verb_vectors, index_to_vector_verb, verb_to_index

    

main_dict_vectors, main_index_to_vector, main_verb_to_index = create_vectors(filtered_main_dict)

names_dict_vectors, names_index_to_vector, names_verb_to_index = create_vectors(filtered_names_dict)



verb_main_vectors_df = pd.DataFrame(main_dict_vectors)

print(verb_main_vectors_df.shape)

print(len(main_index_to_vector))



verb_names_vectors_df = pd.DataFrame(names_dict_vectors)

print(verb_names_vectors_df.shape)

print(len(names_index_to_vector))
from sklearn.cluster import KMeans



n = round(len(verb_main_vectors_df) / 20)

kmeans = KMeans(n_clusters=n).fit(verb_main_vectors_df)



cluster_map = pd.DataFrame()

cluster_map['data_index'] = verb_main_vectors_df.index.values

cluster_map['cluster'] = kmeans.labels_
similar_verbs_dict = {}



for i in range(n):

    similar_verbs_list = []

    relevant_cluster = cluster_map[cluster_map.cluster == i]

    for ind in relevant_cluster.index.values:

        verb = main_index_to_vector[ind]

        similar_verbs_list.append(verb)

    for v in similar_verbs_list:

        similar_verbs_dict[v] = [x for x in similar_verbs_list if x != v]

print(similar_verbs_dict['convince'])
def print_verb_complements(verb):

    verb_dict = filtered_main_dict[verb]

    ordered_dict = collections.OrderedDict(reversed(sorted(verb_dict.items(),

                                           key=operator.itemgetter(1))))

    print("===========" + verb + "===============")

    for x,y in ordered_dict.items():

        print(x, '===>', y)



def get_word_vector(verb):

    relevant_index = main_verb_to_index[verb]

    return verb_main_vectors_df.iloc[relevant_index]



def get_verb_details(verb):

    print_verb_complements(verb)

    print(get_word_vector(verb))





print_verb_complements("encourage")

print_verb_complements("persuade")
def get_dict_distinct(dictio):

    for verb in dictio:

        count = dictio[verb]['COUNT']

        if count > 100:

            verb_dict = dictio[verb]

            for name in verb_dict:

                if name == 'COUNT':

                    continue

                val = verb_dict[name]

                if val / count > 0.90:

                    print(verb, name)

        

print(get_dict_distinct(names_dict))
import numpy as np

def subtract_verb_vectors(v1, v2):

    index1 = main_verb_to_index[v1]

    index2 = main_verb_to_index[v2]

    return np.subtract(main_dict_vectors[index1], main_dict_vectors[index2]) 



def get_verbs_diff(v1, v2):

    res = []

    s = subtract_verb_vectors(v1, v2)

    for i, label in enumerate(labels):

        if abs(s[i]) > 0.2:

            higher = v1

            if s[i] < 0:

                higher = v2

            res.append((higher, label, s[i]))

    return res
s = get_verbs_diff("persuade", "enable")

print(s)
def find_prefix_pairs(prefix):

    res = []

    for verb in filtered_main_dict:

        new_verb = prefix + verb

        if new_verb in filtered_main_dict:

            res.append((verb, new_verb)) 

    return res

        

def find_prefix_charachter(prefix):

    res = []

    pairs = find_prefix_pairs(prefix)

    vecs = []

    for v1, v2 in pairs:

        vecs.append(subtract_verb_vectors(v1, v2))

    average_vec = np.median(vecs, axis=0)

    for i, label in enumerate(labels):

        if abs(average_vec[i]) > 0.1:

            res.append((label, average_vec[i]))

    return len(vecs), res



print(find_prefix_pairs("re"))
prefixes = ["re", "en", "dis", "de", "un",  "out", "co", "under", "pre", "mis", "sub"]



for prefix in prefixes:

    instances_count, res = find_prefix_charachter(prefix)

    if len(res) > 0:

        print(prefix + ": ", instances_count)

        print("======================")

        for line in res:

            print(line[0] + ": ", line[1])

        print(" ")

print(len(labels))

print(len(main_dict_vectors[0]))