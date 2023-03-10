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
inputs = [

({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'},     False),

({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'},    False),

({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'},      True),

({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'},   True),

({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'},       True),

({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'},      False),

({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'},         True),

({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'},   False),

({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'},       True),

({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'},  True),

({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),

({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'},     True),

({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'},       True),

({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'},  False)

]



from collections import defaultdict

from collections import Counter

from functools import partial

import math



def entropy(class_probabilities):

    """dada uma lista de probabilidades de classe, compute a entropia"""

    return sum(-p * math.log(p, 2)

               for p in class_probabilities if p) # ignora probabilidades zero



def class_probabilities(labels):

    total_count = len(labels)

    return [count / total_count

          for count in Counter(labels).values()]



def data_entropy(labeled_data):

    labels = [label for _, label in labeled_data]

    probabilities = class_probabilities(labels)

    return entropy(probabilities)



def partition_entropy(subsets):

    """encontre a entropia desta divis??o de dados em subconjuntos

    subconjunto ?? uma lista de listas de dados rotulados"""

    total_count = sum(len(subset) for subset in subsets)

    return sum( data_entropy(subset) * len(subset) / total_count

               for subset in subsets )



def partition_by(inputs, attribute):

    """cada entrada ?? um par (attribute_dict, label).

    retorna uma dict: attribute_value ->inputs"""

    groups = defaultdict(list)

    for input in inputs:

          key = input[0][attribute]  # pega o valor do atributo especificado

          groups[key].append(input)  # ent??o adiciona essa entrada ?? lista correta

    return groups



def partition_entropy_by(inputs, attribute):

    """computa a entropia correspondente ?? parti????o dada"""

    partitions = partition_by(inputs, attribute)

    return partition_entropy(partitions.values())



for key in ['level','lang','tweets','phd']:

    print (key, partition_entropy_by(inputs, key))
senior_inputs = [(input, label)

                 for input, label in inputs if input["level"] == "Senior"]



for key in ['lang', 'tweets', 'phd']:

    print (key, partition_entropy_by(senior_inputs, key))
def classify(tree, input):

    """classifica a entrada usando a ??rvore de decis??o fornecida"""

    # se for um n?? folha, retorna seu valor

    if tree in [True, False]:

        return tree

    # sen??o, esta ??rvore consiste de uma caracter??stica para dividir

    # e um dicion??rio cujas chaves s??o valores daquela caracter??stica

    # e cujos valores s??o sub-??rvores para considerar depois

    attribute, subtree_dict = tree

    subtree_key = input.get(attribute)   # None se estiver faltando caracter??stica

    if subtree_key not in subtree_dict:  # se n??o h?? sub-??rvore para chave,

        subtree_key = None            # usaremos a sub-??rvore None

    subtree = subtree_dict[subtree_key]  # escolha a sub-??rvore apropriada

    return classify(subtree, input)     # e use para classificar a entrada



def build_tree_id3(inputs, split_candidates=None):

    # se este ?? nosso primeiro passo,

    # todas as chaves da primeira entrada s??o candidatos divididos

    if split_candidates is None:

        split_candidates = inputs[0][0].keys()

    # conta Trues e Falses nas entradas

    num_inputs = len(inputs)

    num_trues = len([label for item, label in inputs if label])

    num_falses = num_inputs - num_trues

    if num_trues == 0: return False      # nenhum True? Retorne uma folha ???False???

    if num_falses == 0: return True      # nenhum False? Retorne uma folha ???True???

    if not split_candidates:             # se n??o houver mais candidatos a dividir

        return num_trues >= num_falses   # retorne a folha majorit??ria

    # sen??o, divida com base na melhor caracter??stica

    best_attribute = min(split_candidates,

                         key=partial(partition_entropy_by, inputs))

    partitions = partition_by(inputs, best_attribute)

    new_candidates = [a for a in split_candidates

                      if a != best_attribute]

    # recursivamente constr??i as sub-??rvores

    subtrees = { attribute_value : build_tree_id3(subset, new_candidates)

                for attribute_value, subset in partitions.items() }

    subtrees[None] = num_trues > num_falses     # caso padr??o

    return (best_attribute, subtrees)



tree = build_tree_id3(inputs)
classify(tree, { "level" : "Junior",

                "lang" : "Java",

                "tweets" : "yes",

                "phd" : "no"} )             # True
classify(tree, { "level" : "Junior",

                "lang" : "Java",

                "tweets" : "yes",

                "phd" : "yes"} )           # False
classify(tree, { "level" : "Intern" } ) # True
classify(tree, { "level" : "Senior" } ) # False