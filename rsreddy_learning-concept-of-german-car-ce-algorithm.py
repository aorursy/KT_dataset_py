# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
cars_data = {'origin': ['Germany', 'Germany', 'USA', 'Japan', 'Germany'], 
        'manufacturer': ['Volkswagen', 'BMW', 'Ford', 'Toyota', 'Volkswagen'], 
        'color': ['blue', 'black', 'red', 'silver', 'grey'], 
        'year': [2017, 2018, 2019, 2017, 2013],
        'model': ['Golf R', 'X7', 'Mustang GT', 'Camry', 'Toureg'],
        'german_made': ['Y', 'Y','N','N','Y']}
cars_example = pd.DataFrame(cars_data, columns = ['origin', 'manufacturer', 'color', 'year', 'model', 'german_made'])
# df.to_csv("../input/cars.csv", index=False, encoding='utf8')
book_data = {'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'], 
        'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'], 
        'Humidity': ['Normal', 'High', 'High', 'High'], 
        'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
        'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
        'Forecast': ['Same', 'Same','Same','Change'],
        'EnjoySport': ['Y','Y','N','Y']}
book_example = pd.DataFrame(book_data, columns = ['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])
shape_data = {'Size':['big','small','small','big','small'],
        'Color': ['red','red','red','blue','blue'],
        'Shape': ['circle','triangle', 'circle', 'circle', 'circle'],
        'isSmallCircle':['N','N','Y','N','Y']}
shape_example = pd.DataFrame(shape_data, columns = ['Size', 'Color', 'Shape', 'isSmallCircle'])
df = cars_example
df

# init G to the set of maximally general hypotheses in H
def g_0(n):
    return ("?",)*n

# init S to the set of maximally specific hypotheses in H
def s_0(n):
    return ('0',)*n  
def more_general(h1, h2):
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == "?" or (x != "0" and (x == y or y == "0"))
        more_general_parts.append(mg)
    return all(more_general_parts)

l1 = [1, 2, 3]
l2 = [3, 4, 5]

list(zip(l1, l2))
# min_generalizations
def fulfills(example, hypothesis):
    ### the implementation is the same as for hypotheses:
    return more_general(hypothesis, example)

def min_generalizations(h, x):
    h_new = list(h)
    for i in range(len(h)):
        if not fulfills(x[i:i+1], h[i:i+1]):
            h_new[i] = '?' if h[i] != '0' else x[i]
    return [tuple(h_new)]
def min_specializations(h, domains, x):
    results = []
    for i in range(len(h)):
        if h[i] == "?":
            for val in domains[i]:
                if x[i] != val:
                    h_new = h[:i] + (val,) + h[i+1:]
                    results.append(h_new)
        elif h[i] != "0":
            h_new = h[:i] + ('0',) + h[i+1:]
            results.append(h_new)
    return results
min_specializations(h=('?', 'x',), 
                    domains=[['a', 'b', 'c'], ['x', 'y']], 
                    x=('b', 'x'))
examples=[]

for row in df.iterrows():
    index, data = row
    examples.append(data.tolist())
    
examples

def get_domains(examples):
    d = [set() for i in examples[0]]
    for x in examples:
        for i, xi in enumerate(x):
            d[i].add(xi)
    return [list(sorted(x)) for x in d]

get_domains(examples)
def candidate_elimination(examples):
    domains = get_domains(examples)[:-1]
    
    G = set([g_0(len(domains))])
    S = set([s_0(len(domains))])
    i=0
    print("\n G[{0}]:".format(i),G)
    print("\n S[{0}]:".format(i),S)
    for xcx in examples:
        i=i+1
        x, cx = xcx[:-1], xcx[-1]  # Splitting data into attributes and decisions
        if cx=='Y': # x is positive example
            G = {g for g in G if fulfills(x, g)}
            S = generalize_S(x, G, S)
        else: # x is negative example
            S = {s for s in S if not fulfills(x, s)}
            G = specialize_G(x, domains, G, S)
        print("\n G[{0}]:".format(i),G)
        print("\n S[{0}]:".format(i),S)
    return 
def generalize_S(x, G, S):
    S_prev = list(S)
    for s in S_prev:
        if s not in S:
            continue
        if not fulfills(x, s):
            S.remove(s)
            Splus = min_generalizations(s, x)
            ## keep only generalizations that have a counterpart in G
            S.update([h for h in Splus if any([more_general(g,h) 
                                               for g in G])])
            ## remove hypotheses less specific than any other in S
            S.difference_update([h for h in S if 
                                 any([more_general(h, h1) 
                                      for h1 in S if h != h1])])
    return S
def specialize_G(x, domains, G, S):
    G_prev = list(G)
    for g in G_prev:
        if g not in G:
            continue
        if fulfills(x, g):
            G.remove(g)
            Gminus = min_specializations(g, domains, x)
            ## keep only specializations that have a conuterpart in S
            G.update([h for h in Gminus if any([more_general(h, s)
                                                for s in S])])
            ## remove hypotheses less general than any other in G
            G.difference_update([h for h in G if 
                                 any([more_general(g1, h) 
                                      for g1 in G if h != g1])])
    return G
candidate_elimination(examples)
