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
import os

import pprint

import numpy as np

import pandas as pd

from collections import OrderedDict

import random

import networkx as nx

import matplotlib.pyplot as plt

from matplotlib.font_manager import weight_dict
#make a word list of common adjectives in data-set



words_dict = {"Uncurable":-1,"love":1,"good":1,"awesome":1,"nice":1,"good quality":1,"classic":1,"pretty":1,"seasoned":1,"lovely":1,"privileged":1,"attentive":1,"friendly":1,"modern":1,"exceptional":1,"enthusiastic":1,"famous":1,"prompt":1,"special":1,"unbelievable":1,"courteous":1,"delightful":1,"efficient":1,"inexpensive":1,"great":1,"pleasant":1,"fresh":1,"cool":1,"refresh":1,"positive":1,"beautiful":1,"wonderful":1,"perfect":1,"best":1,"amazing":1,"excellent":1,"impressive":1,"impressed":1,"pleased":1,"overwhelmed":1,"negative":-1,"mean":-1,"bad":-1,"sad":-1,"poor":-1,"frustrated":-1,"low":-1,"worse":-1,"worst":-1,"horrible":-1,"cheap":-1,"ridiculous":-1,"overpriced":-1,"costly":-1,"pneumatic":-1,"strange":-1,"unprofessional":-1,"nasty":-1,"late":-1,"low quality":-1,"bad quality":-1,"disappointed":-1,"disappointing":-1,"angry":-1}



#extract common adjectives from the different data sets 



def get_adjectives(text):

    blob = TextBlob(text)

    adjectives = list()

    for word, tag in blob.tags:

        if tag == 'JJ':

            adjectives.append(word.lower())

    return set(adjectives)
#get the unique attribute from is dataset and map them based on similarity and buid a bipartite sentiment graph



def get_unique_attributes(file):

    result = {}

    unique_attr = {}



    with open(file,'r') as file:



        for i, line in enumerate(file):



            result[i] = set()

            words = line.split()



            for w, word  in enumerate (map (lambda word :  word.lower().replace(".",""), words)):

                if word in words_dict:

                    if 'no' == words[w-1].lower() or 'not' == words[w-1].lower():

                        if 'not ' + word in unique_attr:

                            unique_attr['not ' + word] +=  -1* words_dict[word]

                        else:

                            unique_attr['not ' + word] =  -1* words_dict[word]

                    else:

                        if word in unique_attr:

                            unique_attr[word]+= words_dict[word]

                        else:

                            unique_attr[word]= words_dict[word]

    return unique_attr
def PercentageMissin(Dataset):

    #"""this function will return the percentage of missing values in a dataset """

    if isinstance(Dataset,pd.DataFrame):

        adict={} #a dictionary conatin keys columns names and values percentage of missin value in the columns

        for col in Dataset.columns:

            adict[col]=(np.count_nonzero(Dataset[col].isnull())*100)/len(Dataset[col])



        return pd.DataFrame(adict,index=['% of missing'],columns=adict.keys())

    else:

        raise TypeError("can only be used with panda dataframe")



if __name__ == "__main__":



    file1 = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/25621281691205eb015383cbac839182b838514f.json"

    file2 = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/7db22f7f81977109d493a0edf8ed75562648e839.json"

    file3 = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/6c3e1a43f0e199876d4bd9ff787e1911fd5cfaa6.json"

    file4 = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/2ce201c2ba233a562ee605a9aa12d2719cfa2beb.json"

    file5 = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/b460e5b511b4e2c3233f9476cd4e0616d6f405ac.json"



    files = [file1,file2,file3,file4,file5]

    fnames = []



    file_results = OrderedDict()

    all_unique_attributes = set()



    for f in files:

        unique_attributes = get_unique_attributes(f)

        file_results[os.path.basename(f)] = unique_attributes
print('##################################### File wise attributes - start #################')

pp = pprint.PrettyPrinter()

pp.pprint(file_results)



print('##################################### File wise attributes - end #################')



for res in file_results.values():

    all_unique_attributes.update(res)

print('')









print('##################################### Unique attributes amongst all reviews #################')

pp.pprint(all_unique_attributes)



    
B = nx.Graph()

B.add_nodes_from(file_results.keys(), bipartite=0)

B.add_nodes_from(all_unique_attributes, bipartite=1)



all_edges = []

for file, weight_dict in file_results.items():

    for attribute, weight in weight_dict.items():

        all_edges.append((file,attribute,weight))

        B.add_weighted_edges_from(all_edges)



    print(B.edges(data=True))



   
pos = {node:[0, i] for i,node in enumerate(file_results.keys())}

pos.update({node:[1, i] for i,node in enumerate(all_unique_attributes)})

nx.draw(B, pos, with_labels=False)

for p in pos:  # raise text positions

    pos[p][1] += 0.25

    nx.draw_networkx_labels(B, pos)



    plt.show()