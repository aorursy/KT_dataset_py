# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from sklearn.cluster import KMeans

from collections import defaultdict

 

import io

    

def file_preprocess(file_name, nwords):

  '''

  function to readtext file and process it to create data for clustering

  

  RETURNS: numpy array of vector dimension and list of words

  '''

  np_arrays = []

  wordlist = []

  with io.open(file_name, mode='r', encoding='utf-8') as f:

    #print("len" , len(f.readlines()))

    for index, line in enumerate(f):

      tokens = line.split()

      wordlist.append(tokens[0])

      np_arrays.append( np.array([float(i) for i in tokens[1:]]) )



      if index == nwords:

        return np.array( np_arrays ), wordlist



  return np.array( np_arrays ), wordlist





def assign_word2cluster(word_list, cluster_labels):

  '''

  RETURNS: dict {"cluster":[words  assigend to cluster]}

  '''

  cluster_to_words = defaultdict(list)

  for index, cluster in enumerate(cluster_labels):

    cluster_to_words[ cluster ].append( word_list[index] )

  return cluster_to_words





if __name__ == "__main__":

    



  cluster_data_file = "/kaggle/input/clusterin_data.txt" 



  #Number of words to analyse according to memory availability

  #n_words = 30000 # Number of words to analyse according to memory availability

  n_words = 20000

  reduction_factor =.1  # Amount of dimension reduction {0,1}

  n_clusters = int( n_words * reduction_factor ) # Number of clusters to make

  cluster_data, wordlist = file_preprocess(cluster_data_file, nwords = n_words)

  

  kmeans_model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=15,random_state=1,max_iter=500,verbose=1)

  kmeans_model.fit(cluster_data)



  cluster_labels  = kmeans_model.labels_ #returns all cluster number assigned to each word respectively

  cluster_to_words  = assign_word2cluster(wordlist, cluster_labels)



  with io.open("/kaggle/working/output.txt",mode='w+',encoding="UTF-8") as file:

    for key in sorted(cluster_to_words.keys()) :

        file.writelines("Cluster "+str(key) +" :: "+ "|".join( k for k in cluster_to_words[key])+"\n")

        print("Cluster "+str(key) , " :: " , "|".join( k for k in cluster_to_words[key]))

    file.close()