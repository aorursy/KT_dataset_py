# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


labels =  ["Where is my Phone","I Cant find my phone","My Phone is lost","This phone is made by apple",
           "This is a brand new phone","How much does this phone cost","What is the price of this phone",
          "Can you give me your phone","How Can I unlock this phone","Will this phone break","How durable is this phone"]
embeddings = embed(labels)

print(embeddings)
import seaborn as sns
import numpy as np

corr = tf.tensordot(embeddings,tf.transpose(embeddings), 1)

sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
import pandas as pd
data = pd.read_csv('../input/flipkart-products/flipkart_com-ecommerce_sample.csv')
name_list = data['product_name'].unique()
name_embeddings = embed(name_list) 
name_embeddings.shape
inner_prod = tf.tensordot(name_embeddings, tf.transpose(name_embeddings), 1)
import random
item = name_list[random.randint(0,len(name_list))]
item
item_index = np.where(name_list == item)[0][0]
temp = inner_prod[item_index]
max_values = tf.math.top_k(temp,k = 6)

name_list[[max_values.indices]]
class ContextRecomenderEngine:
    
    products = []
    embeddings = []
    path = ""
    embed = []
    inner_prod = []
    name_embeddings = []
    index = 0
    
    def __init__(self,embedding_tf_hub_path):
        self.path = embedding_tf_hub_path
    
    def fit(self,products):
        self.products = products
        self.embed = hub.load(self.path)
        self.name_embeddings = self.embed(products)
        self.inner_prod = tf.tensordot(name_embeddings, tf.transpose(name_embeddings), 1)
        print("Similarity Matrix size is {}".format(inner_prod.shape))
        
    def transform(self,item,top_k):
        self.index = np.where(self.products == item)[0][0]
        temp = self.inner_prod[self.index]
        max_values = tf.math.top_k(temp,k = top_k+1)
        return self.products[[max_values.indices]][1:]
        
engine = ContextRecomenderEngine("https://tfhub.dev/google/universal-sentence-encoder-large/5")
engine.fit(name_list)
item = name_list[random.randint(0,len(name_list))]
print(item)
engine.transform(item,50)