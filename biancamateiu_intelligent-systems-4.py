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
data=pd.read_csv("/kaggle/input/intelligent-systems-3/mycsvfile.csv")

data.head()
import tensorflow as tf

import tensorflow_hub as hub



module_url='https://tfhub.dev/google/universal-sentence-encoder/2'

embed_module = hub.load(module_url)



def get_embed_content(content):

      

    tensor = tf.convert_to_tensor([content])

    

    embed = embed_module.signatures['default'](tensor)

    

    return embed['default'].numpy()[0]



from scipy import spatial



def find_similarity(section):

    return 1 - spatial.distance.cosine(query_embedded, np.fromstring(section[1:-1], dtype=np.float32, sep=' '))
analyze_data_dict = {"section_body": [], "embedding" : [], "similarity": []}

analyze_data = pd.DataFrame.from_dict(analyze_data_dict)



analyze_data['section_body'] = data['section_body']

analyze_data['embedding'] = data['embedding']



query = "What do we know about COVID-19 risk factors"



query_embedded = get_embed_content(query)



analyze_data['similarity'] = data['embedding'].apply(find_similarity)



analyze_data.sort_values(by=['similarity'], ascending=False, inplace=True)



final_data = analyze_data.head()



print('Query = ', query)



for index, row in final_data.iterrows():

    print(row['similarity'])

    print(row['section_body'])

    print()