# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import tensorflow as tf

import tensorflow_hub as hub



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/project-is-2-0-1/mycsvfile.csv")

data.head()
data_embedded = pd.read_csv("/kaggle/input/project-is-2-1/mycsvfile.csv")

data_embedded.head()
data.shape
data_embedded.shape
is_info = data['grammar_label'] == 'INFORMATION'

data_info = data[is_info]

data_info.shape
is_gibberish = data['grammar_label'] == 'GIBBERISH'

data_gibberish = data[is_gibberish]

data_gibberish.shape
indexNames = data[ data['grammar_label'] != 'INFORMATION' ].index

 

# Delete these row indexes from dataFrame

data.drop(indexNames , inplace=True)



data.shape
data_merged = pd.merge(data_embedded, data, on = ['paper_id', 'section_id', 'section_body', 'tag_label'])

data_merged.head()
data_merged.shape




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



analyze_data['section_body'] = data_merged['section_body']

analyze_data['embedding'] = data_merged['embedding']



query = "What is known about transmission, incubation, and environmental stability"



query_embedded = get_embed_content(query)



analyze_data['similarity'] = data_merged['embedding'].apply(find_similarity)



analyze_data.sort_values(by=['similarity'], ascending=False, inplace=True)



final_data = analyze_data.head()



print('Query 1 = ', query)



for index, row in final_data.iterrows():

    print(row['similarity'])

    print(row['section_body'])

    print()
query = "What do we know about COVID-19 risk factors"



query_embedded = get_embed_content(query)



analyze_data['similarity'] = data_merged['embedding'].apply(find_similarity)



analyze_data.sort_values(by=['similarity'], ascending=False, inplace=True)



final_data = analyze_data.head()



print('Query 2 = ', query)



for index, row in final_data.iterrows():

    print(row['similarity'])

    print(row['section_body'])

    print()
query = "What do we know about virus genetics, origin, and evolution"



query_embedded = get_embed_content(query)



analyze_data['similarity'] = data_merged['embedding'].apply(find_similarity)



analyze_data.sort_values(by=['similarity'], ascending=False, inplace=True)



final_data = analyze_data.head()



print('Query 3 = ', query)



for index, row in final_data.iterrows():

    print(row['similarity'])

    print(row['section_body'])

    print()
query = "What do we know about vaccines and therapeutics"



query_embedded = get_embed_content(query)



analyze_data['similarity'] = data_merged['embedding'].apply(find_similarity)



analyze_data.sort_values(by=['similarity'], ascending=False, inplace=True)



final_data = analyze_data.head()



print('Query 4 = ', query)



for index, row in final_data.iterrows():

    print(row['similarity'])

    print(row['section_body'])

    print()
query = "What has been published about medical care"



query_embedded = get_embed_content(query)



analyze_data['similarity'] = data_merged['embedding'].apply(find_similarity)



analyze_data.sort_values(by=['similarity'], ascending=False, inplace=True)



final_data = analyze_data.head()



print('Query 5 = ', query)



for index, row in final_data.iterrows():

    print(row['similarity'])

    print(row['section_body'])

    print()
query = "What do we know about non-pharmaceutical interventions"



query_embedded = get_embed_content(query)



analyze_data['similarity'] = data_merged['embedding'].apply(find_similarity)



analyze_data.sort_values(by=['similarity'], ascending=False, inplace=True)



final_data = analyze_data.head()



print('Query 6 = ', query)



for index, row in final_data.iterrows():

    print(row['similarity'])

    print(row['section_body'])

    print()
query = "What do we know about diagnostics and surveillance"



query_embedded = get_embed_content(query)



analyze_data['similarity'] = data_merged['embedding'].apply(find_similarity)



analyze_data.sort_values(by=['similarity'], ascending=False, inplace=True)



final_data = analyze_data.head()



print('Query 7 = ', query)



for index, row in final_data.iterrows():

    print(row['similarity'])

    print(row['section_body'])

    print()
query = "What has been published about information sharing and inter-sectoral collaboration"



query_embedded = get_embed_content(query)



analyze_data['similarity'] = data_merged['embedding'].apply(find_similarity)



analyze_data.sort_values(by=['similarity'], ascending=False, inplace=True)



final_data = analyze_data.head()



print('Query 8 = ', query)



for index, row in final_data.iterrows():

    print(row['similarity'])

    print(row['section_body'])

    print()
query = "What has been published about ethical and social science considerations"



query_embedded = get_embed_content(query)



analyze_data['similarity'] = data_merged['embedding'].apply(find_similarity)



analyze_data.sort_values(by=['similarity'], ascending=False, inplace=True)



final_data = analyze_data.head()



print('Query 9 = ', query)



for index, row in final_data.iterrows():

    print(row['similarity'])

    print(row['section_body'])

    print()