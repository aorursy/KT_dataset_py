# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import tensorflow_hub as hub



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/intelligent-systems-2/mycsvfile.csv")

data.head(10)
data.shape

data.describe()
new_data = data.fillna(value={'grammar_label' : 'INFORMATION'})
is_info = new_data['grammar_label'] == 'INFORMATION'

data_info = new_data[is_info]

data_info.shape
new_data.drop_duplicates(subset="section_body", inplace = True)



indexNames = new_data[ new_data['grammar_label'] != 'INFORMATION' ].index

 

# Delete these row indexes from dataFrame

new_data.drop(indexNames , inplace=True)



df1 = new_data.describe(include = 'all')



df1.loc['dtype'] = new_data.dtypes

df1.loc['size'] = len(new_data)

df1.loc['% count'] = new_data.isnull().sum()



print (df1)
new_data['section_body'].describe()
analyze_data_dict = {"section_body": [], "grammar_label" : [], "length": []}

analyze_data = pd.DataFrame.from_dict(analyze_data_dict)



analyze_data['section_body'] = new_data['section_body']

analyze_data['grammar_label'] = new_data['grammar_label']





analyze_data['length'] = new_data['section_body'].apply(len)



analyze_data.sort_values(by = ['length'], inplace=True)



analyze_data.head(10)
module_url='https://tfhub.dev/google/universal-sentence-encoder/2'

embed_module = hub.load(module_url)


def get_embed_content(content):

      

    tensor = tf.convert_to_tensor([content])

    

    embed = embed_module.signatures['default'](tensor)

    

    return embed['default'].numpy()[0]
new_data['embedding'] = new_data['section_body'].apply(get_embed_content)

new_data.head()
new_data.shape
new_data.to_csv('mycsvfile.csv',index=False)