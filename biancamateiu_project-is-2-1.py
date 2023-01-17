# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/proect-is-2-0/mycsvfile.csv")

data.head()
import tensorflow as tf

import tensorflow_hub as hub



module_url='https://tfhub.dev/google/universal-sentence-encoder/2'

embed_module = hub.load(module_url)
def get_embed_content(content):

      

    tensor = tf.convert_to_tensor([content])

    

    embed = embed_module.signatures['default'](tensor)

    

    return embed['default'].numpy()[0]
data['embedding'] = data['section_body'].apply(get_embed_content)
data.head()
data.drop(['grammar_label'], axis=1, inplace=True)

data.head()
data.shape
data.to_csv('mycsvfile.csv',index=False)