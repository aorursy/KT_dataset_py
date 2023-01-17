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
# Version 2 - I added the data set

# Version 3 - I added code to create a CSV

# Version 4 through 6 - I cleaned up a lot of comments.

#  This is the advice someone has: If you are using python, its much easier if you convert the JSON to a Pandas data frame:



# The following is the code that they recommeded



# import pandas, json



# json_data = json.load(open('train.json'))



# df=pandas.io.json.json_normalize(json_data)
# This is the code advice that someone else also recommeded

# df = pandas.read_json('~/myfile.json')

# df.to_csv('~/myfile.csv')
# df = pandas.read_json('/kaggle/input/test.json')

# df.to_csv('/kaggle/working/test.csv')





import json

import os



import numpy as np

import pandas as pd

from tqdm import tqdm
data = {'id': [], 'sequence': [], 'structure': [], 'predicted_loop_type': [], 'signal_to_noise': [], 'SN_filter': [], 'seq_length': [], 'seq_scored': [], 

        'reactivity_error': [], 'deg_error_Mg_pH10': [], 'deg_error_pH10': [], 'deg_error_Mg_50C': [], 'deg_error_50C': [],

        'reactivity': [], 'deg_Mg_pH10': [], 'deg_pH10': [], 'deg_Mg_50C': [], 'deg_50C': []}



with open('../input/stanford-covid-vaccine/train.json') as f:

    for line in tqdm(f):

        review = json.loads(line)

        data['id'].append(review['id'])

        data['sequence'].append(review['sequence'])

        data['structure'].append(review['structure'])

        data['predicted_loop_type'].append(review['predicted_loop_type'])

        data['signal_to_noise'].append(review['signal_to_noise'])

        data['SN_filter'].append(review['SN_filter'])

        data['seq_length'].append(review['seq_length'])

        data['seq_scored'].append(review['seq_scored'])

        data['reactivity_error'].append(review['reactivity_error'])

        data['deg_error_Mg_pH10'].append(review['deg_error_Mg_pH10'])

        data['deg_error_pH10'].append(review['deg_error_pH10'])

        data['deg_error_Mg_50C'].append(review['deg_error_Mg_50C'])

        data['deg_error_50C'].append(review['deg_error_50C'])

        data['reactivity'].append(review['reactivity'])

        data['deg_Mg_pH10'].append(review['deg_Mg_pH10'])

        data['deg_pH10'].append(review['deg_pH10'])

        data['deg_Mg_50C'].append(review['deg_Mg_50C'])

        data['deg_50C'].append(review['deg_50C'])

        

        
df = pd.DataFrame(data)



print(df.shape)

df.head()
df.to_csv('train.csv', index=False)
# Someone included the following code but I found that it was not necessary to utilize the code to create a dataframe and a CSV

# df['stars'] = df['stars'].astype('category')

# df['text'] = df['text'].astype(str)