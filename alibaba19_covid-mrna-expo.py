# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pathlib import Path

import collections

import matplotlib.pyplot as plt
input_path = Path('../input/stanford-covid-vaccine')

os.listdir(input_path)
train_df = pd.read_json(input_path/'train.json', lines=True)

test_df = pd.read_json(input_path/'test.json', lines=True)

sample_sub = pd.read_csv(input_path/'sample_submission.csv')
train_df.head()
test_df.head()
#we're getting alot of information in addition to this though

list(train_df.columns)
train_df.shape, test_df.shape, sample_sub.shape
#way lower than length of submission file bc it features rows that

#are not present and are meant to be ignored (idk why included)

3634*107 #num test samples * num of basepairs
#number of positions used in scoring -- 

train_df['seq_scored'].unique()
#length of sequence -- same for all test samples

train_df['seq_length'].unique()
#all of the sequences have 107 basepairs (characters)

len(train_df['sequence'][0]), train_df['sequence'][0]
#distribution of nucleotides in first training sample

collections.Counter(train_df['sequence'][0])
train_df['structure'][0]
sample_sequence = train_df.iloc[0]['sequence']

sample_structure = train_df.iloc[0]['structure']

sample_sequence, sample_structure
sample_sequence[98:], sample_structure[70:80]
#the number of positions used in scoring with predicted values

test_df['seq_scored'].value_counts()
test_df.columns
train_df['predicted_loop_type'][0]
plt.title('Loop types for single sequence of mRNA')

plt.hist(np.array(list(train_df['predicted_loop_type'][0])));
sample = train_df.iloc[0]

sample
(np.array(sample['deg_Mg_50C']) > np.array(sample['deg_50C'])).sum() / len(sample['deg_50C'])
(np.array(sample['deg_error_Mg_pH10']) / np.array(sample['deg_error_pH10'])).mean()
train_df.shape, train_df['reactivity'].shape, len(train_df['reactivity'][0])
#reminder of the columns in our training, test and submission dataframes

list(train_df.columns), list(test_df.columns), list(sample_sub.columns)
one_sample = train_df.iloc[0]

sequence = list(one_sample['sequence'])[0:68]

reactivity = one_sample['reactivity']

deg_50C = one_sample['deg_50C']

sample_df = pd.DataFrame([sequence, reactivity, deg_50C]).T

sample_df.columns= ['basepair', 'reactivity', 'deg_50C']

sample_df
seq_pos = []

for x in range(68):

    seq_pos.append(one_sample['id'] + '_' + str(x))

seq_pos[0:4]
def extract_sample(sample):

    seq_pos = []

    for x in range(68):

        seq_pos.append(sample['id'] + '_' + str(x))

    sequence = list(sample['sequence'])[0:68]

    structure = list(sample['structure'])[0:68]

    predicted_loop_type = list(sample['predicted_loop_type'])[0:68]

    reactivity = sample['reactivity']

    deg_Mg_pH10 = sample['deg_Mg_pH10']

    deg_pH10 = sample['deg_pH10']

    deg_Mg_50C = sample['deg_Mg_50C']

    deg_50C = sample['deg_50C']

    sample_df = pd.DataFrame([seq_pos, sequence, structure,

                               predicted_loop_type, reactivity,

                              deg_Mg_pH10, deg_pH10, deg_Mg_50C,

                              deg_50C]).T

    sample_df.columns= ['seq_pos', 'basepair', 'structure',

                        'predicted_loop_type', 'reactivity', 

                        'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C',

                        'deg_50C']

                       

    return sample_df
extract_sample(one_sample)
%%time

training_data = pd.DataFrame()

for x in range(0, train_df.shape[0]):

    df = extract_sample(train_df.iloc[x])

    training_data = training_data.append(df).reset_index(drop=True)
training_data.shape