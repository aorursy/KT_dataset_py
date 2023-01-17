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
import json

import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm
path = ['biorxiv_medrxiv', 'comm_use_subset', 'custom_license', 'noncomm_use_subset']

#path = '../input/CORD-19-research-challenge'

docs = []

for d in path:

    for file in os.listdir(f"{'../input/CORD-19-research-challenge/'+d}/{d}"):

        file_name = f"{'../input/CORD-19-research-challenge/'+d}/{d}/{file}"

        j = json.load(open(file_name, 'rb'))

        for text in j['body_text']:

            docs.append([text['text'] + '\n\n'])
print(len(docs))
df = pd.DataFrame(docs, columns = ['text'])

print(df.head())
incubation = df[df['text'].str.contains('incubation')]

print(incubation)
text = incubation['text'].values

for x in text:

    print(x)

    break
import re

incubation_times = []

for t in text:

    for sentences in t.split('. '):

        if 'incubation' in sentences:

            single_day = re.findall(r"\d{1,2}\.?\d{1,2} day[s]", sentences)

            if len(single_day) > 0:

                for x in single_day:

                    incubation_times.append(x.split()[0])

print(incubation_times)
len(incubation_times)


plt.hist(incubation_times, bins = 10)

plt.xlabel('incubation_times(days)')

plt.ylabel('bins count')

plt.show()