# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

drugs = pd.read_csv('../input/drugsComTest_raw.csv')
drugs.head()
# Any results you write to the current directory are saved as output.
top10conditions = drugs["condition"].value_counts().head(10)
top10conditions
top10conditions/drugs.shape[0]*100
# top 10 conditions by number of reviews

top10conditions.plot.bar()
top10conditions = top10conditions.to_frame()
#for row in top10conditions.iterrows():
#    top10conditions['bestdrugs'] = 
#    drugs.loc[drugs['condition'] == row[0]]
#    print(row[0])

tmp_df = drugs.loc[drugs['condition'] == 'Depression']
tmp_df.head()
best_drug_dict = dict()
for row in tmp_df.iterrows():
    if row[1]['drugName'] not in best_drug_dict:
        best_drug_dict[row[1]['drugName']] = 0
    else:
        best_drug_dict[row[1]['drugName']] += row[1]['rating']

best_drug_dict
best_drug_df = pd.DataFrame.from_dict(best_drug_dict, orient='index')
best_drug_df.columns = ['rating sum']
best_drug_df = best_drug_df.sort_values(by=['rating sum'], ascending=False)
best_drug_df.head()
from collections import Counter
Counter(best_drug_dict).most_common(5)
# find the best drugs for top 10 condition

data = []

for row in top10conditions.iterrows():
    tmp_df = drugs.loc[drugs['condition'] == row[0]]
    #print(tmp_df.head())
    best_drug_dict = dict()
    for row in tmp_df.iterrows():
        if row[1]['drugName'] not in best_drug_dict:
            best_drug_dict[row[1]['drugName']] = 0
        else:
            best_drug_dict[row[1]['drugName']] += row[1]['rating']
    #print(Counter(best_drug_dict).most_common(5))
    #data.append(Counter(best_drug_dict).most_common(5))
    data.append(list())
    for k, v in Counter(best_drug_dict).most_common(5):
        data[-1].append(k)
print(data)
# this is the table showing the top 10 condition based on the sum of ratings for each drug for each condition 

top10 = top10conditions.assign(best_drugs=pd.Series(data).values)

top10




