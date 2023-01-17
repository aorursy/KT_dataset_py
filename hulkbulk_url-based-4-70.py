# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
perfect = pd.read_csv('../input/perfect/perfect_submission .csv')
perfect.to_csv('sub_perfect.csv',index=False)
perfect.head()
train = pd.read_csv("../input/gendered-pronoun-resolution/test_stage_1.tsv", delimiter="\t")
submission = pd.read_csv('../input/gendered-pronoun-resolution/sample_submission_stage_1.csv')
train['n_A_in_Text'] = train[['Text', 'A']].apply(lambda row : row['Text'].count(row['A']), axis=1)
train['n_B_in_Text'] = train[['Text', 'B']].apply(lambda row :  row['Text'].count(row['B']), axis=1)
train['Pronoun_count'] = train[['Text', 'Pronoun']].apply(lambda row :  row['Text'].count(row['Pronoun']), axis=1)
len(set(set(train['A'].unique()) | set(train['B'].unique())))
len(train)
train[['URL']].head()
train['URL_end'] = train['URL'].apply(lambda url : url.replace("http://en.wikipedia.org/wiki/", ""))
train.head()
def count_in_url(row,field):

    pronoun = row[field].lower().split()

    url_clean = row["URL_end"].lower().split('_')

    counter = 0

    for p in pronoun:

        if p in url_clean:

            counter += 1

    return counter
train['A_count_url'] = train[['A','URL_end']].apply(lambda row : count_in_url(row,"A"), axis=1)

train['B_count_url'] = train[['B','URL_end']].apply(lambda row : count_in_url(row,"B"), axis=1)
train[["A_count_url", "B_count_url"]].head(50)

prediction = train[['ID',"A_count_url", "B_count_url"]]
def get_proba(row):

    sum_ = row["A_count_url"] + row['B_count_url']

    if sum_ == 0:

        

        return 1/3.

    else :

        return row["A_count_url"]/ sum_

def get_proba_B(row):

    sum_ = row["A_count_url"] + row['B_count_url']

    if sum_ == 0:

        

        return 1/3.

    else :

        return row["B_count_url"]/ sum_

def get_proba_neither(row):

    return  1- row["A"] - row["B"]
prediction['A'] = prediction.apply(lambda row : get_proba(row), axis=1)

prediction['B'] = prediction.apply(lambda row : get_proba_B(row), axis=1)

prediction['NEITHER'] = prediction.apply(lambda row : get_proba_neither(row), axis=1)
prediction[['ID',"A",'B', "NEITHER"]].to_csv('sub.csv', index=False)