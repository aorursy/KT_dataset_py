# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
grouped = train_data.groupby('is_duplicate').aggregate('sum').reset_index()

grouped = grouped.is_duplicate.count_values()
cos1 = train_data['question1'].copy()

cos2 = train_data['question2'].copy()



from nltk.corpus import stopwords

stop = stopwords.words('english')

for index in cos1:

    temp = index.split()

    for i in temp:

        if i not in stop:

            index.replace(i," ")

        else:

            pass

        
cos1 = train_data['question1'].copy()

cos2 = train_data['question2'].copy()

td_matrix = {}

f_matrix={}

i=0

for index in cos1:

    tc = nltk.TextCollection(index)

    fdist = nltk.FreqDist(index)

    for term in fdist:

        td_matrix[term] = tc.tf_idf(term,index)

        f_matrix.update({i:td_matrix[term]})

        i=i+1
print(cos1.head())