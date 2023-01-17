# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

import sklearn

import sklearn.metrics



from sklearn.decomposition import PCA

from sklearn.preprocessing import label_binarize





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# reads chunks of data from test.csv and makes a DataFrame for the test set



test_df = pd.DataFrame()

chunks = [test_df]



for chunk in pd.read_csv('../input/test.csv', sep=',', chunksize=1e6):

    chunks += [chunk]



test_df = pd.concat(chunks)

test_df.head()
# summary statistics for test data set



test_df.describe()
train_df = pd.DataFrame()

chunks = [train_df]



sample_size = int(1e5)

bin_num = 38

bin_sample_size = int(sample_size // bin_num)



for chunk in pd.read_csv('../input/train.csv', sep=',', chunksize=1e6):

    if sample_size - (2 * bin_sample_size) < 0:

        temp = chunk.sample(sample_size)

    else:

        temp = chunk.sample(bin_sample_size)

        

    print(len(chunks))

    

    sample_size -= bin_sample_size

    chunks += [temp]





train_df = pd.concat(chunks)

train_df.head()
train_df.describe()
dest_df = pd.DataFrame()



for chunk in pd.read_csv('../input/destinations.csv', sep=',', chunksize=1e6):

    dest_df = pd.concat([dest_df, chunk])



dest_df.describe()
# frequency count of 5 most popular hotel clusters from training sample



freq_df = pd.DataFrame(train_df['hotel_cluster'].copy())

freq_df.columns = ['actual']





freq_predictions = train_df['hotel_cluster'].value_counts().head().index.tolist()

freq_df['freq_predict'] = [freq_predictions[0] for i in range(freq_df.shape[0])]

temp = [freq_predictions[0] for i in range(freq_df.shape[0])]

freq_df.head()

y = freq_df['actual'].values

y = label_binarize(y, classes=list(range(0,100)))

x = label_binarize(temp, classes=list(range(0,100)))



sklearn.metrics.average_precision_score(y, x)
# check Pearson's Correlation Coefficient (r) values for every search feature against each other



rcorrs = train_df.corr()



for i, r in rcorrs.iterrows():

    for j in range(rcorrs.shape[1]):

        if np.abs(r[j]) < 0.4 or np.abs(r[j]) == 1:

            r[j] = np.NaN

    

rcorrs
dest_pca = PCA(n_component=5)

p = list(range(1,100))

print(p)