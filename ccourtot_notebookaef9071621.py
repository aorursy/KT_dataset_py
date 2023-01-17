# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.naive_bayes import GaussianNB

from scipy.stats.mstats import normaltest
train_numeric_part = pd.read_csv('../input/train_numeric.csv', nrows=10000)

header = train_numeric_part.columns.tolist()
def file_len(fname):

    with open(fname) as f:

        for i, l in enumerate(f):

            pass

    return i + 1
nline = file_len('../input/train_numeric.csv')
col = []

train_numeric_part_gauss = pd.DataFrame(index=train_numeric_part.index)

train_numeric_part_gauss_X = pd.DataFrame(index=train_numeric_part.index)

for c in train_numeric_part.columns:

    if train_numeric_part[c].dropna().size >= 0.946*train_numeric_part.shape[0]:

        if normaltest(train_numeric_part[c].dropna()).pvalue < 0.05:

            train_numeric_part_gauss[c] = train_numeric_part[c]

            col.append(c)
train_numeric_part_gauss["Response"] = train_numeric_part['Response']

train_numeric_part_gauss = train_numeric_part_gauss.dropna()

train_numeric_part_gauss.set_index("Id",inplace=True)

train_numeric_part_gauss_X = train_numeric_part_gauss.loc[:,train_numeric_part_gauss.columns != "Response"]
gnb = GaussianNB()
gnb = gnb.fit(train_numeric_part_gauss_X,train_numeric_part_gauss.Response)
reader = pd.read_csv('../input/train_numeric.csv', chunksize=10000)



for chunk in reader:

    train_numeric_part_gauss = chunk[col]

    train_numeric_part_gauss["Response"] = chunk['Response']

    train_numeric_part_gauss = train_numeric_part_gauss.dropna()

    train_numeric_part_gauss.set_index("Id",inplace=True)

    train_numeric_part_gauss_X = train_numeric_part_gauss.loc[:,train_numeric_part_gauss.columns != "Response"]

    gnb = gnb.partial_fit(train_numeric_part_gauss_X,train_numeric_part_gauss.Response, classes=np.unique(train_numeric_part_gauss.Response))
file_len('../input/test_numeric.csv')
test_numeric_part = pd.read_csv('../input/test_numeric.csv',nrows=1000)
res = pd.DataFrame(columns=["Id","Response"])

res.set_index("Id",inplace=True)

with open('res.csv', 'a') as f:

             res.to_csv(f, header=True)
reader = pd.read_csv('../input/test_numeric.csv', chunksize=10000)

#col.remove("Response")

for chunk in reader:

    indata = chunk[col]

    indata.set_index("Id",inplace=True)

    indata = indata.dropna()

    indata['res'] = gnb.predict(indata)

    with open('res.csv', 'a') as f:

             indata[['res']].to_csv(f, header=False)
#print(check_output(["head","res.csv"]).decode("utf8"))

res = pd.DataFrame(columns=["Id","Response"])

res.set_index("Id",inplace=True)

with open('res.csv', 'a') as f:

             res.to_csv(f, header=True)
file_len('res.csv')
train_categorical_part = pd.read_csv('../input/test_categorical.csv', nrows=10000, low_memory=False)
ngcol = []

train_numeric_part_ngauss = pd.DataFrame(index=train_numeric_part.index)

train_numeric_part_ngauss_X = pd.DataFrame(index=train_numeric_part.index)

for c in train_numeric_part.columns:

    if train_numeric_part[c].dropna().size >= 0.94*train_numeric_part.shape[0]:

        if normaltest(train_numeric_part[c].dropna()).pvalue >= 0.05:

            train_numeric_part_ngauss[c] = train_numeric_part[c]

            ngcol.append(c)

            

ngcol.append("Response")

ngcol.append("Id")

print(ngcol)
from sklearn import linear_model
reader = pd.read_csv('../input/train_numeric.csv', chunksize=10000)



failcount = 0



for chunk in reader:

    failcount += chunk.loc[chunk.Response == 1, "Response"].size
s = file_len('../input/test_categorical.csv')

cweights = {0 : (s - failcount)/s,

           1 : failcount/s}
reader = pd.read_csv('../input/train_numeric.csv', chunksize=10000)



perc = linear_model.Perceptron(class_weight='balanced')



for chunk in reader:

    train_numeric_part_ngauss = chunk[ngcol]

    train_numeric_part_ngauss["Response"] = chunk['Response']

    train_numeric_part_ngauss = train_numeric_part_ngauss.dropna()

    train_numeric_part_ngauss.set_index("Id",inplace=True)

    train_numeric_part_ngauss_X = train_numeric_part_ngauss.loc[:,train_numeric_part_ngauss.columns != "Response"]

    perc = perc.partial_fit(train_numeric_part_ngauss_X,train_numeric_part_ngauss.Response, classes=np.unique(train_numeric_part_ngauss.Response))