# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/candy-data.csv')

df.head()
df.describe()
cols = df.columns[1:]

data = df[cols].values

features = data[:,:-1]

target = data[:,-1]
def normalize(feat):

    return (feat-np.min(feat, axis=0))/(np.max(feat, axis=0)-np.min(feat, axis=0))
n_feat = normalize(features) 
partitions=np.linspace(0.3,0.95,6)

x=np.random.binomial(1,0.8,size=len(df))

train_feat = n_feat[x==1,:]

train_t = target[x==1]

test_feat = n_feat[x==0,:]

test_t = target[x==0]

# Baseline Model

baseline = np.sqrt(np.sum((test_t-np.mean(train_t))**2))/np.sqrt(len(test_t))

print ('The Error of the baseline Model is: {}'.format(baseline))

for prob in partitions:

    print (prob)

    x=np.random.binomial(1,prob,size=len(df))

    train_feat = n_feat[x==1,:]

    train_t = target[x==1]

    test_feat = n_feat[x==0,:]

    test_t = target[x==0]

    lm = LinearRegression()

    lm.fit(train_feat,train_t)

    rmse = np.sqrt(np.sum((test_t-lm.predict(test_feat))**2))/np.sqrt(len(test_t))

    print ('The Error of the linear model is: {}'.format(rmse))

    plt.scatter(test_t,lm.predict(test_feat))

    plt.title('how the predictions are scattered: predicted v/s real target')

    plt.show()
