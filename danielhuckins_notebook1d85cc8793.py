# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# given weights

# every call return a new weight based on the given dist

weights = {}

weights['horse'] = lambda: max(0, np.random.normal(5,2,1)[0])

weights['ball'] = lambda: max(0, 1 + np.random.normal(1,0.3,1)[0])

weights['bike'] = lambda: max(0, np.random.normal(20,10,1)[0])

weights['train'] = lambda: max(0, np.random.normal(10,5,1)[0])

weights['coal'] = lambda: 47 * np.random.beta(0.5,0.5,1)[0]

weights['book'] = lambda: np.random.chisquare(2,1)[0]

weights['doll'] = lambda: np.random.gamma(5,1,1)[0]

# var name was 'block', changed to blocks

weights['blocks'] = lambda: np.random.triangular(5,10,20,1)[0]

weights['gloves'] = lambda: 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
df = pd.read_csv('../input/gifts.csv', engine='c')
def mean_weights(w:str, k:int=10000):

    return sum(weights[w]() for i in range(k))/k
df['group'] = df['GiftId'].map(lambda x: x[:x.index('_')])

df['approx_weight'] = df['group'].map(lambda x: mean_weights(x))
df.groupby('group')['approx_weight'].mean()