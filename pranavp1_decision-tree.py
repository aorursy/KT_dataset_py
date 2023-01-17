# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')  # train set

test_df  = pd.read_csv('../input/test.csv')   # test  set

test_df.keys()

train_df.keys()
train_y = np.array(train_df['label'])

train_X = train_df.drop(['label'], axis=1)

train_X = np.array(train_X) 

test_X = np.array(test_df)

neural_net = MLPClassifier(solver='sgd', alpha=1e-5, random_state=1)

neural_net.fit(train_X, train_y)
predictions = neural_net.predict(test_X)
df = pd.DataFrame(predictions)

df.index += 1

df.columns = ['Label']

df["ImageId"] = df.index

cols = df.columns.tolist()

df = df[cols[::-1]]

df.to_csv('results.csv', header=True, index=False)