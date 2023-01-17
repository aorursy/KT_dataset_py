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
all_data = pd.read_csv("../input/voice.csv")
rand_indices = np.random.permutation(len(all_data))

features = [feat for feat in all_data.columns if feat != 'label']

output = 'label'

num_datapoints = len(all_data)

test_total = int(num_datapoints * 0.1)



test_indices = rand_indices[-test_total:]

train_indices = rand_indices[:-test_total]



test_data = all_data[features].iloc[test_indices]

train_data = all_data[features].iloc[train_indices]



test_labels = all_data[output].iloc[test_indices]

train_labels = all_data[output].iloc[train_indices]



print(num_datapoints, len(train_data), len(test_data))

print(features)

from sklearn import linear_model

logistic = linear_model.LogisticRegression(C=1e5)



logistic.fit(train_data, train_labels)

for i, f in enumerate(features):

    print(features[i], logistic.coef_[0][i])
predictions = logistic.predict(test_data)

from sklearn.metrics import accuracy_score

accuracy_score(test_labels, predictions)

#[predictions[i] != test_labels.iloc[i] for i in range(len(predictions))]