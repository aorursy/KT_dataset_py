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
input_data = pd.read_csv('../input/party_in_nyc.csv')

input_data['Incident Zip'] = input_data['Incident Zip'].fillna(0).astype(int)

input_data = pd.get_dummies(input_data[['Location Type', 'City', 'Borough', 'Incident Zip']])



def make_dataset(name):

    output = pd.read_csv(name)

    output['id'] = output['id'].str[2:].astype(int)

    return pd.merge(input_data, output, left_index=True, right_on='id')



train_parties = make_dataset('../input/train_parties.csv')

test_parties = make_dataset('../input/test_parties.csv')

train_parties.shape
from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier()

avg_comp = train_parties['num_complaints'][lambda x: x > 0].mean()

weights = train_parties['num_complaints'].copy()

weights[lambda w: w == 0] = avg_comp

train_x = train_parties.drop('num_complaints', axis=1)

train_y = train_parties['num_complaints'] > 0

classifier.fit(train_x, train_y, sample_weight=weights)

train_y
from sklearn import metrics

test_x = test_parties.drop('num_complaints', axis=1)

test_y = test_parties['num_complaints'] > 0

metrics.confusion_matrix(classifier.predict(test_x), test_y)