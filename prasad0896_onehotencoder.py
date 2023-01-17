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
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer
from tpot.builtins import OneHotEncoder

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('../input/train.csv', dtype=np.float64)
features = tpot_data.drop('label', axis=1).values
#training_features, testing_features, training_target, testing_target = train_test_split(features, tpot_data['target'].values, random_state=None)
training_features = features
testing_features = pd.read_csv('../input/test.csv')
training_target = tpot_data['label'].values
# Average CV score on the training set was:0.9580165883858065

exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
    Binarizer(threshold=0.0),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.15000000000000002, min_samples_leaf=4, min_samples_split=19, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
t = results.tolist()
t = list(map(int,t))
df = pd.DataFrame()
df['Label'] = t
df['ImageId'] = np.arange(1,28001)
df = df[['ImageId', 'Label']]
df
