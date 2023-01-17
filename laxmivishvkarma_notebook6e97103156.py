# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.head()
train["keyword"].fillna("No Entry",inplace=True)

train["location"].fillna("No Entry",inplace=True)

test["keyword"].fillna("No Entry",inplace=True)

test["location"].fillna("No Entry",inplace=True)
train.head()
from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer()



X = vector.fit_transform(train["text"])

Y = train["target"].copy()

X_test = vector.transform(test["text"])
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=None, max_features='auto',

                       max_leaf_nodes=None, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=100,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)

model.fit(X, Y)

predictions = model.predict(X_test)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

accuracy = (accuracy_score(sample_submission["target"], predictions, normalize=False))/(len(predictions))

print(accuracy)
output = pd.DataFrame({'id': test.id, 'target': predictions})

output.to_csv('my_submission.csv',index=False)