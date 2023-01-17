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
from sklearn import feature_extraction



count_vectorizer = feature_extraction.text.CountVectorizer()



train_data = pd.read_csv ('../input/nlp-getting-started/train.csv')

test_data = pd.read_csv ('../input/nlp-getting-started/test.csv')



train_data.head ()
test_data.head ()
train_vectors = count_vectorizer.fit_transform(train_data["text"])



test_vectors = count_vectorizer.transform(test_data["text"])
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier (random_state = 1)

model.fit (train_vectors, train_data["target"])

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sol = model.predict (test_vectors)
sol
sol = pd.DataFrame ({'id': test_data.id, 'target' : sol})

sol.head ()
sol.to_csv ('submission1.csv', index=False)