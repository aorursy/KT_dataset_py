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
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import spacy

nlp = spacy.load("en_core_web_sm")

fake_data = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

true_data = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fake_data.head()
true_data.head()
fake_data['label'] = 0

true_data['label'] = 1
frames = [fake_data, true_data]

data = pd.concat(frames)
with nlp.disable_pipes():

    vectors = np.array([nlp(each.title).vector for idx, each in data.iterrows()])
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





X_train, X_test, y_train, y_test = train_test_split(vectors, data.label, test_size=0.1, random_state=1)

model = LinearSVC(random_state=1, dual=False)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, vectors, data.label, cv=5)

scores