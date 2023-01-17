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
# %% Imports

import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
# %%
dataset = pd.read_csv('/kaggle/input/enron-email-dataset/enron.csv')
dataset.describe()

authors = {"chris.germany@enron.com","sara.shackleton@enron.com"}

convos = []

for d in dataset.values:
	if d[2] in authors:
		convos.append({ 'author' : ('Chris' if d[2] == "chris.germany@enron.com" else 'Sara'),
			'message' : d[5]})

word_data_author = pd.DataFrame(convos)
# %% 


features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data_author['message'], word_data_author['author'], test_size=0.1, random_state=42)

### text vectorization--go from strings to lists of numbers
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, labels_train)
features_train_transformed = selector.transform(features_train_transformed).toarray()
features_test_transformed  = selector.transform(features_test_transformed).toarray()
# %%

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import  accuracy_score

clf = GaussianNB()
clf.fit(features_train_transformed, labels_train)
pred = clf.predict(features_test_transformed)
accuracy_score(pred, labels_test)