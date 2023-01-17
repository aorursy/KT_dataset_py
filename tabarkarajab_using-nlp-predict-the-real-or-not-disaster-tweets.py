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
import numpy as np 
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train=filenames[0]
test=filenames[1]
print(train)
print(test)
train_pd=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_pd=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
print(train_pd)

print(train_pd.head())
train_pd[train_pd["target"] == 1]["text"].values[3]
train_pd[train_pd["target"] == 1]["text"].values[4]
count_vectorizer = feature_extraction.text.CountVectorizer()
print(count_vectorizer)
## let's get counts for the first 5 tweets in the data
count_words = count_vectorizer.fit_transform(train_pd["text"][0:10])
print(count_words[0].todense().shape)
train_vectors = count_vectorizer.fit_transform(train_pd["text"])
test_vectors = count_vectorizer.transform(test_pd["text"])
model_linear = linear_model.RidgeClassifier()
predict = model_selection.cross_val_score(model_linear, train_vectors, train_pd["target"], cv=3, scoring="f1")
predict
model_linear.fit(train_vectors, train_pd["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission.head()
sample_submission["target"] = model_linear.predict(test_vectors)
sample_submission.head()
sample_submission.to_csv("submission_competition.csv", index=False)