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
data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
data
sample_submission
data.text[0]
from sklearn.feature_extraction.text import CountVectorizer

bag = CountVectorizer()

bag.fit(data.text[0:3])
bag.vocabulary_
from sklearn.model_selection import train_test_split

X = data.drop(["target", "id", "keyword", "location"], axis = 1)

y = data.target

X_train, X_val, y_train, y_val = train_test_split(X,y)

X_train
from sklearn.pipeline import Pipeline

from sklearn.linear_model import RidgeClassifier

model = Pipeline(steps= [("vectorizer", CountVectorizer()), ("ridge", RidgeClassifier())])
y_val.shape
X_train.shape

X_train = X_train.to_numpy()

X_train.shape

X_train = X_train.reshape(5709)

X_val = X_val.to_numpy().reshape(1904)
model.fit(X_train, y_train)
model.score(X_val, y_val)
parameters = {"ridge__alpha": 10**np.arange(1,2,0.1, dtype=float)}

CountVectorizer().get_params()
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(model, parameters, cv=5)

search = grid_search.fit(X_train, y_train)
search
search.best_score_
search.best_params_
search.cv_results_["mean_test_score"]
grid_search.score(X_val, y_val)
y_val.shape
X = X.to_numpy().reshape(X.shape[0])
y.shape
grid_search.fit(X,y)
grid_search.best_params_
submission_id = test.id

test = test.drop(["id", "keyword", "location"], axis= 1)

test = test.to_numpy().reshape(test.shape[0])

test
predictions = grid_search.predict(test)

predictions
submission = pd.DataFrame({"target": predictions}, index= submission_id)

submission.index.rename("id")

submission
submission.to_csv("submission.csv")