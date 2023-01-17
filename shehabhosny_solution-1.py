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
train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")



# Fill in missing values by a default value ("No Entry")

train_data["keyword"].fillna("No Entry", inplace = True)

train_data["location"].fillna("No Entry", inplace = True)



train_data.head()
test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")



# Fill in missing values by a default value ("No Entry")

test_data["keyword"].fillna("No Entry", inplace = True)

test_data["location"].fillna("No Entry", inplace = True)



test_data.head()
from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer()



Y = train_data["target"].copy()



X = vector.fit_transform(train_data["text"])

X_test = vector.transform(test_data["text"])



print("Text Sample: " + train_data["text"][0])
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV



model = RandomForestClassifier()



# Create a Dictionary containing n_estimators

dictionary = dict(n_estimators=[200, 300, 400])



# Use GridSearch to find the best parameters for the decision tree

# cv -> Number of cross-validation you have to try for each selected set of hyperparameters

hyperparameters = GridSearchCV(model, dictionary, cv=3)

hyperparameters.fit(X, Y)
model
model.fit(X, Y)

predictions = model.predict(X_test)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
output = pd.DataFrame({'id': test_data.id, 'target': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")