# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# libraries

import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import scale
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test_data = pd.read_csv("../input/digit-recognizer/test.csv")

training_data = pd.read_csv("../input/digit-recognizer/train.csv")
training_data.shape
training_data.info()
training_data.head()
training_data.max().sort_values()
training_data.isna().sum().sort_values(ascending=False)
training_data.duplicated().sum()
training_data.columns
count_table = training_data.label.value_counts()

count_table = count_table.reset_index().sort_values(by='index')

count_table
plt.figure(figsize=(10, 5))

sns.barplot(x='index', y='label', data=count_table)
digit_means = training_data.groupby('label').mean()

digit_means.head()

plt.figure(figsize=(18, 10))

sns.heatmap(digit_means)
# average feature values

round(training_data.drop('label', axis=1).mean(), 2).sort_values()
# splitting into X and y

X = training_data.drop("label", axis = 1)

y = training_data['label']
# scaling the features

X_scaled = scale(X)



# train test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)
# model with optimal hyperparameters



# model

model = SVC(C=10, gamma = 0.001, kernel="rbf")



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# metrics

print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")
# scaling test data

# splitting into X and y

X_test_data = test_data

X_test_data = scale(X_test_data)

y_test_pred = model.predict(X_test_data)

y_test_pred
output = pd.DataFrame({"ImageId": i+1 , "Label": y_test_pred[i]} for i in range(0, X_test_data.shape[0]))

output.to_csv('submission.csv', index=False)