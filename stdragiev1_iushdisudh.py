# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
labeled_data = pd.read_csv("../input/train.csv")
labeled_y = labeled_data['label']
labeled_X = labeled_data[labeled_data.columns[1:]]

training_X, test_X, training_y, test_y = train_test_split(
    labeled_X, labeled_y, test_size=0.15, random_state=42)

submission_X = pd.read_csv("../input/test.csv")

print('Starting LDA')
lda = LDA()
lda.fit(training_X,training_y)
print('Done training, starting with predictions')
training_predictions = lda.predict(training_X)
test_predictions = lda.predict(test_X)
submission_predictions = lda.predict(submission_X)
print('Done with training and prediction!')

import sklearn.metrics as metrics
print('Report:')
print('#######')
print('Training data: ')
print(metrics.classification_report(training_y, training_predictions))
print(' ')
print('Test data: ')
print(metrics.classification_report(test_y, test_predictions))

print(submission_predictions)