import pandas as pd
import numpy as np
from sklearn import svm
from patsy import dmatrices
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

#target_name = np.array(['Response'])

#train = train[train['is_train']==True]
all_columns = "+".join(train.columns - ["Response"])
formula = 'Response ~ ' + all_columns
y, X = dmatrices(formula, train, return_type="dataframe")

y = np.ravel(y)

model = svm.SVC(kernel='rbf')
model = model.fit(X, y)

# check the accuracy on the training set
model.score(X, y)

