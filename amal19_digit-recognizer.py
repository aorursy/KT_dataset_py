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
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
split = 0.8

limit = int(len(train)*split)

print(limit)
training = train.iloc[0:limit,1:]

training_label = train.iloc[0:limit,0]



testing = train.iloc[limit:,1:]

actual_label = train.iloc[limit:,0]
print(testing)
clf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=2, min_samples_leaf=1)



clf.fit(training,training_label)
predictions = clf.predict_proba(testing).astype(float)

predictions = predictions.argmax(axis=1)
print(confusion_matrix(actual_label,predictions))

print(classification_report(actual_label,predictions))
test= test.iloc[0:int(len(test))]

predictions = clf.predict_proba(test).astype(float)

predictions = predictions.argmax(axis=1)
submission = pd.DataFrame({

    "ImageId": range(1,28001),

    "Label": predictions

})



submission .to_csv('submission.csv', index=False)