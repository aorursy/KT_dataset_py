import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

submission = pd.read_csv("../input/titanic/gender_submission.csv")
import pandas_profiling as pp
pp.ProfileReport(train)
pp.ProfileReport(test)
pp.ProfileReport(submission)