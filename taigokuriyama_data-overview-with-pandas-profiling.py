import numpy as np

import pandas as pd

import pandas_profiling as pdp
X_train = pd.read_csv('../input/X_train.csv')

y_train = pd.read_csv('../input/y_train.csv')

X_test = pd.read_csv('../input/X_test.csv')
profile_X_train = pdp.ProfileReport(X_train)

profile_X_train
profile_y_train = pdp.ProfileReport(y_train)

profile_y_train
profile_X_test = pdp.ProfileReport(X_test)

profile_X_test
profile_X_train.to_file(outputfile="X_train.html")

profile_y_train.to_file(outputfile="y_train.html")

profile_X_test.to_file(outputfile="X_test.html")