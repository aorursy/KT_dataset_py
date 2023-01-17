# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Any results you write to the current directory are saved as output.
#test_df
#print(train_df['pixel12'].sum())
X = train_df.drop('label', axis=1)
y = train_df['label']
X_train,X_test,y_train,y_test = train_test_split(X,y)

nn = MLPClassifier()
nn.fit(X_train,y_train)
res = nn.predict(X_test)

print(accuracy_score(res,y_test, normalize=False))

test_result = nn.predict(test_df)
test_result = pd.DataFrame(test_result)
sub = test_result.to_csv('sub.csv', index=False)