# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



# Any results you write to the current directory are saved as output.





df = pd.read_csv('../input/glass.csv')

data = df.as_matrix(columns=['Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])

classes = df.as_matrix(columns=['Type'])

x_train, x_test, y_train, y_test = train_test_split(data, classes, train_size=0.08, random_state=len(data))



clf = SVC(kernel='rbf',C=10000.0)

clf.fit(x_test, y_test.ravel())

predicted = clf.predict(x_test)

accuracy = accuracy_score(y_test, predicted)

print (accuracy)
print(df.head(15))