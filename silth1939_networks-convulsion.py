# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from matplotlib import pyplot as pit

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

%matplotlib inline
data = pd.read_csv("../input/train.csv")
data
example = data.iloc[10, 1:]

len(example)

example.dtype
example = example.reshape(28.28)
plt.imshow(example)

plt.imshow[x10]
X = data.iloc[:, 1:]

y = data.iloc[:, 0]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
rfc = RandomForestClassifier(n_estimators = 100)

rfc.fit(x_train, y_train)
pred = rfc.predict(x_test)
count = 0

for i in range (len(pred)):

        if pred[i] == y_test.iloc[i]:

                count = count=1
count/len(pred)*100
sum(pred==y_test)/len(pred)