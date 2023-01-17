# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
path = '../input/train.csv'
df = pd.read_csv(path)
data = df[df.columns[1:]]
target = df['label']
mnist = {'data':data.values, 'target': target.values}
X, y = mnist['data'], mnist['target']
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary,
               interpolation='nearest')
    plt.axis('off')
    plt.show()
index = 41000
some_digit = X[index]
print(y[index])
plot_digit(some_digit)
X_train, X_test, y_train, y_test = X[:21000], X[21000:], y[:21000], y[21000:]
shuffle_index = np.random.permutation(21000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
from sklearn.svm import SVC
clf = SVC(kernel='poly', degree=2)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(pred)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, y_test)
print(acc)
clf.score(X_test, y_test)
test_data = pd.read_csv('../input/test.csv')
X_test = test_data.values
y_pred_test = clf.predict(X_test)
result_df = pd.DataFrame(data=y_pred_test, index=np.arange(1, 28001))
result_df.index.name = 'ImageId'
result_df.columns = ['Label']
result_df.to_csv(path_or_buf='output.csv')