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
df = pd.read_csv('../input/challenge_dataset.txt', names=['X', 'Y'])
df.shape
df.info()
df.describe()
df.hist()
df.head()
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df.X.reshape(-1, 1), 
    df.Y.reshape(-1, 1), 
    test_size=0.25
)
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
score = model.score(predicted, y_test)
print('Score: ', score)
import matplotlib.pyplot as plt
import seaborn as sns

x_line = np.arange(5, 25).reshape(-1,1)
sns.regplot(x=df.X, y=df.Y, data=df, fit_reg=False)
plt.plot(x_line, model.predict(x_line))
plt.show()
from sklearn import datasets

ds = datasets.load_iris()
from sklearn import svm

cv_result = model_selection.cross_val_score(
    svm.SVC(),
    ds.data,
    ds.target,
    scoring='accuracy'
)
print('Mean: ', cv_result.mean())
print('Standard Deviation: ', cv_result.std())
