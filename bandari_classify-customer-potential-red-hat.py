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
#fetch train data

train = pd.read_csv("../input/act_train.csv")

'''

from sklearn.naive_bayes import GaussianNB

>>> gnb = GaussianNB()

>>> y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

>>> print("Number of mislabeled points out of a total %d points : %d"

...       % (iris.data.shape[0],(iris.target != y_pred).sum()))

Number of mislabeled points out of a total 150 points : 6

http://scikit-learn.org/stable/modules/naive_bayes.html

'''



from sklearn.navie_bayes import GaussianNB

model = GaussianNB()

model.fit()