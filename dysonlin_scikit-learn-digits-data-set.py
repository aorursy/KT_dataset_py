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
from sklearn import datasets

digits = datasets.load_digits()

print(digits.data)

print(digits.target)
from sklearn import svm

model = svm.SVC(gamma = 0.0001, C = 100)
model.fit(digits.data[:-2], digits.target[:-2])
print(model.predict(digits.data[-2:]))