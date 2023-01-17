100# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

from sklearn import svm

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



digits = datasets.load_digits()

clf = svm.SVC(gamma=0.01, C=100.)



clf.fit(digits.data[:-1], digits.target[:-1])



clf.predict(digits.data[-2:-1])
