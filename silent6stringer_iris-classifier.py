# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Features of input:
# Id
# SepalLengthCm
# SepalWidthCm
# PetalLengthCm
# PetalWidthCm
# Species

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
iris_data = pd.read_csv('../input/Iris.csv', header=0)
iris_data = iris_data[features]
label = features.pop()
_in = iris_data[features]
_out = iris_data[label]
clf = RandomForestClassifier(n_estimators=10)
clf.fit(_in, _out)
sample = pd.DataFrame(data = {'SepalLengthCm': 6.3, 'SepalWidthCm': 4.2, 'PetalLengthCm': 3.8, 'PetalWidthCm': 0.4}, index=[0])
print(sample)
