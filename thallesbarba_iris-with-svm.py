# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from sklearn import svm


df = pd.read_csv ('../input/Iris.csv')

ri = np.arange(0, len (df.index) )
np.random.shuffle ( ri )

train_set = df.ix[ ri[0: round(len(df)*0.8)] ]
test_set  = df.ix[ ri[round(len(df)*0.8):] ]




C = 1.0
svc = svm.SVC (kernel='poly', degree=3, C=C)


svc.fit ( train_set[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] , train_set.Species )

Z = svc.predict ( test_set[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] )

test_set['Predicted'] = Z


for i in Z:
    print (i)


# Any results you write to the current directory are saved as output.