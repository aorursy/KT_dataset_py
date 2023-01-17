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
%matplotlib inline
import pandas as pd
from sklearn import svm
df = pd.read_csv('../input/train.csv')
#print(df.columns)
clf = svm.SVC()
clf.fit(df[['Elevation','Slope']], df['Cover_Type'])
testdf = pd.read_csv('../input/test.csv')
testdf['Cover_Type'] = clf.predict(testdf[['Elevation','Slope']])
#print(testdf[['Id','Cover_Type']])
testdf[['Id','Cover_Type']].to_csv('submission.csv', index=False)