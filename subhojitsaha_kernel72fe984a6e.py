# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
sub = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
train
X = train.iloc[:,1:].values
Y = train.iloc[:,0].values
Y
t=test.iloc[:,:].values
test
sub
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X,Y)
clf.score(X[:2000,:],Y[:2000])
X[:2000,:]
pred = clf.predict(t)
res = pd.DataFrame(pred)
res.columns=["Label"]
res["imageId"]=sub["ImageId"]
sub["ImageId"]
res
res.to_csv("sub.csv",index=False)
