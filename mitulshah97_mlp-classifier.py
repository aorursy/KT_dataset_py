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

import pandas as pd

df = pd.read_csv("../input/train.csv")
df2 = pd.read_csv("../input/test.csv")
X = df.iloc[:,1:784]
y = df.iloc[:,0]
X_test=df2.iloc[:,0:783]

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,20), random_state=1)
clf.fit( X, y ) 
y_pred = clf.predict(X_test)
import numpy as np

ImageId = np.arange(1,28001)
Label = y_pred

ImageId = pd.Series(ImageId)
Label = pd.Series(Label)

submit = pd.concat([ImageId,Label],axis=1, ignore_index=True)
submit.columns=['ImageId','Label']

submit.to_csv("submit3.csv",index=False)