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
df = pd.read_csv("../input/train.csv")
X = df.iloc[:,1:784]
y = df.iloc[:,0]
test= pd.read_csv("../input/test.csv")
test1=test.iloc[:,0:783]
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

model_rf = XGBClassifier(random_state=1211)
model_rf.fit( X,y )
y_pred = model_rf.predict(test1)

import numpy as np

ImageId = np.arange(1,28001)
Label = y_pred

ImageId = pd.Series(ImageId)
Label = pd.Series(Label)

submit = pd.concat([ImageId,Label],axis=1, ignore_index=True)
submit.columns=['ImageId','Label']

submit.to_csv("submitSampXGB.csv",index=False)
