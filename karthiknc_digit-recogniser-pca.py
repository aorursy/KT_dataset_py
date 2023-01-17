# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from collections import Counter

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
train=pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv",index_col = None)
train.head()
train.shape
trainx = train.drop('label',axis = 1)
sc=StandardScaler()

trainx = sc.fit_transform(trainx)

testx = sc.fit_transform(test)
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(trainx)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principalcomponent1',

                                                                  'principalcomponent2'])

label = pd.DataFrame(data = train['label'])

principalDf = pd.concat([principalDf,label],axis = 1,ignore_index=True)



principalDf.columns = ["principalcomponent1", "principalcomponent2", "label"] 
sns.lmplot( x='principalcomponent1', y='principalcomponent2', data=principalDf, fit_reg=False, 

           hue='label', legend=False, palette="Blues")

plt.figure(figsize=(13,10))
pca2 = PCA(.95)



pca2.fit(trainx)



train_2 = pca.transform(trainx)

test_2 = pca.transform(testx)
logr = LogisticRegression(solver = 'lbfgs') 

logr.fit(train_2, train['label'])

test_pred=logr.predict(test_2)
submission = pd.DataFrame({

        "ImageId": np.arange(1,28001),

        "Label": test_pred

    })

submission.to_csv('submission.csv', index=False)