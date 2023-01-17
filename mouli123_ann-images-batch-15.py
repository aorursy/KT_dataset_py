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
digits=pd.read_csv('../input/digit-recognizer/train.csv')
digits.info()
digits.head()
labels=digits['label']
labels
digits.drop('label',axis=1,inplace=True)
import sklearn.neural_network as nn

import sklearn.model_selection as ms
x_train,x_test,y_train,y_test=ms.train_test_split(digits,labels,test_size=0.2,random_state=22)
x_train.shape
ANN=nn.MLPClassifier()
ANN.fit(x_train,y_train)
ANN.score(x_test,y_test)
test=pd.read_csv('../input/digit-recognizer/test.csv')
answers=ANN.predict(test)
submission=pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submission.head()
submission['Label']=answers
submission.to_csv('1st.csv',index=False)