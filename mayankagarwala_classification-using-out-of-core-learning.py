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
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import SGDClassifier
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False
def stream_docs(path, path2):
    with open(path, 'r') as csv1, open(path2,'r') as csv2:
        next(csv1)
        next(csv2)
        for linex,liney in zip(csv1,csv2):
            data=linex.split(",")
            k=len(data)
            text=[]
            for i in range(k):
                if(is_number(data[i])==True):
                    text.append(float(data[i]))
                else:
                    text.append(0)
            label=int(liney)
            yield text, label
next(stream_docs(path='../input/xtrain.csv',path2='../input/ytrain.csv'))
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for i in range(size):
            text, label = next(doc_stream)
            docs.append(text)            
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y
clf=SGDClassifier(eta0=0.01, warm_start=True, average=True, n_iter=1000, random_state=8)
doc_stream = stream_docs(path='../input/xtrain.csv',path2='../input/ytrain.csv')
import pyprind
pbar = pyprind.ProgBar(100)
classes = np.array([0,1])
for _ in range(100):
    X_train, y_train = get_minibatch(doc_stream, size=7500)
    #print (X_train)
    if not X_train:
        break
    clf.partial_fit(X_train, y_train, classes)
    pbar.update()
X_test, y_test = get_minibatch(doc_stream, size=150000)
print('Accuracy: %.3f' % clf.score(X_test, y_test))
