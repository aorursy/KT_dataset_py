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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import pylab as pl



# 读取对应数据集

train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

val= pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')





# 将train数据集划分为训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size=0.2)



# from sklearn.decomposition import PCA

# pca = PCA(n_components=0.7)

# 调用sklearn库中的逻辑回归模型并用训练集进行拟合

from sklearn.linear_model import LogisticRegression

ModelLR = LogisticRegression(solver='newton-cg', multi_class='multinomial')

ModelLR.fit(X_train, y_train)



# 获得在测试集上得到的预测值

y_predLR = ModelLR.predict(X_test)



# Accuracy score

print('accuracy is',accuracy_score(y_predLR,y_test))



score = accuracy_score(y_predLR,y_test)



# 应用在私有测试集

test_x = test.values[:,1:]



preds = ModelLR.predict(test_x)



submission['label'] = preds

submission.to_csv('submission.csv',index=False)


