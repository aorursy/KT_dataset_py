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
# 加载相关数据文件

train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

val= pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')

# 显示训练数据集文件的头几条

train.head()
# 显示测试集文件的头几条

test.head()
# 对训练数据集进行数据，标签分离

X_train = train.drop('label', axis=1)

Y_train = train['label']
# 将训练数据集划分为训练集和验证集

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)
# 调用XGBoost分类器并在训练集上拟合模型

from xgboost import XGBClassifier

model = XGBClassifier()

eval_set = [(X_val,Y_val)]

model.fit(X_train, Y_train, early_stopping_rounds=5, eval_set=eval_set, verbose=True)
# 将训练过的模型应用在验证集上，得到预测值

Y_pred = model.predict(X_val)

predictions = [round(value) for value in Y_pred] 
# 通过对比预测值和真实值，得到在验证集上的预测准确率

from sklearn.metrics import accuracy_score

# evaluate predictions

accuracy = accuracy_score(Y_val, predictions)

print("Accuracy XGBOOST: %.2f%%" % (accuracy * 100.0))
test = test.drop('id', axis=1)
test.head()
# 将模型应用在测试集上

test_pred = model.predict(test)
# 获得提交文件

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = test_pred

submission.to_csv('submission.csv', index=False)
submission.head(5)