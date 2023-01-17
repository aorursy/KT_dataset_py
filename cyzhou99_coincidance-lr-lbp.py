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
# 读取数据集文件

train=pd.read_csv('../input/Kannada-MNIST/train.csv')

test=pd.read_csv('../input/Kannada-MNIST/test.csv')
# 将训练数据集中的特征和标签分离

X_train = train.drop('label', axis=1)

Y_train = train['label']
# 调用库函数将训练集数据集划分为训练集和验证集

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)
from skimage import feature as skft

%matplotlib inline

import matplotlib.pyplot as plt
# 将图像转换为灰度图并显示

train1 = X_train.iloc[1,:].to_numpy().reshape(28, 28)

plt.imshow(train1.astype(np.uint8), cmap='gray')
# 对上面显示的图像进行LBP处理

lbp = skft.local_binary_pattern(train1,8,1,'ror');

plt.imshow(lbp.astype(np.uint8), cmap='gray')
# 对训练集和验证集的图像进行LBP处理

X_train_LBP = skft.local_binary_pattern(X_train,8,1,'ror');

X_val_LBP = skft.local_binary_pattern(X_val,8,1,'ror');
# 调用sklearn库中的逻辑回归模型并在经过LBP处理的训练集上进行拟合

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200, multi_class='multinomial', solver='lbfgs')

model.fit(X_train_LBP, Y_train)
# 将训练过的模型应用在验证集上，得到预测值

Y_pred = model.predict(X_val_LBP)
# 获得准确率

from sklearn.metrics import accuracy_score

acc = accuracy_score(Y_pred, Y_val)

print("accuray is:", acc)
# 对测试集进行处理

test = test.drop('id', axis=1)

test_LBP = skft.local_binary_pattern(test,8,1,'ror');
# 获得在测试集上的预测结果

test_pred = model.predict(test_LBP)

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = test_pred

submission.to_csv('submission.csv', index=False)