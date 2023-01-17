

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn import neighbors
import pandas as pd
data = pd.read_csv("../input/train.csv")
Test = pd.read_csv("../input/test.csv")
Test = Test.drop('Id', axis=1)
label_pd = data.get('Cover_Type')
train_pd = data.drop(['Cover_Type', 'Id'], axis=1)
label = label_pd
train = train_pd
Test = Test

isExists=os.path.exists('../data')
# 判断结果
if not isExists:
    os.makedirs('../data') 
    print (' 创建成功')
else:
    # 如果目录存在则不创建，并提示目录已存在
    print (' 目录已存在')
clf = neighbors.KNeighborsClassifier()
clf.fit(train, label)
predict = clf.predict(Test)
print(predict.dtype)
print(predict.shape)
predict = pd.Series(predict, name='Cover_Type')
print(predict.shape)
sub = pd.read_csv('../input/sample_submission.csv')
sub.drop('Cover_Type', axis=1, inplace=True)#drop掉原先样式中的最后一列，接下来再增加这一列把预测的结果放进去
print(sub.shape)
sub = pd.concat([sub, predict], axis=1)
sub.head()

print(sub.head())
sub.to_csv('../data/sample_submission.csv')

