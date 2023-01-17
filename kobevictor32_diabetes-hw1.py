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
#前處理將 - class轉換成數字

raw = pd.read_csv("../input/diabetes/diabetes.csv")

raw = pd.get_dummies(raw, columns=['class'], drop_first=True)
raw.describe()
raw.head()
x = raw.drop(['class_tested_positive'], axis=1)

y = raw['class_tested_positive']
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, classification_report

gnb = GaussianNB()

gnb.fit(x, y)

x_predicted = gnb.predict(x)



print('分類演算法:GaussianNB')

print('預測 class_tested_positive\n')

print('正確率:')

print(accuracy_score(y,x_predicted ))

print('\n')

print('混淆矩陣:')

print(confusion_matrix(y,x_predicted))

print('\n')

print('分類報告:')

print(classification_report(y,x_predicted))
#丟入資料'2','1','0','0','2','1','2','20'預測結果

t = pd.DataFrame(['2','1','0','0','2','1','2','20']).T

x_predicted1 = gnb.predict(t)

print('資料2,1,0,0,2,1,2,20之分類結果為:')

print(x_predicted1) 