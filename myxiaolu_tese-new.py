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
#读取数据
Train_Data = pd.read_csv('../input/train.csv')
Test_Data = pd.read_csv('../input/test.csv')

ID = Test_Data["Id"]
#删除训练集和测试集中的ID idhogar
del Train_Data["Id"]
del Train_Data["idhogar"]

del Test_Data["Id"]
del Test_Data["idhogar"]

Train_Data.head(10)
#将空值数据用 0 填充
Train_Data = Train_Data.fillna(0)
Test_Data = Test_Data.fillna(0)
Train_Data.head(10)
#将 dependency edjefa edjefe 三列中的yes 改为 1， no改为 0.
import numpy as np

Train_Data[Train_Data == "yes"] = np.nan
Train_Data = Train_Data.fillna(1)
Train_Data[Train_Data== "no" ] = np.nan
Train_Data = Train_Data.fillna(0)

Test_Data[Test_Data == "yes"] = np.nan
Test_Data = Test_Data.fillna(1)
Test_Data[Test_Data== "no" ] = np.nan
Test_Data = Test_Data.fillna(0)

Train_Data["dependency"] = Train_Data["dependency"].astype("float64")
Train_Data["edjefe"] = Train_Data["edjefe"].astype("float64")
Train_Data["edjefa"] = Train_Data["edjefa"].astype("float64")

Test_Data["dependency"] = Test_Data ["dependency"].astype("float64")
Test_Data["edjefe"] = Test_Data["edjefe"].astype("float64")
Test_Data["edjefa"] = Test_Data["edjefa"].astype("float64")

#将标签为1，2，3，4 的统计出来。
train_levels = Train_Data.loc[(Train_Data['Target'].notnull())]
label_counts = train_levels['Target'].value_counts().sort_index().to_frame()
target = label_counts
sum = target.sum()
target
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

label_list = ["Extreme Poverty", 
              "Moderate Poverty", 
              "Vulnerable", 
              "Non Vulnerable"]    # 各部分标签

size = [target['Target'][1]/sum*100, 
        target['Target'][2]/sum*100, 
        target['Target'][3]/sum*100,
        target['Target'][4]/sum*100]    # 各部分大小

color = ["red", "green", "blue","yellow"]     # 各部分颜色

explode = [0.05, 0, 0,0]   # 各部分突出值
"""
绘制饼图
explode：设置各部分突出
label:设置各部分标签
labeldistance:设置标签文本距圆心位置，1.1表示1.1倍半径
autopct：设置圆里面文本
shadow：设置是否有阴影
startangle：起始角度，默认从0开始逆时针转
pctdistance：设置圆内文本距圆心距离
返回值
l_text：圆内部文本，matplotlib.text.Text object
p_text：圆外部文本
"""
patches, l_text, p_text = plt.pie(size, explode=explode, 
                                  colors=color,
                                  labels=label_list, 
                                  labeldistance=1.1,
                                  autopct="%1.1f%%", 
                                  shadow=False,
                                  startangle=90,
                                  pctdistance=0.6)

plt.axis("equal")    # 设置横轴和纵轴大小相等，这样饼才是圆的

plt.legend()
plt.show()
X = Train_Data.drop(["Target"],1)
Y = Train_Data["Target"]
X.dtypes.value_counts()
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

seed = 9
test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test,predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
Y = Test_Data
X.dtypes.value_counts()
y_pred_test = model.predict(Y)
sum_ = len(y_pred_test)
sum_
one = 0
two = 0
th  = 0
fr  = 0
for i in y_pred:
    if i == 1:
        one =one + 1
    if i == 2:
        two = two + 1
    if i == 3:
        th =th + 1
    if i == 4:
        fr = fr + 1
one
two
th
fr
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

label_list = ["Extreme Poverty", 
              "Moderate Poverty", 
              "Vulnerable", 
              "Non Vulnerable"]    # 各部分标签

size = [one/sum_*100, 
        two/sum_*100, 
        th/sum_*100,
        fr/sum_*100]    # 各部分大小

color = ["red", "green", "blue","yellow"]     # 各部分颜色

explode = [0.05, 0, 0,0]   # 各部分突出值
"""
绘制饼图
explode：设置各部分突出
label:设置各部分标签
labeldistance:设置标签文本距圆心位置，1.1表示1.1倍半径
autopct：设置圆里面文本
shadow：设置是否有阴影
startangle：起始角度，默认从0开始逆时针转
pctdistance：设置圆内文本距圆心距离
返回值
l_text：圆内部文本，matplotlib.text.Text object
p_text：圆外部文本
"""
patches, l_text, p_text = plt.pie(size, explode=explode, 
                                  colors=color,
                                  labels=label_list, 
                                  labeldistance=1.1,
                                  autopct="%1.1f%%", 
                                  shadow=False,
                                  startangle=90,
                                  pctdistance=0.6)

plt.axis("equal")    # 设置横轴和纵轴大小相等，这样饼才是圆的

plt.legend()
plt.show()
Test_se = {
    "Id":ID,
    "target":y_pred_test
}

TEST = pd.DataFrame(Test_se)
TEST.to_csv("sample_submission.csv",index=False)
