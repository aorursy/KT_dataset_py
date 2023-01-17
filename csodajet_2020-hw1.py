import sys

import pandas as pd

import numpy as np

#导入数据集,big5为繁体中文字符标准,data:<class 'pandas.core.frame.DataFrame'>

data = pd.read_csv('../input/ml2020spring-hw1/train.csv',encoding="big5")

#截取csv文件数值部分

data = data.iloc[:, 3:]

#pandas逻辑运算，取数值为'NR'的子集置0

data[data == 'NR'] = 0

#raw_data:<class 'numpy.ndarray'>

raw_data = data.to_numpy()
month_data = {}

for month in range(12):

    #创建18*480大小的临时数组，存储每个月里 20天*24小时*18个指标 的数据

    sample = np.empty([18, 480])

    #按天数移动数据

    for day in range(20):

        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]

    #将每个月的数据存放在month_data字典里

    month_data[month] = sample
#x存储重新分配的数据，将每个月480小时，按每9小时一组，划分为471组，每组数据大小为 9小时*18个指标。

x = np.empty([12 * 471, 18 * 9], dtype = float)

#y存储每组第10小时的值

y = np.empty([12 * 471, 1], dtype = float)



for month in range(12):

    for day in range(20):

        for hour in range(24):

            if day == 19 and hour > 14:

                continue

            #reshape(1，-1)将每组 9小时*18个指标 的数组重新组织为一行

            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) 

            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] 
#mean()求均值，axis=0对各列求均值

mean_x = np.mean(x, axis = 0) 

#std()求标准差，axis=0d对各列求标准差

std_x = np.std(x, axis = 0)

#标准差归一化，参考https://www.cnblogs.com/LBSer/p/4440590.html

for i in range(len(x)): # 12*471

    for j in range(len(x[0])): 

        if std_x[j] != 0:

            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

print(x)

print(y)
#18*9个权重+1个偏置

dim = 18 * 9 + 1

#zeros()生成全0数组，ones()生成全1数组

w = np.zeros([dim, 1])

#concatenate()数组拼接，axis=1对行操纵

#astype()数据类型转换

x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)



learning_rate = 100 #学习率

iter_time = 1000    #迭代次数



adagrad = np.zeros([dim, 1])    

eps = 0.0000000001

for t in range(iter_time):

    yt = np.dot(x, w)

    #损失函数

    #power()数组元素n次方，dot()点乘，sum()所有元素和

    loss = np.sqrt(np.sum(np.power(yt - y, 2))/471/12)



    if(t%100==0):#每迭代100次输出一次loss

        print(str(t) + ":" + str(loss))

    

    #Adagrad算法

    #transpose()数组转置

    gradient = 2 * np.dot(x.transpose(), yt - y)

    adagrad += gradient ** 2

    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)   #eps避免分母为0

#保存参数

np.save('./weight.npy', w)
testdata = pd.read_csv('../input/ml2020spring-hw1/test.csv', header = None, encoding = 'big5')

testdata = testdata.iloc[:, 2:]

testdata[testdata == 'NR'] = 0

test_data = testdata.to_numpy()



test_x = np.empty([240, 18*9], dtype = float)

for i in range(240):

    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)

for i in range(len(test_x)):

    for j in range(len(test_x[0])):

        if std_x[j] != 0:

            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]

test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
w = np.load('./weight.npy')

ans_y = np.dot(test_x, w)
import csv

with open('./submit.csv', mode='w', newline='') as submit_file:

    csv_writer = csv.writer(submit_file)

    header = ['id', 'value']

    print(header)

    csv_writer.writerow(header)

    for i in range(240):

        row = ['id_' + str(i), ans_y[i][0]]

        csv_writer.writerow(row)

        print(row)