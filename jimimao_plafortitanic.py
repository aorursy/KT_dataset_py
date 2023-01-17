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
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train = df_train.drop(['Name','Ticket','Cabin'],axis =1 )

df_train.loc[df_train["Sex"] == 'male','Sex'] = 1

df_train.loc[df_train["Sex"] == 'female','Sex'] = 0

df_train.loc[df_train["Embarked"] == 'C','Embarked'] = 0

df_train.loc[df_train["Embarked"] == 'Q','Embarked'] = 1

df_train.loc[df_train["Embarked"] == 'S','Embarked'] = 2

df_train = df_train.fillna(0)

traindata = df_train.iloc[:,2:]

traindata = (traindata - traindata.min()) / (traindata.max() - traindata.min())

print(traindata)

trainlabel = df_train.iloc[:,1:2]

trainlabel.loc[trainlabel['Survived'] == 0] =-1

print(trainlabel)

# print(df_train.isna())

# df_train

# pd.read_csv('/kaggle/input/titanic/train.csv')
# df_train.columns.values
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test = df_test.drop(['Name','Ticket','Cabin'],axis =1 )

df_test.loc[df_test["Sex"] == 'male','Sex'] = 1

df_test.loc[df_test["Sex"] == 'female','Sex'] = 0

df_test.loc[df_test["Embarked"] == 'C','Embarked'] = 0

df_test.loc[df_test["Embarked"] == 'Q','Embarked'] = 1

df_test.loc[df_test["Embarked"] == 'S','Embarked'] = 2

df_test = df_test.fillna(0)

testdata = df_test.iloc[:,1:]

testdata = (testdata - testdata.min()) / (testdata.max() - testdata.min())

print(testdata)

testPassengerId = df_test['PassengerId']

print(testPassengerId)
#权重矩阵初始化

def defineweight(n):

    w = [0] * (n+1)

    w = np.mat(w)

    return w



# t 为阈值

def predict(weight,data):

    a = np.array([1] * (data.shape[0]))

    X = np.array(data)

    X = np.insert(X,0,values=a,axis=1)

    # print(X)

    #给数据添加上一行1，乘上权重的第一个数值，表示最后的  -threshold，

    res = np.mat(weight) * np.transpose(np.mat(X)) ## * == np.dot(),矩阵乘法

    prediction = np.sign(res)

    return prediction



#更新权重矩阵

def renewweight(weight,prediction,target,lr,Train_data,m):

    a = np.array([1] * Train_data.shape[0])

    X = np.array(Train_data)

    X = np.insert(X,0,values=a,axis=1)



    prediction = np.transpose(prediction)

    for i in range(0,m):

        weight = np.mat(weight) + lr * (target[i] - prediction[i]) * np.mat(X[i])

    return weight



#计算正确率

def acc(pre,data_target):

    l = data_target.shape[0]

    tt = np.array(pre) + np.transpose(np.array(data_target))

    wrong = np.sum(tt == 0)

    return 1- wrong/l

if __name__ == "__main__":

    #超参数

    lr = 0.01

    epoch = 200

    Train_data = traindata.values

    Train_target = trainlabel.values

    print(Train_target)

    Test_data = testdata.values

    m = Train_data.shape[0] #数据 训练集长度

    n = Train_data.shape[1] #数据 维度

       #训练

    weight = defineweight(n) # 初始化权重矩阵

    # print(weight)

    prediction = [0]*m

    for i in range(0,epoch):

        # 预测结果

        prediction = predict(weight,Train_data)

        # 对于一条数据的 n 个维度，都要更新权重, t作为threshold 也要更新

        weight = renewweight(weight, prediction, Train_target, lr, Train_data,m)

        # print(weight)

        #利用目前的weight与threshold，对现有的训练数据集进行预测，并记录

        pre_train = predict(weight,Train_data)

        acc_train = acc(pre_train,Train_target)

        # print(pre_train)

        print("TrainACC%d:%.5f"%(i,acc_train))

#         acclist.append(acc_train * 100)

#         print(weight[0,0])

    pre = predict(weight,Test_data)

#     print(pre)

    pre_target = pd.DataFrame(pre.transpose(),columns=['Survived'])

    submission = pd.concat([testPassengerId,pre_target],axis = 1)

    submission['Survived'] = submission['Survived'].astype("int")

    submission.loc[submission['Survived']== -1,'Survived'] = 0

    print(submission)

    submission.to_csv('submission.csv', header=True, index=False)