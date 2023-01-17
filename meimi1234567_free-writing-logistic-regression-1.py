from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = load_breast_cancer()
x = data.data
y = data.target
x.shape #查看資料型態
lrl1 = LR(penalty='l1',solver='liblinear',C = 0.5,max_iter = 1000)
lrl2 = LR(penalty='l2',solver='liblinear',C = 0.5,max_iter = 1000)
#使用邏輯回歸的重要屬性coef_查看每個特徵所對應的參數
lrl1 = lrl1.fit(x,y)
lrl1.coef_
(lrl1.coef_!=0).sum(axis = 1)#總和不為0的係數有多少個（30個係數有10個系數不為0）
lrl2 = lrl2.fit(x,y)
lrl2.coef_
(lrl2.coef_!=0).sum(axis = 1)#總和不為0的係數有多少個
l1 = []
l2 = []
l1test = []
l2test = []
#監控訓練集和預測的結果差異
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state=420)

#C的學習曲線
for i in np.linspace(0.05,2,19):   #0.05~1之間取19個數字
    lrl1 = LR(penalty='l1',solver='liblinear',C = i,max_iter = 1000)
    lrl2 = LR(penalty='l2',solver='liblinear',C = i,max_iter = 1000)
    lrl1 = lrl1.fit(xtrain,ytrain)
    l1.append(accuracy_score(lrl1.predict(xtrain),ytrain))
    l1test.append(accuracy_score(lrl1.predict(xtest),ytest))
    lrl2 = lrl2.fit(xtrain,ytrain)
    l2.append(accuracy_score(lrl2.predict(xtrain),ytrain))
    l2test.append(accuracy_score(lrl2.predict(xtest),ytest))
graph = [l1,l2,l1test,l2test]
color = ['green','black','lightgreen','blue']
label = ['L1','L2','L1test','L2test']

plt.figure(figsize=(6,6))
for i in range(len(graph)):
    plt.plot(np.linspace(0.05,2,19)
             ,graph[i],color[i],label = label[i])
plt.legend(loc=4)   #loc=圖例位置在哪,4為右下角
plt.show() 
#0.8以後模型在未知數據集上的表現開始下跌，但訓練集卻持續向上，此時出現過擬和現象