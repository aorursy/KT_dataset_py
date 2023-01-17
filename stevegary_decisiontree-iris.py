#coding=utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

mpl.rcParams['font.sans-serif']=['simhei']
mpl.rcParams['axes.unicode_minus']=False
#利用决策树对鸢尾花数据分类
#只使用鸢尾花数据的花瓣长度和花瓣宽度进行预测
data = load_iris()
x_train,x_test,y_train,y_test = train_test_split(data.data[:,2:4],data.target,test_size=0.3)
pipe = Pipeline([("tree",DecisionTreeClassifier(criterion="entropy"))])


#首先选择最佳深度
#第一种
y = []
for i in range(1,15):
    pipe.set_params(tree__max_depth=i).fit(x_train,y_train)
    y.append(pipe.score(x_test,y_test))
    
plt.xlabel(u"depth")
plt.ylabel(u"acc")
plt.title(u"one-shot val")
plt.plot(range(1,15),y,marker="*",ms=10)
plt.show()
#第二种交叉验证
y = []
for i in range(1,15):
    pipe.set_params(tree__max_depth=i)
    score = cross_val_score(pipe._final_estimator,data.data[:,2:4],data.target,cv=10)
    y.append(np.mean(score))
plt.xlabel(u"depth")
plt.ylabel(u"acc")
plt.title(u"ten-entropy val")
plt.plot(range(1,15),y,marker="*",ms=10)
plt.show()



# 可以得出最佳的深度是4
data0 = data.data[data.target==0][:,2:4]
data1 = data.data[data.target==1][:,2:4]
data2 = data.data[data.target==2][:,2:4]
#训练模型
pipe.set_params(tree__max_depth=4).fit(data.data[:,2:4],data.target)
#用颜色在图中区分每一个区域
#定义10000个网格点，用模型去预测
x=np.linspace(np.min(data.data[:,2]),np.max(data.data[:,2]),100)
y=np.linspace(np.min(data.data[:,3]),np.max(data.data[:,3]),100)
x,y = np.meshgrid(x,y)
pre_x = np.stack([x.ravel(),y.flat],axis=1)
pre_y = pipe.predict(pre_x)
pre_data0 = pre_x[pre_y==0]
pre_data1 = pre_x[pre_y==1]
pre_data2 = pre_x[pre_y==2]

plt.title(u"classification")
plt.xlabel(u"petal_length")
plt.ylabel(u"petal_width")
plt.scatter(pre_x[pre_y==0][:,0],pre_x[pre_y==0][:,1],color="lightgray")
plt.scatter(pre_x[pre_y==1][:,0],pre_x[pre_y==1][:,1],color="wheat")
plt.scatter(pre_x[pre_y==2][:,0],pre_x[pre_y==2][:,1],color="lightskyblue")
plt.scatter(data0[:,0],data0[:,1],marker="*")
plt.scatter(data1[:,0],data1[:,1],marker="^")
plt.scatter(data2[:,0],data2[:,1],marker="v")
plt.xlim(np.min(data.data[:,2]),np.max(data.data[:,2]))
plt.ylim(np.min(data.data[:,3]),np.max(data.data[:,3]))
plt.show()
#随机森林
pipe = Pipeline([("rf",RandomForestClassifier(max_features=2))])
y=[]
for i in range(1,21):
    pipe.set_params(rf__n_estimators=i).fit(x_train,y_train)
    y.append(pipe.score(x_test, y_test))
plt.plot(range(1,21),y)
plt.title(u"The effect of the number of subtrees on the accuracy rate")
plt.xlabel(u"the number of subtrees")
plt.ylabel(u"acc")
plt.show()