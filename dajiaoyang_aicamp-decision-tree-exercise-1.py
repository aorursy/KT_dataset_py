"""

DO NOT edit the code below

"""



%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
"""

DO NOT edit the code below

"""



def plot_decision_boundary(X, model):

    h = .02

    

    x_min, x_max = X[:, 0].min() -1, X[:,0].max() + 1

    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    z = z.reshape(xx.shape)

    

    plt.contour(xx, yy, z,  colors=['#808080'])

    

np.random.seed(10)    # 固定随机种子



N = 800     #  数据点的个数

D = 2    #  数据的维度 

X = np.random.randn(N,D)    #  创建N * D 的高斯随机数据



delta = 1.5



# 赋值数据和标签

X[:N // 2] += np.array([delta, delta])

X[N // 2:] += np.array([-delta, -delta])   

Y = np.array([0] * (N // 2) + [1] * (N // 2))



# 画图



"""

# DO NOT edit code below

"""



plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

plt.show()




model = DecisionTreeClassifier() # 直接调用决策树模型

model.fit(X, Y)

print("score for basic tree:", model.score(X, Y))


plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model)

plt.show()
model_depth_3 =  DecisionTreeClassifier(criterion='entropy', max_depth=3)   # 调用决策树模型并限制树深度为3

#TODO 

#模型在X, Y数据集上fit

model_depth_3.fit(X, Y)

print("score for basic tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model_depth_3)

plt.show()
model_depth_5 = DecisionTreeClassifier(criterion='entropy', max_depth=5) # 调用决策树模型并限制树深度为5

#TODO 

#模型在X, Y数据集上fit

model_depth_5.fit(X, Y)

print("score for basic tree:", model_depth_5.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model_depth_5)

plt.show()
np.random.seed(20)    # 固定随机种子



N = 800     #  数据点的个数

D = 2    #  数据的维度 

X = np.random.randn(N,D)    #  创建N * D 的高斯随机数据



delta = 1.75



# 赋值数据和标签

X[:200] += np.array([delta, delta])

X[200:400] += np.array([delta, -delta])   

X[400:600] += np.array([-delta, delta])   

X[600:] += np.array([-delta, -delta])   

Y = np.array([0] * 200 + [1] * 600)







#可视化



plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

plt.show()
# 将模型分别在X, Y数据上fit

#TODO

model.fit(X, Y)

model_depth_3.fit(X, Y)

model_depth_5.fit(X, Y)





# 打印决策树模型的表现

print("score for basic tree:", model.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model)

plt.show()



# 打印深度为3的决策树模型的表现

print("score for depth_3 tree:", model_depth_3.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model_depth_3)

plt.show()



# 打印深度为5的决策树模型的表现

#TODO

print("score for depth_5 tree:", model_depth_5.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model_depth_5)

plt.show()



np.random.seed(20)    # 固定随机种子



N = 800     #  数据点的个数

D = 2    #  数据的维度 

X = np.random.randn(N,D)    #  创建N * D 的高斯随机数据



delta = 1.75



# 赋值数据和标签

X[:200] += np.array([delta, delta])

X[200:400] += np.array([delta, -delta])   

X[400:600] += np.array([-delta, delta])   

X[600:] += np.array([-delta, -delta])   

Y = np.array([0] * 200 + [1] * 400 + [0] * 200)





#可视化

"""

DO NOT edit code below

"""

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

plt.show()
# 将模型分别在X, Y数据上fit

#TODO

model.fit(X, Y)

model_depth_3.fit(X, Y)

model_depth_5.fit(X, Y)





# 打印决策树模型的表现

print("score for basic tree:", model.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model)

plt.show()



# 打印深度为3的决策树模型的表现

print("score for depth_3 tree:", model_depth_3.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model_depth_3)

plt.show()



# 打印深度为5的决策树模型的表现

#TODO

print("score for depth_5 tree:", model_depth_5.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model_depth_5)

plt.show()
np.random.seed(10)



N = 800

D = 2

X = np.random.randn(N, D)



R_smaller = 6  # 内圈的半径大小

R_larger = 10    # 外圈的半径大小



R1 = np.random.randn(N // 2) + R_smaller       # 内圈半径 + 随机扰动

theta = 2 * np.pi * np.random.random(N // 2)    # 随机生成夹角

X[:N // 2] = np.concatenate([[R1 * np.cos(theta)], [R1*np.sin(theta)]]).T   #创建内圈数据点





R2 = np.random.randn(N // 2) + R_larger     # 外圈半径 + 随机扰动

theta = 2 * np.pi * np.random.random(N//2)  # 随机生成夹角

X[N // 2:] = np.concatenate([[R2 * np.cos(theta)], [R2*np.sin(theta)]]).T  #创建外圈数据点



Y = np.array([0] * (N // 2) + [1] * (N // 2))  #赋值标签



# 可视化

"""

Do Not edit code below

"""

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

plt.show()
model_depth_4 = DecisionTreeClassifier(criterion='entropy', max_depth=4)

model_depth_8 = DecisionTreeClassifier(criterion='entropy', max_depth=8)

# 将模型分别在X, Y数据上fit

#TODO

model.fit(X, Y)

model_depth_3.fit(X, Y)

model_depth_4.fit(X, Y)

model_depth_5.fit(X, Y)

model_depth_8.fit(X, Y)



# 打印决策树模型的表现

print("score for basic tree:", model.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model)

plt.show()



# 打印深度为3的决策树模型的表现

print("score for depth_3 tree:", model_depth_3.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model_depth_3)

plt.show()



# 打印深度为4的决策树模型的表现

print("score for depth_4 tree:", model_depth_4.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model_depth_4)

plt.show()



# 打印深度为5的决策树模型的表现

#TODO

print("score for depth_5 tree:", model_depth_5.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model_depth_5)

plt.show()



# 打印深度为8的决策树模型的表现

#TODO

print("score for depth_8 tree:", model_depth_8.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=50, c=Y, alpha=0.5)

#TODO 

#调用plot_decision_boundary 函数画出决策边界

plot_decision_boundary(X, model_depth_8)

plt.show()