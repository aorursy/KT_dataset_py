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
    y_min, y_max = X[:, 1].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    plt.contour(xx, yy, z, cmap=plt.cm.Paired)
    

np.random.seed(10)    # 固定随机种子

N = 500    #  数据点的个数
D = 2    #  数据的维度 
X = np.random.randn(N,D)    #  创建N * D 的高斯随机数据

delta = 1.5

# 赋值数据和标签
X[:N//2] += np.array((delta, delta))
X[N//2:] += np.array((-delta, -delta))   
Y = np.array([0] * (N//2) + [1] * (N//2))

# 画图
"""
DO NOT edit code below
"""
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()


model = DecisionTreeClassifier()     # 直接调用决策树模型
model.fit(X, Y)
print("score for basic tree:", model.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model)
plt.show()
model_depth_3 = DecisionTreeClassifier(criterion='entropy',max_depth=3)    # 调用决策树模型并限制树深度为3
#TODO 
#模型在X, Y数据集上fit
model_depth_3.fit(X,Y)
print("score for basic tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model_depth_3)
plt.show()
model_depth_5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)# 调用决策树模型并限制树深度为5
#TODO 
#模型在X, Y数据集上fit
model_depth_5.fit(X, Y)
print("score for basic tree:", model_depth_5.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model_depth_5)
plt.show()

np.random.seed(10)   #固定随机种子
"""
画出分散的四个点集
"""
N = 500    #  数据点的个数
D = 2    #  数据的维度 
X = np.random.randn(N,D)    #  创建N * D 的高斯随机数据

delta = 1.75

# 赋值给X和标签Y
"""
这里X的划分我想按照第一区间～第四区间的顺序来，但是会影响后面model.score的结果
"""
X[:125] += np.array([delta, delta])
X[125:250] += np.array([-delta, delta])
X[250:375] += np.array([-delta, -delta])
X[375:] += np.array([delta, -delta])
Y = np.array([0] * 125 + [1] * 125 + [1] * 125 + [1] * 125)


#可视化
"""
DO NOT edit code below
"""
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
# 将模型分别在X, Y数据上fit
#TODO
model.fit(X, Y)

# 打印决策树模型的表现
print("score for basic tree:", model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model)
plt.show()


model_depth_3.fit(X,Y)

# 打印深度为3的决策树模型的表现
print("score for basic tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model_depth_3)
plt.show()

# 打印深度为5的决策树模型的表现
#TODO
model_depth_5.fit(X,Y)
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
print("score for basic tree:", model_depth_5.score(X, Y))
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model_depth_5)
plt.show()


np.random.seed(10)   #固定随机种子
"""
画出分散的四个点集
"""
N = 500    #  数据点的个数
D = 2    #  数据的维度 
X = np.random.randn(N,D)    #  创建N * D 的高斯随机数据

delta = 1.75

# 赋值给X和标签Y
"""
这里仍然按第一～第四区间：(+,+),(-,+),(-,-),(+,-)进行X的划分
但是最大深度为3和5的模型的model.score与课堂上(+,+),(+,-),(-,+),(-,-)划分得到的差异极大..
这是因为训练集不同造成的偶然结果，还是因为我这种方式划分得到的训练集不够科学？（后面两个cells里PO出了inclass的划分方案作为对比）
"""
X[:125] += np.array([delta, delta])
X[125:250] += np.array([-delta, delta])
X[250:375] += np.array([-delta, -delta])
X[375:] += np.array([delta, -delta])
Y = np.array([0]*125 + [1]*125 + [0]*125 + [1]*125)


#可视化
"""
DO NOT edit code below
"""
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
# 将模型分别在X, Y数据上fit
#TODO
model.fit(X, Y)
model_depth_3.fit(X, Y)
model_depth_5.fit(X, Y)

# 可视化每个模型的决策边界
#TODO
print('score for basic tree: ', model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()

print('score for basic tree: ', model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()

print('score for basic tree: ', model_depth_5.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_5)
plt.show()
np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

delta = 1.75
X[:125] += np.array([delta, delta])
X[125:250] += np.array([delta, -delta])
X[250:375] += np.array([-delta, delta])
X[375:] += np.array([-delta, -delta])
Y = np.array([0] * 125 + [1]*125 + [1]*125 + [0] * 125)

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
# 将模型分别在X, Y数据上fit
#TODO
model.fit(X, Y)
model_depth_3.fit(X, Y)
model_depth_5.fit(X, Y)

# 可视化每个模型的决策边界
#TODO
print('score for basic tree: ', model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()

print('score for basic tree: ', model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()

print('score for basic tree: ', model_depth_5.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_5)
plt.show()
np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

R_smaller = 5   # 内圈的半径大小
R_larger = 10    # 外圈的半径大小

R1 = R_smaller + np.random.randn(N//2)       # 内圈半径 + 随机扰动
theta = 2 * np.pi * np.random.random(N//2)    # 随机生成夹角
X[:250] = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T   #创建内圈数据点


R2 = R_larger + np.random.randn(N//2)     # 外圈半径 + 随机扰动
theta = 2 * np.pi * np.random.random(N//2)  # 随机生成夹角
X[250:] = np.concatenate([[R2 * np.cos(theta)], [R2*np.sin(theta)]]).T  #创建外圈数据点

Y = np.array([0] * 250 + [1] * 250)  #赋值标签

# 可视化
"""
Do Not edit code below
"""
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
# 将模型分别在X, Y数据上fit
#TODO
model.fit(X, Y)
model_depth_3.fit(X, Y)
model_depth_5.fit(X, Y)


# 可视化每个模型的决策边界
#TODO
print('basic tree score: ', model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()

print('basic tree score: ', model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()

print('basic tree score: ', model_depth_5.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_5)
plt.show()


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, Y)

print('logistic regression score: ', model.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()