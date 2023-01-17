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
    
    plt.contour(xx, yy, z, cmap=plt.cm.Paired)
    

np.random.seed(10)    # 固定随机种子

N =  500   #  数据点的个数
D =  2   #  数据的维度 
X =  np.random.randn(N, D)   #  创建N * D 的高斯随机数据

delta = 1.5

# 赋值数据和标签
X[:N//2] += [delta, delta]
X[N//2:] += [-delta, -delta]
Y = [0]*(N//2) + [1]* (N//2)

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
model_depth_3 = DecisionTreeClassifier(max_depth = 3) 
model_depth_3.fit(X, Y)
# 调用决策树模型并限制树深度为3
#TODO 
#模型在X, Y数据集上fit
print("score for basic tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model_depth_3)
plt.show()
model_depth_5 = DecisionTreeClassifier(max_depth = 5)  # 调用决策树模型并限制树深度为5
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
X[:125] += [delta, delta]
X[125:250] += [delta, -delta]
X[250:375] += [-delta, delta]
X[375:] += [-delta, -delta]
Y = [0]*125 + [1]*375


#可视化
"""
DO NOT edit code below
"""
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
# 将模型分别在X, Y数据上fit
#TODO
model.fit(X, Y)
model_depth_3.fit(X,Y)
model_depth_5.fit(X,Y)


# 打印决策树模型的表现
print("score for basic tree:", model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model)
plt.show()




# 打印深度为3的决策树模型的表现
print("score for maxdepth=3 tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model_depth_3)
plt.show()

# 打印深度为5的决策树模型的表现
#TODO
print('score for maxdepth =5 tree:', model_depth_5.score(X,Y))
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
X = np.random.randn(N, D)    #  创建N * D 的高斯随机数据

delta = 1.75

# 赋值给X和标签Y
X[:125] += [delta, delta]
X[125:250] += [delta, -delta]
X[250:375] += [-delta, delta]
X[375:] += [-delta, -delta]
Y = [0]*125 + [1]*250 + [0]*125


#可视化
"""
DO NOT edit code below
"""
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
# 将模型分别在X, Y数据上fit
#TODO
model.fit(X, Y)
model_depth_3.fit(X,Y)
model_depth_5.fit(X,Y)


# 可视化每个模型的决策边界
#TODO
# 打印决策树模型的表现
print("score for basic tree:", model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model)
plt.show()




# 打印深度为3的决策树模型的表现
print("score for maxdepth=3 tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model_depth_3)
plt.show()

# 打印深度为5的决策树模型的表现
#TODO
print('score for maxdepth =5 tree:', model_depth_5.score(X,Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model_depth_5)
plt.show()

np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)
print(X.shape)

R_smaller = 1   # 内圈的半径大小
R_larger =  5   # 外圈的半径大小

R1 =  R_smaller + np.random.randn(250)      # 内圈半径 + 随机扰动
print(R1.shape)
theta =   np.random.random(250)*2*np.pi  # 随机生成夹角
print(np.concatenate(([R1 * np.cos(theta)], [R1*np.sin(theta)]), axis = 0).shape)
X[:250] = np.concatenate([[R1 * np.cos(theta)], [R1*np.sin(theta)]]).T   #创建内圈数据点


R2 =  R_larger + np.random.randn(250)    # 外圈半径 + 随机扰动
theta =   np.random.random(250)*2*np.pi # 随机生成夹角
X[250:] = np.concatenate([[R2 * np.cos(theta)], [R2*np.sin(theta)]]).T  #创建外圈数据点

Y =  [0]*250 + [1]*250 #赋值标签

# 可视化
"""
Do Not edit code below
"""
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
# 将模型分别在X, Y数据上fit
#TODO
model.fit(X, Y)
model_depth_3.fit(X,Y)
model_depth_5.fit(X,Y)


# 可视化每个模型的决策边界
#TODO
# 打印决策树模型的表现
print("score for basic tree:", model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model)
plt.show()




# 打印深度为3的决策树模型的表现
print("score for maxdepth=3 tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model_depth_3)
plt.show()

# 打印深度为5的决策树模型的表现
#TODO
print('score for maxdepth =5 tree:', model_depth_5.score(X,Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
#调用plot_decision_boundary 函数画出决策边界
plot_decision_boundary(X, model_depth_5)
plt.show()

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
# linear dividable data
np.random.seed(10)    # 固定随机种子

N =  500   #  数据点的个数
D =  2   #  数据的维度 
X =  np.random.randn(N, D)   #  创建N * D 的高斯随机数据

delta = 1.5

# 赋值数据和标签
X[:N//2] += [delta, delta]
X[N//2:] += [-delta, -delta]
Y = [0]*(N//2) + [1]* (N//2)

log_model.fit(X, Y)
plt.scatter(X[:,0], X[:,1], c = Y, s = 100, alpha = 0.5)
plot_decision_boundary(X, log_model)
plt.show()

# linear undividable data
np.random.seed(10)   #固定随机种子
"""
画出分散的四个点集
"""
N = 500    #  数据点的个数
D = 2    #  数据的维度 
X = np.random.randn(N,D)    #  创建N * D 的高斯随机数据

delta = 1.75

# 赋值给X和标签Y
X[:125] += [delta, delta]
X[125:250] += [delta, -delta]
X[250:375] += [-delta, delta]
X[375:] += [-delta, -delta]
Y = [0]*125 + [1]*375

log_model.fit(X, Y)
plt.scatter(X[:,0], X[:,1], c = Y, s = 100, alpha = 0.5)
plot_decision_boundary(X, log_model)
plt.show()
# linear undividable (advanced)

np.random.seed(10)   #固定随机种子
"""
画出分散的四个点集
"""
N = 500    #  数据点的个数
D = 2    #  数据的维度 
X = np.random.randn(N, D)    #  创建N * D 的高斯随机数据

delta = 1.75

# 赋值给X和标签Y
X[:125] += [delta, delta]
X[125:250] += [delta, -delta]
X[250:375] += [-delta, delta]
X[375:] += [-delta, -delta]
Y = [0]*125 + [1]*250 + [0]*125

log_model.fit(X, Y)
plt.scatter(X[:,0], X[:,1], c = Y, s = 100, alpha = 0.5)
plot_decision_boundary(X, log_model)
plt.show()
# donut shape data
np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)
print(X.shape)

R_smaller = 1   # 内圈的半径大小
R_larger =  5   # 外圈的半径大小

R1 =  R_smaller + np.random.randn(250)      # 内圈半径 + 随机扰动
print(R1.shape)
theta =   np.random.random(250)*2*np.pi  # 随机生成夹角
print(np.concatenate(([R1 * np.cos(theta)], [R1*np.sin(theta)]), axis = 0).shape)
X[:250] = np.concatenate([[R1 * np.cos(theta)], [R1*np.sin(theta)]]).T   #创建内圈数据点


R2 =  R_larger + np.random.randn(250)    # 外圈半径 + 随机扰动
theta =   np.random.random(250)*2*np.pi # 随机生成夹角
X[250:] = np.concatenate([[R2 * np.cos(theta)], [R2*np.sin(theta)]]).T  #创建外圈数据点

Y =  [0]*250 + [1]*250 #赋值标签


log_model.fit(X, Y)
plt.scatter(X[:,0], X[:,1], c = Y, s = 100, alpha = 0.5)
plot_decision_boundary(X, log_model)
plt.show()