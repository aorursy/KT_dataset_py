%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error as mse
Num_datasets = 50
noise_level = 0.5

N = 25

trainN = int(N * 0.9)
np.random.seed(2)
# common util function used to draw the boundary for classcification
def plot_decision_boundary(X, model):
    # 步长为0.02 下面np.arange(x_min, x_max, h)generate数组时候用到
    h = .02 
    # context: 在该实验中,每个样本是X(feature1, feature2).feature1 表示x 轴坐标，feature2表示y轴坐标.一共500个样本。label为2类用0，1表示
    # 因为这里要为所有数据点画边界，所以要找出每个数据样本在x轴 和y轴上的最大，最小值
    # X[:, 0].min(): 第0列 最小的， X[:, 0].max()： 第0列最大的，
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # X[:, 1].min(): 第1列 最小的， X[:, 1].max()： 第1列最大的
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # x_min, x_max 第0列 即feataure1 x坐标的最小的和最大的
    # y_min, y_max 第1列 最小的和最大的 即feataure2 y坐标的最小的和最大的
    
    # np.arange(x_min, x_max, h).shape (550,) np.arange(y_min, y_max, h).shape (534,)
    # 由于上面两个shape不一样，所以需要meshgrid. after meshgrid,output出来的xx 和yy 的shape 一模一样 都是 (534, 550)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # np.c_ 表示：xx.ravel()变成一位数组后，作为第一列，yy.ravel()变成一位数组后，作为第二列
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z.shape (293700,)
    Z = Z.reshape(xx.shape)
    # Z.shape (534, 550)
    
    # xx表示每个数据点的x 坐标， yy  表示每个数据点的y坐标 X (xx,yy) = X (featur1, feature2) Z 表示预测出来的类别
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contour.html
    # xx, yy must both be 2-D with the same shape as Z (e.g. created via numpy.meshgrid()),
    # The height values over which the contour is drawn
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
class BaggedTreeClassifier:
    def __init__(self, M):
        self.M = M # M个决策树
        
    def fit(self, X, Y):
        N = len(X)
        self.models = list() # 袋子,用来装models
        for m in range(self.M): # 对于每个model, 有放回的采样N个数据
            # return index
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]
            model = DecisionTreeClassifier(max_depth=5)
            model.fit(Xb, Yb)
            self.models.append(model)
        # 创建深度为5的决策树

    def predict(self,X):
        # 二分类预测
        # predictions list 里每个item代表一个样本的的预测值 0 OR 1. 为了方便理解  可以假想成 2维表： M * len(X) 
        # 相当于每个model过来都把X里所有sample 预测一遍,最后一行 所有预测的和 i.e predictions += model.predict(X),
        #  np.round(predictions / self.M) 和巧妙的四舍五入 相当于投票的效果，大于1半models投哪个类，结果就是哪个类别
        #    X1, X2, X3, X4
        # M1  0   1   0   1
        # M2  1   0   1   0
        # M3  1   1   0   0
        #     2   2   1   1
        predictions = np.zeros(len(X))
        
        # AICamp Ensemble Exercise 2 ada710 里面有解释过
        # 这里是二分类问题，输出的算法参考 Week6.Session1.Ensemble_Theory.pdf page36
        # 对于多分类问题，输出，参考Week6.Session1.Ensemble_Theory.pdf page 33- 36
        for model in self.models:
            predictions += model.predict(X)
        return np.round(predictions / self.M)
    
    def score(self,X,Y):
        result = self.predict(X)
        return np.mean(result == Y)
        
## 决策树的代码 

# after setting seed, 每次random函数generate出来的值是一样的
np.random.seed(10)

N = 500
D = 2
# randn 标准0，1正态分布 N 函数，D列数
X = np.random.randn(N, D) # X.shape (500,2)

delta = 1.75 # 实际数据有些噪音
X[:125] += np.array([delta, delta])
X[125:250] += np.array([delta, -delta])
X[250:375] += np.array([-delta, delta])
X[375:] += np.array([-delta, -delta])

X.shape
# 由于X.shape (500, 2), 这里generate 500lable. i.e. 125个0， 125 个1， 125 个1， 125 个0 -> 125 * 4 = 500 个label
Y_array = [0] * 125 + [1]*125 + [1]*125 + [0] * 125
print (Y_array)
Y = np.array(Y_array)
Y.shape
# scatter reference doc: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html

# s: scalar or arry_like. shape(n,) optional
# c: color. sequence of color c=Y, 由于Y里只有0和1 两类，所以只有两种颜色. 每种颜色表中一个类别
# alpha:透明度 scalar, optional, default: None. The alpha blending value, between 0 (transparent) and 1 (opaque).
# X.shape=(500, 2). i.e. X[:,0] 表示X的第一列数据点 作为 x axis. X[:,1]表示X的第二列数据点 作为y axis. X(X1, X2...X500) X1(col1, col2), col1表示x轴坐标，col2表示y轴坐标
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
# without setting s=100
plt.scatter(X[:,0], X[:,1],c=Y, alpha=0.5)
plt.show()
# without setting color:
plt.scatter(X[:,0], X[:,1], s=100, alpha=0.5)
plt.show()
# Fixing random state for reproducibility
np.random.seed(19680801)

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
print(colors)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
# generate 原始数据点集 和 label 
# after setting seed, 每次random函数generate出来的值是一样的
np.random.seed(10)

N = 500
D = 2
# randn 标准0，1正态分布 N 函数，D列数
X = np.random.randn(N, D) # X.shape (500,2)

delta = 1.75 # 实际数据有些噪音
# X.shape (500,2)
# X[:125] += np.array([delta, delta]) 就是0 到124行，每一行的第一列 +delta， 第二列+delta
X[:125] += np.array([delta, delta])
# e.g 140行  [ 3.08730551e+00, -2.39016584e+00] --> [ 1.33730551e+00, -6.40165842e-01],
X[125:250] += np.array([delta, -delta])
X[250:375] += np.array([-delta, delta])
X[375:] += np.array([-delta, -delta])
Y = np.array([0] * 125 + [1]*125 + [1]*125 + [0] * 125)

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()

# 基础树模型
model = DecisionTreeClassifier()
# use different model and compare the boundaries.
#model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
model.fit(X, Y)
print("score for basic tree:", model.score(X, Y))

# plot data with boundary
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
# 调用plot_decision_boundary函数 draw 两个类别的boundary “Understand plot_decision_boundary function”
# 具体解释见： https://www.kaggle.com/jungan/numpy-matplotlib-complete-version-cd7c52
plot_decision_boundary(X, model)
plt.show()

# 树的深度为3的模型
model_depth_3 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model_depth_3.fit(X, Y)

print("score for tree of depth 3:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()
# 树的深度为5的模型
model_depth_5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
model_depth_5.fit(X, Y)

print("score for tree of depth 5:", model_depth_5.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_5)
plt.show()
#logistic regression模型
model_logistic = LogisticRegression()
model_logistic.fit(X, Y)

print("score for logistic regression model:", model_logistic.score(X, Y))

# 可视化
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_logistic)
plt.show()  
# bagged tree 模型
baggedTree = BaggedTreeClassifier(200)
baggedTree.fit(X, Y)
print("score for bagged tree model:", baggedTree.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, baggedTree)
plt.show()
## 决策树代码

# 生成甜甜圈数据点 - start
np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

R_smaller = 5
R_larger = 10

R1 = np.random.randn(N//2) + R_smaller
theta = 2 * np.pi * np.random.random(N//2)
X[:250] = np.concatenate([[R1 * np.cos(theta)], [R1*np.sin(theta)]]).T


R2 = np.random.randn(N//2) + R_larger
theta = 2 * np.pi * np.random.random(N//2)
X[250:] = np.concatenate([[R2 * np.cos(theta)], [R2*np.sin(theta)]]).T

Y = np.array([0] * (N//2) + [1] * (N//2))


# plot the data
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
# 生成甜甜圈数据点 - end

# 基础树模型
model = DecisionTreeClassifier()
model.fit(X, Y)
print("score for basic tree:", model.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()


# 树的深度为3的模型
model_depth_3 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model_depth_3.fit(X, Y)

print("score for tree of depth 3:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()

# 树的深度为5的模型
model_depth_5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
model_depth_5.fit(X, Y)

print("score for tree of depth 5:", model_depth_5.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_5)
plt.show()


# logistic regression 模型
model_logistic = LogisticRegression()
model_logistic.fit(X, Y)

print("score for logistic regression model:", model_logistic.score(X, Y))

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_logistic)
plt.show()

# bagged tree 模型
#TODO
baggedTree = BaggedTreeClassifier(200)
baggedTree.fit(X,Y)
print("score for bagged tree:", baggedTree.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, baggedTree)
plt.show()
