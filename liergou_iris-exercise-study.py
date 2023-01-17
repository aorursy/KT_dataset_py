import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from sklearn.tree import DecisionTreeClassifier #分类树
from sklearn.model_selection  import train_test_split#测试集和训练集
from sklearn.feature_selection import SelectKBest #特征选择
from sklearn.feature_selection import chi2 #卡方统计量

from sklearn.preprocessing import MinMaxScaler  #数据归一化
from sklearn.decomposition import PCA #主成分分析

#网格搜索交叉验证，用于选择最优的参数
from sklearn.model_selection import GridSearchCV
# 管道
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore', category=FutureWarning)

iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature_C = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
#读取数据
path = '../input/iris.data'
data = pd.read_csv(path, header=None)

data.head()
x=data[list(range(4))]#获取X变量
print(data[4].value_counts())
#把Y转换成分类型的0,1,2
y=pd.Categorical(data[4]).codes
print("总样本数目：%d;特征属性数目:%d" % x.shape)

print(data.head(5))

#数据进行分割（训练数据和测试数据）
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size=0.8, random_state=14)
x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))
## 因为需要体现是分类模型，因为DecisionTreeClassifier是分类算法，要求y必须是int类型
y_train = y_train.astype(np.int)
y_test = y_test.astype(np.int)
type(y_train),type(x_train)

#数据标准化
#StandardScaler (基于特征矩阵的列，将属性值转换至服从标准正态分布 N(0, 1))
#标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下
#常用与基于正态分布的算法，比如回归

#数据归一化
#MinMaxScaler （区间缩放，基于最大最小值，将数据转换到-1,1区间上的）
#提升模型收敛速度，提升模型精度
#常见用于神经网络
ss = MinMaxScaler()
#用标准化方法对数据进行处理并转换
## scikit learn中模型API说明：
### fit: 模型训练；基于给定的训练集(X,Y)训练出一个模型；该API是没有返回值；eg: ss.fit(X_train, Y_train)执行后ss这个模型对象就训练好了
### transform：数据转换；使用训练好的模型对给定的数据集(X)进行转换操作；一般如果训练集进行转换操作，那么测试集也需要转换操作；这个API只在特征工程过程中出现
### predict: 数据转换/数据预测；功能和transform类似，都是对给定的数据集X进行转换操作，只是transform中返回的是一个新的X, 而predict返回的是预测值Y；这个API只在算法模型中出现
### fit_transform: fit+transform两个API的合并，表示先根据给定的数据训练模型(fit)，然后使用训练好的模型对给定的数据X进行转换操作
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
print ("原始数据各个特征属性的调整最小值:",ss.min_)
print ("原始数据各个特征属性的缩放数据值:",ss.scale_)

#Normalizer （基于矩阵的行，将样本向量转换为单位向量）
#其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准
#常见用于文本分类和聚类、logistic回归中也会使用，有效防止过拟合


#特征选择：从已有的特征中选择出影响目标值最大的特征属性
# 类比：l1正则 线性回归 稀疏解
#常用方法：
# { 分类：F统计量、卡方系数，互信息mutual_info_classif
#{ 连续 回归：皮尔逊相关系数 F统计量 互信息mutual_info_classif

#SelectKBest（卡方系数）
#在当前的案例中，使用SelectKBest这个方法从4个原始的特征属性，选择出来3个
ch2 = SelectKBest(chi2, k=3)
#K默认为10
#如果指定了，那么就会返回你所想要的特征的个数
x_train = ch2.fit_transform(x_train, y_train)#训练并转换
x_test = ch2.transform(x_test)#转换

select_name_index = ch2.get_support(indices=True)
print ("对类别判断影响最大的三个特征属性分布是:", ch2.get_support(indices=False))
print(select_name_index)

#降维：对于数据而言，如果特征属性比较多，在构建过程中，会比较复杂，
# 这个时候考虑将多维（高维）映射到低维的数据
# 特征进行融合 所有的特征都考虑进去 得到新的特征
#常用的方法：
#PCA：主成分分析（无监督）
#LDA：线性判别分析（有监督）类内方差最小，人脸识别，通常先做一次pca

pca = PCA(n_components=2)#构建一个pca对象，设置最终维度是2维
# #这里是为了后面画图方便，所以将数据维度设置了2维，一般用默认不设置参数就可以

x_train = pca.fit_transform(x_train)#训练并转换
x_test = pca.transform(x_test)#转换
#模型的构建
model = DecisionTreeClassifier(
    criterion='entropy', # default is gini
    splitter='best',#default is best. or random.  split strategy
    max_depth=None, # the maximun depth of the tree. if none ,the nodes are expanded until all leaves is pure or until all leaves contains less than min_samples_split samples
    min_samples_split=2, #default is 2. the minimun number of samples required to split an internal node.
    min_samples_leaf=1, #default is 1. the mininum number of samoles required tobe at a leaf node.
    min_weight_fraction_leaf=0,# default is 0 .the mininum weighted fraction of the sum total of weights. default is  equal weight.
    max_features=None,# default is none ,can be  int  float auto(sqrt) sqrt, log2 . default is  max_features = n_features
    max_leaf_nodes=None,# if none the unlimted number of leaf nodes.
    random_state=0,
    
)#另外也可选gini
#模型训练
model.fit(x_train, y_train)
#模型预测
y_test_hat = model.predict(x_test)

#模型结果的评估
y_test2 = y_test.reshape(-1)
result = (y_test2 == y_test_hat)
print(result)
print ("准确率:%.2f%%" % (np.mean(result) * 100))
#实际可通过参数获取
print ("Score：", model.score(x_test, y_test))#准确率
print ("Classes:", model.classes_)
print("获取各个特征的权重:", end='')
print(model.feature_importances_)
#画图
N = 100  #横纵各采样多少个值
x1_min = np.min((x_train.T[0].min(), x_test.T[0].min()))
x1_max = np.max((x_train.T[0].max(), x_test.T[0].max()))
x2_min = np.min((x_train.T[1].min(), x_test.T[1].min()))
x2_max = np.max((x_train.T[1].max(), x_test.T[1].max()))

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)

x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
# 网格相交的点
print(x1)
print('*'*58)
print(x2)
# flat 扁平化操作
x_show = np.dstack((x1.flat, x2.flat))[0] #测试点

print(x_show.shape)

y_show_hat = model.predict(x_show) #预测值

y_show_hat = y_show_hat.reshape(x1.shape)  #使之与输入的形状相同
print(y_show_hat.shape)
print(y_show_hat[0])

# print(y_show_hat[0])
# print(y_show_hat[55])
# print(y_show_hat[58])
# print(x1[1])
#参数优化
pipe = Pipeline([
            ('mms', MinMaxScaler()),
            ('skb', SelectKBest(chi2)),
            ('pca', PCA()),
            ('decision', DecisionTreeClassifier(random_state=0))
        ])

# 参数
parameters = {
    "skb__k": [1,2,3,4],
    "pca__n_components": [0.5,0.99],#设置为浮点数代表主成分分析所占最小比例的阈值，这里不建议设置为数值
    "decision__criterion": ["gini", "entropy"],
    "decision__max_depth": [1,2,3,4,5,6,7,8,9,10]
}
#数据
x_train2, x_test2, y_train2, y_test2 = x_train1, x_test1, y_train1, y_test1
#模型构建：通过网格交叉验证，寻找最优参数列表， param_grid可选参数列表，cv：进行几折交叉验证
# GridSearchCV：网格交叉验证，主要用于模型开发阶段找出模型的最优参数的一种方式，内部会利用交叉验证
#
gscv = GridSearchCV(pipe, param_grid=parameters, cv=3)
#模型训练
gscv.fit(x_train2, y_train2)
#算法的最优解
print("最优参数列表:", gscv.best_params_)
print("score值：",gscv.best_score_)
print("最优模型:", end='')
print(gscv.best_estimator_)
#预测值
# y_test_hat2 = gscv.predict(x_test2)

#应用最优参数看效果
mms_best = MinMaxScaler()
skb_best = SelectKBest(chi2, k=3)
pca_best = PCA(n_components=0.99)
decision3 = DecisionTreeClassifier(criterion='gini', max_depth=4)
#构建模型并训练模型
x_train3, x_test3, y_train3, y_test3 = x_train1, x_test1, y_train1, y_test1
x_train3 =pca_best.fit_transform(skb_best.fit_transform(mms_best.fit_transform(x_train3), y_train3))
x_test3 = pca_best.transform(skb_best.transform(mms_best.transform(x_test3)))
decision3.fit(x_train3, y_train3)

print("最优参数看效果-正确率:", decision3.score(x_test3, y_test3))
# 基于原始数据前2列比较一下决策树在不同深度的情况下错误率
x_train4, x_test4, y_train4, y_test4 = train_test_split(x.iloc[:, :2], y, train_size=0.7, random_state=14)

depths = np.arange(1, 14)
err_list_test = []
err_list_train = []
for d in depths:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=d,
                                 min_samples_split=10)  # 仅设置了这二个参数，没有对数据进行特征选择和降维，所以跟前面得到的结果不同
    clf.fit(x_train4, y_train4)
    ## 计算的是在测试集上的模型预测能力
    score1 = clf.score(x_test4, y_test4)
    # 计算的是在训练集上的模型
    score2 = clf.score(x_train4, y_train4)
    err_list_test.append(1 - score1)
    err_list_train.append(1 - score2)
    print("%d深度，训练集上正确率%.5f" % (d, score2))
    print("%d深度，测试集上正确率%.5f\n" % (d, score1))



## 画图
plt.figure(facecolor='w')
plt.plot(depths, err_list_test, 'ro-', lw=3, label = 'test data set')
plt.plot(depths, err_list_train, 'bo-', lw=3, label = 'train data set')
plt.xlabel(u'decision deep', fontsize=16)
plt.ylabel(u'error rate', fontsize=16)
plt.legend(loc = 'upper right')
plt.grid(True)
plt.title(u'decision deep vs over fit', fontsize=18)
plt.show()




