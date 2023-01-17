%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()
with tf.Session() as sess:  # with this method, you don't need to call "session.close" explicitly 
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
print(result)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
print(result)
init = tf.global_variables_initializer() # sess.run
with tf.Session() as sess:
    init.run()
    print(x.eval())
    print(y.eval())
    print(f.eval())
    result = sess.run(f, feed_dict={x:1, y:0}) # through feed_dict: change the previouly set values for x and y
print(result)

def plot_decision_boundary(X, model):
    h = .02 

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))


    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

# linear function to be predicated/learned
def f(X):
    """
    input: x
    output: y = 3x + 4
    """
    return 3*X + 4
N = 40
noise_level = 0.8
trainX = np.linspace(-4.0, 4.0, N) # from -4 to 4, N=40 datapoints
np.random.shuffle(trainX)
trainY = f(trainX) + np.random.randn(N) * noise_level # generate training labels by adding some noise. randn  高斯分布

learning_rate = 0.01
training_epochs = 1000 # run 1000 time against all training data
display_step = 50 # show logs every 50 epochs 
plt.scatter(trainX, trainY)
np.random.randn()
class LinearRegressionTF(BaseEstimator):
    def __init__(self, learning_rate, training_epochs, display_step, annotate=False):
        self.annotate = annotate
        self.sess = tf.Session() # Tensflow session instance
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.display_step = display_step
        
        
    def fit(self, trainX,trainY):
        N = trainX.shape[0] # see how many training data i.e. row number
        # 图的输入
        self.X = tf.placeholder("float")
        self.Y = tf.placeholder("float")
        
        # 参数的定义, also give intial value to those parameters.之所以定义为变量，因为在train的过程中会变化
        #  y = 3 * X + 4
        self.W = tf.Variable(np.random.randn(), name="weight") # randomly initialize a value for weight
        self.b = tf.Variable(np.random.randn(), name="bias") # randomly initialize a value for bias
        
        # 线性模型 i.e. Y = W * X + b. 这里也可以直接用+号，tf.add是可以在tensorboard上显示的，debug的时候能看见
        self.pred = tf.add(tf.multiply(self.X, self.W), self.b) 
        
        # Mean Squre Error 分母是2N 或 N 都可以
        cost = tf.reduce_sum(tf.pow(self.pred-self.Y, 2))/(2*N)
        
        # 反向梯度优化时的优化器
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        
        # 初始化所有的参数
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # intial graph before trainning data to fit 
        if self.annotate:
            plt.plot(trainX, trainY, 'ro', label='Original data')
            #self.sess.run(self.W) can view/monitor each individual variable value
            plt.plot(trainX, self.sess.run(self.W) * trainX + self.sess.run(self.b), label='Fitted line')
            plt.legend()
            plt.title("This is where model starts to learn!!")
            plt.show()
            
        # 训练开始
        for epoch in range(self.training_epochs):
            for (x, y) in zip(trainX, trainY): # go through each training data sample in one epoch
                # regular one: result = sess.run(f, feed_dict={x:1, y:0})
                # pass in optimizer function wihch is defined ealier, and each trian sample data points
                self.sess.run(optimizer, feed_dict={self.X: x, self.Y: y}) # set self.X = x, self.Y = y

            #展示训练结果 for each epoch 
            if (epoch+1) % display_step == 0: # because epoch start with 0, so here epoch +1. i.e. epoch =49 means 50 epoch
                # pass in cost function, which is defined earlier. When calculate cost, we need all data points..that's why self.X = trainX
                c = self.sess.run(cost, feed_dict={self.X: trainX, self.Y:trainY}) # set self.X = trainX, self.Y = trainY
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
            "W=", self.sess.run(self.W), "b=", self.sess.run(self.b))      # \ means change line, 
                
            #显示拟合的直线 for each epoch, 和initial graph会不同，因为变量 W 和 b 是一直改变的
                if self.annotate:
                    plt.plot(trainX, trainY, 'ro', label='Original data')
                    plt.plot(trainX, self.sess.run(self.W) * trainX + self.sess.run(self.b), label='Fitted line')
                    plt.legend()
                    plt.show()
                #plt.pause(0.5)

        print("Optimization Finished!")
        # 训练结束，看一下cost. self.sess.run 可以显示一个特定变量，cost变量计算需要输入trainX, trainY数据
        training_cost = self.sess.run(cost, feed_dict={self.X: trainX, self.Y: trainY})
        print("Training cost=", training_cost, "W=", self.sess.run(self.W), "b=", self.sess.run(self.b), '\n')

        
    def predict(self, testX):
        #self.pred is defined in fit function: self.pred = tf.add(tf.multiply(self.X, self.W), self.b)
        # 虽然self.pred 用到，X, W,b 由于训练结束后，W, b都是已知的了，所以只需要传入textX 就可以了
        prediction = self.sess.run(self.pred,feed_dict={self.X: testX}) # just pass in real testing data i.e. testX，
        return prediction
    
    def score(self, testX, testY):
        result = self.predict(testX)
        return r2_score(testY, result) # score (actual lable, predicated value)
        
    
lr = LinearRegressionTF(learning_rate, 1000, display_step, annotate=False) # False OR True
lr.fit(trainX, trainY)
from sklearn.model_selection import cross_val_score
cross_val_score(lr, trainX, trainY, cv=2).mean()

tf.reset_default_graph() # have to reset it after previous tensflow run
N = 100
D = 2
# trainX.shape: (100 * 2), also it follows standard normal distribution
trainX = np.random.randn(N, D) # Return a samples from the “standard normal” distribution, N rows, D columns

# add some noise to the trainning data
delta = 1.75
trainX[:N//2] += np.array([delta, delta]) # add delta for each 2 columns for first 50 rows
trainX[N//2:] += np.array([-delta, -delta])

trainY = np.array([0] * (N//2) + [1] * (N//2))
plt.scatter(trainX[:,0], trainX[:,1], s=100, c=trainY, alpha=0.5)
plt.show()
# 100 lables, generate 50 ZERO & 50 ONE
original_label = np.array([0] * (N//2) + [1] * (N//2)) # 5//2 =2 5/2 = 2.5
print(original_label)
print(len(original_label))
from sklearn.metrics import accuracy_score
class LogisticRegressionTF(BaseEstimator):
    def __init__(self, learning_rate, training_epochs, display_step, annotate=False):
        self.annotate = annotate
        self.sess = tf.Session() # like linear regression, always createa session instance in __init__ method
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.display_step = display_step
        
        
    def fit(self, trainX,trainY):
        N, D = trainX.shape # 在我们的例子里D=2,也就表示每个X 有2个features
        _, c = trainY.shape # c 多少个类别.在我们的例子里 c=2.由于one hot encoded, 用2列表示
        # 图的输入. here "None" means I'm not sure how many rows/training data
        self.X = tf.placeholder(tf.float64, shape=[None, D]) # D列features
        self.Y = tf.placeholder(tf.float64, shape=[None, c]) # c = trainY.shape defined earlier.通常来说label就1列，但是由于做了one hot encoding, label多少个种类就有多少列
        
        # 参数的定义 and also assign initial values to those newly defined variables/parameters
        # 解释为什么定义这个randn(D,c)shape. trainX(N * D) * W (D * c) -> label (N * c) 所以这里需要定义W 为（D, c）shape
        self.W = tf.Variable(np.random.randn(D,c), name="weight") # D： trainX的feature columns 列
        self.b = tf.Variable(np.random.randn(c), name="bias") # label (N * c), 每一列都需要一个bias,所以bias shape为 c
        
        # logistic prediction
        #self.pred = tf.sigmoid(tf.add(tf.matmul(self.X, self.W), self.b))
        # matmul VS multiply: matmul: MATRIX multiplication AND multiple: element-wise multiplication which is used in above linear regression,只能用于单行 或单列矩阵相乘
        # https://stackoverflow.com/questions/47583501/tf-multiply-vs-tf-matmul-to-calculate-the-dot-product
        output_logits = tf.add(tf.matmul(self.X, self.W), self.b) # 这里 X, W 顺序不能变换
        # different from linear regression, here we have to go through sigmoid function to get the probability 
        #这里返回的就是2个值，也就是每一类的概率 （prediction function 里面用了np.argmax(prediction, axis=1) 表示每一行结果2个值？）
        self.pred = tf.sigmoid(output_logits)   # turn logits to probability through sigmoid function
        
        # 交叉熵loss (differnt from linear regressin's MSE: mean squred error)
        #cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
        #??? 
        cost= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_logits, labels=self.Y))
        
        # 优化器
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        
        # 初始化所有的参数
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        
        # 可视化初始化的模型边界
        if self.annotate:
            assert len(trainX.shape) == 2, "Only 2d points are allowed!!"

            plt.scatter(trainX[:,0], trainX[:,1], s=100, c=original_label, alpha=0.5) 

            h = .02 
            x_min, x_max = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1
            y_min, y_max = trainX[:, 1].min() - 1, trainX[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
            plt.title("This is where model starts to learn!!")
            plt.show()

        

        # 训练开始
        for epoch in range(self.training_epochs):
            for (x, y) in zip(trainX, trainY):
                # have to call np.asmatrix(x) ?, in above linear regression, there is no asmatrix,因为没有 shape的问题
                # 需要指明shape的问题，所以需要转化为matrix
                self.sess.run(optimizer, feed_dict={self.X: np.asmatrix(x), self.Y: np.asmatrix(y)})

            #展示训练结果 for each epoch
            if (epoch+1) % display_step == 0:
                c = self.sess.run(cost, feed_dict={self.X: trainX, self.Y:trainY})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
            "W=", self.sess.run(self.W), "b=", self.sess.run(self.b))
                
            #显示拟合的直线 for each epoch. it's actualy just the "plot_decision_boundaryg" function in below link:
            # https://www.kaggle.com/jungan/ensemble-bagging-tree-plot-decision-boundary
                if self.annotate:
                    assert len(trainX.shape) == 2, "Only 2d points are allowed!!"

                    plt.scatter(trainX[:,0], trainX[:,1], s=100, c=original_label, alpha=0.5) 
             
                    h = .02 
                    x_min, x_max = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1
                    y_min, y_max = trainX[:, 1].min() - 1, trainX[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
                    # in plot_decision_boundary function, model.predcit where model is passed into this function
                    Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
                    plt.show()



        print("Optimization Finished!")
        training_cost = self.sess.run(cost, feed_dict={self.X: trainX, self.Y: trainY})
        print("Training cost=", training_cost, "W=", self.sess.run(self.W), "b=", self.sess.run(self.b), '\n')

        
    def predict(self, testX):
        # 0.4， 0.6 -> 1 去概率值大的列号
        # 0.7， 0.3 -> 0   
        # 为什么prediction返回两列？ 是我们tensorflow的这个graph定义的，之前我们train的时候用的数据的label是2列的，那么在predict的时候返回的就是一样大小的值
        prediction = self.sess.run(self.pred,feed_dict={self.X: testX}) # prediction 有两列，每一个种类的概率
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.argmax.html
        # return the max value's index in each row (note: not the actual value...instead just index)
        return np.argmax(prediction, axis=1)
    
    def score(self, testX, testY):
        #suppose the testY has been one hot encoded. testY = label of the training data
        #eg:#0: [1,0]  -> 0
            #1: [1,0]  -> 0
            #2: [0,1]  -> 1
        _ , true_result = np.where(testY == 1) # which index = 1, e.g. 0, 1 -> index =1, class1, 1,0 -> index =0, class 0
        result = self.predict(testX)
        return accuracy_score(true_result, result)
from sklearn.preprocessing import OneHotEncoder
trainY


le = OneHotEncoder()
# below fit and transform can be done in one line
le.fit(trainY.reshape(N,-1)) # -1 means not sure how many columns and columsn will be inferred from another dimension
trainY = le.transform(trainY.reshape(N,-1)).toarray() # toarray()  变成 numpy 的ndarray type
trainY

# lalel=0 -> [1, 0] 
# label=1 -> [0, 1]
logisticTF = LogisticRegressionTF(learning_rate, 1000, display_step, annotate=False)
logisticTF.fit(trainX, trainY)
from sklearn.model_selection import cross_val_score
cross_val_score(logisticTF, trainX, trainY, cv=2).mean() # will run 5 times. each time will 1000 epochs
tf.reset_default_graph()
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#trainX, trainY = mnist.train.next_batch(5000) #5000个数据作为近邻集合
#testX, testY = mnist.test.next_batch(200) #200个数据用于测试
import os
data_folder = "../input/ninechapterdigitsub"
#data_folder = "data"
trainX = np.genfromtxt(os.path.join(data_folder, "digit_mnist_trainx.csv"), delimiter=',')
trainY = np.genfromtxt(os.path.join(data_folder, "digit_mnist_trainy.csv"), delimiter=',')
testX = np.genfromtxt(os.path.join(data_folder, "digit_mnist_testx.csv"), delimiter=',')
testY = np.genfromtxt(os.path.join(data_folder, "digit_mnist_testy.csv"), delimiter=',')
trainX.shape, trainY.shape, testX.shape, testY.shape, type(trainX)
#为什么label 有10 cloumns,  因为label 做了one hot encodingl了 0， 1，2，3，4，5，6，7，8，9 具体是哪个数字，那个对应位置的值就是1
trainY[:2]
xtr = tf.placeholder("float", [None, 784]) # 前面用法：self.X = tf.placeholder(tf.float64, shape=[None, D])
xte = tf.placeholder("float", [784]) # 对于测试数据，肯定只有1行，所以不用None
# xtr: (5000, 784) xte(200, 784)
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xtr, xte)), reduction_indices=1)) # 设置维度为1

# example： 每个test data 去和所有的training data 计算距离， 因为基于上面的定义：xte = tf.placeholder("float", [784])， xte 就表示1维数据，也就1行数据
# train 0: [1,...1]
# train 1: [0,...0]
# test : [1,...1]
# tf.subtract(xtr, xte):
#  0: [0,...0]
#  1: [-1,...-1]
# tf.square:
#  0: [0,...0]
#  1: [1,...1]
#  tf.reduce_sum(tf.square(tf.subtract(xtr, xte)), reduction_indices=1):
#  0: [0]
#  1: [784]
# 因为是topk大的值，这里distance取负号, 表示取举例最小的K个
KVALUE = 2
# based on the https://www.tensorflow.org/api_docs/python/tf/math/top_k,  tf.nn.top_k 函数 返回 values  and index 两个值
pred= tf.nn.top_k(-distance, k=KVALUE) #上面其他几个模型实现的时候，由于定义了init, fit, precit functions, 所以他们都放在了predict 函数里了
from collections import Counter
accuracy = 0.

# 初始化参数
init = tf.global_variables_initializer()

# 开始训练 采用了上面"intialize Variable - Method 4 - Recommened Way"
with tf.Session() as sess:
    sess.run(init) # 也可以用init.run

    # 预测测试数据的标签 (passive learner)
    for i in range(len(testX)):
        # 最近邻的序号
        # 为什么知道这里能返回哪些东西，有的时候是1个值，有的时候2个值, 根据就是tensorflow具体API 的函数的用法
        # 根据上面pred的定义： top_k返回2个值： https://www.tensorflow.org/api_docs/python/tf/math/top_k
        # 这里的feed_dict变量名字要和前面定义的 对应 i.e.xtrde = tf.placeholder("float", [None, 784]) 
        values, knn_index = sess.run(pred, feed_dict={xtr: trainX, xte: testX[i, :]}) # testX的i行，所有列

        # 拿到k个邻居后做全民公投，得票最多的为预测标签，
        # e.g y =1 [0, 1,0, 0] np.argmax -> 1
        #     y =0 [1, 0, 0, 0] np.argmax -> 0
        #     y =3 [0, 0, 0, 1] np.argmax -> 3
        # 因为lable,做了oneHotEncoding, 只有0，1 两个值，np.argmax(trainY[knn_index]，就可以知道1对应的位置，也就是实际的数字，因为数字就是从0开始 到9
        c = Counter(np.argmax(trainY[knn_index], axis=1)) # e.g. K=2时，sample output: Counter({7: 1, 1: 1})相当于dict类型
        # 对于Counter({7: 1, 1: 1})， Counter.most_commmon(2) 返回数组类型Counter [(7, 1), (1, 1)]， 这样 加上[0][0] 就娶到实际的值 7了
        # another exmaple: {0:2, 1:4}, after most_common((1, 4), (0,2))
        result = c.most_common(KVALUE)[0][0]
        # 计算最近邻的标签和真实标签值
        print("Test", i, "Prediction:", result, \
            "True Class:", np.argmax(testY[i]))
        # 正确率
        if result == np.argmax(testY[i]):
            accuracy += 1./len(testX)
            
    print("Done!")
    print("Accuracy:", accuracy)
tf.reset_default_graph()
N = 120
D = 2
trainX = np.random.randn(N, D) # generate 120 samples , each sample has 2 features

delta = 2
#trainX[:N//3] += np.array([delta, delta])
#trainX[N//3:N*2//3] += np.array([-delta, delta])
#trainX[N*2//3:] += np.array([0, -delta])

delta = 1.75
trainX[:N//2] += np.array([delta, delta])
trainX[N//2:] += np.array([-delta, -delta])

trainY = np.array([0] * (N//2) + [1] * (N//2))
plt.scatter(trainX[:,0], trainX[:,1], s=100, c=trainY, alpha=0.5)
plt.show()
from sklearn.metrics import accuracy_score
from matplotlib import colors
from sklearn.utils.fixes import logsumexp

class NaiveBayesTF(BaseEstimator):
    
    def __init__(self):
        self.dist = None
        self.sess = tf.Session() # as always, initialize Tensforflow session instance in __init__

    def fit(self, trainX, trainY):
        # Separate training points by class (nb_classes * nb_samples * nb_features)
        unique_classes = np.unique(trainY) # check how many unique classess in labels
        # points_by_class shape: (2, 60, 2) i.e. (nb_classes * nb_samples * nb_features), 2 classess, each class has 60 samples, each sample has 2 features
        points_by_class = np.array([
            [x for x, y in zip(trainX, trainY) if y == c]
            for c in unique_classes])
        input_x = tf.placeholder(tf.float64, shape=points_by_class.shape) #points_by_class shape: (2, 60, 2)
        # 估计每个类底下每一种feature的均值和方差
        # shape: num_classes * nb_features
        
        #???  how to determin axes=[1] for 2 * 60 * 2 dimensions
        # https://www.tensorflow.org/api_docs/python/tf/nn/moments
        moments = tf.nn.moments(input_x, axes=[1]) # tf.nn.moments return: mean and variance,  axes=[1]:each column/feature of input_x
        mean, var = self.sess.run(moments, feed_dict={input_x:points_by_class})
        # print(mean.shape) (2, 2) 每个类底下每一种feature的均值和方差 -> 2 * 2
        # print(var.shape) (2, 2)
        
        # 点集实验里为2类，每个数据点有2个特征 
        # known mean and variance. https://www.tensorflow.org/api_docs/python/tf/distributions/Normal
        self.dist = tf.distributions.Normal(loc=mean, scale=tf.sqrt(var))
        # print(self.dist.scale.shape) (2, 2)
        # print(self.dist.scale) -> Tensor("Normal_8/scale:0", shape=(2, 2), dtype=float64)

    def predict(self, testX):
        assert self.dist is not None
        num_classes, num_features = map(int, self.dist.scale.shape)

        # 条件概率 log P(x|c)
        # (nb_samples, nb_classes)
        # ??? how to determin:  axis=2
        cond_probs = tf.reduce_sum(
            # https://www.tensorflow.org/api_docs/python/tf/distributions/Normal#log_prob
            # https://www.tensorflow.org/api_docs/python/tf/tile
            # self.dist is normal distrubtion
            self.dist.log_prob(
                # tf.tile(testX, [1, num_classes]), 1 means along x asix, duplicate num_classes times
                tf.reshape(
                    tf.tile(testX, [1, num_classes]), [-1, num_classes, num_features])), # -1 means: not sure the dimension
            axis=2)
        
        # 第一个点: 2.0,3.5
        # 第二个点: 0.5,1.4
        
        # tf.tile (num_classes = 2): 
        # ===>
        # 第一个点: 2.0,3.5,2.0,3.5
        # 第二个点: 0.5,1.4,0.5,1.4
        
        # tf.reshape:
        # ===>
        # 第一个点: 2.0,3.5 
        #         2.0,3.5
        # 第二个点：0.5,1.4
        #         0.5,1.4
        
        # e.g: point1, c = 0  log(P(x0=2|c=0)) + log(P(x1=3.5|c=0))
        #      point1, c = 1  log(P(x0=2|c=1)) + log(P(x1=3.5|c=1))
        # bascially, dupliate each row, then it's easier to calculate prob for each row for each class  
        
        # P(C) 均匀分布  # priors array([0.5, 0.5]), i.e each class has 50% in all samples
        priors = np.log(np.array([1. / num_classes] * num_classes))

        # 后验概率取log, log P(C) + log P(x|C)
        posterior = tf.add(priors, cond_probs)
        
        # 取概率最大的那一个
        # 第一个点: 2.0,3.5 
        #         2.0,3.5
        # each data point has two posterior, i.e. each row (each class) has one.
        # axis=1 means, max index for for each row
        result = self.sess.run(tf.argmax(posterior, axis=1)) # axis=1, 表示顺着y轴， 对每一行求max

        return result
    
    
    def score(self, testX, testY):
        result = self.predict(testX)
        return accuracy_score(testY, result) # accuracy_score(label, predicated values)

np.array([1. / 2] * 2)
tf_nb = NaiveBayesTF()
tf_nb.fit(trainX, trainY)
x_min, x_max = trainX[:, 0].min() - .5, trainX[:, 0].max() + .5
y_min, y_max = trainX[:, 1].min() - .5, trainX[:, 1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                     np.linspace(y_min, y_max, 30))
Z = tf_nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.scatter(trainX[:,0], trainX[:,1], s=100, c=trainY, alpha=0.5)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
plt.show()
tf_nb.score(trainX, trainY)


from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import pandas as pd
import numpy as np
import os
data_folder = "../input/fashionmnist"
#data_folder = "./data"
train_data = pd.read_csv(os.path.join(data_folder, "fashion-mnist_train.csv"))
test_data = pd.read_csv(os.path.join(data_folder, "fashion-mnist_test.csv"))
train_data.head()
test_data.head()
trainX = np.array(train_data.iloc[:, 1:])
trainY = np.array(train_data.iloc[:, 0])
testX = np.array(test_data.iloc[:, 1:])
testY = np.array(test_data.iloc[:, 0])
IMAGE_CLASSES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
import matplotlib.pyplot as plt
img_size = 28 # 28 * 28 = 784 which is he feature number
for img, label in zip(trainX[:10], trainY[:10]):   # take a look at at first 10 images in the train set
    plt.imshow(img.reshape(img_size,img_size),cmap='gray')
    plt.title(IMAGE_CLASSES[label])
    plt.show()
# 参数设定
#The 10 categories
#784 Each image is 28x28 pixels
num_steps = 100# Total steps to train 
batch_size = 1024 # The number of samples per batch 
num_trees = 10
max_nodes = 1000
tf.reset_default_graph() # as always, reset the tensorflow
class RandomForestTF(BaseEstimator):
    
    # why not define  self.sess = tf.Session() wihtin __init__
    def __init__(self, num_trees):
        self.num_trees = num_trees # 
        
    def fit(self, X, Y, num_steps, batch_size,max_nodes):
        num_classes = 10   #len(IMAGE_CLASSES)
        num_data = X.shape[0] # how many samples
        num_features = X.shape[1] # how many features
        
        self.X = tf.placeholder(tf.float32, shape=[None, num_features]) # None means not sure how many samples
        self.Y = tf.placeholder(tf.int32, shape=[None])
        
        # 随机森林的参数
        hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=self.num_trees,
                                      max_nodes=max_nodes).fill()
        
        # 随机森林的计算图
        forest_graph = tensor_forest.RandomForestGraphs(hparams)
        
        train_operation = forest_graph.training_graph(self.X, self.Y)
        loss_operation = forest_graph.training_loss(self.X, self.Y)
        
        # inference_graph will return probabilities, decision path and variance
        self.infer_op, _, _ = forest_graph.inference_graph(self.X) # inference_graph API tell you return data
        # infer_op includes inrerred prob for each type, pick the max prob type. then check against with the original label
        # tf.equal: https://www.dotnetperls.com/equal-tensorflow
        correct_prediction = tf.equal(tf.argmax(self.infer_op, 1), tf.cast(self.Y, tf.int64))
        print("correct_prediction")
        print(correct_prediction)
        # https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        
        # 将初始化的操作和树的参数初始化 作为一个整体操作
        init_vars = tf.group(tf.global_variables_initializer(),
                   resources.initialize_resources(resources.shared_resources()))
        #init_vars = tf.global_variables_initializer()
        
        self.sess = tf.Session()
        self.sess.run(init_vars)

        # 开始训练
        cnt = 0
        for i in range(1, num_steps + 1): # step = epoch
            # Prepare Data
            # 每次学习一个batch的MNIST data
            #batch_x, batch_y = training_set.next_batch(batch_size)
            start, end = ((i-1) * batch_size) % num_data, (i * batch_size) % num_data
            
            batch_x, batch_y = X[start:end], Y[start:end]
            _, l = self.sess.run([train_operation, loss_operation], feed_dict={self.X: batch_x, self.Y: batch_y})
            if i % 50 == 0 or i == 1:
                acc = self.sess.run(self.accuracy_op, feed_dict={self.X: batch_x, self.Y: batch_y})
                print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
                
    def predict(self, testX):
        results = self.sess.run(self.infer_op, feed_dict={self.X:testX})
        return np.argmax(results, axis=1)
    
    def score(self, testX, testY):
        accuracy = self.sess.run(self.accuracy_op, feed_dict={self.X: testX, self.Y: testY})
        return accuracy
rftf = RandomForestTF(num_trees)
rftf.fit(trainX, trainY, num_steps, batch_size, max_nodes)
rftf.score(testX, testY)
