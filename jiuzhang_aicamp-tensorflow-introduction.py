%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
print(result)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
print(result)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(x.eval())
    print(y.eval())
    result = sess.run(f, feed_dict={x:1, y:0})
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


def f(X):
    """
    input: x
    output: y = 3x + 4
    """
    return 3*X + 4



N = 40
noise_level = 0.8
trainX = np.linspace(-4.0, 4.0, N)
np.random.shuffle(trainX)
trainY = f(trainX) + np.random.randn(N) * noise_level


learning_rate = 0.01
training_epochs = 1000
display_step = 50
plt.scatter(trainX, trainY)
from sklearn.base import BaseEstimator
class LinearRegressionTF(BaseEstimator):
    def __init__(self, learning_rate, training_epochs, display_step, annotate=False):
        self.annotate = annotate
        self.sess = tf.Session()
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.display_step = display_step
        
        
    def fit(self, trainX,trainY):
        N = trainX.shape[0]
        # 图的输入
        self.X = tf.placeholder("float")
        self.Y = tf.placeholder("float")
        
        
        # 参数的定义
        self.W = tf.Variable(np.random.randn(), name="weight")
        self.b = tf.Variable(np.random.randn(), name="bias")
        
        # 线性模型
        self.pred = tf.add(tf.multiply(self.X, self.W), self.b)
        
        # mean squre error
        cost = tf.reduce_sum(tf.pow(self.pred-self.Y, 2))/(2*N)
        
        # 优化器
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        
        # 初始化所有的参数
        init = tf.global_variables_initializer()
        self.sess.run(init)

        
        if self.annotate:
            plt.plot(trainX, trainY, 'ro', label='Original data')
            plt.plot(trainX, self.sess.run(self.W) * trainX + self.sess.run(self.b), label='Fitted line')
            plt.legend()
            plt.title("This is where model starts to learn!!")
            plt.show()
            
        # 训练开始
        for epoch in range(self.training_epochs):
            for (x, y) in zip(trainX, trainY):
                self.sess.run(optimizer, feed_dict={self.X: x, self.Y: y})

            #展示训练结果
            if (epoch+1) % display_step == 0:
                c = self.sess.run(cost, feed_dict={self.X: trainX, self.Y:trainY})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
            "W=", self.sess.run(self.W), "b=", self.sess.run(self.b))
                
            #显示拟合的直线
                if self.annotate:
                    plt.plot(trainX, trainY, 'ro', label='Original data')
                    plt.plot(trainX, self.sess.run(self.W) * trainX + self.sess.run(self.b), label='Fitted line')
                    plt.legend()
                    plt.show()
                #plt.pause(0.5)

        print("Optimization Finished!")
        training_cost = self.sess.run(cost, feed_dict={self.X: trainX, self.Y: trainY})
        print("Training cost=", training_cost, "W=", self.sess.run(self.W), "b=", self.sess.run(self.b), '\n')

        
    def predict(self, testX):
        prediction = self.sess.run(self.pred,feed_dict={self.X: testX})
        return prediction
    
    def score(self, testX, testY):
        result = self.predict(testX)
        return r2_score(testY, result)
        
    
lr = LinearRegressionTF(learning_rate, 1000, display_step, annotate=False)
lr.fit(trainX, trainY)
from sklearn.model_selection import cross_val_score
cross_val_score(lr, trainX, trainY, cv=5).mean()

tf.reset_default_graph()
N = 100
D = 2
trainX = np.random.randn(N, D)

delta = 1.75
trainX[:N//2] += np.array([delta, delta])
trainX[N//2:] += np.array([-delta, -delta])

trainY = np.array([0] * (N//2) + [1] * (N//2))
plt.scatter(trainX[:,0], trainX[:,1], s=100, c=trainY, alpha=0.5)
plt.show()
original_label = np.array([0] * (N//2) + [1] * (N//2))
from sklearn.metrics import accuracy_score
class LogisticRegressionTF(BaseEstimator):
    def __init__(self, learning_rate, training_epochs, display_step, annotate=False):
        self.annotate = annotate
        self.sess = tf.Session()
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.display_step = display_step
        
        
    def fit(self, trainX,trainY):
        N, D = trainX.shape
        _, c = trainY.shape
        # 图的输入
        self.X = tf.placeholder(tf.float64, shape=[None, D])
        self.Y = tf.placeholder(tf.float64, shape=[None, c])
        
        
        # 参数的定义
        self.W = tf.Variable(np.random.randn(D,c), name="weight")
        self.b = tf.Variable(np.random.randn(c), name="bias")
        
        # logistic prediction
        #self.pred = tf.sigmoid(tf.add(tf.matmul(self.X, self.W), self.b))
        output_logits = tf.add(tf.matmul(self.X, self.W), self.b)
        self.pred = tf.sigmoid(output_logits)   # turn logits to probability
        
        # 交叉熵loss
        #cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
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
                self.sess.run(optimizer, feed_dict={self.X: np.asmatrix(x), self.Y: np.asmatrix(y)})

            #展示训练结果
            if (epoch+1) % display_step == 0:
                c = self.sess.run(cost, feed_dict={self.X: trainX, self.Y:trainY})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
            "W=", self.sess.run(self.W), "b=", self.sess.run(self.b))
                
            #显示拟合的直线
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
                    plt.show()



        print("Optimization Finished!")
        training_cost = self.sess.run(cost, feed_dict={self.X: trainX, self.Y: trainY})
        print("Training cost=", training_cost, "W=", self.sess.run(self.W), "b=", self.sess.run(self.b), '\n')

        
    def predict(self, testX):
        prediction = self.sess.run(self.pred,feed_dict={self.X: testX})
        return np.argmax(prediction, axis=1)
    
    def score(self, testX, testY):
        # suppose the testY has been one hot encoded
        #eg:#0: [1,0]  -> 0, 0
            #1: [1,0]  -> 1, 0
            #2: [0,1]  -> 2, 1
        _ , true_result = np.where(testY == 1)
        result = self.predict(testX)
        return accuracy_score(true_result, result)
from sklearn.preprocessing import OneHotEncoder
le = OneHotEncoder()
le.fit(trainY.reshape(N,-1))
trainY = le.transform(trainY.reshape(N,-1)).toarray()
logisticTF = LogisticRegressionTF(learning_rate, 1000, display_step, annotate=False)
logisticTF.fit(trainX, trainY)
from sklearn.model_selection import cross_val_score
cross_val_score(logisticTF, trainX, trainY, cv=5).mean()
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
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xtr, xte)), reduction_indices=1))

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


# 因为是topk大的值，这里distance取负号
KVALUE = 1
pred= tf.nn.top_k(-distance, k=KVALUE)
from collections import Counter
accuracy = 0.

# 初始化参数
init = tf.global_variables_initializer()


# 开始训练
with tf.Session() as sess:
    sess.run(init)

    # 预测测试数据的标签 (passive learner)
    for i in range(len(testX)):
        # 最近邻的序号
        values, knn_index = sess.run(pred, feed_dict={xtr: trainX, xte: testX[i, :]})

        # 拿到k个邻居后做全民公投，得票最多的为预测标签
        c = Counter(np.argmax(trainY[knn_index], axis=1))
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
trainX = np.random.randn(N, D)

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
        self.sess = tf.Session()

    def fit(self, trainX, trainY):
        # Separate training points by class (nb_classes * nb_samples * nb_features)
        unique_classes = np.unique(trainY)
        points_by_class = np.array([
            [x for x, y in zip(trainX, trainY) if y == c]
            for c in unique_classes])
        
        input_x = tf.placeholder(tf.float64, shape=points_by_class.shape)
        # 估计每个类底下每一种feature的均值和方差
        # shape: num_classes * nb_features
        
        moments = tf.nn.moments(input_x, axes=[1])
        mean, var = self.sess.run(moments, feed_dict={input_x:points_by_class})
        #print(mean.shape)
        #print(var.shape)
        
        # 点集实验里为2类，每个数据点有2个特征 
        # known mean and variance
        self.dist = tf.distributions.Normal(loc=mean, scale=tf.sqrt(var))
        

    def predict(self, testX):
        assert self.dist is not None
        num_classes, num_features = map(int, self.dist.scale.shape)

        # 条件概率 log P(x|c)
        # (nb_samples, nb_classes)
        cond_probs = tf.reduce_sum(
            self.dist.log_prob(
                tf.reshape(
                    tf.tile(testX, [1, num_classes]), [-1, num_classes, num_features])),
            axis=2)
        
        # 第一个点: 2.0,3.5
        # 第二个点: 0.5,1.4
        # tf.tile (num_classes = 2):
        # 第一个点: 2.0,3.5,2.0,3.5
        # 第二个点: 0.5,1.4,0.5,1.4
        # tf.reshape:
        # 第一个点: 2.0,3.5 
        #         2.0,3.5
        # 第二个点：0.5,1.4
        #         0.5,1.4

        # P(C) 均匀分布
        priors = np.log(np.array([1. / num_classes] * num_classes))

        # 后验概率取log, log P(C) + log P(x|C)
        posterior = tf.add(priors, cond_probs)
        
        # 取概率最大的那一个
        result = self.sess.run(tf.argmax(posterior, axis=1))

        return result
    
    
    def score(self, testX, testY):
        result = self.predict(testX)
        return accuracy_score(testY, result)

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
img_size = 28
for img, label in zip(trainX[:10], trainY[:10]):
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

tf.reset_default_graph()
class RandomForestTF(BaseEstimator):
    
    def __init__(self, num_trees):
        self.num_trees = num_trees
        
    def fit(self, X, Y, num_steps, batch_size,max_nodes):
        num_classes = 10   #len(IMAGE_CLASSES)
        num_data = X.shape[0]
        num_features = X.shape[1]
        
        self.X = tf.placeholder(tf.float32, shape=[None, num_features]) 
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
        self.infer_op, _, _ = forest_graph.inference_graph(self.X)
        correct_prediction = tf.equal(tf.argmax(self.infer_op, 1), tf.cast(self.Y, tf.int64))
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        
        # 将初始化的操作和树的参数初始化 作为一个整体操作
        init_vars = tf.group(tf.global_variables_initializer(),
                   resources.initialize_resources(resources.shared_resources()))
        #init_vars = tf.global_variables_initializer()
        
        self.sess = tf.Session()
        self.sess.run(init_vars)

        # 开始训练
        cnt = 0
        for i in range(1, num_steps + 1):
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
