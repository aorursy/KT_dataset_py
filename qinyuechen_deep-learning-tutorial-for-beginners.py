# 我们这个 notebook 使用的python3的环境安装了很多有用的用于数据分析的库

# 这个环境被封装成docker镜像"kaggle/python"：https://github.com/kaggle/docker-python

# 比如，我们在这里载入几个有用的库



import numpy as np # linear algebra 线性代数库

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 数据处理，CSV 文件 I/O

import matplotlib.pyplot as plt



# 数据输入文件在"../input/"目录中

# 比如调用下面的代码会列出输入目录中的文件(通过点击 run 按钮 或者 按下快捷键 Shift+Enter)

# 载入 warning 库

import warnings

# 过滤警告

warnings.filterwarnings('ignore')

from subprocess import check_output

# 如果是 kaggle 环境

print(check_output(["ls", "../input"]).decode("utf8"))

# 如果是 github 下载的，你需要自己下载[Sign-language-digits数据集](https://www.kaggle.com/ardamavi/sign-language-digits-dataset)

#print(check_output(["ls", "input"]).decode("utf8"))

# 你写到 input目录中的文件都会被打印在下方。
# 译者注：载入npy格式的数组文件，这是numpy存储数组的一种格式

# 如果是 kaggle 环境

x_l = np.load('../input/sign-language-digits-dataset/X.npy')

Y_l = np.load('../input/sign-language-digits-dataset/Y.npy')

# 如果是 github 下载的，

#x_l = np.load('input/Sign-language-digits-dataset/X.npy')

# Y_l = np.load('input/Sign-language-digits-dataset/Y.npy')

img_size = 64

plt.subplot(1, 3, 1)

# 译者注：大家可以通过改变x_l序号，看看不同手势符号

# 符号0图片

plt.imshow(x_l[205].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 3, 2)

# 符号1图片

plt.imshow(x_l[823].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 3, 3)

# 译者注：符号3图片，你还可以试试其他序号

plt.imshow(x_l[1].reshape(img_size, img_size))

plt.axis('off')
# Join a sequence of arrays along an row axis.

# 把一串数组的连接起来，统一编号

X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # 从 0 到 204 是符号”0“ 从205到410是符号”1“

z = np.zeros(205)

o = np.ones(205)

Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)

# 译者笔记：这里将z和o连接起来,然后变成一个410x1的数组

print(X.shape)

print(Y.shape)

# 去掉注释就可以阅读到np.reshape的帮助

# help(np.reshape)
# 让我们创建数组 x_train, y_train, x_test, y_test 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]

print(number_of_train)

print(number_of_test)
X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])

X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])

print("X train flatten",X_train_flatten.shape)

print("X test flatten",X_test_flatten.shape)
x_train = X_train_flatten.T

x_test = X_test_flatten.T

y_train = Y_train.T

y_test = Y_test.T

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
# 简单的定义方法示例

# 译者注：如果你已经学会 python, 不用管他

def dummy(parameter):

    dummy_parameter = parameter + 5

    return dummy_parameter

result = dummy(3)     # result = 8



# 让我们初始化参数吧

# 我们需要一个4096维的数组作为我们这个初始化权重方法的参数，每一维对应着一个像素。

# 译者注：initialize_weights_and_bias函数接受dimension做为参数，比如 dimension=4096, 

# 然后通过 np.full 创建一个4096维的长度为1的数组，并且通通赋值为0.01

# 译者注：np.full 返回一个根据指定shape和type,并用fill_value填充的新数组。

def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w, b

#help(np.full)
w,b = initialize_weights_and_bias(4096)

# 译者注：我们来看看参数的形状吧

print(w.shape)

print(b)
# 计算z值

#z = np.dot(w.T,x_train)+b

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
y_head = sigmoid(0)

y_head
# 前向传播步骤:

# 计算 z = w.T*x+b

# y_head = sigmoid(z)

# loss(error) = loss(y,y_head)

# cost = sum(loss)

def forward_propagation(w,b,x_train,y_train):

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z) # probabilistic 0-1

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    return cost 
# 在后序传播中，我们会使用前向传播得到的 y_head 

# 因此，我们把前向传播和后向传播绑在一起，而不是再写个后向传播的函数。

def forward_backward_propagation(w,b,x_train,y_train):

    # 前向传播

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    # 后向传播

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients
# 更新学习参数

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    # 更新学习参数需要number_of_iterarion次迭代

    for i in range(number_of_iterarion):

        # 做前向和后向传播来寻找代价和梯度

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # 让我们来做更新

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    # 我们更新了学习参数权重 weights 和偏置值 bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list

#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)
 # 预测

def predict(w,b,x_test):

    # x_test 是前向传播的输入

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # 如果 z 大于 0.5, 我们的预测结果是符号1 (y_head=1),

    # 如果 z 小于 0.5,, 我们的预测结果是符号0 (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

# predict(parameters["weight"],parameters["bias"],x_test)
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # 初始化

    dimension =  x_train.shape[0]  # that is 4096

    w,b = initialize_weights_and_bias(dimension)

    # 不要去改变学习率

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)



    # 打印训练测试错误

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)
from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)

print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))

print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
# 初始化参数和层大小

def initialize_parameters_and_layer_sizes_NN(x_train, y_train):

    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,

                  "bias1": np.zeros((3,1)),

                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,

                  "bias2": np.zeros((y_train.shape[0],1))}

    return parameters


def forward_propagation_NN(x_train, parameters):



    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]

    A1 = np.tanh(Z1)

    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]

    A2 = sigmoid(Z2)



    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache

# 计算代价

def compute_cost_NN(A2, Y, parameters):

    logprobs = np.multiply(np.log(A2),Y)

    cost = -np.sum(logprobs)/Y.shape[1]

    return cost

# 后向传播

def backward_propagation_NN(parameters, cache, X, Y):



    dZ2 = cache["A2"]-Y

    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]

    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]

    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))

    dW1 = np.dot(dZ1,X.T)/X.shape[1]

    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]

    grads = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2}

    return grads
# 更新参数

def update_parameters_NN(parameters, grads, learning_rate = 0.01):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],

                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],

                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],

                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}

    

    return parameters
# 预测

def predict_NN(parameters,x_test):

    # x_test 是前向传播的输入

    A2, cache = forward_propagation_NN(x_test,parameters)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # 如果z>0.5, 我们预测的结果是符号1

    # 如果z<0.5, 我们预测的结果是符号0,

    for i in range(A2.shape[1]):

        if A2[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
# 两层神经网络

def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):

    cost_list = []

    index_list = []

    #初始化参数和层大小

    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)



    for i in range(0, num_iterations):

         # 前向传播

        A2, cache = forward_propagation_NN(x_train,parameters)

        # 计算开销

        cost = compute_cost_NN(A2, y_train, parameters)

         # 后向传播

        grads = backward_propagation_NN(parameters, cache, x_train, y_train)

         # 更新参数

        parameters = update_parameters_NN(parameters, grads)

        

        if i % 100 == 0:

            cost_list.append(cost)

            index_list.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    plt.plot(index_list,cost_list)

    plt.xticks(index_list,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    

    # 预测

    y_prediction_test = predict_NN(parameters,x_test)

    y_prediction_train = predict_NN(parameters,x_train)



    # 打印训练和测试的结果

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters



parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)
# 转置

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
# 评估ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # 载入神经网络库

from keras.layers import Dense # 载入我们层库

def build_classifier():

    classifier = Sequential() # 载入神经网络

    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))