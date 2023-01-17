%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# 引入mean_square_error

#TODO

from sklearn.metrics import mean_squared_error as mse

# 数据集个数

#TODO

Num_datasets = 50

noise_level = 0.5



# 最大的degree

#TODO

max_degree = 10



# 每个数据集里的数据个数

#TODO

N = 25



# 用于训练的数据数

#TODO

trainN = int(N * 0.9)

np.random.seed(2)
def make_poly(x, degree):

    """

    input: x  N by 1

    output: N by degree + 1

    """

    N = len(x)

    result = np.empty((N, degree+1))

    for d in range(degree + 1):

        result[:,d] = x ** d

        if d > 1:

            result[:, d] = (result[:, d] - result[:, d].mean()) / result[:,d].std()

    return result



def f(X):

    #TODO

    return np.sin(X)
x_axis = np.linspace(-np.pi, np.pi, 100)#TODO

y_axis = f(x_axis)#TODO



# 可视化

#TODO

plt.plot(x_axis, y_axis)
# 基本训练集

# TODO

X = np.linspace(-np.pi, np.pi, 25)

np.random.shuffle(X)

f_X = f(X)



# 创建全部的数据

allData = make_poly(X, max_degree)



train_scores = np.zeros((Num_datasets, max_degree))

test_scores = np.zeros((Num_datasets, max_degree))



train_predictions = np.zeros((trainN, Num_datasets, max_degree))

prediction_curves = np.zeros((100, Num_datasets, max_degree))



model = LinearRegression()
# TODO

plt.scatter(X, f_X)

plt.show()
for k in range(Num_datasets):

    

    # 每个数据集不失pattern的情况下稍微不一样~

    Y = f_X + np.random.randn(N) * noise_level

    

    trainX, testX = allData[:trainN], allData[trainN:]

    trainY, testY = Y[:trainN], Y[trainN:]

    

    # 用不同的模型去学习当前数据集

    for d in range(max_degree):

        

        # 模型学习

        # TODO

        model.fit(trainX[:, :d+2], trainY)

        

        # 在allData上的预测结果

        all_predictions = model.predict(allData[:, :d+2])

        

        # 预测并记录一下我们的目标函数

        x_axis_poly = make_poly(x_axis, d + 1)   # true poly x

        axis_predictions = model.predict(x_axis_poly)   # true y

        prediction_curves[:, k, d] = axis_predictions

        

        train_prediction = all_predictions[:trainN]

        test_prediction = all_predictions[trainN:]

        

        train_predictions[:, k, d] = train_prediction # 用于计算bias and varaince 

        

        

        #计算并存储训练集和测试集上的分数

        train_score = mse(train_prediction, trainY)#TODO

        test_score = mse(test_prediction, testY)#TODO

        train_scores[k, d] = train_score

        test_scores[k, d] = test_score       

    

    
for d in range(max_degree):

    for k in range(Num_datasets):

        # TODO

        # 给定当前模型，画出它在所有数据集上的表现

        plt.plot(x_axis, prediction_curves[:, k, d], color='green', alpha=0.5)

        

    

    # TODO

    # 给定当前模型，画出它在所有数据集上的表现的平均

    plt.plot(x_axis, prediction_curves[:, :, d].mean(axis=1), color='blue', linewidth=2)

    

    plt.title("curves for degree=%d" %(d+1))

    plt.show()
#TODO 每一个模型的bias

average_train_prediction = np.zeros((trainN, max_degree))# 模型的平均表现

squared_bias = np.zeros(max_degree)



trueY_train = f_X[:trainN]# 真值

#TODO

for d in range(max_degree):

    for i in range(trainN):

        average_train_prediction[i,d] = train_predictions[i, :, d].mean()

    squared_bias[d] = ((average_train_prediction[:, d] - trueY_train) ** 2).mean()

        

        

variances = np.zeros((trainN, max_degree))

for d in range(max_degree):

    for i in range(trainN):

        #TODO

        difference = train_predictions[i, :, d] - average_train_prediction[i, d]

        variances[i,d] = np.dot(difference, difference) / Num_datasets





# TODO

variance = variances.mean(axis=0)
degrees = np.arange(max_degree) + 1

best_degree = np.argmin(test_scores.mean(axis=0)) + 1
#TODO

plt.plot(degrees, squared_bias, label='squared bias')

plt.plot(degrees, variance, label='variance')

plt.plot(degrees, test_scores.mean(axis=0), label='test scores')

plt.plot(degrees, squared_bias + variance, label='squared bias + variance')

plt.axvline(x=best_degree, linestyle='--', label='best complexity')

plt.legend()

plt.show()