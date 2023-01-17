%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error as mse



# 数据集个数

Num_datasets = 50

noise_level = 0.5



# 最大的degree

max_degree = 10



# 每个数据集里的数据个数

N = 25



# 用于训练的数据数 i.e. 90% 的数据用于train, 10%用于validation

trainN = int(N * 0.9)



np.random.seed(2)
# 结合老师PPT里的解释

def make_poly(x, degree):

    """

    input: x  N by 1

    output: N by degree + 1

    """

    N = len(x)

    result = np.empty((N, degree+1)) #e.g. degree=10， 生产额外的11列 从feature0, feature1,...feature10

    for d in range(degree + 1):

        result[:,d] = x ** d # 每一列是原始数据的d次方

        if d > 1: # 这里做normalization, 去均值，除以方差

            result[:, d] = (result[:, d] - result[:, d].mean()) / result[:,d].std()

    return result



def f(X):

    """

    input: x

    output: sin(x)

    """

    return np.sin(X)

        
x_axis = np.linspace(-np.pi, np.pi, 100)

y_axis = f(x_axis)

plt.plot(x_axis, y_axis)
# 基础点集

X = np.linspace(-np.pi, np.pi, N) # from -np.pi to np.pi, generate N(25)个点

np.random.shuffle(X)

f_X = f(X)



# allData shape: N * 11

allData = make_poly(X, max_degree)





# 每一个训练集上的训练得分与测试得分

train_scores = np.zeros((Num_datasets, max_degree))

test_scores = np.zeros((Num_datasets, max_degree))





train_predictions = np.zeros((trainN, Num_datasets, max_degree))

prediction_curves = np.zeros((100, Num_datasets, max_degree))



model = LinearRegression()

plt.scatter(X, f_X)
for k in range(Num_datasets):

    

    # 每个数据集需要不一样~, 在不失pattern的情况下，加点noise

    Y = f_X + np.random.randn(N) * noise_level

    

    trainX, testX = allData[:trainN], allData[trainN:]

    trainY, testY = Y[:trainN], Y[trainN:]

    

    for d in range(max_degree):

        

        # 模型学习 根据degree 取 前d+1列 作为feature 列

        model.fit(trainX[:,:d+2], trainY)

        

        # 在allData上的预测结果

        all_predictions = model.predict(allData[:, :d+2])

        

        # 预测并记录一下我们的目标函数

        x_axis_poly = make_poly(x_axis, d+1)    # true poly x

        axis_predictions = model.predict(x_axis_poly)   # true y

        prediction_curves[:, k, d] = axis_predictions

        

        train_prediction = all_predictions[:trainN]

        test_prediction = all_predictions[trainN:]

        

        train_predictions[:, k, d] = train_prediction # 用于计算bias and varaince 

        

        

        #计算并存储训练集和测试集上的分数

        train_score = mse(train_prediction, trainY)

        test_score = mse(test_prediction, testY)

        train_scores[k, d] = train_score

        test_scores[k, d] = test_score

            
for d in range(max_degree):

    for k in range(Num_datasets):

        plt.plot(x_axis, prediction_curves[:,k,d], color='green', alpha=0.5)

    plt.plot(x_axis, prediction_curves[:,:,d].mean(axis=1), color='blue', linewidth=2)

    plt.title("curves for degree = %d" % (d + 1))

    plt.show()
average_train_prediction = np.zeros((trainN, max_degree)) # 模型的平均表现

squared_bias = np.zeros(max_degree)



trueY_train = f_X[:trainN] # 真值

# 每个模型的bias

for d in range(max_degree):

    for i in range(trainN):

        average_train_prediction[i,d] = train_predictions[i,:,d].mean()

    squared_bias[d] = ((average_train_prediction[:,d] - trueY_train) ** 2).mean()
variances = np.zeros((trainN, max_degree))

for d in range(max_degree):

    for i in range(trainN):

        difference = train_predictions[i,:,d] - average_train_prediction[i,d]

        variances[i,d] = np.dot(difference, difference) / Num_datasets

variance = variances.mean(axis=0)
degrees = np.arange(max_degree) + 1

best_degree = np.argmin(test_scores.mean(axis=0)) + 1



plt.plot(degrees, squared_bias, label='squared bias')

plt.plot(degrees, variance, label = 'variance')

plt.plot(degrees, test_scores.mean(axis=0), label='test scores')

plt.plot(degrees, squared_bias + variance, label='squared bias + variance')

plt.axvline(x=best_degree, linestyle='--', label='best complexity')

plt.legend()

plt.show()
plt.plot(degrees, train_scores.mean(axis=0), label='train scores')

plt.plot(degrees, test_scores.mean(axis=0), label= 'test scores')

plt.axvline(x=best_degree, linestyle='--', label='best complexity')

plt.legend()

plt.show()