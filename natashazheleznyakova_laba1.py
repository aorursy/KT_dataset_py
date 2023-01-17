import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt
import sys

sys.path.append("..")
with open('/kaggle/input/lab-1-data/train.npy', 'rb') as fin:

    X = np.load(fin)

    

with open('/kaggle/input/lab-1-data/target.npy', 'rb') as fin:

    y = np.load(fin)



plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)

plt.show()
def expand(X):

    """

    Добавим квадратичные признаки. 

    Это позволит линейной модели разделить линейно неразделимые данные

    

    Для каждого объекта (строки в матрице) сформируйте расширенную строку:

    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]

    

    :param X: матрица признаков размера [n_samples,2]

    :returns: расширенная матрица признаков размера [n_samples,6]

    """

    X_expanded = np.zeros((X.shape[0], 6))

    

    # TODO:<your code here>

    for i in range(X.shape[0]):

        X_expanded[i] = [X[i][0], X[i][1], X[i][0]**2, X[i][1]**2, X[i][0]*X[i][1], 1]   

                   

    return X_expanded

    
X_expanded = expand(X)
# simple test on random numbers



dummy_X = np.array([

        [0,0],

        [1,0],

        [2.61,-1.28],

        [-0.59,2.1]

    ])



# call your expand function

dummy_expanded = expand(dummy_X)



# what it should have returned:   x0       x1       x0^2     x1^2     x0*x1    1

dummy_expanded_ans = np.array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  1.    ],

                               [ 1.    ,  0.    ,  1.    ,  0.    ,  0.    ,  1.    ],

                               [ 2.61  , -1.28  ,  6.8121,  1.6384, -3.3408,  1.    ],

                               [-0.59  ,  2.1   ,  0.3481,  4.41  , -1.239 ,  1.    ]])



#tests

assert isinstance(dummy_expanded,np.ndarray), "please make sure you return numpy array"

assert dummy_expanded.shape == dummy_expanded_ans.shape, "please make sure your shape is correct"

assert np.allclose(dummy_expanded,dummy_expanded_ans,1e-3), "Something's out of order with features"



print("Seems legit!")

def probability(X, w):

    """

    Given input features and weights

    return predicted probabilities of y==1 given x, P(y=1|x), see description above

        

    Don't forget to use expand(X) function (where necessary) in this and subsequent functions.

    

    Принимает на вход признаки и веса

    Возвращает прогнозируемые вероятности y == 1 для заданных x, P (y = 1 | x), см. описание выше

        

    Не забудьте использовать expand(X) (при необходимости) в этой и последующих функциях.

    

    :param X: матрица признаков размера [n_samples,6] (расширенная)

    :param w: вектор весов w размера [6] для каждого из признаков

    :returns: массив предсказанных верятностей в интервале [0,1].

    """



    # TODO:<your code here>

    return 1 / (1 + np.exp(-np.dot(X, w)))

        
import math



dummy_weights = np.linspace(-1, 1, 6)

ans_part1 = probability(X_expanded[:1, :], dummy_weights)[0]

def compute_loss(X, y, w):

    """

    Принимает матрицу признаков X [n_samples,6], целевой вектор [n_samples] of 1/0,

    и вектор весов w [6], вычисляет скалярную функцию потерь по формуле L используя формулу выше.

    Имейте в виду, что потери усредняются по всем выборкам (строкам) в X.

    """

    # TODO:<your code here>

    p = probability(X, w)

    

    return -1 / X.shape[0] * np.sum(y * np.log(p) + (1-y) * np.log(1-p))
ans_part2 = compute_loss(X_expanded, y, dummy_weights)
def compute_grad(X, y, w):

    """

    Принимает матрицу признаков X [n_samples,6], целевой вектор [n_samples] состоящий из значений 1/0,

    и вектор весов w [6], вычисляет вектор [6] производных L по каждому весу.

    Имейте в виду, что потери усредняются по всем выборкам (строкам) в X.

    """

    

    # TODO<your code here>

    p = probability(X, w)

    

    return 1 / X.shape[0] * np.dot(X.T, p-y)
ans_part3 = np.linalg.norm(compute_grad(X_expanded, y, dummy_weights))
from IPython import display



h = 0.01

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



def visualize(X, y, w, history):

    """draws classifier prediction with matplotlib magic"""

    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)

    Z = Z.reshape(xx.shape)

    plt.subplot(1, 2, 1)

    plt.contourf(xx, yy, Z, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())

    

    plt.subplot(1, 2, 2)

    plt.plot(history)

    plt.grid()

    ymin, ymax = plt.ylim()

    plt.ylim(0, ymax)

    display.clear_output(wait=True)

    plt.show()
visualize(X, y, dummy_weights, [0.5, 0.5, 0.25])
# Использовать np.random.seed(42), eta=0.1, n_iter=100 и batch_size=4



np.random.seed(42)

w = np.array([0, 0, 0, 0, 0, 1])



eta= 0.1 # learning rate



n_iter = 100

batch_size = 4

loss = np.zeros(n_iter)

plt.figure(figsize=(12, 5))



for i in range(n_iter):

    ind = np.random.choice(X_expanded.shape[0], batch_size)

    loss[i] = compute_loss(X_expanded, y, w)

    if i % 10 == 0:

        visualize(X_expanded[ind, :], y[ind], w, loss)



    # Имейте в виду, что compute_grad уже делает усреднение по выборке

    # TODO:<your code here>

    w = w - eta * compute_grad(X_expanded[ind,:], y[ind], w)



visualize(X, y, w, loss)

plt.clf()
ans_part4 = compute_loss(X_expanded, y, w)
# please use np.random.seed(42), eta=0.05, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results

np.random.seed(42)

w = np.array([0, 0, 0, 0, 0, 1])



eta = 0.05 # learning rate

alpha = 0.9 # momentum

nu = np.zeros_like(w)



n_iter = 100

batch_size = 4

loss = np.zeros(n_iter)

plt.figure(figsize=(12, 5))



for i in range(n_iter):

    ind = np.random.choice(X_expanded.shape[0], batch_size)

    loss[i] = compute_loss(X_expanded, y, w)

    if i % 10 == 0:

        visualize(X_expanded[ind, :], y[ind], w, loss)



    # TODO:<your code here>

    nu = alpha * nu + eta * compute_grad(X_expanded, y, w)

    w = w - nu



visualize(X, y, w, loss)

plt.clf()
ans_part5 = compute_loss(X_expanded, y, w)
# Используйте np.random.seed(42), eta=0.1, alpha=0.9, n_iter=100 и batch_size=4

np.random.seed(42)



w = np.array([0, 0, 0, 0, 0, 1.])



eta = 0.1 # скорость обучения

alpha = 0.9

g2 = None # начинаем с None, чтобы корректно обновить это значение на первой итерации

eps = 1e-8



n_iter = 100

batch_size = 4

loss = np.zeros(n_iter)

plt.figure(figsize=(12,5))



g = compute_grad(X_expanded, y, w)

g_squared = np.square(g)

g2 = g*g

g2 = alpha * g2 + (1 - alpha) * g_squared

w = w - eta * g / np.sqrt(g2 + eps)



for i in range(n_iter):

    ind = np.random.choice(X_expanded.shape[0], batch_size)

    loss[i] = compute_loss(X_expanded, y, w)

    if i % 10 == 0:

        visualize(X_expanded[ind, :], y[ind], w, loss)



    # TODO:<your code here>

    g = compute_grad(X_expanded, y, w)

    g_squared = np.square(g)

    g2 = alpha * g2 + (1 - alpha) * g_squared

    w = w - eta * g / np.sqrt(g2 + eps)



visualize(X, y, w, loss)

plt.clf()
ans_part6 = compute_loss(X_expanded, y, w)