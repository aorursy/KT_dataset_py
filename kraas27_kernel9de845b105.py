import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf

import numpy as np

from sklearn.datasets import make_classification

import matplotlib.pyplot as plt

%matplotlib inline
NUM_SAMPLES = 1000

NUM_FEATURES = 2
X, y = make_classification(n_samples = NUM_SAMPLES, 

                           n_features = NUM_FEATURES, 

                           n_informative = NUM_FEATURES, 

                           n_redundant = 0, 

                           n_classes = 2, 

                           n_clusters_per_class = 1, 

                           class_sep = 0.75,

                           random_state = 54312)
y = y.reshape(-1, 1)

y.shape
ones = np.where(y == 1)   # индексы объектов класса '1'

zeros = np.where(y == 0)  # индексы объектов класса '0'
plt.plot(X[ones, 0], X[ones, 1], 'ob');

plt.plot(X[zeros, 0], X[zeros, 1], 'or');
import string



def py_func_with_grad(func, inp, Tout, grad, name=None, stateful=False, graph=None):

    name_prefix = ''.join(np.random.choice(list(string.ascii_letters), size = 10)) # генерит новое имя

    name = '%s_%s' % (name_prefix, name or '')

    grad_func_name = '%s_grad' % name # к имени из 10 букв пребавляет '_grad'

    

    tf.RegisterGradient(grad_func_name)(grad) # декоратор для регистрации функции градиента для типа op

    

    g = graph or tf.get_default_graph()

    with g.gradient_override_map({'PyFunc': grad_func_name, 

                                  'PyFuncStateless': grad_func_name}):

        with tf.name_scope(name, 'PyFuncOp', inp):

            return tf.py_func(func, inp, Tout, stateful = stateful, name = name) # оборачиваем нашу 

                                                                # функцию в качестве тензора
def linear_op_forward(X, W, b):

    '''Реализация линейной операции'''

    return np.dot(X, W.T) + b # аргументы являются numpy-массивами



def linear_op_backward(op, grads):

    '''Реализация вычисления градиента линейной операции'''

    X = op.inputs[0]  # тензор входных данных

    W = op.inputs[1]  # тензор параметров модели

    b = op.inputs[2]

    dX = tf.multiply(grads, W)

    dW = tf.reduce_sum(tf.multiply(X, grads),

                      axis = 0,

                      keep_dims = True)

    db = tf.reduce_sum(tf.multiply(1., grads),

                      axis = 0,

                      keep_dims = True)

    return dX, dW, db



def sigmoid_op_forward(X):

    return 1 / (1 + np.exp(-X))



def sigmoid_op_backward(op, grads):

    sigmoid = op.outputs[0]

    sigmoid_deriv = sigmoid * (1 - sigmoid)

    d_grad = tf.reduce_sum(tf.multiply(sigmoid_deriv, grads),

                       axis = 0,

                       keep_dims = True)

    return d_grad
y.shape
BATCH_SIZE = NUM_SAMPLES // 10



weights = None # в этой переменной мы сохраним результат обучения модели

bias = None

learning_curve = [] # значение ошибки на каждой итерации обучения



with tf.Session(graph = tf.Graph()) as sess: # инициализируем сессию вычислений

    

    # создаем placeholder'ы, через них мы будем передавать внешние данные в граф вычислений

    plh_X = tf.placeholder(dtype = tf.float32, shape = [None, NUM_FEATURES])

    plh_labels = tf.placeholder(dtype = tf.float32, shape = [None, 1])

    

    # создаём переменную для хранения весов модели

    # эти веса будут изменяться в процессе обучения

    var_W = tf.Variable(tf.random_uniform(shape = [1, NUM_FEATURES], 

                                          dtype = tf.float32, 

                                          seed = 27))

    

    var_b = tf.Variable(tf.random_normal(shape = [1, 1], 

                                         mean = 0.5, 

                                         dtype = tf.float32, 

                                         seed = 27))

    

#     var_b = tf.Variable(initial_value=1.0)

    

    # создаем переменную для результата предсказания модели

    var_Pred = py_func_with_grad(linear_op_forward, 

                                 [plh_X, var_W, var_b], 

                                 [tf.float32], 

                                 name = 'linear_op', 

                                 grad = linear_op_backward, 

                                 graph = sess.graph)

    

    # создаем переменную для результата операции sigmoid

    var_Sigmoid = py_func_with_grad(sigmoid_op_forward, 

                                    [var_Pred], 

                                    [tf.float32], 

                                    name = 'sigmoid_op', 

                                    grad = sigmoid_op_backward, 

                                    graph = sess.graph)

    

    # кросс-энтропийная функция потерь для бинарной классификации

    cost = tf.losses.sigmoid_cross_entropy(plh_labels, var_Sigmoid)

    

    # инициализируем оптимизатор и указываем скорость обучения

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.4).minimize(cost)

    

    # инициализируем placeholder'ы и переменные

    sess.run(tf.global_variables_initializer())

    

    indices = np.arange(len(X)) # массив индексов объектов

    

    # выполняем итерации по 10 эпохам

    for epoch in range(250):

        

        # в начале каждой эпохи перемешиваем индексы

        np.random.shuffle(indices)

        

        # внутри каждой эпохи данные разбиваются на батчи

        for batch in range(len(X) // BATCH_SIZE):

            

            # выбираем индексы очередного батча:

            batch_indices = indices[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]

            

            # выполняем шаг обучения: вычисляем ошибку и обновляем веса

            loss, _ = sess.run([cost, optimizer], # указываем какие операции необходимо выполнить

                               feed_dict = {plh_X: X[batch_indices], # передаем входные данные

                                            plh_labels: y[batch_indices]})

            

            # сохраняем значение ошибки для построения кривой обучения

            learning_curve.append(loss)

            

            # выводим текущее значение ошибки для каждого 10-го шага

            steps = len(learning_curve) - 1

            if steps % 250 == 0:

                print ('[%03d] loss=%.3f weights=%s bias=%s' % (steps, loss, var_W.eval(), var_b.eval()))

                

        # сохраняем обученные веса

        weights = var_W.eval()

        bias = var_b.eval()

            
plt.xlabel('step')

plt.ylabel('loss')

plt.title('Learning curve')

plt.plot(learning_curve);
bias[0, 0]
y_pred = - X[:, 0] * weights[0, 0]/ weights[0, 1]  - bias[0, 0]/ weights[0, 1]



order = np.argsort(X[:, 0])



plt.xlabel('x')

plt.ylabel('y')

plt.plot(X[ones, 0], X[ones, 1], 'ob',

         X[zeros, 0], X[zeros, 1], 'or',

         X[order, 0], y_pred[order], '-g');
X.shape