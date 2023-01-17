# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import division

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt



years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]

gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# cria um gráfico de linha, anos no eixo x, gdp no eixo y

plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

# adiciona um título

plt.title("GDP Nominal")

# adiciona um selo no eixo y

plt.ylabel("Bilhões de $")

plt.show()
friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67]

minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# nomeia cada posição

for label, friend_count, minute_count in zip(labels, friends, minutes):

    plt.annotate(label, 

                 xy=(friend_count, minute_count), # coloca o rótulo com sua posição

                 xytext=(5, -5), # mas compensa um pouco

                 textcoords='offset points')

plt.title("Minutos Diários vs. Número de Amigos")

plt.xlabel("# de amigos")

plt.ylabel("minutos diários passados no site")

plt.show()
data = [(0.7,48000,1),(1.9,48000,0),(2.5,60000,1),(4.2,63000,0),(6,76000,0),(6.5,69000,0),(7.5,76000,0),(8.1,88000,0),(8.7,83000,1),(10,83000,1),(0.8,43000,0),(1.8,60000,0),(10,79000,1),(6.1,76000,0),(1.4,50000,0),(9.1,92000,0),(5.8,75000,0),(5.2,69000,0),(1,56000,0),(6,67000,0),(4.9,74000,0),(6.4,63000,1),(6.2,82000,0),(3.3,58000,0),(9.3,90000,1),(5.5,57000,1),(9.1,102000,0),(2.4,54000,0),(8.2,65000,1),(5.3,82000,0),(9.8,107000,0),(1.8,64000,0),(0.6,46000,1),(0.8,48000,0),(8.6,84000,1),(0.6,45000,0),(0.5,30000,1),(7.3,89000,0),(2.5,48000,1),(5.6,76000,0),(7.4,77000,0),(2.7,56000,0),(0.7,48000,0),(1.2,42000,0),(0.2,32000,1),(4.7,56000,1),(2.8,44000,1),(7.6,78000,0),(1.1,63000,0),(8,79000,1),(2.7,56000,0),(6,52000,1),(4.6,56000,0),(2.5,51000,0),(5.7,71000,0),(2.9,65000,0),(1.1,33000,1),(3,62000,0),(4,71000,0),(2.4,61000,0),(7.5,75000,0),(9.7,81000,1),(3.2,62000,0),(7.9,88000,0),(4.7,44000,1),(2.5,55000,0),(1.6,41000,0),(6.7,64000,1),(6.9,66000,1),(7.9,78000,1),(8.1,102000,0),(5.3,48000,1),(8.5,66000,1),(0.2,56000,0),(6,69000,0),(7.5,77000,0),(8,86000,0),(4.4,68000,0),(4.9,75000,0),(1.5,60000,0),(2.2,50000,0),(3.4,49000,1),(4.2,70000,0),(7.7,98000,0),(8.2,85000,0),(5.4,88000,0),(0.1,46000,0),(1.5,37000,0),(6.3,86000,0),(3.7,57000,0),(8.4,85000,0),(2,42000,0),(5.8,69000,1),(2.7,64000,0),(3.1,63000,0),(1.9,48000,0),(10,72000,1),(0.2,45000,0),(8.6,95000,0),(1.5,64000,0),(9.8,95000,0),(5.3,65000,0),(7.5,80000,0),(9.9,91000,0),(9.7,50000,1),(2.8,68000,0),(3.6,58000,0),(3.9,74000,0),(4.4,76000,0),(2.5,49000,0),(7.2,81000,0),(5.2,60000,1),(2.4,62000,0),(8.9,94000,0),(2.4,63000,0),(6.8,69000,1),(6.5,77000,0),(7,86000,0),(9.4,94000,0),(7.8,72000,1),(0.2,53000,0),(10,97000,0),(5.5,65000,0),(7.7,71000,1),(8.1,66000,1),(9.8,91000,0),(8,84000,0),(2.7,55000,0),(2.8,62000,0),(9.4,79000,0),(2.5,57000,0),(7.4,70000,1),(2.1,47000,0),(5.3,62000,1),(6.3,79000,0),(6.8,58000,1),(5.7,80000,0),(2.2,61000,0),(4.8,62000,0),(3.7,64000,0),(4.1,85000,0),(2.3,51000,0),(3.5,58000,0),(0.9,43000,0),(0.9,54000,0),(4.5,74000,0),(6.5,55000,1),(4.1,41000,1),(7.1,73000,0),(1.1,66000,0),(9.1,81000,1),(8,69000,1),(7.3,72000,1),(3.3,50000,0),(3.9,58000,0),(2.6,49000,0),(1.6,78000,0),(0.7,56000,0),(2.1,36000,1),(7.5,90000,0),(4.8,59000,1),(8.9,95000,0),(6.2,72000,0),(6.3,63000,0),(9.1,100000,0),(7.3,61000,1),(5.6,74000,0),(0.5,66000,0),(1.1,59000,0),(5.1,61000,0),(6.2,70000,0),(6.6,56000,1),(6.3,76000,0),(6.5,78000,0),(5.1,59000,0),(9.5,74000,1),(4.5,64000,0),(2,54000,0),(1,52000,0),(4,69000,0),(6.5,76000,0),(3,60000,0),(4.5,63000,0),(7.8,70000,0),(3.9,60000,1),(0.8,51000,0),(4.2,78000,0),(1.1,54000,0),(6.2,60000,0),(2.9,59000,0),(2.1,52000,0),(8.2,87000,0),(4.8,73000,0),(2.2,42000,1),(9.1,98000,0),(6.5,84000,0),(6.9,73000,0),(5.1,72000,0),(9.1,69000,1),(9.8,79000,1),]

data = list(map(list, data)) # change tuples to lists

data
x = [[1] + row[:2] for row in data] # cada elemento é [1, experience, salary]

y = [row[2] for row in data]        # cada elemento é paid_account
from collections import Counter

from functools import partial, reduce

# from linear_algebra import dot, vector_add

# from gradient_descent import maximize_stochastic, maximize_batch

# from working_with_data import rescale

# from machine_learning import train_test_split

# from multiple_regression import estimate_beta, predict

import math, random



# helpers



## stats



def mean(x):

    return sum(x) / len(x)



def standard_deviation(x):

    return math.sqrt(variance(x))



def variance(x):

    """assumes x has at least two elements"""

    n = len(x)

    deviations = de_mean(x)

    return sum_of_squares(deviations) / (n - 1)



def de_mean(x):

    """translate x by subtracting its mean (so the result has mean 0)"""

    x_bar = mean(x)

    return [x_i - x_bar for x_i in x]



##



## linear algebra

def dot(v, w):

    """v_1 * w_1 + ... + v_n * w_n"""

    return sum(v_i * w_i for v_i, w_i in zip(v, w))



def vector_add(v, w):

    """adds two vectors componentwise"""

    return [v_i + w_i for v_i, w_i in zip(v,w)]



def vector_subtract(v, w):

    """subtracts two vectors componentwise"""

    return [v_i - w_i for v_i, w_i in zip(v,w)]



def make_matrix(num_rows, num_cols, entry_fn):

    """returns a num_rows x num_cols matrix

    whose (i,j)-th entry is entry_fn(i, j)"""

    return [[entry_fn(i, j) for j in range(num_cols)]

            for i in range(num_rows)]



def shape(A):

    num_rows = len(A)

    num_cols = len(A[0]) if A else 0

    return num_rows, num_cols



def get_column(A, j):

    return [A_i[j] for A_i in A]



def sum_of_squares(v):

    """v_1 * v_1 + ... + v_n * v_n"""

    return dot(v, v)



def scalar_multiply(c, v):

    return [c * v_i for v_i in v]



## linear algebra



## gradient_descent



def negate(f):

    """return a function that for any input x returns -f(x)"""

    return lambda *args, **kwargs: -f(*args, **kwargs)



def negate_all(f):

    """the same when f returns a list of numbers"""

    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]



def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

    data = list(zip(x, y))

    theta = theta_0                             # initial guess

    alpha = alpha_0                             # initial step size

    min_theta, min_value = None, float("inf")   # the minimum so far

    iterations_with_no_improvement = 0

    # if we ever go 100 iterations with no improvement, stop

    while iterations_with_no_improvement < 100:

        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )

        if value < min_value:

            # if we've found a new minimum, remember it

            # and go back to the original step size

            min_theta, min_value = theta, value

            iterations_with_no_improvement = 0

            alpha = alpha_0

        else:

            # otherwise we're not improving, so try shrinking the step size

            iterations_with_no_improvement += 1

            alpha *= 0.9

        # and take a gradient step for each of the data points

        for x_i, y_i in in_random_order(data):

            gradient_i = gradient_fn(x_i, y_i, theta)

            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta



def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

    return minimize_stochastic(negate(target_fn),

                               negate_all(gradient_fn),

                               x, y, theta_0, alpha_0)



def in_random_order(data):

    """generator that returns the elements of data in random order"""

    indexes = [i for i, _ in enumerate(data)]  # create a list of indexes

    random.shuffle(indexes)                    # shuffle them

    for i in indexes:                          # return the data in that order

        yield data[i]

        

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):

    """use gradient descent to find theta that minimizes target function"""



    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]



    theta = theta_0                           # set theta to initial value

    target_fn = safe(target_fn)               # safe version of target_fn

    value = target_fn(theta)                  # value we're minimizing



    while True:

        gradient = gradient_fn(theta)

        next_thetas = [step(theta, gradient, -step_size)

                       for step_size in step_sizes]



        # choose the one that minimizes the error function

        next_theta = min(next_thetas, key=target_fn)

        next_value = target_fn(next_theta)



        # stop if we're "converging"

        if abs(value - next_value) < tolerance:

            return theta

        else:

            theta, value = next_theta, next_value

        

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):

    return minimize_batch(negate(target_fn),

                          negate_all(gradient_fn),

                          theta_0,

                          tolerance)



def safe(f):

    """define a new function that wraps f and return it"""

    def safe_f(*args, **kwargs):

        try:

            return f(*args, **kwargs)

        except:

            return float('inf')         # this means "infinity" in Python

    return safe_f



def step(v, direction, step_size):

    """move step_size in the direction from v"""

    return [v_i + step_size * direction_i

            for v_i, direction_i in zip(v, direction)]



## gradient_descent



## working_data



def rescale(data_matrix):

    """rescales the input data so that each column

    has mean 0 and standard deviation 1

    ignores columns with no deviation"""

    means, stdevs = scale(data_matrix)



    def rescaled(i, j):

        if stdevs[j] > 0:

            return (data_matrix[i][j] - means[j]) / stdevs[j]

        else:

            return data_matrix[i][j]



    num_rows, num_cols = shape(data_matrix)

    return make_matrix(num_rows, num_cols, rescaled)



def scale(data_matrix):

    num_rows, num_cols = shape(data_matrix)

    means = [mean(get_column(data_matrix,j))

             for j in range(num_cols)]

    stdevs = [standard_deviation(get_column(data_matrix,j))

              for j in range(num_cols)]

    return means, stdevs



## working_data



## machine_learning



def split_data(data, prob):

    """split data into fractions [prob, 1 - prob]"""

    results = [], []

    for row in data:

        results[0 if random.random() < prob else 1].append(row)

    return results



def train_test_split(x, y, test_pct):

    data = list(zip(x, y))                        # pair corresponding values

    train, test = split_data(data, 1 - test_pct)  # split the dataset of pairs

    x_train, y_train = list(zip(*train))          # magical un-zip trick

    x_test, y_test = list(zip(*test))

    return x_train, x_test, y_train, y_test



## machine_learning



## multiple_regression



def predict(x_i, beta):

    return dot(x_i, beta)



def estimate_beta(x, y):

    beta_initial = [random.random() for x_i in x[0]]

    return minimize_stochastic(squared_error,

                               squared_error_gradient,

                               x, y,

                               beta_initial,

                               0.001)



def error(x_i, y_i, beta):

    return y_i - predict(x_i, beta)



def squared_error(x_i, y_i, beta):

    return error(x_i, y_i, beta) ** 2



def squared_error_gradient(x_i, y_i, beta):

    """the gradient corresponding to the ith squared error term"""

    return [-2 * x_ij * error(x_i, y_i, beta)

            for x_ij in x_i]



## multiple_regression



# helpers



def logistic(x):

    return 1.0 / (1 + math.exp(-x))



def logistic_prime(x):

    return logistic(x) * (1 - logistic(x))



def logistic_log_likelihood_i(x_i, y_i, beta):

    if y_i == 1:

        return math.log(logistic(dot(x_i, beta)))

    else:

        return math.log(1 - logistic(dot(x_i, beta)))



def logistic_log_likelihood(x, y, beta):

    return sum(logistic_log_likelihood_i(x_i, y_i, beta)

               for x_i, y_i in zip(x, y))



def logistic_log_partial_ij(x_i, y_i, beta, j):

    """here i is the index of the data point,

    j the index of the derivative"""



    return (y_i - logistic(dot(x_i, beta))) * x_i[j]



def logistic_log_gradient_i(x_i, y_i, beta):

    """the gradient of the log likelihood

    corresponding to the i-th data point"""



    return [logistic_log_partial_ij(x_i, y_i, beta, j)

            for j, _ in enumerate(beta)]



def logistic_log_gradient(x, y, beta):

    return reduce(vector_add,

                  [logistic_log_gradient_i(x_i, y_i, beta)

                   for x_i, y_i in zip(x,y)])
rescaled_x = rescale(x)

beta = estimate_beta(rescaled_x, y) # [0.26, 0.43, -0.43]

predictions = [predict(x_i, beta) for x_i in rescaled_x]

plt.scatter(predictions, y)

plt.xlabel("prevista")

plt.ylabel("realizada")

plt.show()
random.seed(0)

x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, 0.33)



# queremos maximizar o log da probabilidade em dados de treinamento

fn = partial(logistic_log_likelihood, x_train, y_train)

gradient_fn = partial(logistic_log_gradient, x_train, y_train)

# escolhemos um ponto de partida aleatório

beta_0 = [random.random() for _ in range(3)]

# e maximizamos usando o gradiente descendente

beta_hat = maximize_batch(fn, gradient_fn, beta_0)

print("beta_batch", beta_hat)
# escolhemos um ponto de partida aleatório

beta_0 = [random.random() for _ in range(3)]

beta_hat = maximize_stochastic(logistic_log_likelihood_i,

               logistic_log_gradient_i,

               x_train, y_train, beta_0)

print("bet stochastic", beta_hat)
true_positives = false_positives = true_negatives = false_negatives = 0

for x_i, y_i in zip(x_test, y_test):

    predict = logistic(dot(beta_hat, x_i))

    if y_i == 1 and predict >= 0.5:   # TP: paga e previmos paga

        true_positives += 1

    elif y_i == 1:                    # FN: paga e previmos não pagantes

        false_negatives += 1

    elif predict >= 0.5:              # FP: não paga e previmos pagantes

        false_positives += 1

    else:                             # TN: não paga e previmos não paga

        true_negatives += 1

        

precision = true_positives / (true_positives + false_positives)

recall = true_positives / (true_positives + false_negatives)



print("precision", precision)

print("recall", recall)
predictions = [logistic(dot(beta_hat, x_i)) for x_i in x_test]

plt.scatter(predictions, y_test)

plt.xlabel("probabilidade prevista")

plt.ylabel("resultado real")

plt.title("Regressão Logística Prevista vs. Real")

plt.show()