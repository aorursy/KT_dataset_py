########################################################################################

# Лабораторная работа 2 по дисциплине МРЗвИС

# Выполнена студентом группы 721702

# БГУИР Гурбович Артём Игоревич

#

# Вариант 9

#

# 25.10.2019



import math

import numpy as np
def fibonacci_function(n):

    if n == 1 or n == 2:

        return float(1.0)

    else:

        return float(float(fibonacci_function(n - 1.0)) + float(fibonacci_function(n - 2.0)))

    

def factorial_function(n):

    return float(math.factorial(n))



def periodic_function(n):

    return float((-1)**n)



def periodic_function_2(n):

    if n % 2 == 0:

        return 0.0

    else:

        return 1.0

    

def periodic_function_3(n):

    a = [1.0, 0.0, 0.5]

    return a[n % 3]



def power_function(n): 

    return float((2)**n)



def create_sequence(sequence_number, sequence_size):

    result = []

    if sequence_number == 1:

        for n in range(1, sequence_size + 1):

            result.append(fibonacci_function(n))

    elif sequence_number == 2:

        for n in range(1, sequence_size + 1):

            result.append(factorial_function(n))

    elif sequence_number == 3:

        for n in range(1, sequence_size + 1):

            result.append(periodic_function(n))

    elif sequence_number == 4:

        for n in range(1, sequence_size + 1):

            result.append(power_function(n))

    elif sequence_number == 5:

        for n in range(1, sequence_size + 1):

            result.append(periodic_function_2(n))

    else:

        for n in range(1, sequence_size + 1):

            result.append(periodic_function_3(n))

    return result
def activation_function(x):

    #return math.log(x + math.sqrt(math.pow(x, 2.0) + 1.0))

    return 0.1*x



def derivative_activation_function(x):

    #return (1.0 / math.sqrt(math.pow(x, 2.0) + 1.0))

    return 0.1
def forward_propagation():

    global W, v_input, Wch_h, v_context_hidden, Wco_h, context_output, T, v_hidden, W_, T_, output

    for j in range(m):

        S = 0.0

        for i in range(p):

            S += W[i][j] * v_input[i]

        for i in range(m):

            S += Wch_h[i][j] * v_context_hidden[i]

        S += Wco_h[0][j] * context_output

        S -= T[j]

        v_hidden[j] = activation_function(S)

    S = 0.0

    for i in range(m):

        S += W_[i][0] * v_hidden[i]

    S -= T_

    output = activation_function(S)

    for i in range(m):

        v_context_hidden[i] = v_hidden[i]

    context_output = output
def back_propagation(val):

    global step, output, W, W_, v_hidden, v_input, Wch_h, v_context_hidden, Wco_h, context_output, T, T_

    diff = float(step * (output - val))

    for i in range(m):

        for j in range(p):

            W[j][i] -= diff * W_[i][0] * derivative_activation_function(v_hidden[i]) * v_input[j]

        for j in range(m):

            Wch_h[j][i] -= diff * W_[i][0] * derivative_activation_function(v_hidden[i]) * v_context_hidden[j]

        W_[i][0] -= diff * v_hidden[i]

        Wco_h[0][i] -= diff * W_[i][0] * derivative_activation_function(v_hidden[i]) * context_output

        T[i] = diff * W_[i][0] * derivative_activation_function(v_hidden[i])

    T_ = diff
def train():

    global v_input, X, output, exp_values

    E = 1000.0

    iteration = 0

    while E > error and iteration <= N:

        iteration += 1

        E = 0.0

        for i in range(L):

            v_input = X[i]

            forward_propagation()

            E += (float(output - exp_values[i]) ** 2.0) / 2.0

            back_propagation(exp_values[i])

        if iteration % 1000 == 0:

            print("Iteration:", iteration, " |  Error:", E)

    print("End. Iteration:", iteration, " |  Error:", E)
def test():

    global k, p, v_input, sequence, r, output, exp_sequence

    res_sequence = []

    j = k - p

    for i in range(p):

        v_input[i] = sequence[j]

        j += 1

    for i in range(r):

        if i > 0:

            for j in range(p - 1):

                v_input[j] = v_input[j + 1]

            v_input[p - 1] = output

        print(v_input)

        forward_propagation()

        res_sequence.append(output)

    print("Result:")

    for i in range(r):

        print("value:", res_sequence[i], "expected:", exp_sequence[k + i], "error:", exp_sequence[k + i] - res_sequence[i])
sequence_number = 1#int(input("Entrer sequence number:\n1) fibonacci\n2) factorial\n3) -1^n\n4) 2^n\n5) 1,0,1,0,...\n6) 0,0.5,1,0,0.5,1,... \nsequence number = "))

k = 6#int(input("Enter size of training sequence\nk = "))

r = 3#int(input("Enter number of elements to predict\nr = "))

sequence = np.array(create_sequence(sequence_number, k), dtype="float64").tolist() # исходная последовательность

exp_sequence = np.array(create_sequence(sequence_number, k + r), dtype="float64").tolist() # ожидаемая выходная последовательность

p = 4#int(input("Enter window size of training sequence\np = ")) # размер окна

m = 10#int(input("Enter m\nm = ")) # количество нейронов скрытого слоя

L = k - p # количество строк в матрице обучения 

step = 0.0001#float(input("Enter alpha\nalpha = ")) # коэффициент альфа

error = 0.01 # максимально допустимая ошибка

N = 100000 # максимальное количество шагов обучения 1000000

v_input = np.zeros(p).tolist() # входной вектор (p)

v_hidden = np.zeros(m).tolist() # выходной вектор из скрытого слоя (m)

output = 0.0 # выходной вектор из выходного слоя (1)

v_context_hidden = np.zeros(m).tolist() # контекстный слой для скрытого слоя (m)

context_output = 0.0 # контекстный слой для выходного слоя (1)

T = np.zeros(m).tolist() # пороговые значения для скрытого слоя

T_ = 0.0 # пороговые значения для выходного слоя

X = np.zeros((L, p)).tolist() # матрица обучения m x p

exp_values = [] # значения, которые необходимо получить при обучении для каждого входного вектора

for i in range(L):

    for j in range(p):

        X[i][j] = sequence[i + j]

    exp_values.append(sequence[i + p])

exp_values = np.array(exp_values, dtype="float64").tolist()

W = np.random.rand(p, m) * 2.0 - 1.0 # матрица весов W на скрытом слое p x m

Wch_h = np.random.rand(m, m) * 2.0 - 1.0 # матрица весов между контекстным с предыдущими значениями скрытого и скрытым слоем m x m

W_ = np.random.rand(m, 1) * 2.0 - 1.0 # матрица весов W_ на выходном слое m x 1

Wco_h = np.random.rand(1, m) * 2.0 - 1.0 # матрица весов между контекстным с предыдущим значением выходного и скрытым слоем 1 x m

W = W.tolist()

Wch_h = Wch_h.tolist()

W_ = W_.tolist()

Wco_h = Wco_h.tolist()

print("training sequence =", sequence)
train()
test()