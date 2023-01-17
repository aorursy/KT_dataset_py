import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/MonthlySales.csv')

sales = list(data['sales'])

data.head()
def cal_mse(real_list, pred_list):

    differ = len(pred_list) - len(real_list)

    MSEs = 0

    range_list = range(differ, len(real_list))

    for i in range_list:

        x_real = real_list[i]

        x_pred = pred_list[i]

        MSEs = (x_pred - x_real) ** 2 + MSEs

    MSE = (MSEs ** (1 / 2)) / int(len(range_list))

    return MSE





def exponential_smoothing(alpha, s):

    '''

    smoothing

    '''

    s_temp = [0 for i in range(len(s))]

    s_temp[0] = (s[0] + s[1] + s[2]) / 3

    for i in range(1, len(s)):

        s_temp[i] = alpha * s[i] + (1 - alpha) * s_temp[i - 1]

    return s_temp





def compute_single(alpha, s):

    '''

    single exponetial smoothing

    '''

    return exponential_smoothing(alpha, s)





def single_pred_list(alpha, s):

    '''pred formula ：x_{t+1}=S_t'''

    single_pred = [None]

    S_1 = compute_single(alpha, s)

    for i in S_1:

        single_pred.append(i)

    return single_pred





def single_best_alpha(s):

    '''get best alpha for single exponential smoothing'''

    mse_list = []

    alpha_list = np.arange(0, 1, 0.01)

    for alpha in alpha_list:

        single_pred = single_pred_list(alpha, s)

        mse = cal_mse(s, single_pred)

        mse_list.append(mse)

    best_alpha = alpha_list[mse_list.index(min(mse_list))]

    min_mse = min(mse_list)

    return best_alpha, min_mse





def compute_double(alpha, s):

    '''

    double exponential smoothing

    '''

    s_single = compute_single(alpha, s)

    s_double = compute_single(alpha, s_single)



    a_double = [0 for i in range(len(s))]

    b_double = [0 for i in range(len(s))]



    for i in range(len(s)):

        a_double[i] = 2 * s_single[i] - s_double[i]

        b_double[i] = (alpha / (1 - alpha)) * (s_single[i] - s_double[i])



    return a_double, b_double





def double_pred_list(alpha, s, t):

    '''pred formula ：x_{t+T}=a_t+B_t*T'''

    a_double, b_double = compute_double(alpha, s)

    double_pred = [None] * t

    for i in range(len(a_double)):

        a_t = a_double[i]

        b_t = b_double[i]

        pred_value = a_t + b_t * t

        double_pred.append(pred_value)

    return double_pred





def double_best_alpha(s, t):

    mse_list = []

    alpha_list = np.arange(0, 1, 0.001)

    for alpha in alpha_list:

        single_pred = double_pred_list(alpha, s, t)

        mse = cal_mse(s, single_pred)

        mse_list.append(mse)

    best_alpha = alpha_list[mse_list.index(min(mse_list))]

    min_mse = min(mse_list)

    return best_alpha, min_mse





def compute_triple(alpha, s):

    '''

    triple exponential smoothing

    '''

    s_single = compute_single(alpha, s)

    s_double = compute_single(alpha, s_single)

    s_triple = exponential_smoothing(alpha, s_double)



    a_triple = [0 for i in range(len(s))]

    b_triple = [0 for i in range(len(s))]

    c_triple = [0 for i in range(len(s))]



    for i in range(len(s)):

        a_triple[i] = 3 * s_single[i] - 3 * s_double[i] + s_triple[i]

        b_triple[i] = (alpha / (2 * ((1 - alpha) ** 2))) * (

                (6 - 5 * alpha) * s_single[i] - 2 * ((5 - 4 * alpha) * s_double[i]) + (4 - 3 * alpha) * s_triple[i])

        c_triple[i] = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single[i] - 2 * s_double[i] + s_triple[i])



    return a_triple, b_triple, c_triple





def triple_pred_list(alpha, s, t):

    '''pred formula ：x_{t+T}=a_t+B_t*T'''

    a_triple, b_triple, c_triple = compute_triple(alpha, s)

    triple_pred = [None] * t

    for i in range(len(a_triple)):

        a_t = a_triple[i]

        b_t = b_triple[i]

        c_t = c_triple[i]

        pred_value = a_t + b_t * t + c_t * (t ** 2)

        triple_pred.append(pred_value)

    return triple_pred





def triple_best_alpha(s, t):

    mse_list = []

    alpha_list = np.arange(0, 1, 0.01)

    for alpha in alpha_list:

        single_pred = triple_pred_list(alpha, s, t)

        mse = cal_mse(s, single_pred)

        mse_list.append(mse)

    best_alpha = alpha_list[mse_list.index(min(mse_list))]

    min_mse = min(mse_list)

    return best_alpha, min_mse





def plot_lines(data, pred_list, title):

    add_value = [None] * (len(pred_list) - len(data))

    xlist = list(range(0, len(pred_list)))

    real = data + add_value

    plt.plot(xlist, real, color='blue', label="actual value")

    plt.plot(xlist, pred_list, color='red', label="predicted value")

#     for m, n in zip(xlist, real):

#         if n != None:

#             plt.text(m, n + 10, '%.0f' % n, ha='center', color='blue', fontsize=9)

#     for m1, n1 in zip(xlist, pred_list):

#         if n1 != None:

#             plt.text(m1, n1 - 10, '%.0f' % n1, ha='center', color='red', fontsize=9)

    plt.legend()

    plt.title(title)

    plt.xlabel('date')

    plt.ylabel('sales')

    plt.show()
# single 

single_alpha, single_mse = single_best_alpha(sales)

single_pred = single_pred_list(single_alpha, sales)

print('single best alpha: ', single_alpha)

print('pred value:' ,str(single_pred[-1]))

single_title = str('single pred with MSE: ' + str(single_mse))

plot_lines(sales, single_pred, single_title)
double_alpha, double_mse = double_best_alpha(sales, t=1)

double_pred = double_pred_list(double_alpha, sales, t=1)

print('double best alpha: ', double_alpha)

print('pred value:' ,str(double_pred[-1]))

double_title = str('double pred with MSE: ' + str(double_mse))

plot_lines(sales, double_pred, double_title)
triple_alpha, triple_mse = triple_best_alpha(sales, t=1)

triple_pred = triple_pred_list(triple_alpha, sales, t=1)

print('triple best alpha: ', triple_alpha)

print('pred value:' ,str(triple_pred[-1]))

triple_title = str('triple pred with MSE: ' + str(triple_mse))

plot_lines(sales, triple_pred, triple_title)
triple_alpha_list = [0.1,0.3,0.5,0.7,0.9]

for triple_alpha in triple_alpha_list:

    triple_pred = triple_pred_list(triple_alpha, sales, t=1)

    triple_mse = cal_mse(sales, triple_pred)

    print('triple best alpha: ', triple_alpha)

    print('pred value:' ,str(triple_pred[-1]))

    triple_title = str('triple pred with MSE: ' + str(triple_mse))

    plot_lines(sales, triple_pred, triple_title)