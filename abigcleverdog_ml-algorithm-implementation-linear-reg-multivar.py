import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# create a testing data pair

x = pd.DataFrame([[1,4],[2,5],[3,6],[3,6]])

y = pd.DataFrame([8,9,12,12])
# iterative implementation. This is more intuitive as we are simply mapping the gradient descent step by step



def cal_sum(h, x, y):

    sum0 = sum1 = sum2 = 0

    for i in range(len(y)):

        sum0 += (h(x.iloc[i,:]) - y[i])

        sum1 += (h(x.iloc[i,:]) - y[i]) * x.iloc[i,0]

        sum2 += (h(x.iloc[i,:]) - y[i]) * x.iloc[i,1]

    return sum0, sum1, sum2



def cal_cost(h, x, y):

    j = 0

    for i in range(len(x)):

        j += (h(x.iloc[i,:]) - y[i]) ** 2

    return j / (2 * len(y))



def lrdg_iterative(theta, x, y, lr, epochs):

    theta_0, theta_1, theta_2 = theta

    h = lambda x: theta_0 + theta_1 * x[0] + theta_2 * x[1] # hypothesis

    paras, costs = [], []

    for i in range(epochs):

        sum0, sum1, sum2 = cal_sum(h, x, y[0])

        theta_0 -= lr / len(y) * sum0

        theta_1 -= lr / len(y) * sum1

        theta_2 -= lr / len(y) * sum2

        paras.append((theta_0, theta_1, theta_2))

        costs.append(cal_cost(h, x, y[0]))

    return paras, costs
theta = [0,0,0]

lr = 0.01 # learning rate

epochs = 20

it_paras, it_costs = lrdg_iterative(theta, x, y, lr, epochs)

plt.plot(it_costs, 'go-', label='cost')

plt.legend()

plt.ylabel('Costs')

plt.xlabel('Iterations')

plt.title('Cost Function');
theta = [0,0,0]

lr = 0.15 # learning rate too big, and cost function will not converge

epochs = 20

bi_paras, bi_costs = lrdg_iterative(theta, x, y, lr, epochs)

plt.plot(bi_costs, 'go-', label='cost')

plt.legend()

plt.ylabel('Costs')

plt.xlabel('Iterations')

plt.title('Cost Function');
%timeit _, _ = lrdg_iterative(theta, x, y, lr, epochs)
# implementation with linear algebra

# for quick review of linear algebra relavent to this part: https://www.holehouse.org/mlclass/03_Linear_algebra_review.html; https://www.youtube.com/watch?v=Dft1cqjwlXE&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=13&t=0s 

def gradientDescent(X, y, theta, alpha, num_iters):

    """

       Performs gradient descent to learn theta

    """

    m = y[0].size  # number of training examples

    paras, costs = [], []

    for i in range(num_iters):

        y_hat = np.dot(X, theta)

        theta = theta - alpha * (1.0/m) * np.dot(X.T, y_hat-y)

        paras.append(theta)

        costs.append(cal_cost(lambda x: theta[0] + theta[1]*x[0] + theta[2]*x[1], x, y[0]))

    return theta, paras, costs
X = pd.concat([pd.DataFrame(np.ones((len(y),1))),x], axis = 1) # adding constant term to each train data

theta = np.zeros((3,1))

epochs = 20 # run 20 single step gd to get cost functions and parameters to compare with the non-linear-algebra approach

lr = 0.01

p, la_paras, la_costs = gradientDescent(X, y, theta, lr, epochs)

plt.plot(la_costs, 'go-', label='Lin-Alg')

plt.plot(it_costs, 'r+', label='Iterative')

plt.legend()

plt.ylabel('Costs')

plt.xlabel('Iterations')

plt.title('Cost Function');
%timeit _, _, _ = gradientDescent(X, y, theta, lr, epochs)
def normalEquation(X, y):

    m = len(y)

    theta = []

    

    # Calculating theta

    theta = np.linalg.pinv(X.T.dot(X))  ### Please note using np.linalg.inv will sometime yield wrong outcomes

    theta = theta.dot(X.T)

    theta = theta.dot(y)



    return theta
ne_para = normalEquation(X,y)

ne_cost = cal_cost(lambda x: ne_para[0] + ne_para[1]*x[0] + ne_para[2]*x[1], x, y[0])
%timeit _ = normalEquation(X,y)
compare_df = pd.DataFrame([np.append(ne_cost, ne_para), 

                           np.append(it_costs[-1], it_paras[-1]), 

                           np.append(la_costs[-1], la_paras[-1])],

                         index = ['Normal Equation', 'Iterative 20 It', 'Linear Algebra 20 It'],

                         columns = ['Cost', 'theta_0', 'theta_1', 'theta_2'])

compare_df['timeit (ms)'] = pd.Series([1.22, 95.4, 67.7], index = compare_df.index)

compare_df
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

df.head()
df.rename(columns={'suicides/100k pop': 'suicides_per_100k_pop',

                  ' gdp_for_year ($) ': 'gdp_for_year',

                  'gdp_per_capita ($)': 'gdp_per_capita'}, inplace=True)

df.head()
country_rate = df.groupby('country').suicides_per_100k_pop.mean().reset_index()

country_gdp_cap = df.groupby('country').gdp_per_capita.mean().reset_index()

country_pop = df.groupby('country').population.mean().reset_index()

new_df = pd.merge(country_rate, country_gdp_cap, on='country')

new_df = pd.merge(new_df, country_pop, on='country')

new_df
new_df['gdp_per_capita'] /= 100000   # too large x values send the y values out of normal range and cause calculation errors

new_df['population'] /= 100000000

new_df
x = new_df[['gdp_per_capita', 'population']]

y = pd.DataFrame(list(new_df.suicides_per_100k_pop))
theta = [0,0,0]

lr = 0.08 # learning rate

epochs = 20

it_paras, it_costs = lrdg_iterative(theta, x, y, lr, epochs)

plt.plot(it_costs, 'go-', label='Iterative')

plt.legend()

plt.ylabel('Costs')

plt.xlabel('Iterations')

plt.title('Cost Function');
X = pd.concat([pd.DataFrame(np.ones((len(y[0]),1))),x], axis = 1) # adding constant term to each train data

theta = np.zeros((3,1))

epochs = 20 # run 20 single step gd to get cost functions and parameters to compare with the non-linear-algebra approach

lr = 0.08

p, la_paras, la_costs = gradientDescent(X, y, theta, lr, epochs)

plt.plot(la_costs, 'go-', label='Lin-Alg')

plt.plot(it_costs, 'r+', label='Iterative')

plt.legend()

plt.ylabel('Costs')

plt.xlabel('Iterations')

plt.title('Cost Function');
costs_vs_lr = []

for i in range(10):

    lr = 0.01 + i * 0.02

    theta = np.zeros((3,1))

    epochs = 20 # run 20 single step gd to get cost functions and parameters to compare with the non-linear-algebra approach

    p, _paras, _costs = gradientDescent(X, y, theta, lr, epochs)

    costs_vs_lr.append(_costs)



for i, costs in enumerate(costs_vs_lr):    

    color = list(np.random.random(size=3))

    plt.plot(range(20), costs, c = color, label='lr={:.2f}'.format(0.01 + i * 0.02))

plt.ylabel('cost')

plt.legend()

plt.ylabel('Costs')

plt.xlabel('Iterations')

plt.title('Cost Function');
ne_para = normalEquation(X,y)

ne_cost = cal_cost(lambda x: ne_para[0] + ne_para[1]*x[0] + ne_para[2]*x[1], x, y[0])

print(ne_cost, it_costs[-1], la_costs[-1])

print(ne_para, it_paras[-1], la_paras[-1])
from sklearn.linear_model import LinearRegression

linReg = LinearRegression()

reg = linReg.fit(x, y)

sk_cost = cal_cost(lambda x: reg.intercept_ + reg.coef_[0][0] *x[0] + reg.coef_[0][1]*x[1], x, y[0])

print(reg.coef_, ne_para)

print(sk_cost, ne_cost)
%timeit _, _ = lrdg_iterative(theta, x, y, lr, epochs)
%timeit _, _, _ = gradientDescent(X, y, theta, lr, epochs)
%timeit _ = normalEquation(X,y)
%timeit _ = linReg.fit(x, y)
compare_df = pd.DataFrame([np.append(it_costs[-1], it_paras[-1]), 

                           np.append(la_costs[-1], la_paras[-1]), 

                           np.append(ne_cost, ne_para), 

                           np.append([sk_cost, reg.intercept_], reg.coef_[0])],

                         index = ['Iterative 20 It', 'Linear Algebra 20 It', 'Normal Equation', 'SKLearn'],

                         columns = ['Cost', 'theta_0', 'theta_1', 'theta_2'])

compare_df['timeit (ms)'] = pd.Series([2460, 647, 1.06, 2.27], index = compare_df.index)

compare_df