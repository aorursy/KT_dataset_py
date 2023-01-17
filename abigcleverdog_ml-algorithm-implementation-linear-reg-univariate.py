import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# create a testing data pair

x = 4 * np.random.rand(20, 1)

y = x + 2 * np.random.rand(20, 1)

sns.scatterplot(list(x), list(y))
# %%timeit

# iterative implementataion



h = lambda x: theta_0 + theta_1 * x # set hypothesis

theta_0 = theta_1 = 0 # initiate the parameters

lr = 0.05 # learning rate

epochs = 20

costs = []

paras = []



def cal_sum(h, x, y):

    sum0 = sum1 = 0

    for i in range(len(x)):

        sum0 += (h(x[i]) - y[i])

        sum1 += (h(x[i]) - y[i]) * x[i]

    return sum0, sum1



def cal_cost(h, x, y):

    j = 0

    for i in range(len(x)):

        j += (h(x[i]) - y[i]) ** 2

    return j / (2 * len(x))



for i in range(epochs):

    sum0, sum1 = cal_sum(h, x[:,0], y[:,0])

    theta_0 -= lr / len(x) * sum0

    theta_1 -= lr / len(x) * sum1

    paras.append((theta_0, theta_1))

    costs.append(cal_cost(h, x[:,0], y[:,0]))

# plt.scatter(x = range(len(costs)), y = costs)

plt.plot(costs, 'go-');
plt.plot(x, y, 'bo')

def plot_line(t0, t1, x):

    y = lambda x: t0 + t1*x

    x_values = [i for i in range(int(min(x))-1, int(max(x))+2)]

    y_values = [y(x) for x in x_values]

    color = list(np.random.random(size=3))

    plt.plot(x_values, y_values, c = color)

for t0, t1 in paras:

    plot_line(t0, t1, x)
X = np.concatenate((np.ones((len(x),1)),x), axis = 1) # adding constant term to each train data
# %%timeit



# implementation with linear algebra

# for quick review of linear algebra relavent to this part: https://www.holehouse.org/mlclass/03_Linear_algebra_review.html; https://www.youtube.com/watch?v=Dft1cqjwlXE&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=13&t=0s 

def gradientDescent(X, y, theta, alpha, num_iters):

    """

       Performs gradient descent to learn theta

    """

    m = y.size  # number of training examples

    for i in range(num_iters):

        y_hat = np.dot(X, theta)

        theta = theta - alpha * (1.0/m) * np.dot(X.T, y_hat-y)

    return theta



theta = np.zeros((2,1))

epochs = 20 # run 20 single step gd to get cost functions and parameters to compare with the non-linear-algebra approach

gd_costs = []

gd_paras = []



for i in range(epochs):

    theta = gradientDescent(X, y, theta, 0.05, 1)

    gd_paras.append(theta)

    gd_costs.append(cal_cost(lambda x: theta[0] + theta[1]*x, x[:, 0], y[:, 0]))

plt.plot(gd_costs, 'go-')

plt.plot(costs, 'r+');
plt.plot(x, y, 'bo')



for t0, t1 in gd_paras:

    plot_line(t0, t1, x)
fig = plt.figure(figsize=(18,9))

plt.subplots_adjust(hspace=.5)



plt.subplot2grid((1,2), (0,0))

plt.plot(x, y, 'bo')



for t0, t1 in paras:

    plot_line(t0, t1, x)

plt.title('Iterative approach')



plt.subplot2grid((1,2), (0,1))

plt.plot(x, y, 'bo')



for t0, t1 in gd_paras:

    plot_line(t0, t1, x)

plt.title('Linear Algebra approach')



plt.show();
X_T = np.transpose(X)

inverse = np.linalg.inv(np.dot(X_T,X))

theta = np.dot(np.dot(inverse, X_T), y)

print(theta)

print(cal_cost(lambda x: theta[0] + theta[1]*x, x[:, 0], y[:, 0]))
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

df.head()
print(df.shape)

print(df.country.unique(), df.country.nunique())
df.rename(columns={'suicides/100k pop': 'suicides_per_100k_pop',

                  ' gdp_for_year ($) ': 'gdp_for_year',

                  'gdp_per_capita ($)': 'gdp_per_capita'}, inplace=True)

df.head()
country_rate = df.groupby('country').suicides_per_100k_pop.mean().reset_index()
country_gdp_cap = df.groupby('country').gdp_per_capita.mean().reset_index()

new_df = pd.merge(country_rate, country_gdp_cap, on='country')

new_df
sns.scatterplot(data = new_df, x = 'gdp_per_capita', y = 'suicides_per_100k_pop');
sns.lmplot(data = new_df, x = 'gdp_per_capita', y = 'suicides_per_100k_pop');
x = new_df.gdp_per_capita/10000 # x turned to be to big that will send y values out of range for python to handle

y = new_df.suicides_per_100k_pop
# iterative implementation



h = lambda x: theta_0 + theta_1 * x # set hypothesis

theta_0 = theta_1 = 0 # initiate the parameters

lr = 0.05 # learning rate

epochs = 200

costs = []

paras = []



def cal_sum(h, x, y):

    sum0 = sum1 = 0

    for i in range(len(x)):

        sum0 += (h(x[i]) - y[i])

        sum1 += (h(x[i]) - y[i]) * x[i]

    return sum0, sum1



def cal_cost(h, x, y):

    j = 0

    for i in range(len(x)):

        j += (h(x[i]) - y[i]) ** 2

    return j / (2 * len(x))



for i in range(epochs):

    sum0, sum1 = cal_sum(h, x, y)

    theta_0 -= lr / len(x) * sum0

    theta_1 -= lr / len(x) * sum1

    paras.append((theta_0, theta_1))

    costs.append(cal_cost(h, x, y))
plt.plot(costs, 'go-');
plt.plot(x, y, 'bo')



for t0, t1 in paras:

    plot_line(t0, t1, x)
X = np.concatenate((np.ones((len(x),1)),pd.DataFrame(x)), axis = 1) # adding constant term to each train data
# implementation with linear algebra

# for quick review of linear algebra relavent to this part: https://www.holehouse.org/mlclass/03_Linear_algebra_review.html; https://www.youtube.com/watch?v=Dft1cqjwlXE&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=13&t=0s 

def gradientDescent(X, y, theta, alpha, num_iters):

    """

       Performs gradient descent to learn theta

    """

    m = y.size  # number of training examples

    for i in range(num_iters):

        y_hat = np.dot(X, theta)

        theta = theta - alpha * (1.0/m) * np.dot(X.T, y_hat-pd.DataFrame(y))

    return theta



theta = np.zeros((2,1))

epochs = 200 # run 20 single step gd to get cost functions and parameters to compare with the non-linear-algebra approach

gd_costs = []

gd_paras = []



for i in range(epochs):

    theta = gradientDescent(X, y, theta, 0.05, 1)

    gd_paras.append(theta)

    gd_costs.append(cal_cost(lambda x: theta[0] + theta[1]*x, x, y))
plt.plot(gd_costs, 'go-')

plt.plot(costs, 'r+');
plt.plot(x, y, 'bo')



for t0, t1 in gd_paras:

    plot_line(t0, t1, x)
fig = plt.figure(figsize=(12,6))

plt.subplots_adjust(hspace=.5)



plt.subplot2grid((1,2), (0,0))

plt.plot(x, y, 'bo')



for t0, t1 in paras:

    plot_line(t0, t1, x)

plt.title('Iterative approach')



plt.subplot2grid((1,2), (0,1))

plt.plot(x, y, 'bo')



for t0, t1 in gd_paras:

    plot_line(t0, t1, x)

plt.title('Linear Algebra approach')



plt.show();
x_transpose = np.transpose(X)   #calculating transpose

x_transpose_dot_x = x_transpose.dot(X)  # calculating dot product

temp_1 = np.linalg.inv(x_transpose_dot_x) #calculating inverse



temp_2 = x_transpose.dot(y)  



para = temp_1.dot(temp_2)

para
plt.plot(x, y, 'bo')

plot_line(para[0], para[1], x)