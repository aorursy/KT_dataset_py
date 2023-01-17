import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# create sample data

x1 = np.random.rand(10, 1)

x2 = 1 + np.random.rand(10, 1)

x = np.append(x1,x2)

y1 = [0] * 10

y2 = [1] * 10

y = y1 + y2



plt.plot(x, y, 'bo')
theta_0 = theta_1 = 0

g = lambda x: theta_0 + theta_1 * x

h = lambda x: 1 / (1 + np.exp(-g(x)))



lr = 0.05 # learning rate

epochs = 2000

costs = []

paras = []



def cal_cost(h, x, y):

    j = 0

    for i in range(len(x)):

        j += y[i] * np.log(h(x[i])) + (1 - y[i]) * np.log(1 - h(x[i]))

    return -j / len(x)



def cal_sum(h, x, y):

    sum_0 = sum_1 = 0

    for i in range(len(x)):

        sum_0 += (h(x[i]) - y[i])

        sum_1 += (h(x[i]) - y[i]) * x[i]

    return sum_0 / len(x), sum_1 / len(x)



for i in range(epochs):

    sum_0, sum_1 = cal_sum(h, x, y)

    theta_0 -= lr * sum_0

    theta_1 -= lr * sum_1

    cost = cal_cost(h, x, y)

    costs.append(cost)

    paras.append([theta_0, theta_1])

    

print(costs[-10:])

print(paras[-10:])
plt.plot(costs)
boundaries = [-i[0]/i[1] for i in paras]

plt.plot(boundaries)
# create sample data

x_base = np.arange(0,2.0, 0.1)

x1 = x_base * x_base * 4

x2 = np.exp(x_base) * 2

y1 = [0] * 10

y2 = [1] * 10

y = y1 + y2

plt.scatter(x1, x2);
df = pd.DataFrame(list(zip(x1, x2, y)), 

               columns =['x1', 'x2', 'y']) 

sns.scatterplot(data = df, x = 'x1', y = 'x2', hue = 'y')
X = df.iloc[:,0:2]

y = df.iloc[:,2]
# iterative implementation

theta = [0, 0, 0]

g = lambda x: theta[0] + theta[1] * x[0] + theta[2] * x[1]

h = lambda x: 1 / (1 + np.exp(-g(x)))



lr = 0.5 # learning rate

epochs = 500

costs = []

paras = []



def cal_cost(h, x, y):

    j = 0

    for i in range(len(x)):

        j += y[i] * np.log(h(x.iloc[i])) + (1 - y[i]) * np.log(1 - h(x.iloc[i]))

    return -j / len(x)



def cal_sum(h, x, y):

    sum_0 = sum_1 = sum_2 = 0

    for i in range(len(x)):

        sum_0 += (h(x.iloc[i]) - y[i])

        sum_1 += (h(x.iloc[i]) - y[i]) * x.iloc[i][0]

        sum_2 += (h(x.iloc[i]) - y[i]) * x.iloc[i][1]

    return sum_0 / len(x), sum_1 / len(x), sum_2 / len(x)



def log_reg(h, x, y, theta, lr, epochs):

    for i in range(epochs):

        sum_0, sum_1, sum_2 = cal_sum(h, x, y)

        theta[0] -= lr * sum_0

        theta[1] -= lr * sum_1

        theta[2] -= lr * sum_2

        cost = cal_cost(h, x, y)

        costs.append(cost)

        paras.append(theta)



    print(costs[-5:])

    print(paras[-5:])

    

log_reg(h, X, y, theta, lr, epochs)
def plot_line(theta, x):

    y = lambda x: -(theta[0] + theta[1] * x)/theta[2]

    x_values = [i for i in range(int(min(x))-1, int(max(x))+2)]

    y_values = [y(x) for x in x_values]

    color = list(np.random.random(size=3))

    plt.plot(x_values, y_values, c = color)

    

    

sns.scatterplot(data = df, x = 'x1', y = 'x2', hue = 'y')

for i,t in enumerate(paras):

    if i%100 == 0: 

        plot_line(t, list(df.iloc[:, 0]))
# implementation with linear algebra

X = np.concatenate((np.ones((X.shape[0], 1)) , X), axis = 1)

theta = np.zeros(X.shape[1])



lr = 0.5 # learning rate

epochs = 500

costs = []

paras = []



def cal_cost(h, x, y):

    return (-y * np.log(h) - (1-y) * np.log(1-h)).mean()    



def log_reg(h, x, y, theta, lr, epochs):

    for i in range(epochs):

        z = np.dot(X, theta)

        h = 1/(1 + np.exp(-z))

        gradient = np.dot(X.T, (h - y)) / y.size

        theta -= lr * gradient

        cost = cal_cost(h, x, y)

        costs.append(cost)

        paras.append(theta)



    print(costs[-5:])

    print(paras[-5:])

    

log_reg(h, X, y, theta, lr, epochs)
sns.scatterplot(data = df, x = 'x1', y = 'x2', hue = 'y')

for i,t in enumerate(paras):

    if i%100 == 0: 

        plot_line(t, list(df.iloc[:, 0]))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 



X = df.iloc[:,0:2]

y = df.iloc[:,2]

model = LogisticRegression()

model.fit(X, y)

predicted_classes = model.predict(X)

accuracy = accuracy_score(y, predicted_classes)

parameters = model.coef_

intercept = model.intercept_



print(accuracy)

print(intercept, parameters)
sns.scatterplot(data = df, x = 'x1', y = 'x2', hue = 'y')

plot_line([intercept, parameters[0][0], parameters[0][1]], list(df.iloc[:, 0]))
df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

df.head()
df.rename(columns={'gdp_per_capita ($)': 'gdp_per_capita'}, inplace = True)

df = df[['gdp_per_capita', 'suicides_no', 'sex']]

def t_c(df):

    if df['sex'] == 'female':

        return 0    

    else:

        return 1

df['sex'] = df.apply(t_c, axis=1)

df['gdp_per_capita'] /= df['gdp_per_capita'].max()*0.01

df['suicides_no'] /= df['suicides_no'].max()*0.01

df.head()
sns.scatterplot(data = df, y = 'suicides_no', x = 'gdp_per_capita', hue = 'sex')
X = df.iloc[:,0:2]

y = df.iloc[:,2]



X = np.concatenate((np.ones((X.shape[0], 1)) , X), axis = 1)

theta = np.zeros(X.shape[1])



lr = 0.15 # learning rate

epochs = 500

costs = []

paras = []



def cal_cost(h, x, y):

    return (-y * np.log(h) - (1-y) * np.log(1-h)).mean()    



def log_reg(h, x, y, theta, lr, epochs):

    for i in range(epochs):

        z = np.dot(X, theta)

        h = 1/(1 + np.exp(-z))

        gradient = np.dot(X.T, (h - y)) / y.size

        theta -= lr * gradient

        cost = cal_cost(h, x, y)

        costs.append(cost)

        paras.append(theta)



    print(costs[-5:])

    print(paras[-5:])

    

log_reg(h, X, y, theta, lr, epochs)
sns.scatterplot(data = df, y = 'suicides_no', x = 'gdp_per_capita', hue = 'sex')

for i,t in enumerate(paras):

    if i%100 == 0: 

        plot_line(t, list(df.iloc[:, 0]))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 



X = df.iloc[:,0:2]

model = LogisticRegression()

model.fit(X, y)

predicted_classes = model.predict(X)

accuracy = accuracy_score(y, predicted_classes)

parameters = model.coef_

intercept = model.intercept_



print(accuracy)

print(intercept, parameters)
sns.scatterplot(data = df, y = 'suicides_no', x = 'gdp_per_capita', hue = 'sex')

plot_line([intercept, parameters[0][0], parameters[0][1]], list(df.iloc[:, 0]))