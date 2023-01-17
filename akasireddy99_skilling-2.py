import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
np.linspace(10, 20, 8)
np.linspace(1, 147, 10)
print(plt.rcParams.get('figure.figsize'))

plt.rcParams['figure.figsize'] = [12, 8]

print(plt.rcParams.get('figure.figsize'))
x = np.linspace(-10, 10, 10)

y = x**3
plt.xlabel('X')

plt.ylabel('Y')

plt.title('Cube function')

plt.plot(x, y, 'bd')

plt.subplot()
x = np.linspace(-10, 10, 20)

y = x**3



plt.subplot(2, 2, 1)

plt.plot(x, y, 'b*-')

plt.subplot(2, 2, 2)

plt.plot(x, y, 'y--')

plt.plot(2, 2, 3, 'b*-')

plt.plot(2, 2, 4, 'y--')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
arr = np.array([

    [1, 2, 3],

    [4, 5, 6],

    [7, 8, 9]

])



print(arr)

print(type(arr))

print(arr.shape)

print(np.flip(arr))
arr = np.arange(1, 16, 1)

print(arr)
arr_one = np.arange(1, 5, 1)

arr_two = np.arange(3, 7, 1)

print('Array 1\n', arr_one)

print('Array 2\n', arr_two)



for item in arr_one:

    if item in arr_two:

        print('Common element -> ' + str(item))

        print('Index in Array 1 -> ' + str(np.where(arr_one == item)[0]))

        print('Index in Array 2 -> ' + str(np.where(arr_two == item)[0]))
print(np.random.randint(10, 100, 10))
print(np.random.randint(0, 100, (4, 4)))
div = np.linspace(1, 100100111, endpoint=True)

div
print('Default size ->', plt.rcParams['figure.figsize'])

plt.rcParams['figure.figsize'] = [16, 9]

print('New size ->', plt.rcParams['figure.figsize'])
x = np.linspace(-100, 100, 100)

y = x**3



plt.plot(x, y)

plt.xlabel('x')

plt.ylabel('y')

plt.title('Cubic function')

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle('Sub plots')

ax1.plot(x, y)

ax2.plot(x, -y)
x = np.linspace(0, 100, 100)

y_one = np.sqrt(x)

y_two = np.sin(x)



plt.plot(x, y_one, '-b', label='Sqrt')

plt.plot(x, y_two, '-r', label='Sine')

plt.legend(loc='best')
y = [2.56422, 3.77284,3.52623, 3.51468, 3.02199]

z = [0.15, 0.3, 0.45, 0.6, 0.75]

n = ['dheeraj', 'vamsi', 'satish', 'basha', 'kartik']



plt.scatter(y, z, data=n)
import seaborn as sns

y = np.array([39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 6445, 57189])



sns.distplot(y, bins=20)
sns.boxplot(y=y)
arr = np.random.randint(0, 100, (10, 5))

columns = ['A', 'B', 'C', 'D', 'E']



df = pd.DataFrame(data=arr, columns=columns)

df.head()
data = [

    [4.1, 57081],

    [4.5, 61111],

    [4.9, 67938],

    [5.1, 66029],

    [5.3, 83088],

    [5.9, 81363]

]

data = pd.DataFrame(data=data, columns=['x', 'y'])



plt.scatter(x=data['x'], y=data['y'])
dishes = ['cup_cake', 'bread', 'cookie']

values = np.random.random(3)



plt.pie(x=values, labels=dishes)
A = [165349.2, 162597.7, 153441.51, 144372.41, 14207.34, 131876.9, 134615.46, 130298.13, 120542.52, 123334.88]

B = [136897.8, 151377.59, 101145.55, 118671.85, 91391.77, 99814.71, 145530.06, 148718.95, 108679.17]

C = [471784.1, 443898.53, 407934.54, 383199.62, 366168.42, 362861.36, 127716.82, 323876.68, 311613.29, 304981.62]

E = [192261.83, 191792.06, 191050.39, 182901.99, 166187.94, 156991.12, 156122.51, 155752.6, 15211.77, 149759.96]



a_mean = np.mean(A)

b_mean = np.mean(B)

c_mean = np.mean(C)

e_mean = np.mean(E)



data = [[a_mean], [b_mean], [c_mean], [e_mean]]

sns.heatmap(data)