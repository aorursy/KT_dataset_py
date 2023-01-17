import numpy as np

import matplotlib.pyplot as plt

figure, axes = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 10))

axes[0][0].plot(np.linspace(-6, 6, 500), 1/(1 + np.exp(-np.linspace(-6, 6, 500))), linestyle='dotted', color = 'red', 

          marker='1', markersize=10, label=r'$y = \dfrac{1}{1 + e^{-x}}$')

axes[0][0].set(xlabel='$x$', ylabel='$y$',title=r'$y = \dfrac{1}{1 + e^{-x}}$')

axes[1][0].plot(np.linspace(-2.5, 2.5, 500), np.linspace(-2.5, 2.5, 500)/((1 + (np.linspace(-2.5, 2.5, 500)) ** 2) ** (1/2)), 

             linestyle='dotted', color = 'blue', marker='2', markersize=10, label = r'$y=\dfrac{x}{\sqrt{1+x^2}}$')

axes[1][0].set(xlabel='$x$', ylabel='$y$',title=r'$y=\dfrac{x}{\sqrt{1+x^2}}$');

axes[0][0].legend(loc=4)

axes[1][0].legend(loc=4)

axes[1][1].plot(np.linspace(-6, 6, 500), 1/(1 + np.exp(-np.linspace(-6, 6, 500))), linestyle='dotted', color = 'red', 

          marker='1', markersize=10, label=r'$y = \dfrac{1}{1 + e^{-x}}$')

axes[1][1].set(xlabel='$x$', ylabel='$y$',title=r'$y = \dfrac{1}{1 + e^{-x}}$')

axes[0][1].plot(np.linspace(-2.5, 2.5, 500), np.linspace(-2.5, 2.5, 500)/((1 + (np.linspace(-2.5, 2.5, 500)) ** 2) ** (1/2)), 

             linestyle='dotted', color = 'blue', marker='2', markersize=10, label = r'$y=\dfrac{x}{\sqrt{1+x^2}}$')

axes[0][1].set(xlabel='$x$', ylabel='$y$',title=r'$y=\dfrac{x}{\sqrt{1+x^2}}$');

axes[1][1].legend(loc=4)

axes[0][1].legend(loc=4);

figure, axes = plt.subplots(figsize=(10, 10))

axes.plot(np.linspace(-2*np.pi, 0, 500), np.sin(-2*np.linspace(-2*np.pi, 0, 500)))

axes.plot(np.linspace(0, 5, 500), np.linspace(0, 5, 500)*np.linspace(0, 5, 500)-np.linspace(0, 5, 500))

figure, axes = plt.subplots(figsize=(10, 10))

x0 = 0

y = [0]

for i in range(1, 101):

    x0 = x0 + np.random.normal(0, 5)

    y.append(x0)

axes.plot(y);
figure, axes = plt.subplots(figsize=(10, 10))

for a in [1, 0.8, 0.6, 0.4, 0.2]:

    x0 = 0

    y = [0]

    for i in range(1, 101):

        x0 = a*x0 + np.random.normal(0, 5)

        y.append(x0)

    axes.plot(y, label = a)

axes.legend();
import os

os.listdir('../input')
data = np.loadtxt(r'../input/cardiovascular-disease-dataset/cardio_train.csv', delimiter = ';', skiprows = 1)

data.dtype
fig, axes = plt.subplots(figsize=(10, 10))

axes.hist(data[:, 1]/365.25)

fig, ax = plt.subplots(figsize=(20, 7))

plt.boxplot(data[:, 4], vert=False);
height = data[:, 3]

weight = data[:, 4] # изначально возраст был в днях



fig, ax = plt.subplots(figsize=(7, 7))

plt.scatter(weight, height)

plt.xlabel('weight, kg')

plt.ylabel('height, cm')

plt.title('Weight vs. Height');
#6. Как отличается средний вес здоровых и больных пациентов? Визуализируйте ответ в виде **bar plot**.



#7. Каково соотношение больных и здоровых в трёх различных группах по уровню холестерина? Визуализируйте ответ в виде **bar plot**.

wez = data[data[:, -1] == 0, 4] 

web = data[data[:, -1] == 1, 4] 





fig, ax = plt.subplots()



plt.bar([1, 2], height=[wez.mean(), web.mean()], color=['red', 'blue'])

ax.set_xticks([1, 2])

ax.set_xticklabels(['wez', 'web']);



coz = data[data[:, -1] == 0, 8] 

cob = data[data[:, -1] == 1, 8] 

coz1 = coz[coz == 1]

coz2 = coz[coz == 2]

coz3 = coz[coz == 3]

cob1 = cob[cob == 1]

cob2 = cob[cob == 2]

cob3 = cob[cob == 3]



fig, ax = plt.subplots()



plt.bar([1, 2, 3, 4, 5, 6], height=[len(coz1), len(cob1), len(coz2), len(cob2), len(coz3), len(cob3)],

       color=['red','green','red','green','red','green'])

ax.set_xticks([1, 2, 3, 4, 5, 6])

ax.set_xticklabels(['coz1', 'cob1', 'coz2', 'cob2', 'coz3', 'cob3']);