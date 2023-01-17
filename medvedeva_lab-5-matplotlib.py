import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np



x = np.linspace(-6, 6, 100)

y = 1/(1+np.exp(-x))

plt.plot(x, y)





plt.show()
x = np.linspace(-6, 6, 100)

y = 1/(1+np.exp(-x))

plt.plot(x, y)



plt.xlabel('x')

plt.ylabel('y')

plt.title(r'График $y=\dfrac{1}{1+e^{-x}}$')



plt.show()
x = np.linspace(-6, 6, 10)

y = 1/(1+np.exp(-x))

plt.plot(x, y, linestyle='dashdot', linewidth=2, color='pink', marker='o', markersize=12, markerfacecolor='white')



plt.xlabel('x')

plt.ylabel('y')

plt.title(r'График $y=\dfrac{1}{1+e^{-x}}$')



plt.show()
x = np.linspace(-2.5, 2.5, 10)

y_1 = 1/(1+np.exp(-x))

y_2 = x/(1+x**2)**0.5

plt.plot(x, y_1, label='$y=\dfrac{1}{1+e^{-x}}$', linestyle='solid', linewidth=2, color='b', marker='o', markersize=12)

plt.plot(x, y_2, label='$y=\dfrac{x}{\sqrt{1+x^2}}$', linestyle='dotted', linewidth=2, color='c', marker='s', markersize=6, markerfacecolor='white')



plt.xlabel('x')

plt.ylabel('y')



plt.legend(loc=4)



plt.show()
fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True)



mpl.rcParams.update({'font.size':22, 'font.family':'Sans Serif', 'font.weight':'bold'})





x = np.linspace(-2.5, 2.5, 10)

y_1 = 1/(1+np.exp(-x))

y_2 = x/(1+x**2)**0.5



ax[0].plot(x, y_1, label='$y=\dfrac{1}{1+e^{-x}}$', linestyle='solid', linewidth=2, color='b', marker='o', markersize=12)

ax[1].plot(x, y_2, label='$y=\dfrac{x}{\sqrt{1+x^2}}$', linestyle='dotted', linewidth=2, color='c', marker='s', markersize=6, markerfacecolor='white')



ax[0].legend(loc=4)

ax[1].legend(loc=4)



ax[0].set(xlabel='x', ylabel='y')

ax[1].set(xlabel='x')



ax[0].set_title(r'График $y=\dfrac{1}{1+e^{-x}}$')

ax[1].set_title(r'График $y=\dfrac{x}{\sqrt{1+x^2}}$');



fig, ax = plt.subplots(2, 1, figsize=(18, 8), sharex=True)



mpl.rcParams.update({'font.size':18, 'font.family':'Sans Serif', 'font.weight':'medium', 'font.style':'italic'})





y_1 = 1/(1+np.exp(-x))

y_2 = x/(1+x**2)**0.5



ax[0].plot(x, y_1, label='$y=\dfrac{1}{1+e^{-x}}$', linestyle='solid', linewidth=2, color='b', marker='o', markersize=12)

ax[1].plot(x, y_2, label='$y=\dfrac{x}{\sqrt{1+x^2}}$', linestyle='dotted', linewidth=2, color='c', marker='s', markersize=6, markerfacecolor='white');



ax[0].legend(loc=2)

ax[1].legend(loc=2)



ax[0].set(ylabel='y')

ax[1].set(xlabel='x', ylabel='y')



ax[0].set_title(r'График $y=\dfrac{1}{1+e^{-x}}$')

ax[1].set_title(r'График $y=\dfrac{x}{\sqrt{1+x^2}}$');
x = np.linspace(-2*np.pi, 0, 10000)

plt.plot(x, np.sin(-2*x))

x = np.linspace(0, 5, 10000)

plt.plot(x, x*x-x)



plt.show()
L_y = [0]

y = 0

for i in range (1, 101):

    y = y + np.random.normal(0, 2)

    L_y.append(y)

plt.plot(L_y)

plt.show()

    


fig, ax = plt.subplots(figsize=(18, 8), sharey=True)

A = [1, 0.8, 0.6, 0.4, 0.2]

for a in A:

    L_y = [0]

    y = 0

    for i in range (1, 101):

        y = a*y + np.random.normal(0, 2)

        L_y.append(y)

    plt.plot(L_y, label=a)

plt.legend(loc=0)

plt.show()
import os

os.listdir('../input')
cardio_data = np.loadtxt('../input/cardiovascular-disease-dataset/cardio_train.csv', delimiter=';', skiprows=1)

print(cardio_data[:5, :])

print(cardio_data.shape)
cardio_data.dtype
age = cardio_data[:, 1] / 365



fig, ax = plt.subplots(figsize=(20, 7))

plt.plot(age);
fig, ax = plt.subplots(figsize=(20, 7))

plt.scatter(cardio_data[:, 0], age, marker='v', s = 100);
fig, ax = plt.subplots(figsize=(20, 7))



ax.hist(age);
weight = cardio_data[:, 4]



fig, ax = plt.subplots(figsize=(20, 7))

plt.plot(weight);
fig, ax = plt.subplots(figsize=(20, 7))

ax.hist(weight);
fig, ax = plt.subplots(figsize=(20, 7))

plt.boxplot(weight, vert=False);
height = cardio_data[:, 3]





fig, ax = plt.subplots(figsize=(14, 7))

plt.scatter(height, weight)

plt.xlabel('height, cm')

plt.ylabel('weight, kg')

plt.title('Height vs. Weight');
fig, ax = plt.subplots()



plt.bar([1, 2], height=[np.mean(cardio_data[cardio_data[:, -1] == 0, 4]), np.mean(cardio_data[cardio_data[:, -1] == 1, 4])], color=['red', 'blue'])

ax.set_xticks([1, 2])

ax.set_xticklabels(['good', 'no_good']);
good = (cardio_data[(cardio_data[:, -1] == 0) & (cardio_data[:, 7] == 1), :].shape[0], cardio_data[(cardio_data[:, -1] == 0) & (cardio_data[:, 7] == 2), :].shape[0], 

        cardio_data[(cardio_data[:, -1] == 0) & (cardio_data[:, 7] == 3), :].shape[0])

no_good = (cardio_data[(cardio_data[:, -1] == 1) & (cardio_data[:, 7] == 1), :].shape[0], cardio_data[(cardio_data[:, -1] == 1) & (cardio_data[:, 7] == 2), :].shape[0], 

           cardio_data[(cardio_data[:, -1] == 1) & (cardio_data[:, 7] == 3), :].shape[0])



ind = np.arange(len(good))

width = 0.35



fig, ax = plt.subplots()

rects1 = ax.bar(ind - width/2, good, width,

                label='good')

rects2 = ax.bar(ind + width/2, no_good, width,

                label='no_good')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Numbers_of_holestirin')

ax.set_xlabel('Groups_of_holestirin')

ax.set_title('Health by group holestirin')

ax.set_xticks(ind)

ax.set_xticklabels(('H1', 'H2', 'H3'))

ax.legend();
