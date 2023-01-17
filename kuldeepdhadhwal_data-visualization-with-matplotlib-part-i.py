# import the required modules

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab
N_points = 100000

n_bins = 20



# Generate a normal distribution, center at x=0 and y=5

x = np.random.randn(N_points)

y = .4 * x + np.random.randn(100000) + 5



fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)



# We can set the number of bins with the `bins` kwarg

axs[0].hist(x, facecolor='yellow',edgecolor='green', bins=n_bins)

axs[1].hist(y, facecolor='yellow',edgecolor='green', bins=n_bins)
plt.hist(x, facecolor='peru',edgecolor='blue', bins=n_bins, cumulative=True)

plt.show()
plt.hist(x, facecolor='peru',edgecolor='blue', bins=n_bins, range=(-2,2))

plt.show()
plt.hist(x, facecolor='peru',edgecolor='blue', bins=n_bins)

plt.hist(y, facecolor='yellow',edgecolor='green', bins=n_bins)

plt.show()
# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'

sizes = [15, 30, 45, 10]

explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
from matplotlib.ticker import FuncFormatter



data = {'Barton LLC': 109438.50,

        'Frami, Hills and Schmidt': 103569.59,

        'Fritsch, Russel and Anderson': 112214.71,

        'Jerde-Hilpert': 112591.43,

        'Keeling LLC': 100934.30,

        'Koepp Ltd': 103660.54,

        'Kulas Inc': 137351.96,

        'Trantow-Barrows': 123381.38,

        'White-Trantow': 135841.99,

        'Will LLC': 104437.60}

group_data = list(data.values())

group_names = list(data.keys())

group_mean = np.mean(group_data)
fig, ax = plt.subplots()

ax.barh(group_names, group_data)
plt.bar([1,3,5,7,9],[5,2,7,8,2], label="Example one")



plt.bar([2,4,6,8,10],[8,6,2,5,6], label="Example two", color='g')

plt.legend()

plt.xlabel('bar number')

plt.ylabel('bar height')



plt.title('Epic Graph\nAnother Line! Whoa')



plt.show()
data1 = [23,85, 72, 43, 52]

data2 = [42, 35, 21, 16, 9]

plt.bar(range(len(data1)), data1)

plt.bar(range(len(data2)), data2, bottom=data1)

plt.show()
np.random.seed(10)

collectn_1 = np.random.normal(100, 10, 200)

collectn_2 = np.random.normal(80, 30, 200)

collectn_3 = np.random.normal(90, 20, 200)

collectn_4 = np.random.normal(70, 25, 200)



## combine these different collections into a list

data_to_plot = [collectn_1, collectn_2, collectn_3, collectn_4]



# Create a figure instance

fig = plt.figure()



# Create an axes instance

ax = fig.add_axes([0,0,1,1])



# Create the boxplot

bp = ax.violinplot(data_to_plot)

plt.show()