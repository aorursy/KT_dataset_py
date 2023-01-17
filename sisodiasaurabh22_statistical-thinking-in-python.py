# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
iris_data = pd.read_csv('../input/iris.data.csv')
iris_data.head()
iris_data.columns
sns.set()
plt.hist(iris_data['5.1'])
plt.xlabel('petal length (cm)')
plt.ylabel('count')
plt.show()
n_data = len(iris_data['5.1'])
n_bins = np.sqrt(n_data)
n_bins = int(n_bins)
plt.hist(iris_data['5.1'], bins=n_bins)
plt.xlabel('petal length (cm)')
plt.ylabel('count')
plt.show()
_ = sns.swarmplot(x='Iris-setosa',y='5.1',data=iris_data)
_ = plt.xlabel('')
_ = plt.ylabel('')
plt.show()

x = np.sort(iris_data['5.1'])
y = np.arange(1,len(x)+1)/len(x)
_=plt.plot(x,y,marker='.',linestyle='none')
_=plt.xlabel('')
_=plt.ylabel('ECDF')
plt.margins(0.2)
plt.show()
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y
percentiles = np.array([2.5,25,50,75,97.5])
ptiles_vers = np.percentile(iris_data['5.1'],percentiles)
print(ptiles_vers)
# Plot the ECDF
_ = plt.plot(x, y, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')
# Show the plot
plt.show()
# Create box plot with Seaborn's default settings

sns.boxplot(x='Iris-setosa',y='5.1',data = iris_data)
# Label the axes

plt.xlabel('species')
plt.ylabel('petal length (cm)')

# Show the plot
plt.show()

np.random.seed(42)
# draw 6 random numbers
random_numbers = np.random.random(size=6)

def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0


    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success

# Seed random number generator

np.random.seed(42)
# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100,.05)


# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()

np.random.binomial(4,.5)
np.random.binomial(4,.5,size=10)

np.random.poisson(6,size=100)

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10

samples_std1 = np.random.normal(20,1,size=100000)
samples_std3 = np.random.normal(20,3,size=100000)
samples_std10 = np.random.normal(20,10,size=100000)

# Make histograms
_=plt.hist(samples_std1,bins=100,normed=True,histtype='step')
_=plt.hist(samples_std3,bins=100,normed=True,histtype='step')
_=plt.hist(samples_std10,bins=100,normed=True,histtype='step')
# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()

# Generate CDFs
x_std1,y_std1=ecdf(samples_std1)
x_std3,y_std3=ecdf(samples_std3)
x_std10,y_std10=ecdf(samples_std10)

# Plot CDFs
_=plt.plot(x_std1,y_std1,marker='.',linestyle='none')
_=plt.plot(x_std3,y_std3,marker='.',linestyle='none')
_=plt.plot(x_std10,y_std10,marker='.',linestyle='none')
# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()


