import pandas as pd
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
recent_grads = pd.read_csv('../input/recent-graduates.csv')
recent_grads.iloc[0]
recent_grads.head(5)
recent_grads.tail(3)
recent_grads.describe()
raw_data_count = recent_grads.shape
raw_data_count
recent_grads = recent_grads.dropna()
cleaned_data_count = recent_grads.shape
cleaned_data_count
cleaned_data_count
X = recent_grads['Sample_size']
Y = recent_grads['Median']

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

a, b = best_fit(X, Y)

plt.scatter(X, Y, c='b')
yfit = [a + b * xi for xi in X]
plt.title('Sample size vs. Median')
plt.xlabel('Sample size')
plt.ylabel('Median')
plt.plot(X, yfit)
recent_grads.plot(x='Sample_size', y='Unemployment_rate', kind='scatter',title="Sample size vs. Unemployment rate")
X = recent_grads['Full_time']
Y = recent_grads['Median']

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

a, b = best_fit(X, Y)

plt.scatter(X, Y, c='b')
yfit = [a + b * xi for xi in X]
plt.title('Full_time vs. Median')
plt.xlabel('Full_time')
plt.ylabel('Median')
plt.plot(X, yfit)
recent_grads.plot(x='ShareWomen', y='Unemployment_rate', kind='scatter',title="ShareWomen vs. Unemployment_rate")
recent_grads.plot(x='Men', y='Median', kind='scatter',title="Men vs. Median")
X = recent_grads['Women']
Y = recent_grads['Median']

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

a, b = best_fit(X, Y)

plt.scatter(X, Y, c='b')
yfit = [a + b * xi for xi in X]
plt.title('Women vs. Median')
plt.xlabel('Women')
plt.ylabel('Median')
plt.plot(X, yfit)
recent_grads['Sample_size'].hist(bins=25, range=(0,5000), color='m')
recent_grads['Median'].hist(bins=25, color='b')
recent_grads['Employed'].hist(bins=25, range=(0,5000), color='black')
recent_grads['Full_time'].hist(bins=25, range=(0,5000), color='g')
recent_grads['ShareWomen'].hist(bins=25, color='violet')
mostly_female = recent_grads['ShareWomen'] > 0.5
mostly_female.value_counts()
76/96 * 100
recent_grads['Unemployment_rate'].hist(bins=25, color='yellow')
recent_grads['Men'].hist(bins=25, range=(0,5000), color='c')
recent_grads['Women'].hist(bins=25, range=(0,5000), color='pink')
from pandas.plotting import scatter_matrix

scatter_matrix(recent_grads[['Sample_size', 'Median']], figsize=(20,20), c='black')
scatter_matrix(recent_grads[['Sample_size', 'Median', 'ShareWomen']], figsize=(20,20), c='black')
recent_grads[:10].plot.barh(x='Major',y='ShareWomen', color='c');
plt.title('Proportion Of Women In The Ten Highest Paying Majors')
plt.xlabel('Percentage')
plt.ylabel('Major')
end = len(recent_grads) - 10
recent_grads[end:].plot.barh(x='Major',y='ShareWomen', color='c');
plt.title('Proportion Of Women In The Ten Lowest Paying Majors')
plt.xlabel('Percentage')
plt.ylabel('Major')
recent_grads[:10].plot.barh(x='Major',y='Unemployment_rate');
plt.title('Proportion Of Unemployed In The Ten Highest Paying Majors')
plt.xlabel('Percentage')
plt.ylabel('Major')
end = len(recent_grads) - 10
recent_grads[end:].plot.barh(x='Major',y='ShareWomen');
plt.title('Proportion Of Unemployed In The Ten Lowest Paying Majors')
plt.xlabel('Percentage')
plt.ylabel('Major')