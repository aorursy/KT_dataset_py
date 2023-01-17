import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

%matplotlib inline
train_data = pd.read_csv('../input/train.csv')
train_data.describe()
train_data.head()
p = plt.hist(acc_df.mean_accuracy, bins=np.arange(min(acc_df.mean_accuracy), max(acc_df.mean_accuracy) + 1, 1))
plt.xlabel('Mean Accuracy')
plt.ylabel('Count')
plt.title('Counts of mean accuracy for places')

plt.show()
p = plt.hist(acc_df.std_x, bins=np.arange(min(acc_df.std_x), max(acc_df.std_x) + .01, .01))
plt.xlabel('Std(y)')
plt.ylabel('Count')
plt.title('Counts of std(x) for places')

plt.show()
p = plt.hist(acc_df.std_y, bins=np.arange(min(acc_df.std_y), max(acc_df.std_y) + .001, .001))
plt.xlabel('Std(y)')
plt.ylabel('Count')
plt.title('Counts of std(y) for places')

plt.show()
p = plt.scatter(acc_df.mean_accuracy,acc_df.std_x)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(x)')
plt.title('std(x) vs Mean accuracy for each place')

plt.show()
p = plt.scatter(acc_df.mean_accuracy, acc_df.std_y)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(y)')
plt.title('std(y) vs mean accuracy for each place')

plt.show()
#Exponential Decay function
def func(x, a, b, c):
    return a*np.exp(-b*(x+c))

popt, pcov = scipy.optimize.curve_fit(func, acc_df.mean_accuracy, acc_df.std_x)
popt
p = plt.scatter(acc_df.mean_accuracy,acc_df.std_x)
l = plt.plot([i for i in range(0,1000)], [func(i, 1.09251753e+00,   4.33988866e-03,   2.86223906e+01) for i in range(0,1000)], 'r', linewidth=1)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(x)')
plt.title('std(x) vs Mean accuracy for each place (exp decay)')

plt.show()
#csch = 1/sinh and 
def func2(x,a,b,c):
    return a*1/(np.sinh(b*x)) + c

popt, pcov = scipy.optimize.curve_fit(func2, acc_df.mean_accuracy, acc_df.std_x)
popt
p = plt.scatter(acc_df.mean_accuracy,acc_df.std_x)
l = plt.plot([i for i in range(0,1000)], [func2(i, 108.75643807,   40.73096679,    0.67921053) for i in range(0,1000)], 'r', linewidth=1)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(x)')
plt.title('std(x) vs Mean accuracy for each place (csch)')

plt.show()
#1/x
def func4(x,a,b,c):
    return a*(1/(b*(x+c)))

popt, pcov = scipy.optimize.curve_fit(func4, acc_df.mean_accuracy, acc_df.std_x)
popt
p = plt.scatter(acc_df.mean_accuracy,acc_df.std_x)
l = plt.plot([i for i in range(0,1000)], [func4(i, 97.59074952,   0.8234309 ,  95.34280786) for i in range(0,1000)], 'r', linewidth=1)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(x)')
plt.title('std(x) vs Mean accuracy for each place (1/x)')

plt.show()
popt, pcov = scipy.optimize.curve_fit(func, acc_df.mean_accuracy, acc_df.std_y)
popt
p = plt.scatter(acc_df.mean_accuracy,acc_df.std_y)
l = plt.plot([i for i in range(0,1000)], [func(i, -0.01277189,  1.4337799 ,  2.71943575) for i in range(0,1000)], 'r', linewidth=1)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(y)')
plt.title('std(y) vs Mean accuracy for each place (exp decay)')

plt.show()
popt, pcov = scipy.optimize.curve_fit(func4, acc_df.mean_accuracy, acc_df.std_y)
popt
p = plt.scatter(acc_df.mean_accuracy,acc_df.std_y)
l = plt.plot([i for i in range(0,1000)], [func(i, 221.24964964,   12.02131292,  804.60004683) for i in range(0,1000)], 'r', linewidth=1)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(y)')
plt.title('std(y) vs Mean accuracy for each place (1/x)')

plt.show()
