import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
# randn returns an array, sampled from a normal distribution

a = np.random.randn(5)

a
a * 5
# x = np.random.randn(1000)

x = 200 * np.random.randn(1000)

plt.hist(x)

plt.show()
bin_number = 100

plt.hist(x, 

         bin_number, 

         range=(-500, 500), 

         histtype='stepfilled', 

         align='mid', 

         color='orange', 

         label='Test Data')

plt.legend()

plt.title('My Histogram')

plt.show()
df = pd.read_excel('data/survey.xls')
df.head()
df['Age'].hist(range=(12, 40))
df = pd.DataFrame({

    'length': [1, 2, 3, 4, 4, 4],

    'width': [1, 1, 1, 2, 3, 4]

    }, index= ['pig', 'rabbit', 'duck', 'chicken', 'horse', 'cat'])

df
df.hist(bins=4)