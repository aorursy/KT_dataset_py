# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



def get_filename():

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            return os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.
filename = get_filename()

df = pd.read_csv(filename)

df.head()
df_males = df.loc[df['Gender'] == 'Male']

df_males.head()
df_females = df.loc[df['Gender'] == 'Female']

df_females.head()
min(df_males['Height']), max(df_males['Height'])
min(df_females['Height']), max(df_females['Height'])
bins = np.linspace(54, 79, 60)



plt.hist(df_males['Height'], bins, alpha=0.5, label='males')

plt.hist(df_females['Height'], bins, alpha=0.5, label='females')

plt.show()
# mean

mean_males = np.mean(df_males['Height'])

mean_females = np.mean(df_females['Height'])



# standard deviation

std_males = np.std(df_males['Height'])

std_females = np.std(df_females['Height'])



# Gaussian hypothesis

plt.hist(df_males['Height'], bins, alpha=0.5, label='males', density=True)

plt.hist(df_females['Height'], bins, alpha=0.5, label='females', density=True)



import scipy.stats as stats

import math



x = np.linspace(54, 79, 100)

plt.plot(x, stats.norm.pdf(x, mean_males, std_males))

plt.plot(x, stats.norm.pdf(x, mean_females, std_females))

plt.show()
# overlap as Gaussian

def solve(m1,m2,std1,std2):

    a = 1/(2*std1**2) - 1/(2*std2**2)

    b = m2/(std2**2) - m1/(std1**2)

    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)

    return np.roots([a,b,c])



from scipy.stats import norm

result = solve(mean_males, mean_females, std_males, std_females)[0]

area = norm.cdf(result,mean_males,std_males) + (1.-norm.cdf(result,mean_females,std_females))



print('Overlap as Gaussian = %.2f%%' % (area / (2-area) * 100))



# real overlap

area = len(df_males.loc[df_males['Height'] <= result]) + len(df_females.loc[df_females['Height'] >= result])

area / (len(df_males) + len(df_females) - area)



print('Real overlap = %.2f%%' % (area / (len(df_males) + len(df_females) - area) * 100))