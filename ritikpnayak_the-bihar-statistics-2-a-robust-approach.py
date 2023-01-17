# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

from scipy import stats

from collections import defaultdict

style.use('ggplot')
gross_enrollment = pd.read_csv('/kaggle/input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv', index_col = 'State_UT')
gross_enrollment.rename({'MADHYA PRADESH' : 'Madhya Pradesh', 'Pondicherry' : 'Puducherry', 'Uttaranchal' : 'Uttarakhand'}, inplace = True)
gross_enrollment.reset_index(inplace = True)
gross_enrollment.head()
gross_enrollment.shape
gross_enrollment.info()
cols_to_convert = ['Higher_Secondary_Boys', 'Higher_Secondary_Girls', 'Higher_Secondary_Total']

np_vals = ['NR', '@']

gross_enrollment[cols_to_convert] = gross_enrollment[cols_to_convert].replace(np_vals,np.nan)

gross_enrollment[cols_to_convert] = gross_enrollment[cols_to_convert].astype('float64')
gross_enrollment.info()
plt.figure(figsize=(15,8))

gross_enrollment.Primary_Total.hist()
plt.figure(figsize=(15,8))



gross_enrollment.Primary_Boys.hist()
plt.figure(figsize=(15,8))



gross_enrollment.Primary_Girls.hist()
gross_enrollment['class'] = gross_enrollment.Primary_Total.apply(lambda x:1 if x<=70

                                                                else 2 if x<=80

                                                                else 3 if x<=90

                                                                else 4 if x<=100

                                                                else 5 if x<=110

                                                                else 6 if x<=120

                                                                else 7 if x<=130

                                                                else 8 if x<=140

                                                                else 9)   



gross_enrollment['class_boys'] = gross_enrollment.Primary_Boys.apply(lambda x:1 if x<=70

                                                                else 2 if x<=80

                                                                else 3 if x<=90

                                                                else 4 if x<=100

                                                                else 5 if x<=110

                                                                else 6 if x<=120

                                                                else 7 if x<=130

                                                                else 8 if x<=140

                                                                else 9)      



gross_enrollment['class_girls'] = gross_enrollment.Primary_Girls.apply(lambda x:1 if x<=70

                                                                else 2 if x<=80

                                                                else 3 if x<=90

                                                                else 4 if x<=100

                                                                else 5 if x<=110

                                                                else 6 if x<=120

                                                                else 7 if x<=130

                                                                else 8 if x<=140

                                                                else 9)                                                        
bihar = gross_enrollment[gross_enrollment.State_UT == 'Bihar']
hist_all = dict(gross_enrollment['class'].value_counts())

hist_bihar = dict(bihar['class'].value_counts())



pmf_all = defaultdict()

pmf_bihar = defaultdict()



n_all = sum(hist_all.values())

n_bihar = sum(hist_bihar.values())



for index, freq in hist_all.items():

    pmf_all[index] = freq/n_all



for index, freq in hist_bihar.items():

    pmf_bihar[index] = freq/n_bihar
pmf_all
pmf_bihar
hist_boys = dict(gross_enrollment['class_boys'].value_counts())

hist_girls = dict(gross_enrollment['class_girls'].value_counts())



pmf_boys = defaultdict()

pmf_girls = defaultdict()



n_boys = sum(hist_boys.values())

n_girls = sum(hist_girls.values())



for index, freq in hist_boys.items():

    pmf_boys[index] = freq/n_boys



for index, freq in hist_girls.items():

    pmf_girls[index] = freq/n_girls
l = []

for i in pmf_boys.keys():

    if i in pmf_girls.keys():

        l.append(i)



diffs = []

for val in l:

    p_boys = pmf_boys[val]

    p_girls = pmf_girls[val]

    diff = 100*(p_boys-p_girls)

    diffs.append(diff)



plt.figure(figsize=(20,8))



plt.bar(l, diffs)
## Defining our cdf function.



def cdf(sample, val):

    count = 0.0

    for value in sample:

        if value <= val:

            count += 1

            

    prob = count/len(sample)

    return prob



sample = [1,2,2,4,5,1,3]



d = {}

for i in set(sample):

    d[i] = cdf(sample, i)
d
## Defining a cdf function that takes a dictionary as sample.

# it will be beneficial because we are dealing with Series object that can be easily converted to dictionary dataframe.



def cdf_dict(sample, val):

    count = 0.0

    for value in sample:

        if sample[value] <= sample[val]:

            count += 1

            

    prob = count/len(sample)

    return prob

sample_boys = dict(gross_enrollment['class_boys'].value_counts())

sample_girls = dict(gross_enrollment['class_girls'].value_counts())



y_boys = {}

for i in set(gross_enrollment['class_boys']):

    cdf = cdf_dict(sample_boys, i)

    y_boys[i] = cdf



y_girls = {}

for i in set(gross_enrollment['class_girls']):

    cdf = cdf_dict(sample_girls, i)

    y_girls[i] = cdf
fig, axs = plt.subplots(2, figsize = (20,8))

fig.suptitle('CDF Subplots')

axs[0].bar(y_boys.keys(), y_boys.values(), color = 'blue')

axs[1].bar(y_girls.keys(), y_girls.values())
stats.probplot(x = gross_enrollment['Upper_Primary_Total'])
## This is how the graph of probplot looks like



stats.probplot(x = gross_enrollment['Upper_Primary_Total'], plot = plt)
fig, axs = plt.subplots(4, figsize = (20,12))

fig.suptitle('Vertically stacked Normal Probability Plots')



axs[0].plot(stats.probplot(x = gross_enrollment['Primary_Total'])[0][0], stats.probplot(x = gross_enrollment['Primary_Total'])[0][1])

axs[1].plot(stats.probplot(x = gross_enrollment['Upper_Primary_Total'])[0][0], stats.probplot(x = gross_enrollment['Upper_Primary_Total'])[0][1])

axs[2].plot(stats.probplot(x = gross_enrollment['Secondary_Total'])[0][0], stats.probplot(x = gross_enrollment['Secondary_Total'])[0][1])

axs[3].plot(stats.probplot(x = gross_enrollment['Higher_Secondary_Total'])[0][0], stats.probplot(x = gross_enrollment['Higher_Secondary_Total'])[0][1])