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
import pandas as pd

WineData = pd.read_csv("../input/red-wine-dataset/wineQualityReds.csv", index_col= 0)
WineData.head()
#Cheking Histogram

import matplotlib

from matplotlib import pyplot 

%matplotlib inline

pyplot.figure(figsize=(14,6))

pyplot.hist(WineData['volatile.acidity'])

pyplot.show()
#Help from Python

from scipy.stats import shapiro



DataToTest = WineData['volatile.acidity']



stat, p = shapiro(DataToTest)



print('stat=%.2f, p=%.30f' % (stat, p))



if p > 0.05:

    print('Normal distribution')

else:

    print('Not a normal distribution')
#Lets genrate normally distributed data from Python

from numpy.random import randn

DataToTest = randn(100)
DataToTest
stat, p = shapiro(DataToTest)



print('stat=%.2f, p=%.30f' % (stat, p))



if p > 0.05:

    print('Normal distribution')

else:

    print('Not a normal distribution')
# Example of the D'Agostino's K^2 Normality Test

from scipy.stats import normaltest

DataToTest = WineData['volatile.acidity']



stat, p = normaltest(DataToTest)



print('stat=%.10f, p=%.10f' % (stat, p))



if p > 0.05:

    print('Normal')



else:

    print('Not Normllay distributed')
FirstSample = WineData[1:30]['total.sulfur.dioxide']

SecondSample = WineData[1:30]['free.sulfur.dioxide']



pyplot.plot(FirstSample,SecondSample)

pyplot.show()
#Spearman Rank Correlation

from scipy.stats import spearmanr

stat, p = spearmanr(FirstSample, SecondSample)



print('stat=%.3f, p=%5f' % (stat, p))

if p > 0.05:

    print('independent samples')

else:

    print('dependent samples')
#pearson correlation

from scipy.stats import pearsonr

stat, p = pearsonr(FirstSample, SecondSample)



print('stat=%.3f, p=%5f' % (stat, p))

if p > 0.05:

    print('independent samples')

else:

    print('dependent samples')
WineData[1:30].corr(method="pearson")
#Tests whether two categorical variables are related or independent.

#Assumptions - independent observation, size in each box of contingency table > 25

# Example of the Chi-Squared Test

LoanData = pd.read_csv("../input/my-dataset/credit_train.csv")
LoanData.head(10)
contingency_data = pd.crosstab(LoanData['Purpose'], LoanData['Loan Status'],margins = False)
contingency_data
from scipy.stats import chi2_contingency
stat, p, dof, expected = chi2_contingency(contingency_data)

print('stat=%.3f, p=%.3f' % (stat, p))

if p > 0.05:

    print('independent categories')

else:

    print('dependent categories')
contingency_data = [[25,125],[1200,240]] #Observe the numbers carefully
stat, p, dof, expected = chi2_contingency(contingency_data)

print('stat=%.3f, p=%.3f' % (stat, p))

if p > 0.05:

    print('independent categories')

else:

    print('dependent categories')
#Scores of me and Virat

my_score = [23,21,31,20,19,35,26,22,21,19]

virat_score = [46,42,62,40,38,70,52,44,42,38]
#Lets check mean of our scores

import numpy as np

print('Mukesh mean score:', np.mean(my_score))

print('Virat mean score:', np.mean(virat_score))
#One Sample T-test

import scipy

scipy.stats.ttest_1samp(my_score,20)
#Independent Sample T-test

scipy.stats.ttest_ind(my_score,virat_score)
my_score_second_Tour = [46,42,62,40,38,70,52,44,42,38]
#Apired sample t-test

scipy.stats.ttest_rel(my_score,my_score_second_Tour)
# Assumption -  Normal distributon, same variance, identical distribution
average_score = [40,44,60,50,48,68,55,46,44,54]
my_score
average_score
virat_score

tstat, p = scipy.stats.f_oneway(my_score, average_score, virat_score)

print('stat=%.3f, p=%.3f' % (stat, p))

if p > 0.05:

    print('Same distribution of scores')

else:

    print('Different distributions of scores')
#Assumptions - Idential distribution, observations can be ranked
class_1_score = [91,90,81,80,76]

class_2_score = [88,86,85,84,83]
tstat, p = scipy.stats.mannwhitneyu(class_1_score, class_2_score)

print('stat=%.3f, p=%.3f' % (stat, p))

if p > 0.05:

    print('Same distribution')

else:

    print('Different distributions')
# Similarly check for Wilcoxon Signed-Rank Test/Kruskal-Wallis H Test
#Augmented Dickey-Fuller Test -  null hypothesis - Series is non stationary
#Definition of stationary time series - constant mean and variance
from statsmodels.tsa.stattools import adfuller

stock_price_data = [121,131,142,121,131,142,121,131,142]

stat, p, lags, obs, crit, t = adfuller(stock_price_data)

print('stat=%.3f, p=%.3f' % (stat, p))

if p > 0.05:

    print('Series is not Stationary')

else:

    print('Series is stationary')