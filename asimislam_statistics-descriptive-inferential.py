import pandas as pd

import numpy as np

from numpy import random

import math

import scipy.stats as stats

import statistics



import matplotlib.pyplot as plt

import seaborn as sns



#  Kaggle directories

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



heart = pd.read_csv('../input/heart-disease-uci/heart.csv')

heart.shape
#  Heart Disease UCI [Age ]

#  frequency distribution table

interval_range = pd.interval_range(start=0, end=100, freq=10)

age_ranges = heart['age'].value_counts(bins = interval_range).sort_values()



freq = pd.DataFrame()

freq['Age'] = age_ranges.index

freq['Absolute Frequency'] = age_ranges.values

freq['Relative Frequency, %'] = round(freq['Absolute Frequency']*100/len(heart['age']),4)

freq = freq.sort_values(by = ['Age']).reset_index(drop=True)



totals_row = {'Age':'-total-', 'Absolute Frequency':freq['Absolute Frequency'].sum(), 'Relative Frequency, %':freq['Relative Frequency, %'].sum()}

freq = freq.append(totals_row, ignore_index=True)



freq
#  Frequency Distribution Plots

fig = plt.figure(figsize=(12,4))

plt.subplot(131)

plt.title('Heart Disease UCI [age]\nfrequency distribution\n(Histogram)')

heart['age'].plot(kind='hist')



plt.subplot(132)

plt.title('Heart Disease UCI [age]\nfrequency distribution\n(Density Plot)')

heart['age'].plot(kind='kde')



plt.subplot(133)

plt.title('Heart Disease UCI [age]\nfrequency distribution\n(Boxplot)')

heart['age'].plot(kind='box')

plt.show()
#  create data frame from tuples

x = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)

y = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,44,66,66,44,19,20)

z = (-20,-19,-44,-66,-66,-44,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1)



df = pd.DataFrame({'x': x,'y':y,'z':z})

df.transpose()  # print dataframe transpose
#  function - print central tendencies values

def getCT(*args):

    for i in args:

        a =  "min/max:   {:,.0f}/{:,.0f}\n".format(min(i),max(i))

        a += "mean   {:,.2f}\n".format(np.mean(i))

        a += "median   {:,.2f}\n".format(np.median(i))

        a += "mode   {}\n".format(stats.mode(i)[0])

        a += "skew   {:,.2f}\n".format(stats.skew(i))

        a += "kurtosis   {:,.2f}".format(stats.kurtosis(i))

    return a





#  density plots of the distributions

fig = plt.figure(figsize=(10,4))

plt.subplot(131)

x_label = getCT(df.x)

plt.title('Symmertical Distribution\n(density plot)')

plt.xlabel("x\n\n{}".format(x_label))

plt.yticks([])

df.x.plot(kind='kde')   #  skew = 0, symmetric



plt.subplot(132)

x_label = getCT(df.y)

plt.title('Negative Skew Distribution\n(density plot)')

plt.xlabel("y\n\n{}".format(x_label))

plt.yticks([])

df.y.plot(kind='kde')   #  skew < 0, left- hand side tail



plt.subplot(133)

x_label = getCT(df.z)

plt.title('Positive Skew Distribution\n(density plot)')

plt.xlabel("z\n\n{}".format(x_label))

plt.yticks([])

df.z.plot(kind='kde')   #  skew > 0, right-hand side tail

plt.show()
#  boxplots plots of the distributions

fig = plt.figure(figsize=(10,4))

plt.subplot(131)

plt.title('Symmertical Distribution\n(boxplot)')

df.x.plot(kind='box')   #  skew = 0, symmetric



plt.subplot(132)

plt.title('Negative Skew Distribution\n(boxplot)')

df.y.plot(kind='box')   #  skew < 0, left- hand side tail



plt.subplot(133)

plt.title('Positive Skew Distribution\n(boxplot)')

df.z.plot(kind='box')   #  skew > 0, right-hand side tail

plt.show()
print(f"\n{'POPULATION - Dispersion for df.x'}",

      f"\n{'Population Variance:':<35}{statistics.pvariance(df.x):>10,.4f}",

      f"\n{'Population Standard Deviation':<35}{statistics.pstdev(df.x):>10,.4f}",

      f"\n{'Population Coefficient of Variance:':<35}{statistics.pstdev(df.x)/np.mean(df.x):>10,.4f}",

    

      f"\n\n{'SAMPLE - Dispersion for df.x'}",

      f"\n{'Sample Variance:':<35}{statistics.variance(df.x):>10,.4f}",

      f"\n{'Sample Standard Deviation':<35}{statistics.stdev(df.x):>10,.4f}",

      f"\n{'Sample Coefficient of Variance:':<35}{statistics.stdev(df.x)/np.mean(df.x):>10,.4f}")
df.cov()  #  covariance
df.corr()   #  correlation
sns.heatmap(df.corr(), annot=True, annot_kws={"size": 16}, fmt='.4f', square=True, cmap='mako')
#  DATA - Heart Disease UCI, Cholesterol

#  plot the data

plt.figure(figsize=(8,4))

plt.title('Heart Disease UCI [chol]\ndata distribution\nz-statistics, population size = {}'.format(len(heart['chol'])), fontsize=14)

plt.xlabel('observations')

plt.ylabel('density')

plt.vlines(np.mean(heart['chol']),0,0.003)

sns.distplot(heart['chol'], hist=False)

plt.show()



#  mean, standard deviation and standard error

mean  = np.mean(heart['chol'])           # mean

sigma = statistics.stdev(heart['chol'])  # standard deviation

sem   = stats.sem(heart['chol'])         # standard error





print('mean:\t\t\t{:.4f}'.format(mean))

print('standard deviation:\t{:.4f}'.format(sigma))

print('standard error:\t\t{:.4f}'.format(sem))
confidenceLevel = .90

numOfTails      = 2

alpha           = (1 - confidenceLevel)/numOfTails



#  Percent Point Function

#  - calculates z-critical from (1-alpha)

z_critical = stats.norm.ppf(1 - alpha)



print('Step 2a:  Calculate z-critical for Confidence Level {:.0%} with {} tails.'.format(confidenceLevel,numOfTails))

print('Confidence Level:\t{:.0%}'.format(confidenceLevel))

print('Number of Tails:\t{}'.format(numOfTails))

print('alpha:\t\t\t{:.4f}'.format(alpha))

print('z-critical value:\t{:.4f}  <---'.format(z_critical))
print('Step 2b:  Calculate z-critical for several Confidence Levels with 1 and 2 tails.')



#--- func for critical value ---

def z_crit(cl,tail):

    alpha = (1-cl)/tail

    z_critical = stats.norm.ppf(1 - alpha)

    print('conf level {:.0%}'.format(cl), end="")

    print('  tails {}'.format(tail), end="")

    print('  alpha {:.4f}'.format(alpha), end="")

    print('\t-->  z-critical:  {:.4f}'.format(z_critical))

    return



#  Confidence Levels at 75%, 90% and 99% with 1 or 2 tails

confidenceLevel = [.75, .90, .99]

numOfTails      = [1,2]



for i in confidenceLevel:

    for j in numOfTails:

        z_crit(i,j)
confidenceLevel = .95

numOfTails      = 2

alpha           = (1 - confidenceLevel)/numOfTails

z_critical      = stats.norm.ppf(1 - alpha)



# confidence interval formula (manual)

lowerCI = mean - (z_critical * sigma)

upperCI = mean + (z_critical * sigma)



#  print confidence intervals

print('Confidence Level:\t{:.0%}'.format(confidenceLevel))

print('Number of Tails:\t{}'.format(numOfTails))

print('alpha:\t\t\t{:.4f}'.format(alpha))

print('z-critical value:\t{:.4f}  <---'.format(z_critical))

print('\nConfidence Interval:\nlower CI\t\t{:.4f}'.format(lowerCI))

print('upper CI:\t\t{:.4f}'.format(upperCI))
#  PLOT CONFIDENCE INTERVAL

plt.figure(figsize=(10,6))

plt.title('Heart Disease UCI [chol]\nz-statistics - Confidence Level = {:.0%} '.format(confidenceLevel), fontsize=16)

plt.xlabel('cholesterol level')

plt.ylabel('density')



labelCI=("lower CI:  {:.4f}\nupper CI:   {:.4f}".format(lowerCI,upperCI))

labelME=("\nmean:  {:.4f}".format(mean))

plt.vlines([lowerCI,upperCI],0,np.mean(stats.norm.pdf(heart['chol'], loc=mean, scale=sigma)),label=labelCI,color='blue',ls='--')

plt.vlines(mean,0,np.mean(stats.norm.pdf(heart['chol'], loc=mean, scale=sigma)*.5),label=labelME,color='green')

plt.legend(loc='best')



sns.distplot(heart['chol'], hist=False)

plt.show()





print("Z-Statistics - CONCLUSION:")

print("{:.0%} of the total patients in the Heart Disease UCI dataset will have cholesterol levels between {} and {}.".format(confidenceLevel, math.floor(lowerCI), math.ceil(upperCI)))
#  DATA - Heart Disease UCI, Cholesterol - sample

n = 20

heart_chol_sample = heart['chol'].sample(n)



#  plot the data

plt.figure(figsize=(8,4))

plt.title('Heart Disease UCI [chol]\ndata distribution\nt-statistics, sample size = {}'.format(len(heart_chol_sample)), fontsize=14)

plt.ylabel('density')

plt.vlines(np.mean(heart_chol_sample),0,.003)

sns.distplot(heart_chol_sample, hist=False)

plt.show()





#  mean, standard deviation and standard error

mean  = np.mean(heart_chol_sample)           # mean

s     = statistics.stdev(heart_chol_sample)  # standard deviation

sem   = stats.sem(heart_chol_sample)         # standard error



print('mean:\t\t\t{:.4f}'.format(mean))

print('standard deviation:\t{:.4f}'.format(s))

print('standard error:\t\t{:.4f}'.format(sem))
confidenceLevel = .95

n               = 10

ddof            = n -1

numOfTails      = 2

alpha           = (1 - confidenceLevel)/numOfTails



#  Percent Point Function

#  - calculates t-critical from alpha and ddof

t_critical = abs(stats.t.ppf(alpha/numOfTails, ddof))



print('Step 2a:  Calculate z-critical for Confidence Level {:.0%} with {} tails.'.format(confidenceLevel,numOfTails))

print('Confidence Level:\t{:.0%}'.format(confidenceLevel))

print('Number of Tails:\t{}'.format(numOfTails))

print('Degrees of Freedom:\t{}'.format(ddof))

print('alpha:\t\t\t{:.4f}'.format(alpha))

print('t-critical value:\t{:.4f}  <---'.format(t_critical))
print('Step 2b:  Calculate t-critical for several Confidence Levels with 1 and 2 tails.')



#--- func for critical value ---

def t_crit(cl,tail,ddof):

    alpha = (1-cl)/tail

    t_critical = abs(stats.t.ppf(alpha/tail,ddof))

    print('conf level {:.0%}'.format(cl), end="")

    print('  tails {}'.format(tail), end="")

    print('  ddof {}'.format(ddof), end="")

    print('  alpha {:.4f}'.format(alpha), end="")

    print('\t-->  t-critical:  {:.4f}'.format(t_critical))

    return





#  Confidence Levels at 75%, 90% and 99% with 1 or 2 tails

confidenceLevel = [.75, .90, .99]

numOfTails      = [1,2]

n               = 10

ddof            = n - 1



for i in confidenceLevel:

    for j in numOfTails:

        t_crit(i,j,ddof)
confidenceLevel = .95

n               = 10

ddof            = n -1

numOfTails      = 2

alpha           = (1 - confidenceLevel)/numOfTails

t_critical      = abs(stats.t.ppf(alpha/numOfTails,ddof))



# confidence interval formula

lowerCI = mean - (t_critical * s)

upperCI = mean + (t_critical * s)



#  print confidence intervals

print('Confidence Level:\t{:.0%}'.format(confidenceLevel))

print('Number of Tails:\t{}'.format(numOfTails))

print('Degrees of Freedom:\t{}'.format(ddof))

print('alpha:\t\t\t{:.4f}'.format(alpha))

print('t-critical value:\t{:.4f}  <---'.format(t_critical))

print('\nConfidence Interval:\nlower CI\t\t{:.4f}'.format(lowerCI))

print('upper CI:\t\t{:.4f}'.format(upperCI))
plt.figure(figsize=(10,6))

plt.title('Heart Disease UCI [chol] (sample)\nt-statistics - Confidence Level = {:.0%} '.format(confidenceLevel), fontsize=16)

plt.ylabel('density')



labelCI=("lower CI: {:.4f}\nupper CI:  {:.4f}".format(lowerCI,upperCI))

labelME=("\nmean:     {:.4f}".format(mean))

plt.vlines([lowerCI,upperCI],0,np.mean(stats.norm.pdf(heart_chol_sample, loc=mean, scale=s)),label=labelCI,color='blue',ls='--')

plt.vlines(mean,0,np.mean(stats.norm.pdf(heart_chol_sample, loc=mean, scale=s)*.5),label=labelME,color='green')

plt.legend(loc='best')



sns.distplot(heart_chol_sample, hist=False, color='k')

plt.show()





print("T-Statistics - CONCLUSION:")

print("Based on a sample of {} patients, {:.0%} of the patients in the Heart Disease UCI dataset will have cholesterol levels between {} and {}.".format(len(heart_chol_sample),confidenceLevel, math.floor(lowerCI), math.ceil(upperCI)))