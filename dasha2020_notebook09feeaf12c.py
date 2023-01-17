# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

file = open("../input/adult-census-income/adult.csv")
def chr_int(a):
    if a.isdigit(): return int(a) 
    else: return 0
data = []
for line in file:
     data1 = line.split(', ') 
     if len(data1) == 15:
        data.append([chr_int(data1[0]), data1[1], chr_int(data1[2]), 
                     data1[3], chr_int(data1[4]), data1[5], data1[6], data1[7], data1[8], 
                     data1[9], chr_int(data1[10]), chr_int(data1[11]), chr_int(data1[12]), 
                     (data1[13]), (data1[14]) ])
file = ('../input/adult-census-income/adult.csv')
df = pd.read_csv(file)
df.shape
df.head
data = pd.DataFrame(df) 
data.columns = ['age', 'workclass', 'fnlwgt', 'education',
              'education.num', 'marital.status', 'occupation',
              'relationship', 'race', 'sex', 'capital.gain',
              'capital.loss', 'hours.per.week', 'native.country', 'income']
data
counts = df.groupby('native.country').size()
print (counts)

ml = df[(df.sex == 'Male')]
fm = df[(df.sex == 'Female')]
ml1 = df[(df.sex == 'Male') & (df.income=='>50K') ]
fm1 = df[(df.sex == 'Female') & (df.income=='>50K')]
df1 = df[(df.income=='>50K')]
print ('The rate of people with high income is: ',
int(len(df1)/float(len(df))*100), '%.')
print ('The rate of men with high income is: ',
int(len(ml1)/float(len(ml))*100), '%.')
print ('The rate of women with high income is: ',
int(len(fm1)/float(len(fm))*100), '%.')
print ('The average age of men is: ', ml['hours.per.week'].mean())
print ('The average age of women is: ', fm['hours.per.week'].mean())
print ('The average age of high-income men is: ',
ml1['hours.per.week'].mean())
print ('The average age of high-income women is: ',
fm1['hours.per.week'].mean())
ml_mu = ml['hours.per.week'].mean()
fm_mu = fm['hours.per.week'].mean()
ml_var = ml['hours.per.week'].var()
fm_var = fm['hours.per.week'].var()
ml_std = ml['hours.per.week'].std()
fm_std = fm['hours.per.week'].std()
print ('Statistics of age for men: mu: ', ml_mu, 'var: ',
ml_var, 'std: ', ml_std)
print ('Statistics of age for women: mu: ', fm_mu, 'var: ',
fm_var, 'std: ', fm_std)
ml_median = ml['hours.per.week'].median()
fm_median = fm['hours.per.week'].median()
print ('Median age per men and women: ', ml_median, fm_median)
ml_median_age = ml1['hours.per.week'].median()
fm_median_age = fm1['hours.per.week'].median()
print ('Median age per men and women with high-income: ',ml_median_age, fm_median_age)
ml_age = ml['hours.per.week']
ml_age.hist(density = 1, histtype = 'stepfilled', bins = 20)
fm_age = fm['hours.per.week']
fm_age.hist(density = 1, histtype = 'stepfilled', bins = 10)
import seaborn as sns
fm_age.hist(density = 0, histtype = 'stepfilled', alpha = .5, bins = 20)
ml_age.hist(density = 0, histtype = 'stepfilled', alpha = .5, color = sns.desaturate("indianred", .75), bins = 10)
fm_age.hist(density = 1, histtype = 'stepfilled', alpha = .5, bins = 20)
ml_age.hist(density = 1, histtype = 'stepfilled', alpha = .5, bins = 10, color = sns.desaturate("indianred", .75))
ml_age.hist(density = 1, histtype = 'step', cumulative = True, linewidth = 3.5, bins = 20, color = 'blue')
fm_age.hist(density = 1, histtype = 'step', cumulative = True, linewidth = 3.5, bins = 20, color = sns.desaturate("indianred", .75))
df2 = df.drop(df.index[(df.income == '>50K') & (df['hours.per.week'] >
df['hours.per.week'].median() + 50) & (df['hours.per.week'] > df['hours.per.week'].median() - 6)])
ml1_age = ml1['hours.per.week']
fm1_age = fm1['hours.per.week']
ml2_age = ml1_age.drop(ml1_age.index[ (ml1_age > df['hours.per.week'].
median() + 50) & (ml1_age > df['hours.per.week'].median() - 6) ])
fm2_age = fm1_age.drop(fm1_age.index[ (fm1_age > df['hours.per.week'].
median() + 50) & (fm1_age > df['hours.per.week'].median() - 6) ])
mu2ml = ml2_age.mean()
std2ml = ml2_age.std()
md2ml = ml2_age.median()
mu2fm = fm2_age.mean()
std2fm = fm2_age.std()
md2fm = fm2_age.median()
print ("Men statistics:")
print ("Mean:", mu2ml, "Std:", std2ml)
print ("Median:", md2ml)
print ("Min:", ml2_age.min(), "Max:", ml2_age.max())
print ("Women statistics:")
print ("Mean:", mu2fm, "Std:", std2fm)
print ("Median:", md2fm)
print ("Min:", fm2_age.min(), "Max:", fm2_age.max())
import matplotlib.pyplot as plt
plt.figure(figsize = (13.4, 5))
df.age[(df.income == '>50K')].plot(alpha = .25, color = 'blue')
df2.age[(df2.income == '>50K')].plot(alpha = .45, color = 'red')
print ('The mean difference with outliers is: %4.2f. '%
(ml_age.mean() - fm_age.mean()))
print ('The mean difference without outliers is: %4.2f.'%
(ml2_age.mean() - fm2_age.mean()))
import numpy as np
countx, divisionx = np.histogram(ml2_age, density = True)
county, divisiony = np.histogram(fm2_age, density = True)
val = [(divisionx[i] + divisionx[i+1])/2
for i in range(len(divisionx) - 1)]
plt.plot(val, countx - county, 'o-')
import numpy as np
countx, divisionx = np.histogram(ml2_age, density = True)
county, divisiony = np.histogram(fm2_age, density = True)
val = [(divisionx[i] + divisionx[i+1])/2
for i in range(len(divisionx) - 1)]
plt.plot(val, countx - county, 'o-')