file = open("../input/newdata/phpMawTba.csv", "r")

def chr_int(a):

    if a.isdigit(): return int(a) 

    else: return 0

data = []

for line in file:

     data1 = line.split(', ') 

     if len(data1) == 15:

        data.append([chr_int(data1[0]), data1[1], chr_int(data1[2]), data1[3], chr_int(data1[4]), data1[5], data1[6], data1[7], data1[8], data1[9], chr_int(data1[10]), chr_int(data1[11]), chr_int(data1[12]), data1[13], data1[14] ])
print (data[1:2])
import pandas as pd

df = pd.DataFrame(data) 

df.columns = ['age', 'type_employer', 'fnlwgt', 'education', 'education_num', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hr_per_week', 'country', 'income']
df.shape
counts = df.groupby('country').size() 

print (counts)
print (counts.head())
ml = df[(df.sex == 'Male')]

fm = df[(df.sex == 'Female')]
ml1 = df[(df.sex == 'Male') & (df.income=='>50K\n') ]

fm1 = df[(df.sex == 'Female') & (df.income=='>50K\n')]
df1 = df[(df.income=='>50K\n')]

print ('The rate of people with high income is: ', int(len(df1)/float(len(df))*100), '%.')

print ('The rate of men with high income is: ', int(len(ml1)/float(len(ml))*100), '%.')

print ('The rate of women with high income is: ', int(len(fm1)/float(len(fm))*100), '%.')
print ('The average age of men is: ', ml['age'].mean())

print ('The average age of women is: ', fm['age'].mean())

print ('The average age of high-income men is: ', ml1['age'].mean())

print ('The average age of high-income women is: ', fm1['age'].mean())
ml_mu = ml['age'].mean()

fm_mu = fm['age'].mean()

ml_var = ml['age'].var()

fm_var = fm['age'].var()

ml_std = ml['age'].std()

fm_std = fm['age'].std()

print ('Statistics of age for men: mu: ', ml_mu, 'var: ', ml_var, 'std: ', ml_std)

print ('Statistics of age for women: mu: ', fm_mu, 'var: ', fm_var, 'std: ', fm_std)
ml_median = ml['age'].median() 

fm_median = fm['age'].median()

print ("Median age per men and women: ", ml_median, fm_median)

ml_median_age = ml1['age'].median() 

fm_median_age = fm1['age'].median()

print ("Median age per men and women with high-income: ", ml_median_age, fm_median_age)
ml_age = ml['age']

ml_age.hist(normed = 0, histtype = 'stepfilled', bins = 20)
fm_age = fm['age']

fm_age.hist(normed = 0, histtype = 'stepfilled', bins = 10)
import seaborn as sns

fm_age.hist(normed = 0, histtype = 'stepfilled', alpha = .5, bins = 20)

ml_age.hist(normed = 0, histtype = 'stepfilled', alpha = .5, color = sns.desaturate("indianred", .75), bins = 10)
fm_age.hist(normed = 1, histtype = 'stepfilled', alpha = .5, bins = 20)

ml_age.hist(normed = 1, histtype = 'stepfilled', alpha = .5, bins = 10, color = sns.desaturate("indianred",

.75))
ml_age.hist(normed = 1, histtype = 'step', cumulative = True, linewidth = 3.5, bins = 20)

fm_age.hist(normed = 1, histtype = 'step', cumulative = True, linewidth = 3.5, bins = 20, color = sns.desaturate("indianred", .75))
df2 = df.drop(df.index[(df.income == '>50K\n') & (df['age'] > df['age'].median() + 35) & (df['age'] > df['age'].median() -15)])

ml1_age = ml1['age'] 

fm1_age = fm1['age']

ml2_age = ml1_age.drop(ml1_age.index[ (ml1_age > df['age'].median() + 35) & (ml1_age > df['age'].median() - 15) ])

fm2_age = fm1_age.drop(fm1_age.index[ (fm1_age > df['age'].median() + 35) & (fm1_age > df['age'].median() - 15) ])
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

df.age[(df.income == '>50K\n')].plot(alpha = .25, color = 'blue') 

df2.age[(df2.income == '>50K\n')].plot(alpha = .45, color = 'red')
print ('The mean difference with outliers is: %4.2f. '% (ml_age.mean() - fm_age.mean()))

print ('The mean difference without outliers is: %4.2f.'% (ml2_age.mean() - fm2_age.mean()))
import numpy as np

countx, divisionx = np.histogram(ml2_age, normed = True)

county, divisiony = np.histogram(fm2_age, normed = True)

val = [(divisionx[i] + divisionx[i+1])/2 

       for i in range(len(divisionx) - 1)]

plt.plot(val, countx - county, 'o-')
def skewness(x):

    res = 0

    m = x.mean ()

    s = x.std ()

    for i in x:

        res += (i-m) * (i-m) * (i-m)

    res /= (len(x) * s * s * s)

    return res

print ("Skewness of the male population = ", skewness(ml2_age))

print ("Skewness of the female population is = ", skewness(fm2_age))
def pearson(x):

    return 3*(x.mean() - x.median())*x.std()

print ("Pearson’s coefficient of the male population = ", pearson(ml2_age))

print ("Pearson’s coefficient of the female population = ", pearson(fm2_age))
x1 = np.random.normal(-1, 0.5, 15)

x2 = np.random.normal(6, 1, 10)

y = np.r_[x1, x2] 

x = np.linspace(min(y), max(y), 100)

s = 0.4 

kernels = np.transpose ([norm.pdf(x, yi, s) for yi in y])

plt.plot(x, kernels, 'k: ')

plt.plot(x, kernels.sum(1), 'r')

plt.plot(y, np.zeros(len(y)), 'bo', ms = 10)
from scipy.stats import kde

density = kde.gaussian_kde(y)

xgrid = np.linspace(x.min(), x.max(), 200) 

plt.hist(y, bins = 28, normed = True) 

plt.plot(xgrid, density(xgrid), 'r-')
NTs = 200

mu = 0.0

var = 1.0

err = 0.0

NPs = 1000

for i in range(NTs):

    x = np.random.normal(mu, var, NPs)

    err += (x.mean()-mu)**2 

print ('MSE: ', err / NTs)