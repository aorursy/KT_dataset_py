import sys

sys.path.append("../input/supporting-files/")



import thinkplot

import thinkstats2

from __future__ import print_function, division



%matplotlib inline



import numpy as np

import pandas as pd 

#import thinkstats2

#import thinkplot



# read the csv file

suicide_data = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")

suicide_data.head()



# finding the number of rows and columns in the dataframe

suicide_data.shape

    
suicide_data.rename(columns = {'HDI for year':'HDI_for_year'}, inplace = True) 

suicide_data.rename(columns = {'suicides/100k pop':'suicides_per_100k_pop'}, inplace = True) 

suicide_data.rename(columns = {'gdp_per_capita ($)':'gdp_per_capita'}, inplace = True) 

suicide_data.rename(columns = {' gdp_for_year ($) ':'gdp_for_year'}, inplace = True) 

suicide_data["HDI_for_year"].fillna(0, inplace = True) 



# finding the coulumn names

suicide_data.columns

# plotting the histogram for HDI_for_year

hist1 = thinkstats2.Hist(suicide_data.HDI_for_year,label = 'HDI for year')

thinkplot.Hist(hist1)

thinkplot.Show(xlabel='HDI_for_year', ylabel='frequency', ylim = [0,100])

# checking for outliers in age

print("smallest values of HDI_for_year, freq :")

for HDI_for_year, freq in hist1.Smallest(10):

    print(HDI_for_year, freq)

    

print("highest values of HDI_for_year, freq :")

for HDI_for_year, freq in hist1.Largest(10):

    print(HDI_for_year, freq)

#methods to compute mean, variance and standard deviation:

mean_HDI = suicide_data.HDI_for_year.mean()

mode_HDI = suicide_data.HDI_for_year.mode()

var_HDI = suicide_data.HDI_for_year.var()

std_HDI = suicide_data.HDI_for_year.std()

print("mean, mode, var, std of HDI_for_year:", mean_HDI, mode_HDI, var_HDI, std_HDI)
#plotting the histogram for suicides_no

hist2 = thinkstats2.Hist(suicide_data.suicides_no,label = 'suicides_no')

thinkplot.Hist(hist2,width = 250)

thinkplot.Show(xlabel='suicides_no', ylabel='frequency')

# replotting the  histogram for suicides_no with limits for clear understanding of the distribution

thinkplot.Hist(hist2, width = 20,label = 'suicides_no')

thinkplot.Show(xlabel='suicides_no', ylabel='frequency', xlim = [0,1000])

# replotting the  histogram for suicides_no with limits for clear understanding of the distribution

thinkplot.Hist(hist2, width = 100,label = 'suicides_no')

thinkplot.Show(xlabel='suicides_no', ylabel='frequency', xlim = [1000,22500], ylim = [0,10])
# checking for outliers in suicides_no

print("smallest values of suicides_no, freq :")

for suicides_no, freq in hist2.Smallest(10):

    print(suicides_no, freq)

print("highest values of suicides_no, freq :")

for suicides_no, freq in hist2.Largest(10):

    print(suicides_no, freq)
#methods to compute mean, variance and standard deviation:

mean_sn = suicide_data.suicides_no.mean()

mode_sn = suicide_data.suicides_no.mode()

var_sn = suicide_data.suicides_no.var()

std_sn = suicide_data.suicides_no.std()

print("mean, mode, var, std of suicides_no:", mean_sn, mode_sn, var_sn, std_sn)
# plotting the histogram for population

hist3 = thinkstats2.Hist(suicide_data.population,label = 'population')

thinkplot.Hist(hist3, width = 5000)

thinkplot.Show(xlabel='population', ylabel='frequency')

# checking for outliers in population

print("smallest values of population, freq :")

for population, freq in hist3.Smallest(10):

    print(population, freq)

print("highest values of population, freq :")

for population, freq in hist3.Largest(10):

    print(population, freq)
#methods to compute mean, variance and standard deviation:

mean_p = suicide_data.population.mean()

mode_p = suicide_data.population.mode()

var_p = suicide_data.population.var()

std_p = suicide_data.population.std()

print("mean, mode, var, std of population:", mean_p, mode_p, var_p, std_p)
# plotting the histogram for suicides_per_100k_pop

hist4 = thinkstats2.Hist(suicide_data.suicides_per_100k_pop,label = 'suicides_per_100k_pop')

thinkplot.Hist(hist4, width = 3)

thinkplot.Show(xlabel='suicides_per_100k_pop', ylabel='frequency')
# replotting the histogram for suicides_per_100k_pop with different limits

thinkplot.Hist(hist4, width = 1, label = 'suicides_per_100k_pop')

thinkplot.Show(xlabel='suicides_per_100k_pop', ylabel='frequency', xlim = [0,30])
# checking for outliers in population

print("smallest values of suicides_per_100k_pop, freq :")

for suicides_per_100k_pop, freq in hist4.Smallest(10):

    print(suicides_per_100k_pop, freq)

print("highest values of suicides_per_100k_pop, freq :")

for suicides_per_100k_pop, freq in hist4.Largest(10):

    print(suicides_per_100k_pop, freq)
#methods to compute mean, variance and standard deviation:

mean_sp = suicide_data.suicides_per_100k_pop.mean()

mode_sp = suicide_data.suicides_per_100k_pop.mode()

var_sp = suicide_data.suicides_per_100k_pop.var()

std_sp = suicide_data.suicides_per_100k_pop.std()

print("mean, mode, var, std of suicides_per_100k_pop:", mean_sp, mode_sp, var_sp, std_sp)
# plotting the histogram for gdp_per_capita

hist5 = thinkstats2.Hist(suicide_data.gdp_per_capita,label='gdp_per_capita')

thinkplot.Hist(hist5, width = 250)

thinkplot.Show(xlabel='gdp_per_capita', ylabel='frequency')
# checking for outliers in population

print("smallest values of gdp_per_capita, freq :")

for gdp_per_capita, freq in hist5.Smallest(10):

    print(gdp_per_capita, freq)

print("highest values of gdp_per_capita, freq :")

for gdp_per_capita, freq in hist5.Largest(10):

    print(gdp_per_capita, freq)
#methods to compute mean, variance and standard deviation:

mean_gdp = suicide_data.gdp_per_capita.mean()

mode_gdp = suicide_data.gdp_per_capita.mode()

var_gdp = suicide_data.gdp_per_capita.var()

std_gdp = suicide_data.gdp_per_capita.std()

print("mean, mode, var, std of gdp_per_capita:", mean_gdp, mode_gdp, var_gdp, std_gdp)
# creating a dataframe for population of age : 35-54 years

suicide_data_age_35_54 = suicide_data[suicide_data.age == "35-54 years"]

type(suicide_data_age_35_54)
# creating a dataframe for population of age except 35-54 years(others)

suicide_data_others = suicide_data[suicide_data.age != "35-54 years"]

type(suicide_data_others)
# creating histograms for age_35_54 and others

age_35_54_hist = thinkstats2.Hist(suicide_data_age_35_54.suicides_no)

others_hist = thinkstats2.Hist(suicide_data_others.suicides_no)
# plotting the histogram for age_35_54 and others

thinkplot.PrePlot(2)

thinkplot.Hist(others_hist, align='left', color='green',label = 'others')

thinkplot.Hist(age_35_54_hist, align='right', color='red',label = 'age_35_54')

thinkplot.Show(xlabel='suicides_no', ylabel='frequency', xlim = [0,500], ylim =[0,1000])



# replotting the histogram for age_35_54 and others by changing the X and Y limits

thinkplot.PrePlot(2)

thinkplot.Hist(others_hist, align='left', color='green',label = 'others')

thinkplot.Hist(age_35_54_hist, align='right', color='red',label = 'age_35_54')

thinkplot.Show(xlabel='suicides_no', ylabel='frequency', xlim = [500,6000], ylim =[0,10])

# dividing the data according to the age group 

suicide_data_15_24_years = suicide_data[suicide_data.age == "15-24 years"]

suicide_data_75_years = suicide_data[suicide_data.age == "75+ years"]
suicide_data_25_34_years = suicide_data[suicide_data.age == "25-34 years"]
suicide_data_5_14_years = suicide_data[suicide_data.age == "5-14 years"]
suicide_data_55_74_years = suicide_data[suicide_data.age == "55-74 years"]
# plotting histograms for different age groups for comparision

age_15_24_hist = thinkstats2.Hist(suicide_data_15_24_years.suicides_no)

age_75_years_hist = thinkstats2.Hist(suicide_data_75_years.suicides_no)

age_25_34__hist = thinkstats2.Hist(suicide_data_25_34_years.suicides_no)

age_5_14_hist = thinkstats2.Hist(suicide_data_5_14_years.suicides_no)

age_55_74_hist = thinkstats2.Hist(suicide_data_55_74_years.suicides_no)
thinkplot.PrePlot(6)

thinkplot.Hist(age_5_14_hist, align='left', color='green',label = 'age_5_14')

thinkplot.Hist(age_15_24_hist, align='right', color='blue',label = 'age_15_24')

thinkplot.Hist(age_25_34__hist, align='left', color='black',label = 'age_25_34')

thinkplot.Hist(age_55_74_hist, align='left', color='pink',label = 'age_55_74')

thinkplot.Hist(age_75_years_hist, align='right', color='violet',label = 'age_75')

thinkplot.Hist(age_35_54_hist, align='right', color='red',label = 'age_35_54')

thinkplot.Show(xlabel='suicides_no', ylabel='frequency', xlim = [0,500], ylim =[0,600])
thinkplot.PrePlot(6)

thinkplot.Hist(age_5_14_hist, align='left', color='green',label = 'age_5_14')

thinkplot.Hist(age_15_24_hist, align='right', color='blue',label = 'age_15_24')

thinkplot.Hist(age_25_34__hist, align='left', color='black',label = 'age_25_34')

thinkplot.Hist(age_35_54_hist, align='right', color='red',label = 'age_35_54')

thinkplot.Hist(age_55_74_hist, align='left', color='pink',label = 'age_55_74')

thinkplot.Hist(age_75_years_hist, align='right', color='violet',label = 'age_75')

thinkplot.Show(xlabel='suicides_no', ylabel='frequency', xlim = [500,6000], ylim =[0,6])
# pmf of age_35_54, others

age_35_54_pmf = thinkstats2.Pmf(suicide_data_age_35_54.suicides_no)

others_pmf = thinkstats2.Pmf(suicide_data_others.suicides_no)
# plotting pmf for age_35_54 and others

thinkplot.PrePlot(2, cols=2)

thinkplot.Hist(others_pmf, align='left',color='green', label = 'others')

thinkplot.Hist(age_35_54_pmf, align='right',color='red', label = 'age_35_54')

thinkplot.Config(title='suicides_no pmf',xlabel='suicides_no',ylabel='probability',axis=[0, 1000, 0, 0.02])

# plotting CDF

age_35_54_cdf = thinkstats2.Cdf(suicide_data_age_35_54.suicides_no, label='age_35_54')

thinkplot.Cdf(age_35_54_cdf)

thinkplot.Show(xlabel='suicides_no', ylabel='CDF',title='suicides_no CDF')
# plotting CDFs for age_35_54, others

age_35_54_cdf = thinkstats2.Cdf(suicide_data_age_35_54.suicides_no, label='age_35_54')

others_cdf = thinkstats2.Cdf(suicide_data_others.suicides_no, label='others')

thinkplot.PrePlot(2)

thinkplot.Cdfs([age_35_54_cdf, others_cdf])

thinkplot.Show(xlabel='suicides_no', ylabel='CDF', title='suicides_no CDF', legend = 'lower right' )
# plotting complementary CDF

thinkplot.Cdf(age_35_54_cdf, complement=True)

thinkplot.Show(xlabel='suicides_no',ylabel='CCDF',yscale='log',loc='upper right')
# plotting Normal probability plot



suicides_no = suicide_data_age_35_54.suicides_no.dropna()

mean, var = thinkstats2.TrimmedMeanVar(suicides_no, p=0.01)

std = np.sqrt(var)



xs = [-4, 4]

fxs, fys = thinkstats2.FitLine(xs, mean, std)

thinkplot.Plot(fxs, fys, linewidth=4, color='0.8')



xs, ys = thinkstats2.NormalProbability(suicides_no)

thinkplot.Plot(xs, ys, label='age_35_54')



thinkplot.Config(title='Normal probability plot',

                 xlabel='Standard deviations from mean',

                 ylabel='suicides_no')

# Lognormal model



def MakeNormalModel(suicides_no):

    """Plots a CDF with a Normal model.



    weights: sequence

    """

    cdf = thinkstats2.Cdf(suicides_no, label='suicides_no')



    mean, var = thinkstats2.TrimmedMeanVar(suicides_no)

    std = np.sqrt(var)

    print('n, mean, std', len(suicides_no), mean, std)



    xmin = mean - 4 * std

    xmax = mean + 4 * std



    xs, ps = thinkstats2.RenderNormalCdf(mean, std, xmin, xmax)

    thinkplot.Plot(xs, ps, label='model', linewidth=4, color='0.8')

    thinkplot.Cdf(cdf)
MakeNormalModel(suicides_no)

thinkplot.Config(title='age_35_54, log scale', xlabel='suicides_no',

                 ylabel='CDF', loc='lower right')
log_suicides_no = np.log10(suicides_no + 1)

MakeNormalModel(log_suicides_no)

thinkplot.Config(title='age_35_54, log scale', xlabel='log10 suicides_no',

                 ylabel='CDF', loc='upper left')
# Scatter plots

#SampleRows chooses a random subset of the data:



sample = thinkstats2.SampleRows(suicide_data, 5000)

suicides_no, gdp_per_capita = sample.suicides_no, sample.gdp_per_capita

population = sample.population
thinkplot.Scatter(suicides_no, gdp_per_capita)

thinkplot.Show(xlabel='suicides_no',

    ylabel=' gdp_per_capita',loc='upper right')

    
# jittering the data to reverse the effect of rounding off



suicides_no = thinkstats2.Jitter(suicides_no, 0.5)

gdp_per_capita = thinkstats2.Jitter(gdp_per_capita, 0.5)



# adding alpha parameter, which makes the points partly transparent

thinkplot.Scatter(suicides_no, gdp_per_capita, alpha = 0.1, s =10)

thinkplot.Show(xlabel='suicides_no',

    ylabel=' gdp_per_capita')

# compute the covariance of  suicides_no and gdp_per_capita 

thinkstats2.Cov(suicides_no, gdp_per_capita)
# compute the correlation of  suicides_no and gdp_per_capita 

thinkstats2.Corr(suicides_no, gdp_per_capita)
thinkstats2.SpearmanCorr(suicides_no, gdp_per_capita)
thinkstats2.Corr(suicides_no, np.log(gdp_per_capita))
thinkplot.Scatter(suicides_no, population)

thinkplot.Show(xlabel='suicides_no',

    ylabel=' population')

    
# jittering the data to reverse the effect of rounding off



suicides_no = thinkstats2.Jitter(suicides_no, 0.5)

population = thinkstats2.Jitter(population, 0.5)



# adding alpha parameter, which makes the points partly transparent

thinkplot.Scatter(suicides_no, population, alpha = 0.5, s =10)

thinkplot.Show(xlabel='suicides_no',

    ylabel='population')

                 
# compute the covariance of  suicides_no and population

thinkstats2.Cov(suicides_no, population)
# compute the pearson's correlation of  suicides_no and population

thinkstats2.Corr(suicides_no, population)
# compute Spearman correlation coeff.

thinkstats2.SpearmanCorr(suicides_no, population)
print(suicide_data_age_35_54.corr())
class DiffMeansPermute(thinkstats2.HypothesisTest):



    def TestStatistic(self, data):

        group1, group2 = data

        test_stat = abs(group1.mean() - group2.mean())

        return test_stat



    def MakeModel(self):

        group1, group2 = self.data

        self.n, self.m = len(group1), len(group2)

        self.pool = np.hstack((group1, group2))



    def RunModel(self):

        np.random.shuffle(self.pool)

        data = self.pool[:self.n], self.pool[self.n:]

        return data
data = suicide_data_age_35_54.suicides_no, suicide_data_others.suicides_no

ht = DiffMeansPermute(data)

pvalue = ht.PValue()

pvalue
import statsmodels.formula.api as smf



formula = 'suicides_no ~ population + HDI_for_year'

model = smf.ols(formula, data = suicide_data)

results = model.fit()

results.summary()