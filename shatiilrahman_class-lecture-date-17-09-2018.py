# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 19:00:12 2018
@author: TANMOY DAS
"""
import os

# greater than
import scipy.stats
mean_normal = 1830
standard_deviation_normal = 460
probability_norm_gt = scipy.stats.norm.sf(2750, mean_normal,standard_deviation_normal) # greater than
#in between
import scipy.stats as st
mean_normal = 1000
sd_normal = 100
probability_norm_lt1 =  st.norm.cdf(1000,mean_normal,sd_normal)  #cdf=Cumulative distribution function   , use for  less than
probability_norm_lt2 =  st.norm.cdf(790,mean_normal,sd_normal)
probability_norm_in_between = probability_norm_lt1 - probability_norm_lt2
print(probability_norm_in_between)
# less than
mean_normal = 1000
standard_deviation_normal = 100
probability_norm_lt = scipy.stats.norm.cdf(790, mean_normal,standard_deviation_normal)
print(probability_norm_lt)
#To find the variate for which the probability is given, let's say the 
#value which needed to provide a 98% probability, you'd use the 
#PPF Percent Point Function
probability_given = scipy.stats.norm.ppf(.98,100,12)
print(probability_given)
import scipy.stats as st
mean_poisson = 300/1000
# prob = poisson.cdf(x, mu); x= random variable; mu = mean 
probability_poisson = st.poisson.cdf(0, mean_poisson)
import scipy.stats as st
mean_poisson = 30*.05
# n is the number of years, 30 in this case.
# \pi is the probability a hurricane meeting the strength criteria comes ashore.
# \mu is the mean or expected number of storms in a 30-year period.
probability_poisson = 1 - st.poisson.cdf(0, mean_poisson)
print(probability_poisson)