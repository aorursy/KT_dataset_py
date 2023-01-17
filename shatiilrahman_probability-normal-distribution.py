#Normal Distribution Math Problem 01 (grater than)
import scipy.stats as st
mean_normal = 1830
sd_normal = 460
probability_norm_gt =  st.norm.sf(2750,mean_normal,sd_normal)  #sf=sarvivle factor   , use for  grater than
print(probability_norm_gt)
#Normal Distribution Math Problem 02 (less than)

import scipy.stats as st
mean_normal = 1000
sd_normal = 100
probability_norm_lt =  st.norm.cdf(790,mean_normal,sd_normal)  #cdf=Cumulative distribution function   , use for  less than
print(probability_norm_lt)
#Normal Distribution Math Problem 03 (in between)

import scipy.stats as st
mean_normal = 1000
sd_normal = 100
probability_norm_lt1 =  st.norm.cdf(1000,mean_normal,sd_normal)  #cdf=Cumulative distribution function   , use for  less than
probability_norm_lt2 =  st.norm.cdf(790,mean_normal,sd_normal)
probability_norm_in_between = probability_norm_lt1 - probability_norm_lt2
print(probability_norm_in_between)