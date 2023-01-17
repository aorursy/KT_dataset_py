import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stat
#Ho:Mean = 200, H1:Mean =! 200 

ost=[211, 572, 558, 250, 478, 307, 184, 435, 460, 308, 188, 111, 676, 326, 142, 255, 205, 77, 190, 320, 407, 333, 488, 374, 409]



#One sample t-test

stat.ttest_1samp(ost, 200)



#Result: pvalue is < 0.05, so we are rejecting Ho i.e Mean=! 200 (May be mean is bigger or smaller than 200)

# np.mean(ost) # Mean is bigger than 200 i.e 330.56
#Raw data for 2 sample t-tset

Rating=[81, 77, 75, 74, 86, 90, 62, 73, 91, 98, 81, 85, 77, 78, 83, 90, 78, 76, 71, 80, 89, 64, 35, 68, 69, 55, 37, 57, 42, 49, 59, 58, 65, 71, 67, 58, 63, 68, 55, 57]

Hospital=['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']

tst=pd.DataFrame()

tst['Hospital']=Hospital

tst['Rating']=Rating

tst.head()
#2 sample t test when 2 sample are in same column

#Ho: Mean1 = Mean2, H1: Ho: Mean1 =! Mean2

stat.ttest_ind(*tst.groupby('Hospital')['Rating'].apply(lambda x:list(x)))



#instated of above use below code it more simple and understandalbe

#stat.ttest_ind(tst['Rating'][tst['Hospital']=='A'], tst['Rating'][tst['Hospital']=='B']) 



#Result: pvalue is < 0.05, so we are rejecting Ho i.e Mean1 =! Mean2 (2 sample's have diff means)

#tst['Rating'].groupby(tst['Hospital']).mean() # 2 sample means are not equal
#Ho: Mean1 = Mean2, H1: Ho: Mean1 =! Mean2

ptt=pd.DataFrame()

ptt['Before']=[68, 76, 74, 71, 71, 72, 75, 83, 75, 74, 76, 77, 78, 75, 75, 84, 77, 69, 75, 65]

ptt['After']=[67, 77, 74, 74, 69, 70, 71, 77, 71, 74, 73, 68, 71, 72, 77, 80, 74, 73, 72, 62]

ptt.head()



#Paired t-test

stat.ttest_rel(ptt['Before'], ptt['After'])



#Result: pvalue is < 0.05, so we are rejecting Ho i.e Mean1 =! Mean2 (2 sample's have diff means)

#ptt.mean() # 2 sample means are not equal
corr=pd.DataFrame()

Hydrogen=[0.1799999923, 0.2099999934, 0.2099999934, 0.2099999934, 0.2199999988, 0.22, 0.2299999893, 0.23, 0.2399999946, 0.24, 0.2499999851, 0.26, 0.27, 0.2799999714]

Porosity=[0.33, 0.4099999964, 0.4499999881, 0.5499999523, 0.4399999976, 0.2399999946, 0.4699999988, 0.6999999881, 0.7999999523, 0.2199999988, 0.8799999952, 0.719999969, 0.7499999404, 0.6999999881]

Strength=[0.8393054215, 1.122528279, 1.113086864, 1.1, 0.7070504494, 0.4975128406, 0.53, 0.5206216785, 0.1928728766, 0.54, 0.3908573703, 0.42, 0.1834787012, 0.24]

corr['Hydrogen']=Hydrogen

corr['Porosity']=Porosity

corr['Strength']=Strength

corr.head()
#Correlation coefficients

corr.corr()



#Correlation tests:

#Ho: No correlation b/w variables (Independent each other)

#H1: There is correlation b/w variables (Dependent each other)



#Pearson’s Correlation Coefficient

stat.pearsonr(corr['Hydrogen'], corr['Porosity'])

stat.pearsonr(corr['Hydrogen'], corr['Strength'])

stat.pearsonr(corr['Porosity'], corr['Strength'])



#Spearman’s Rank Correlation #only in this test we can pass 2D array like Dataframe 

stat.spearmanr(corr)[1]

stat.spearmanr(corr)[1]<0.05 # it gives pvalues are < 0.05 or not?



#Kendall’s Rank Correlation

stat.kendalltau(corr['Hydrogen'], corr['Porosity'])

stat.kendalltau(corr['Hydrogen'], corr['Strength'])

stat.kendalltau(corr['Porosity'], corr['Strength'])



#Result: Almost in every test pvalue < 0.05, so we are rejecting Ho i.e there is correlation b/w variables(Dependent each other)
#Ho: Sample follows normal distribution, H1: Sample not follows normal distribution

PercentFat=[15.2, 12.4, 15.4, 16.5, 15.9, 17.1, 16.9, 14.3, 19.1, 18.2, 18.5, 16.3, 20, 19.2, 12.3, 12.8, 17.9, 16.3, 18.7, 16.2]



#Shapiro-Wilk Test

stat.shapiro(PercentFat)



#D’Agostino’s K^2 Test

stat.normaltest(PercentFat)



#Anderson-Darling Test

stat.anderson(PercentFat)



#Result: In Anderson-Darling Test 'statistic' value is < 'critical_values' at 5% 'significance_level', so fail to reject Ho, i.e 

# sample follows normal distribution.

#And another two tests above pvalues are > 0.05, so we fail to reject Ho, i.e sample follows normal distribution
#Ho: two sample distribution are same (Two sample medians are identical)

#H1: two sample distribution are not same (Two sample medians are not identical)

MWU=pd.DataFrame()

BrandA=[35.6, 37, 34.9, 36, 36.6, 36.1, 35.8, 34.9, 38.8, 36.5, 34.9]

BrandB=[37.2, 39.7, 37.2, 38.8, 37.7, 36.4, 37.5, 40.5, 38.2, 37.5, 0]

MWU['BrandA']=BrandA

MWU['BrandB']=BrandB

MWU.head()



#Mann-Whitney U Test

stat.mannwhitneyu(MWU['BrandA'], MWU['BrandB'])

#Result: pvalue is < 0.05, so we are rejecting Ho i.e Median1 =! Median2 (2 sample's medians are not identical)
#Ho: the distributions of both samples are equal. (Medians are equal)

#H1: the distributions of both samples are not equal. (Medians are not equal)



#Wilcoxon Signed-Rank Test (used paired t-test data)

stat.wilcoxon(ptt['Before'], ptt['After'])



#Result: pvalue is < 0.05, so we are rejecting Ho i.e Median1 =! Median2 (2 sample's medians are not identical)
#Ho: All samples medians are equal

#H1: At least one sample median id diff from others



KWT=pd.DataFrame()

KWT['Beds']=[6, 37, 3, 17, 11, 30, 15, 16, 29, 25, 5, 34, 28, 41, 13, 40, 31, 9, 32, 39, 27, 31, 13, 35, 19, 4, 29, 0, 7, 5, 33, 17, 24]

KWT['Hospital']=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

KWT.head()



#Kruskal-Wallis H Test

stat.kruskal(KWT['Beds'][KWT['Hospital']==1], KWT['Beds'][KWT['Hospital']==2], KWT['Beds'][KWT['Hospital']==3])



#Result: pvalue is < 0.05, so we are rejecting Ho i.e medians are diff with each other
#Ho: the distributions of all samples are equal. (medians are equal)

#H1: the distributions of all samples are not equal. (medians are not equal)

FMT=pd.DataFrame()

FMT['Response']=[7.2, 9.4, 4.3, 11.3, 3.3, 4.2, 5.9, 6.2, 4.3, 10, 2.2, 6.3, 10.1, 8.2, 5.1, 6.5, 8.7, 6, 12.3, 11.1, 6, 12.1, 6.3, 4.3, 15.7, 18.3, 11.2, 19, 9.2, 10.5, 8.7, 14.3, 3.1, 18.8, 5.7, 20.2]

FMT['Company']=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

FMT['Advtype']=['direct-mail', 'direct-mail', 'direct-mail', 'direct-mail', 'direct-mail', 'direct-mail', 'direct-mail', 'direct-mail', 'direct-mail', 'direct-mail', 'direct-mail', 'direct-mail', 'magazine', 'magazine', 'magazine', 'magazine', 'magazine', 'magazine', 'magazine', 'magazine', 'magazine', 'magazine', 'magazine', 'magazine', 'newspaper', 'newspaper', 'newspaper', 'newspaper', 'newspaper', 'newspaper', 'newspaper', 'newspaper', 'newspaper', 'newspaper', 'newspaper', 'newspaper']

FMT.head()



##imp# Diff b/w Kruskal-Wallis H Test and Friedman Test is:

# Kruskal-Wallis H Test: works for one factor (like in above Eg 'Beds')

# Friedman Test: works for two factors (like in below Eg 'Response', 'Company')

# And in Friedman Test we will not get pvaues for each factor and interaction effect (we will get for overall)



#Friedman Test:

#without interaction effect

stat.friedmanchisquare(FMT['Response'][FMT['Advtype']=='direct-mail'], FMT['Company'][FMT['Advtype']=='direct-mail'], 

                       FMT['Response'][FMT['Advtype']=='magazine'], FMT['Company'][FMT['Advtype']=='magazine'], 

                       FMT['Response'][FMT['Advtype']=='newspaper'], FMT['Company'][FMT['Advtype']=='newspaper'])



#with interaction effect

stat.friedmanchisquare(FMT['Response'][FMT['Advtype']=='direct-mail'], FMT['Company'][FMT['Advtype']=='direct-mail'], 

                       FMT['Response'][FMT['Advtype']=='magazine'], FMT['Company'][FMT['Advtype']=='magazine'], 

                       FMT['Response'][FMT['Advtype']=='newspaper'], FMT['Company'][FMT['Advtype']=='newspaper'],

                       FMT['Response'][FMT['Advtype']=='newspaper']*FMT['Company'][FMT['Advtype']=='newspaper'])



#Result: In above two cases the pvalue < 0.05, so we rejecting Ho i.e distributions of all samples are not equal (Medians are not equal)
#Ho: All sample medians are equal, H1: All sample medians are not equal

MMT=pd.DataFrame()

MMT['Weight']=[22, 18, 22, 24, 16, 18, 19, 15, 21, 26, 16, 25, 17, 14, 28, 21, 19, 24, 23, 17, 18, 13, 20, 21, 18]

MMT['Temp']=[38, 38, 38, 38, 38, 38, 38, 42, 42, 42, 42, 42, 42, 46, 46, 46, 46, 46, 46, 50, 50, 50, 50, 50, 50]

MMT.head()



#Mood;s, Median Test

stat.median_test(MMT['Weight'][MMT['Temp']==38], MMT['Weight'][MMT['Temp']==42], MMT['Weight'][MMT['Temp']==46],

                MMT['Weight'][MMT['Temp']==50])



#Result: pvalue is > 0.05 so, we are fail to reject Ho i.e All sample medians are equal