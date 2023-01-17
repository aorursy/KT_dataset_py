import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
diet = pd.read_csv('../input/dietstudy.csv')

diet.head()
print('pre diet weight: ', diet.wgt0.mean())

print('pre diet triglyceride level: ', diet.tg0.mean())
print('post 1 month diet weight: ', diet.wgt1.mean())

print('post 1 month diet triglyceride level: ', diet.tg1.mean())
month1_wt = stats.ttest_rel(a=diet.wgt0, b=diet.wgt1)

month1_tg = stats.ttest_rel(a=diet.tg0, b=diet.tg1)
print(month1_wt.pvalue < 0.05)

print(month1_tg.pvalue < 0.05)
print('post 2 month diet weight: ', diet.wgt2.mean())

print('post 2 month diet triglyceride level: ', diet.tg2.mean())
month2_wt = stats.ttest_rel(a=diet.wgt0, b=diet.wgt2)

month2_tg = stats.ttest_rel(a=diet.tg0, b=diet.tg2)
print(month2_wt.pvalue < 0.05)

print(month2_tg.pvalue < 0.05)
print('post 3 months diet weight: ', diet.wgt3.mean())

print('post 3 months diet triglyceride level: ', diet.tg3.mean())
month3_wt = stats.ttest_rel(a=diet.wgt0, b=diet.wgt3)

month3_tg = stats.ttest_rel(a=diet.tg0, b=diet.tg3)
print(month3_wt.pvalue < 0.05)

print(month3_tg.pvalue < 0.05)
print('post 4 months diet weight: ', diet.wgt4.mean())

print('post 4 months diet triglyceride level: ', diet.tg4.mean())
month4_wt = stats.ttest_rel(a=diet.wgt0, b=diet.wgt4)

month4_tg = stats.ttest_rel(a=diet.tg0, b=diet.tg4)
print(month4_wt.pvalue < 0.05)

print(month4_tg.pvalue < 0.05)
credit = pd.read_csv('../input/creditpromo.csv')

credit.head()
standard = credit.dollars[credit['insert']=='Standard']

new_promo = credit.dollars[credit['insert']=='New Promotion']
print('Spend of customer with standard promo: ', standard.mean())

print('Spend of customer with new promo: ', new_promo.mean())


credit_equalv = stats.ttest_ind(a= standard, b=new_promo, equal_var=True)

credit_unequalv = stats.ttest_ind(a= standard, b=new_promo, equal_var=False)
credit_equalv.statistic - credit_unequalv.statistic
credit_equalv.pvalue < 0.05
pol = pd.read_csv('../input/pollination.csv')

pol.head()
pol.Seed_Yield_Plant.mean()

seed_yield = stats.ttest_1samp(a=pol.Seed_Yield_Plant, popmean=200)
seed_yield.pvalue < 0.05
natural_pol = pol.loc[pol.Group == 'Natural']



hand_pol = pol.loc[pol.Group == 'Hand']

natural_pol.Seedling_length.mean()
hand_pol.Seedling_length.mean()
seed_length = stats.ttest_ind(a=natural_pol.Seedling_length, b=hand_pol.Seedling_length)

seed_length
seed_length.pvalue
print(natural_pol.Fruit_Wt.mean())

print(hand_pol.Fruit_Wt.mean())



print(natural_pol.Seed_Yield_Plant.mean())

print(hand_pol.Seed_Yield_Plant.mean())
fruit_wt = stats.ttest_ind(a=natural_pol.Fruit_Wt, b=hand_pol.Fruit_Wt)

seed_yield = stats.ttest_ind(a=natural_pol.Seed_Yield_Plant, b=hand_pol.Seed_Yield_Plant)
fruit_wt.pvalue
seed_yield.pvalue
dvd = pd.read_csv('../input/dvdplayer.csv')

dvd.head()
dvd.agegroup.value_counts()
grp1 = dvd.dvdscore.loc[dvd.agegroup == 'Under 25']

grp2 = dvd.dvdscore.loc[dvd.agegroup == '25-34']

grp3 = dvd.dvdscore.loc[dvd.agegroup == '35-44']

grp4 = dvd.dvdscore.loc[dvd.agegroup == '45-54']

grp5 = dvd.dvdscore.loc[dvd.agegroup == '55-64']

grp6 = dvd.dvdscore.loc[dvd.agegroup == '65 and over']
dvd_anova = stats.f_oneway(grp1, grp2, grp3, grp4, grp5, grp6)

dvd_anova
sample_data = pd.read_csv('../input/sample_survey.csv')

sample_data.head(2)
sample_data.info()
wrk_mar_xtab = pd.crosstab(sample_data.wrkstat, sample_data.marital, margins=True)

wrk_mar_xtab
wrk_mar_test = stats.chi2_contingency(observed=wrk_mar_xtab)

wrk_mar_test
degree_mar_xtab = pd.crosstab(sample_data.degree, sample_data.marital, margins=True)

degree_mar_xtab
degree_mar_test = stats.chi2_contingency(observed=degree_mar_xtab)

degree_mar_test[1]
happy_mar_xtab = pd.crosstab(sample_data.happy, sample_data.marital, margins=True)

happy_mar_xtab
happy_mar_test = stats.chi2_contingency(observed=happy_mar_xtab)

happy_mar_test
happy_income_xtab = pd.crosstab(sample_data.happy, sample_data.income, margins=True)

happy_income_xtab
happy_income_test = stats.chi2_contingency(observed=happy_income_xtab)

happy_income_test