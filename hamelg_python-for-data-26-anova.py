import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats as stats
np.random.seed(12)



races =   ["asian","black","hispanic","other","white"]



# Generate random data

voter_race = np.random.choice(a= races,

                              p = [0.05, 0.15 ,0.25, 0.05, 0.5],

                              size=1000)



voter_age = stats.poisson.rvs(loc=18,

                              mu=30,

                              size=1000)



# Group age data by race

voter_frame = pd.DataFrame({"race":voter_race,"age":voter_age})

groups = voter_frame.groupby("race").groups



# Etract individual groups

asian = voter_age[groups["asian"]]

black = voter_age[groups["black"]]

hispanic = voter_age[groups["hispanic"]]

other = voter_age[groups["other"]]

white = voter_age[groups["white"]]



# Perform the ANOVA

stats.f_oneway(asian, black, hispanic, other, white)
import statsmodels.api as sm

from statsmodels.formula.api import ols



model = ols('age ~ race',                 # Model formula

            data = voter_frame).fit()

                

anova_result = sm.stats.anova_lm(model, typ=2)

print (anova_result)
np.random.seed(12)



# Generate random data

voter_race = np.random.choice(a= races,

                              p = [0.05, 0.15 ,0.25, 0.05, 0.5],

                              size=1000)



# Use a different distribution for white ages

white_ages = stats.poisson.rvs(loc=18, 

                              mu=32,

                              size=1000)



voter_age = stats.poisson.rvs(loc=18,

                              mu=30,

                              size=1000)



voter_age = np.where(voter_race=="white", white_ages, voter_age)



# Group age data by race

voter_frame = pd.DataFrame({"race":voter_race,"age":voter_age})

groups = voter_frame.groupby("race").groups   



# Extract individual groups

asian = voter_age[groups["asian"]]

black = voter_age[groups["black"]]

hispanic = voter_age[groups["hispanic"]]

other = voter_age[groups["other"]]

white = voter_age[groups["white"]]



# Perform the ANOVA

stats.f_oneway(asian, black, hispanic, other, white)
# Alternate method

model = ols('age ~ race',                 # Model formula

            data = voter_frame).fit()

                

anova_result = sm.stats.anova_lm(model, typ=2)

print (anova_result)
# Get all race pairs

race_pairs = []



for race1 in range(4):

    for race2  in range(race1+1,5):

        race_pairs.append((races[race1], races[race2]))



# Conduct t-test on each pair

for race1, race2 in race_pairs: 

    print(race1, race2)

    print(stats.ttest_ind(voter_age[groups[race1]], 

                          voter_age[groups[race2]]))
from statsmodels.stats.multicomp import pairwise_tukeyhsd



tukey = pairwise_tukeyhsd(endog=voter_age,     # Data

                          groups=voter_race,   # Groups

                          alpha=0.05)          # Significance level



tukey.plot_simultaneous()    # Plot group confidence intervals

plt.vlines(x=49.57,ymin=-0.5,ymax=4.5, color="red")



tukey.summary()              # See test summary