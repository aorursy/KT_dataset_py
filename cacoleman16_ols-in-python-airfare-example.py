# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as SBI

import statsmodels.formula.api as sm1 # for OLS

import statsmodels.api as sm # for other models 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Read in airplanes with pandas

airplanes = pd.read_csv("../input/airplanes-2017/Airplanes.csv")
# look at the data characteristics

airplanes.describe()
## now take a look at what the data looks like



airplanes.head()
## it looks like there are a lot of "NaN"columns, let's drop them for now. and take another look at the format

airplanes = airplanes.iloc[:,0:15]

airplanes.head()
# it looks like there's one more. 

airplanes = airplanes.drop(airplanes.columns[9],1)

airplanes.head()
#Now let's try to visualize some relationships. For staters we are going to consdier a linear relationship between airfare(Y) and  Distance(in hundreds of miles)

airplanes.plot(x='Miles (in hundreds)', y = 'Fare', style= 'o')

plt.title('Airefare by Distance in Hundreds of Miles')

plt.xlabel('Distance')

plt.ylabel('Airefaire')

plt.show()
# looks like there's a postive relationship between distance and airefare. Makes sense. flying to Tokoyo from NYC is probably more expensive than flying to Paris!

#Now let's look at the distribution

plt.figure(figsize=(15,10))

plt.tight_layout()

SBI.distplot(airplanes['Fare'])
#before running OLS, im going to rename Miles (in hundreds) to distance, because the other name is too much to type 

airplanes= airplanes.rename( columns={'Miles (in hundreds)' : 'distance'}) 

# Now that we have an idea about the data, let's run OLS and see the results of our first model

OLS_results = sm.OLS.from_formula(formula= "Fare ~ distance",data= airplanes).fit()

print(OLS_results.summary()) # these are the results

      

                    

                    
## Get robust standard errors 

results_robust = OLS_results.get_robustcov_results(cov_type='HC1')

print(results_robust.summary())

## These match STATA reg, robust 
## Note, the last results are assuming homoskedasticity. But we don't beleive in that. So lets try with Huber white cov matrix

robust_results= sm.RLM.from_formula(formula= "Fare ~ distance",data= airplanes).fit()

print(robust_results.summary())
## Now, we'd probably imagine airefare is priced by a lot of things, not just the distance. For my next step, I'll add passengers to the equation. 

OLS_results2 = sm.OLS.from_formula(formula= "Fare ~ distance + Passengers",data= airplanes).fit()

results_robust2 = OLS_results2.get_robustcov_results(cov_type='HC1')

print(results_robust2.summary()) 

# For example, the time of year may be important, so let's add the quarter. 

OLS_results3 = sm.OLS.from_formula(formula= "Fare ~ distance + Passengers + Q1 + Q2 + Q3",data= airplanes).fit()

results_robust3 = OLS_results3.get_robustcov_results(cov_type='HC1')

print(results_robust3.summary()) 




