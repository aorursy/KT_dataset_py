import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import modules

import pandas as pd

import matplotlib.pyplot as plt
# Read datasets/yearly_deaths_by_clinic.csv into yearly

yearly= pd.read_csv('../input/survey-data/yearly_deaths_by_clinic.csv')
print(yearly)
yearly['proportion_deaths']=yearly['deaths']/yearly['births']
# Extract clinic 1 data into yearly1 and clinic 2 data into yearly2

yearly1 = yearly.loc[:5,:]

yearly2 = yearly.loc[6:,:]
# This makes plots appear in the notebook

%matplotlib inline



# Plot yearly proportion of deaths at the two clinics

plt.figure()

ax = yearly1.plot(x="year" , y="proportion_deaths" ,label="cilinic1")

yearly2.plot(x="year" , y="proportion_deaths" ,label="colinic2", ax=ax)

ax.set_ylabel("proportion of deaths per no. births")
# Read datasets/monthly_deaths.csv into monthly

monthly = pd.read_csv('../input/survey-data/monthly_deaths.csv' ,parse_dates=["date"]) 



# Calculate proportion of deaths per no. births

monthly['proportion_deaths']=monthly['deaths']/monthly['births']



# Print out the first rows in monthly

monthly.head()
# Plot monthly proportion of deaths

plt.figure()

ax = monthly.plot(x="date" , y="proportion_deaths" ,label="cilinic1")

ax.set_ylabel("proportion of deaths")
# Date when handwashing was made mandatory

handwashing_start = pd.to_datetime('1847-06-01')



# Split monthly into before and after handwashing_start

before_washing = monthly[monthly["date"] < handwashing_start]

after_washing = monthly[monthly["date"] >= handwashing_start]



# Plot monthly proportion of deaths before and after handwashing

plt.figure()

ax = before_washing.plot(x='date' , y="proportion_deaths" ,label="cilinic1")



after_washing.plot(x='date' , y="proportion_deaths" ,label="colinic2", ax=ax)

ax.set_ylabel("proportion of deaths")
# Difference in mean monthly proportion of deaths due to handwashing

before_proportion = before_washing['proportion_deaths']

after_proportion = after_washing['proportion_deaths']

mean_diff = after_proportion.mean() - before_proportion.mean()

mean_diff
# A bootstrap analysis of the reduction of deaths due to handwashing

boot_mean_diff = []

for i in range(3000):

    boot_before = before_proportion.sample(frac=1 ,replace=True)

    boot_after = after_proportion.sample(frac=1 ,replace=True)

    boot_mean_diff.append( boot_after.mean() - boot_before.mean() )



# Calculating a 95% confidence interval from boot_mean_diff 

confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])

confidence_interval
