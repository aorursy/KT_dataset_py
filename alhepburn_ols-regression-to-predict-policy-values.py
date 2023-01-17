import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.formula.api as smf
insurance = pd.read_csv('/kaggle/input/insurance/insurance.csv')

insurance.head()
len(insurance)
insurance.isnull().sum() #No missing data, whatsoever!
insur_rel = insurance.corr()

f, ax = plt.subplots(figsize=(11,7))



sns.heatmap(insur_rel, square=True, linewidths=3, annot=True, cmap="YlGnBu")
# The feature 'Smoker', contains boolean data

# It may help us gain more info about our dataset



smoker = insurance['smoker']== 'yes'

non_smoker = insurance['smoker']=='no'



in_smoke = insurance[smoker]

in_non_smoke = insurance[non_smoker]

#Two new dataframes -- other than insurance



prob_smoke = len(in_smoke) / len(insurance)

prob_non_smoke = len(in_non_smoke) / len(insurance)

print("The percentage of policy holders smoking is {:.2f}%".format(prob_smoke * 100))

print("The percentage of policy holders not smoking is {:.2f}%".format(prob_non_smoke* 100))
in_smoke.head()
in_non_smoke.head()
# Policy Value Conditions 

high_charge = insurance['charges'] > 30000

low_charge = insurance['charges'] <= 30000
smoke_high = insurance[ smoker & high_charge]

smoke_low =  insurance[ smoker &  low_charge]

non_smoke_high = insurance[ non_smoker & high_charge]

non_smoke_low = insurance[ non_smoker & low_charge]
print("High Insurance policies for Smokers range from ${:.2f}".format(min(smoke_high['charges'])), " to ${:.2f}".format(max(smoke_high['charges'])))

print("Low Insurance policies for Smokers range from ${:.2f}".format(min(smoke_low['charges'])), " to ${:.2f}".format(max(smoke_low['charges'])))
print("High Insurance policies for Non-Smokers range from ${:.2f}".format(min(non_smoke_high['charges'])), " to ${:.2f}".format(max(non_smoke_high['charges'])))

print("Low Insurance policies for Non-Smokers range from ${:.2f}".format(min(non_smoke_low['charges'])), " to ${:.2f}".format(max(non_smoke_low['charges'])))
smoke_high.age.hist(bins=10)

plt.show()
smoke_low.age.hist(bins=10)

plt.show()
non_smoke_high.age.hist(bins=10)

plt.show()
non_smoke_low.age.hist(bins=10)

plt.show()
less_kid = insurance['children'] < 3

many_kid = insurance['children'] >= 3

#Children Conditions

small_fam = insurance[less_kid]

many_fam =  insurance[many_kid]
sns.catplot(x = 'smoker', y='children', data=insurance, height=7, kind='boxen', linewidth=2.0)

plt.title("Number of Children for Smokers/Non-Smokers")

#It looks like boxplot shapes change at 3 children, so I'll investigate
#Around 0 - 2 kids, it's pretty even until the groups reach 3 children

sns.catplot(x = 'smoker', y='children', data=many_fam, height=7, kind='boxen', linewidth=2.0)

plt.title("Smokers/Non-Smokers with 3 or more Children")
#Reg Line using 'Age' to predict 'Charges'

ac = smf.ols(formula = 'charges ~ age', data=insurance).fit()

ac.params
sns.lmplot(x = 'age', y = 'charges', data = insurance, hue="smoker", height=11)

plt.title("Age vs. Price")
ac.summary()
# Predict insurance values using list of ages, as a dataframe

work_age = pd.DataFrame({'age':[18, 30, 45, 61]})

print(ac.predict(work_age))
#Reg Line using 'BMI' to predict 'Charges' 

bc = smf.ols(formula = 'charges ~ bmi', data = insurance).fit()

bc.params
sns.lmplot(x = 'bmi', y = 'charges', data = insurance, hue="smoker", height=11)

plt.title("BMI vs. Price")
bc.summary()
bmi_cge = pd.DataFrame({'bmi':[20, 30, 40, 50]})

print(bc.predict(bmi_cge))