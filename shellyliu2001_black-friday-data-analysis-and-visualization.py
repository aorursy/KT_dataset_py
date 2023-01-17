# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from scipy.stats import ttest_1samp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
##Uploading the dataset and summarizing
blackFriday = pd.read_csv("../input/BlackFriday.csv")
blackFriday = blackFriday.drop(columns = ["Product_Category_2", "Product_Category_3"])
##the head() function shows the first few rows of the dataset
blackFriday.head(10)
##the describe() function summarizes the dataset using statistical values
blackFriday.describe()
ages = blackFriday.Age.unique()
age_count = blackFriday['Age'].value_counts()
city = ["City A", "City B", "City C"]
city_count = blackFriday.groupby("City_Category").City_Category.count()

##positioning where each graph is going to be
plt.figure(figsize=(12, 5), dpi= 80, facecolor='w', edgecolor='k')

##creating a bar graph
plt.subplot(1, 2, 1)
plt.bar(ages, age_count)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age of Shoppers')

##creating a pie chart
plt.subplot(1, 2, 2)
plt.pie(city_count)
plt.axis("equal")
plt.legend(city, loc = "upper right")
plt.title("Pie Chart of Frequency of City Categories of Shoppers")

plt.subplots_adjust(wspace = 1)

##printing both charts out
plt.show()
price_by_category = blackFriday.groupby("Product_Category_1").Purchase.mean()

##the standard error of a single bar on the graph is a single standard deviation from the mean
yerr = blackFriday.groupby("Product_Category_1").Purchase.std()

plt.clf()
##configuring the size of the graph
plt.figure(figsize=(12, 7), dpi= 80, facecolor='w', edgecolor='k')

##creating the bar graph with error bars
plt.bar(range(1,19), price_by_category, yerr = yerr, capsize = 3, color = "orchid")
plt.xlabel('Product Category')
plt.ylabel('Purchase in dollars')
plt.title('Price Paid for purchases in each category')

##choosing the x and y axis labels
ax = plt.subplot()
ax.set_xticks(range(1, 19))
ax.set_xticklabels(range(1,19))

plt.show()
purchase = blackFriday.groupby(blackFriday.Occupation).Purchase.mean()

##finding equation line of best fit
lm_original = np.polyfit(range(0,21), purchase, 1)

# calculate the y values based on the co-efficients from the model
r_x, r_y = zip(*((i, i*lm_original[0] + lm_original[1]) for i in range(0,21)))


##choosing graph size and color
plt.figure(figsize=(12, 7), dpi= 80, facecolor='w', edgecolor='k')

##creating scatter plot
plt.scatter(range(0, 21), purchase)
plt.plot(r_x, r_y, color = "Red")

plt.xlabel('Occupation')
plt.ylabel('Purchase in dollars')
plt.title('Relationship between the occupation of the customer to the amount in dollars of the purchase')

plt.show()
a = blackFriday.loc[blackFriday['Age'] == '0-17'].Purchase
b = blackFriday.loc[blackFriday['Age'] == '18-25'].Purchase
c = blackFriday.loc[blackFriday['Age'] == '26-35'].Purchase
d = blackFriday.loc[blackFriday['Age'] == '36-45'].Purchase
e = blackFriday.loc[blackFriday['Age'] == '46-50'].Purchase
f = blackFriday.loc[(blackFriday['Age'] == '51-55')].Purchase
g = blackFriday.loc[(blackFriday['Age'] == '55+')].Purchase

ttest, pval = f_oneway(a, b, c, d, e, f, g)

print(pval)
tukey_result = pairwise_tukeyhsd(blackFriday.Purchase, blackFriday.Age, 0.05)
print(tukey_result)
tukey_plot = tukey_result.plot_simultaneous()
##creating contingency table
cont_table = pd.crosstab(blackFriday.Occupation, blackFriday.Product_Category_1)
print(cont_table)

##chi square testing
chi2, pval, dof, expected = chi2_contingency(cont_table)
print("{} is the p value for the chi square test.".format(pval))
male = blackFriday[blackFriday.Gender == "M"].Purchase
female = blackFriday[blackFriday.Gender == "F"].Purchase

ttest, pval = ttest_ind(male, female)

print("The p-value is: " + str(pval))
print("The t-statistic is: " + str(ttest))

print("The mean of male purchases is: " + str(male.mean()) + ", and the mean of female purchases is: " + str(female.mean()))
gender = ["Male", "Female"]

plt.figure(figsize=(12, 7), dpi= 80, facecolor='w', edgecolor='k')

plt.hist(male, 35)
plt.hist(female, 35)

plt.xlabel('Purchase in dollars')
plt.ylabel('Frequency')
plt.title('Difference in amount purchased between genders')
plt.legend(gender)

plt.show()
sample = 10000
male_sample = np.random.choice(male, size=sample, replace=True)
female_sample = np.random.choice(female, size=sample, replace=True)

ttest, m_pval = ttest_1samp(male_sample, male.mean())
ttest, f_pval = ttest_1samp(female_sample, female.mean())

print(m_pval)
print(f_pval)

print(male_sample.mean())
print(male.mean())

print(female_sample.mean())
print(female.mean())


plt.figure(figsize=(12, 7), dpi= 80, facecolor='w', edgecolor='k')

plt.hist(male_sample, 35,  alpha = 0.5)
plt.hist(female_sample, 35,  alpha = 0.5)

plt.xlabel('Purchase in dollars')
plt.ylabel('Frequency')
plt.title('Difference in amount purchased between genders(sampled)')
plt.legend(gender)

plt.show()