# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Age dependency data

age_depend = pd.read_csv("../input/health-economy-dataset/Age_Dependency (2).csv")

#GDP data

gdp = pd.read_csv("../input/health-economy-dataset/GDP.csv")

#GDP growth data

gdp_growth = pd.read_csv("../input/health-economy-dataset/GDP_Growth.csv")

#GPD per capita data

gdp_cap = pd.read_csv("../input/health-economy-dataset/gdp_per_capita.csv")

#Government health expenditure per capita

gov_health = pd.read_csv("../input/health-economy-dataset/gov_health_spend.csv")

#Health consumption expenditures per capita, U.S. dollars data

health_per_capita = pd.read_csv("../input/health-economy-dataset/health_cons_US.csv")



health_exp_gdp = pd.read_csv("../input/health-economy-dataset/health_exp_gdp.csv")

#Women survival data - Survival to age 65, female (% of cohort) 

women_survival = pd.read_csv("../input/health-economy-dataset/women_survival.csv")
gdp_cap.describe()
plt.plot(gdp_cap["Year"], gdp_cap["United States"])

plt.plot(gdp_cap["Year"], gdp_cap["India"])

plt.plot(gdp_cap["Year"], gdp_cap["Switzerland"])

plt.plot(gdp_cap["Year"], gdp_cap["Brazil"])

plt.plot(gdp_cap["Year"], gdp_cap["Iceland"])

plt.plot(gdp_cap["Year"], gdp_cap["Australia"])

plt.plot(gdp_cap["Year"], gdp_cap["China"])

plt.title('GDP per Capita through the years')

plt.xlabel('Years')

plt.ylabel('GDP per Capita')

plt.legend(['United States', 'India', 'Switzerland', 'Brazil', 'Iceland', 'Australia', "China"], loc='upper left')
gdp.describe()
plt.plot(gdp["Year"], gdp["United States"])

plt.plot(gdp["Year"], gdp["India"])

plt.plot(gdp["Year"], gdp["Switzerland"])

plt.plot(gdp["Year"], gdp["Brazil"])

plt.plot(gdp["Year"], gdp["Iceland"])

plt.plot(gdp["Year"], gdp["Australia"])

plt.plot(gdp["Year"], gdp["China"])

plt.legend(['United States', 'India', 'Switzerland', 'Brazil', 'Iceland', 'Australia', "China"], loc='upper left')

plt.title('GDP through the years')

plt.xlabel('Years')

plt.ylabel('GDP')
gdp_growth.describe()
plt.plot(gdp_growth["Year"], gdp_growth["United States"])

plt.plot(gdp_growth["Year"], gdp_growth["India"])

plt.plot(gdp_growth["Year"], gdp_growth["Switzerland"])

plt.plot(gdp_growth["Year"], gdp_growth["Brazil"])

plt.plot(gdp_growth["Year"], gdp_growth["Iceland"])

plt.plot(gdp_growth["Year"], gdp_growth["Australia"])

plt.plot(gdp_growth["Year"], gdp_growth["China"])

plt.legend(['United States', 'India', 'Switzerland', 'Brazil', 'Iceland', 'Australia', "China"], loc='upper left')

plt.title('GDP growth through the years')

plt.xlabel('Years')

plt.ylabel('GDP growth percentage')

plt.show()

health_per_capita
sns.set_style("darkgrid")

ax = sns.barplot(x="Health consumption expenditures per capita, U.S. dollars, PPP adjusted, 2017", y="Country", data=health_per_capita)
health_exp_gdp
sns.set()

ax = health_exp_gdp.set_index('Type').T.plot(kind='bar', stacked=True)

ax.set(xlabel='Country', ylabel='Health expenditures as GDP%')

ax.set_title("Total health expenditures as percent of GDP by public vs. private spending, 2016 by country")
#Government health expenditure per capita

gov_health.describe()
plt.plot(gov_health["Year"], gov_health["United States"])

plt.plot(gov_health["Year"], gov_health["India"])

plt.plot(gov_health["Year"], gov_health["Switzerland"])

plt.plot(gov_health["Year"], gov_health["Brazil"])

plt.plot(gov_health["Year"], gov_health["Iceland"])

plt.plot(gov_health["Year"], gov_health["Australia"])

plt.plot(gov_health["Year"], gov_health["China"])

plt.title('Domestic general government health expenditure per capita (current USD)')

plt.xlabel('Years')

plt.ylabel('Government health expenditure')

plt.legend(['United States', 'India', 'Switzerland', 'Brazil', 'Iceland', 'Australia', "China"], loc='upper left')



plt.show()
age_depend.describe()
plt.plot(age_depend["Year"], age_depend["United States"])

plt.plot(age_depend["Year"], age_depend["India"])

plt.plot(age_depend["Year"], age_depend["Switzerland"])

plt.plot(age_depend["Year"], age_depend["Brazil"])

plt.plot(age_depend["Year"], age_depend["Iceland"])

plt.plot(age_depend["Year"], age_depend["Australia"])

plt.plot(age_depend["Year"], age_depend["China"])

plt.title('Age Dependency Ratio')

plt.xlabel('Years')

plt.ylabel('Dependency Ratio')

plt.legend(['United States', 'India', 'Switzerland', 'Brazil', 'Iceland', 'Australia', "China"], loc='upper left')



plt.show()
print("Correlation between general government health expenditure per capita and age dependency ratio is given by")

print(age_depend.iloc[:,1:8].corrwith(gov_health.iloc[:,1:8] , axis = 0) )
fig = plt.figure(figsize=(16,6))



plt.subplot(1, 2, 1)

plt.plot(age_depend["Year"], age_depend["India"])

plt.plot(age_depend["Year"], age_depend["Brazil"])

plt.plot(age_depend["Year"], age_depend["China"])

plt.legend(['India', 'Brazil', 'Iceland', "China"], loc='upper left')

plt.xlabel('Years')

plt.ylabel('Age dependency ratio')



plt.subplot(1, 2, 2)

plt.plot(gov_health["Year"], gov_health["India"])

plt.plot(gov_health["Year"], gov_health["Brazil"])

plt.plot(gov_health["Year"], gov_health["China"])

plt.legend(['India', 'Brazil', 'Iceland', "China"], loc='upper left')

plt.xlabel('Years')

plt.ylabel('Domestic General Government Health expenditure')



plt.show()
age_depend
fig = plt.figure(figsize=(16,6))



plt.subplot(1, 2, 1)

plt.plot(age_depend["Year"], age_depend["India"])

plt.plot(age_depend["Year"], age_depend["United States"])

plt.plot(age_depend["Year"], age_depend["China"])

plt.legend(['India', 'United States', 'China'], loc='upper left')

plt.xlabel('Years')

plt.ylabel('Age Dependency Ratio')



plt.subplot(1, 2, 2)

plt.plot(gdp_cap["Year"], gdp_cap["India"])

plt.plot(gdp_cap["Year"], gdp_cap["United States"])

plt.plot(gdp_cap["Year"], gdp_cap["China"])

plt.legend(['India', 'United States', 'China'], loc='upper left')

plt.xlabel('Years')

plt.ylabel('GDP per capita')



plt.show()
