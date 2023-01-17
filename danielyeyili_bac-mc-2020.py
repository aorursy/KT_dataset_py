# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
definitions = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "definitions")

homeowner_income = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "homeowner income")

renter_income = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "renter income")

poverty_rate = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "poverty rate")

population = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "population")

price_single_family_home = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "price single family home")

price_condominium = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "price condominium")

single_family_sale_volume = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "single family sale volume")

condominium_sale_volume = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "condominium sale volume")

home_ownership_rate = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "home ownership rate")

FHA_VA_backed_loans = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "FHA VA backed loans")

LMI_borrowers = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "LMI borrowers")

housing_units = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "housing units")

rent_burden = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "rent burden")

choice_vouchers = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "choice vouchers")

percent_public_housing = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "% public housing")

serious_crime = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "serious crime")

english_scores = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "english scores")

math_scores = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "math scores")

subway_access = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "subway access")

park_access = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "park access")

percent_asian = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "percent asian")

percent_black = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "percent black")

percent_hispanic = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "percent hispanic")

percent_white = pd.read_excel("/kaggle/input/nychousingdata/PHASE-ONE_NYC-housing-data.xlsx", sheet_name = "percent white")
# what has been happening in the past 10-12 years?
#2 home ownership trends -> Sub-borough Area

home_ownership_rate.drop(['short_name','long_name','Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'home ownership rate'], axis = 1, inplace = True)
home_ownership_rate.head()
df = home_ownership_rate.describe().transpose()
df['year'] = df.index
df
import matplotlib.pyplot as plt



plt.plot(df['year'], df['mean']*100, color='black', marker='o')

plt.xlabel('year')

plt.ylabel('avg % home ownership')
#What is happening to home prices and rental costs?

#home prices - Community District - price single family home

price_single_family_home.drop(['Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'price single family home'], axis = 1, inplace = True)
price_single_family_home
df2 = price_single_family_home.describe().transpose()

df2.loc[:,'null'] = price_single_family_home.isnull().sum()

df2.loc[:,'median'] = price_single_family_home.median()

df2['year'] = df2.index

df2 = df2.round(2)

df2

# you have to be careful since data set has null median house sale values which can explain dip in 2001 mean housing values

# fill them later with average values for the specific property
plt.plot(df2['year'], df2['median'], color='black', marker='o')

plt.xlabel('year')

plt.ylabel('median single family home price')
#rental costs - Sub-Borough Area - Rent Burden

rent_burden.drop(['Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'rent burden'], axis = 1, inplace = True)
rent_burden
df3 = rent_burden.describe().transpose()

df3.loc[:,'null'] = rent_burden.isnull().sum()

df3.loc[:,'median'] = rent_burden.median()

df3['year'] = df3.index

df3 = df3.round(2)

df3
plt.plot(df3['year'], df3['mean']*100, color='black', marker='o')

plt.xlabel('year')

plt.ylabel('mean rent burden %')



# rent burden: The median percentage of gross, pre-tax income spent on gross rent by NYC renter households.
#Are incomes keeping pace with increased housing costs?

homeowner_income.head()
df4 = homeowner_income.describe().transpose()

df4.loc[:,'null'] = homeowner_income.isnull().sum()

df4.loc[:,'median'] = homeowner_income.median()

df4['year'] = df4.index

df4 = df4.round(2)

df4
plt.plot(df4['year'], df4['median'], color='black', marker='o')

plt.xlabel('year')

plt.ylabel('median income')
plt.plot(df2['year'], df2['median'], color='black', marker='o')

plt.xlabel('year')

plt.ylabel('median single family home price')