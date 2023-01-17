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







import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

sns.set(color_codes=True)
df = pd.read_csv("/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv", sep = ',')
df.shape
df.head()
#Changing banking_crisis to 0 and 1



df['banking_crisis'] = np.where(df.banking_crisis == 'crisis', 1, 0)
#Checking missing data



print(df.isnull().sum())
df.describe()
#Cross-Tables Economic Variables



print(pd.crosstab(df.systemic_crisis, df.domestic_debt_in_default,margins=True, margins_name="Total"),2*"\n")



print(pd.crosstab(df.systemic_crisis, df.sovereign_external_debt_default,margins=True, margins_name="Total"),2*"\n")



print(pd.crosstab(df.systemic_crisis, df.independence,margins=True, margins_name="Total"),2*"\n")



print(pd.crosstab(df.systemic_crisis, df.currency_crises,margins=True, margins_name="Total"),2*"\n")



print(pd.crosstab(df.systemic_crisis, df.inflation_crises,margins=True, margins_name="Total"),2*"\n")



print(pd.crosstab(df.systemic_crisis, df.banking_crisis,margins=True, margins_name="Total"))
# #Cross-Tables Economic Variables Bar Graph



pd.crosstab(df.domestic_debt_in_default,df.systemic_crisis, ).plot.bar(stacked = True);



pd.crosstab(df.sovereign_external_debt_default,df.systemic_crisis).plot.bar(stacked = True);



pd.crosstab( df.independence,df.systemic_crisis).plot.bar(stacked = True);



pd.crosstab(df.currency_crises,df.systemic_crisis).plot.bar(stacked = True);



pd.crosstab(df.inflation_crises,df.systemic_crisis).plot.bar(stacked = True);



pd.crosstab(df.banking_crisis,df.systemic_crisis).plot.bar(stacked = True);
#Cross-Tables Country



print(pd.crosstab(df.systemic_crisis, df.country,margins=True, margins_name="Total"))



pd.crosstab(df.country,df.systemic_crisis).plot.bar(stacked = True);
#Histogram Year



systemic_crisis_by_year = df.groupby('year').size()

plot_by_day = systemic_crisis_by_year.plot(title='Systemic crisis by year')

plot_by_day.set_xlabel('year')

plot_by_day.set_ylabel('Systemic_crisis_by_year');
# Measures of Spread of Variables grouped by systemic crisis



print("exch_usd Mean\n By",df.groupby('systemic_crisis')['exch_usd'].mean(),"\n")



print("inflation_annual_cpi Mean\n By",df.groupby('systemic_crisis')['inflation_annual_cpi'].mean(),"\n")



print("gdp_weighted_default Mean\n By",df.groupby('systemic_crisis')['gdp_weighted_default'].mean(),"\n")



print("\n","exch_usd Median\n By",df.groupby('systemic_crisis')['exch_usd'].median(),"\n")



print("inflation_annual_cpi Median\n By",df.groupby('systemic_crisis')['inflation_annual_cpi'].median(),"\n")



print("gdp_weighted_default Median\n By",df.groupby('systemic_crisis')['gdp_weighted_default'].median(),"\n")
#Blox-Spot



sns.boxplot(y='exch_usd', x= 'systemic_crisis', data=df, palette="colorblind").set_title('Blox-Spot exch_usd\n By Systemic_Crisis');
sns.boxplot(y='inflation_annual_cpi', x= 'systemic_crisis', data=df, palette="colorblind").set_title('Blox-Spot inflation_annual_cpi\n By Systemic_Crisis');
sns.boxplot(y='gdp_weighted_default', x= 'systemic_crisis', data=df, palette="colorblind").set_title("Blox-Spot gdp_weighted_default\n By Systemic_Crisis ");
# Correlations

plt.figure(figsize=(15,10))



c= df.corr()



sns.heatmap(c,cmap="RdBu_r",annot=True)



c