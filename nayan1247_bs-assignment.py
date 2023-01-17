# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

df_quarter = pd.read_excel("../input/gdsbsnew/BS_data.xlsx",sheet_name="Consolidated Data1")
df_year = pd.read_excel("../input/gdsbsnew/BS_data.xlsx",sheet_name="ConsolidatedData2")

df_quarter.head()
## Preprocess and Combine Data

# Transpose df_quarter
df_quarter_processed = pd.melt(df_quarter, id_vars=["Geography", "Category","Data Type"],var_name='Quarter', value_name='values')
df_quarter_processed['Year'] = df_quarter_processed['Quarter'].str[3:]
df_quarter_processed= df_quarter_processed.pivot(values='values',index=['Geography','Data Type','Quarter','Year'],columns='Category').reset_index()
df_quarter_processed['Year'] = df_quarter_processed['Year'].astype(int)

# Transpose df_year
df_year_processed = pd.melt(df_year, id_vars=["Geography", "Category","Data Type"],var_name='Year', value_name='values')
df_year_processed= df_year_processed.pivot(values='values',index=['Geography','Data Type','Year'],columns='Category').reset_index()
df_year_processed['Year'] = df_year_processed['Year'].astype(int)

# Combined the two data
df_combined = pd.merge(df_quarter_processed,df_year_processed, on=['Geography','Data Type', 'Year'])
df_combined.replace(to_replace ="-", value = None,inplace=True) 
df_combined['Consumer Confidence Index'] = pd.to_numeric(df_combined['Consumer Confidence Index'])
df_combined['Real GDP Growth'] = pd.to_numeric(df_combined['Real GDP Growth'])
df_combined['Ease of Doing Business Ranking'] = pd.to_numeric(df_combined['Ease of Doing Business Ranking'])
df_combined['Employment Rate'] = pd.to_numeric(df_combined['Employment Rate'])
df_combined['Total Population'] = pd.to_numeric(df_combined['Total Population'])
df_combined['Inflation'] = pd.to_numeric(df_combined['Inflation'])



df_combined.columns
# Timeseries
import seaborn as sns
import matplotlib.pyplot as plt


## For a Country
g = sns.relplot(x="Quarter", y="Real GDP Growth", kind="line", data=df_combined.query("Geography=='India'"))
g.fig.autofmt_xdate()
g.fig.set_size_inches(15,5)
g.set(title='GDP Growth for India', xlabel='Quarter of the Year', ylabel='GDP')




## Metric For a group.. Eg Developed

g = sns.boxplot(x="Quarter", y="Real GDP Growth", data=df_combined.query("`Data Type`=='Developed'"),color='lightgreen')
g.set(title='GDP Growth for Developed Countries', xlabel='Quarter of the Year', ylabel='GDP')
g.set_xticklabels(g.get_xticklabels(),rotation=30)


## Metric mean for a group.. Eg Developed

df_combined_temp = df_combined.groupby(['Data Type','Quarter']).agg({'Real GDP Growth':np.mean}).reset_index()

g = sns.relplot(x="Quarter", y="Real GDP Growth", kind="line", data=df_combined_temp.query("`Data Type`=='Developed'"))
g.fig.autofmt_xdate()
g.fig.set_size_inches(15,5)
g.set(title='GDP Growth for Developed Countries', xlabel='Quarter of the Year', ylabel='GDP')

## Metric mean for a group.. All Together

df_combined_temp = df_combined.groupby(['Data Type','Quarter']).agg({'Real GDP Growth':np.mean}).reset_index() # Use this construct to aggregate data to mean/median

g = sns.relplot(x="Quarter", y="Real GDP Growth", kind="line", data=df_combined_temp,hue='Data Type')
g.fig.autofmt_xdate()
g.fig.set_size_inches(20,5)
g.set(title='GDP Growth for Developed Countries', xlabel='Quarter of the Year', ylabel='GDP')


# Scatter Plot Eg: GDP vs Ease of Business Combined

g = sns.lmplot(x="Ease of Doing Business Ranking", y="Real GDP Growth", hue="Data Type", data=df_combined);
g.fig.autofmt_xdate()
g.fig.set_size_inches(15,5)


# Scatter Plot Eg: GDP vs Ease of Business Separate

g = sns.lmplot(x="Ease of Doing Business Ranking", y="Real GDP Growth",  data=df_combined.query("`Data Type`=='Developed'"));
g.fig.autofmt_xdate()
g.fig.set_size_inches(15,5)


# Linear Regression for Developed Ref : https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/

import statsmodels.formula.api as smf

df_combined_copy = df_combined.copy()
df_combined_copy.dropna(inplace=True)
df_combined_copy = df_combined_copy.query("`Data Type`=='Developed'")

# Model initialization
X = df_combined_copy.query("`Data Type`=='Developed'")[['Consumer Confidence Index', 'Inflation', 
       'Ease of Doing Business Ranking', 'Employment Rate',
       'Total Population']]
y = df_combined_copy.query("`Data Type`=='Developed'")['Real GDP Growth']

lm1 = smf.ols(formula=' Q("Real GDP Growth") ~Q("Consumer Confidence Index") + Q("Inflation") +Q("Ease of Doing Business Ranking") + Q("Employment Rate") +Q("Total Population")', data=df_combined_copy).fit()
# print the coefficients
lm1.params
### STATSMODELS ###

# print the p-values for the model coefficients
lm1.pvalues



# print the R-squared value for the model
lm1.rsquared
