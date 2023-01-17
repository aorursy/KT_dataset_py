import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

sns.set(font_scale=2)

# https://docs.google.com/spreadsheets/d/18K135CiBoO4w85qgR9eQOCHornqQzoeGXrojF4tDl0Y/edit?usp=sharing
df = pd.read_csv('../input/nba_team_win.csv')
df['Year'] = df['Year'].map(lambda year: int(year.split('-')[0]) + 1)

# the % sign throws off statsmodels
df.columns = ['Year', 'Team', 'Record', 'Win']

df.head()
# generate variance grouped by year
var_df = df.groupby('Year')['Win'].apply(np.var).to_frame(name="Win")
var_df['Year'] = var_df.index
plt.figure(figsize=(45,8))
p = sns.barplot(var_df['Year'], var_df['Win'], color="b").set(xlabel='Year', ylabel='Variance of Win%', title="Variance of Win% By Year")
# if we look at the variance of win%, we can see it's basically flat
plt.figure(figsize=(16,8))
sns.regplot(data=var_df, x='Year', y='Win').set(xlabel='Year', ylabel='Variance of Win%',title="Variance of Win% By Year (with trendline)")
sm.ols(formula="Win ~ Year", data=var_df).fit().summary()