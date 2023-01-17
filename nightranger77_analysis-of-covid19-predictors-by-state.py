!pip install pingouin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

pd.set_option('display.max_columns', None)
df = pd.read_csv('../input/covid19-state-data/COVID19_state.csv')
df.head()
NUM_ROWS = 6
NUM_COLS = 4

cols_of_interest = df[df.columns[1:-1]]
col_names = cols_of_interest.columns

f, axes = plt.subplots(6, 4, figsize=(9, 9), sharex=False)
c = 0
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        g = sns.distplot(cols_of_interest[col_names[c]], ax=axes[i, j], axlabel=False)
        g.tick_params(left=False, bottom=False)
        g.set(yticklabels=[])
        g.set_title(col_names[c], fontsize=12)
        c += 1
f.tight_layout()
fig, ax = plt.subplots(ncols=2, figsize=(8,4))
pg.qqplot(df['Infected'], dist='expon', ax=ax[0], confidence=False)
ax[0].set_title('QQ Plot Infections in All States + DC')

df_no_ol = df[(df.State != 'New York') & (df.State != 'New Jersey')]

pg.qqplot(df_no_ol['Infected'], dist='expon', ax=ax[1])
ax[1].set_title('QQ Plot Infections with Outliers Removed')
[ax[i].set_xlabel('Exponential Distribution Quantiles') for i in range(0,len(ax))]
[ax[i].set_ylabel('Infected Sample Quantiles') for i in range(0,len(ax))]
fig.tight_layout()
ax = pg.qqplot(df_no_ol['Smoking Rate'], dist='norm')
ax.set_xlabel('Normal Distribution Quantiles')
ax.set_ylabel('Smoking Rate Sample Quantiles')
ax.set_title('QQ Plot Smoking Rate in All States + DC')
plt.show()
f, ax = plt.subplots(1, 3)
df['Deaths'].plot.box(grid=True, ax=ax[0])
df['Infected'].plot.box(grid=True, ax=ax[1])
df['Tested'].plot.box(grid=True, ax=ax[2])
f.tight_layout()
def make_corr_map(df, title='Correlation Heat Map', size=(9,7)):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr)) # for upper triangle
    f, ax = plt.subplots(figsize=size)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(title)
make_corr_map(df)
X = df[df.columns[4:-1]]
X_s = RobustScaler().fit_transform(X) # scale the data
rfr = RandomForestRegressor(max_depth=10).fit(X_s, df['Deaths'])
plt.bar(X.columns, rfr.feature_importances_)
plt.xticks(fontsize=10, rotation=90)
plt.title('Feature Importances based on Random Forest Regressor (y=Deaths)')
plt.show()
plt.scatter(np.log(df['Deaths'], where=df['Deaths'] != 0), np.log(df['Physicians']))
plt.xlabel('Deaths')
plt.ylabel('Physicians')
plt.title('Log Log Plot of Deaths vs Physicians')
plt.show()
socio = ['Population', 'Pop Density', 'Gini', 'Income', 'GDP', 'Unemployment', 'Sex Ratio',
         'Health Spending', 'Urban', 'Age 0-25', 'Age 26-54', 'Age 55+']
health = ['ICU Beds', 'Smoking Rate', 'Flu Deaths', 'Respiratory Deaths', 'Physicians', 'Hospitals', 
          'Health Spending', 'Pollution', 'Age 0-25', 'Age 26-54', 'Age 55+', 'Tested']

df_socio, df_health = df[socio], df[health]
make_corr_map(df_socio, 'Socioeconomic Predictors Correlation', (6,6))
X = df_socio.drop(['GDP', 'Gini', 'Income'], axis=1)
X_s = RobustScaler().fit_transform(X) # scale the data
rfr = RandomForestRegressor().fit(X_s, df['Deaths'])
plt.bar(X.columns, rfr.feature_importances_)
plt.xticks(fontsize=10, rotation=90)
plt.title('Feature Importances based on Random Forest Regressor')
plt.show()
plt.scatter(np.log(df['Deaths'], where=df['Deaths'] != 0), np.log(df['Population']))
plt.xlabel('Deaths')
plt.ylabel('Population')
plt.title('Log Log Plot of Deaths vs Population')
plt.show()
make_corr_map(df_health, 'Public Health Predictors Correlation', (6,6))
X = df_health.drop(['Age 0-25', 'Physicians', 'Tested'], axis=1)
X_s = RobustScaler().fit_transform(X) # scale the data
rfr = RandomForestRegressor().fit(X_s, df['Deaths'])
plt.bar(X.columns, rfr.feature_importances_)
plt.xticks(fontsize=10, rotation=90)
plt.title('Feature Importances based on Random Forest Regressor')
plt.show()
df['School Closure Date'] = pd.to_datetime(df['School Closure Date'])
df['Date Encoding'] = df['School Closure Date'].dt.day

fig, ax = plt.subplots(figsize=(12,5))
g = sns.stripplot(x='State', y='Date Encoding', data=df, ax=ax, color='blue', size=8);
ax.set_title('School Closures in March 2020 due to COVID19')
ax.set_xticklabels(labels=df['State'], rotation=90)
ax.set_ylabel('School Closure Date (March)')
ax.set_xlabel('')
plt.grid()
plt.show()