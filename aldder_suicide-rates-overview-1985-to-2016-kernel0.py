import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from tqdm import tqdm_notebook

print(os.listdir("../input"))

df = pd.read_csv('../input/master.csv')
df.tail(10)
df.info()
df.describe()
CONTINUOUS = ['suicides/100k pop','HDI for year','gdp_per_capita ($)']

CATEGORICALS = []

TO_DELETE = []
plt.figure()

sns.distplot(df['suicides/100k pop'])

plt.show()
low = np.quantile(df['suicides/100k pop'], .01)

high = np.quantile(df['suicides/100k pop'], .99)

print(low, high)



df['suicides/100k pop'] = df['suicides/100k pop'].apply(lambda x: x if x <= high and x >= low else np.NaN)

print(df.shape)

df.dropna(subset=['suicides/100k pop'], inplace=True)

print(df.shape)
print(f"Country number of unique values: {df['country'].nunique()}")



plt.figure(figsize=(20, 5))

sns.countplot(df['country'], order=df['country'].value_counts().index)

plt.title("Countries CountPlot")

plt.xticks(rotation=90)

plt.show()
print("Top 30 suicides countries")

top_suicides_countries = list(df.groupby(['country']).mean()['suicides/100k pop'].sort_values(ascending=False)[:30].index)

print(top_suicides_countries)



print("Bottom 30 sucides countries")

bottom_suicides_countries = list(df.groupby(['country']).mean()['suicides/100k pop'].sort_values(ascending=True)[:30].index)

print(bottom_suicides_countries)
fig, ax = plt.subplots(2,1,figsize=(15,15))

sns.boxplot(x='country', y='suicides/100k pop', 

            data=df.loc[df['country'].isin(top_suicides_countries)], ax=ax[0])

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)

ax[0].set_title('Top 30 suicides countries')

sns.boxplot(x='country', y='suicides/100k pop', 

            data=df.loc[df['country'].isin(bottom_suicides_countries)], ax=ax[1])

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)

ax[1].set_title('Bottom 30 suicides countries')

plt.tight_layout()

plt.show()
temp = pd.qcut(x=df.groupby(['country']).mean()['suicides/100k pop'].sort_values(ascending=False), 

               q=10, labels=range(10))

df['country_group'] = df['country'].map(temp)

df['country_group'] = pd.to_numeric(df['country_group'])



plt.figure(figsize=(10,6))

sns.boxplot(x=df['country_group'], y=df['suicides/100k pop'])

plt.title("Suicides frequencies per country group (the higher the most frequency suicides)")

plt.show()
CATEGORICALS.append('country_group')

TO_DELETE.append('country')
plt.figure(figsize=(20, 5))

sns.barplot(df['year'], df['suicides/100k pop'])

plt.title("Average suicides per year")

plt.xticks(rotation=90)

plt.show()
df = df.loc[df['year']!=2016]
temp = df.groupby(['year']).mean()[['suicides/100k pop','suicides_no']]



fig, ax1 = plt.subplots(figsize=(15,4))

temp['suicides/100k pop'].plot(c='b', ax=ax1)

ax2 = ax1.twinx()

temp['suicides_no'].plot(c='r', ax=ax2)

ax1.figure.legend()

ax1.set_title("Suicides trends over years")

plt.show()
plt.figure(figsize=(20, 5))

sns.barplot(pd.Series(pd.qcut(df['year'],10), name='year'), df['suicides/100k pop'])

plt.title("Suicides trend per grouped year (equally distribuited)")

plt.xticks(rotation=90)

plt.show()
df['year_range'] = pd.qcut(df['year'],10)
CATEGORICALS.append('year_range')

TO_DELETE.append('year')
plt.figure(figsize=(10,5))

sns.violinplot(x='sex', y='suicides/100k pop', data=df)

plt.title("Suicides distribution per sex")

plt.show()
plt.figure(figsize=(20,5))

sns.violinplot(x='sex', y='suicides/100k pop', hue='country_group', data=df[df['country_group']>=5])

plt.title("Suicides distribution per sex by top 5 suicide rates country group")

plt.show()
CATEGORICALS.append('sex')
print(f"Age number of unique values: {df['age'].nunique()}")

print(df['age'].value_counts())



display(df.groupby(['age']).agg(['sum','mean'])[['suicides_no','suicides/100k pop']])



fig, ax = plt.subplots(1, 2, figsize=(20, 5))

sns.barplot(df['age'], df['suicides/100k pop'], orient='v', ax=ax[0])

sns.barplot(df['age'], df['suicides_no'], orient='v', ax=ax[1])

fig.suptitle("Average suicides per age")

plt.show()



plt.figure(figsize=(20,6))

sns.boxplot(df['age'], df['suicides/100k pop'])

plt.title("Suicides distribution per age")

plt.show()
CATEGORICALS.append('age')
plt.figure()

sns.heatmap(df[['population','suicides_no','suicides/100k pop']].corr(), annot=True, cmap='RdBu_r')

plt.title("Suicides/Population correlation plot")

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(df['HDI for year'], orient='v', ax=ax[0])

sns.violinplot(df['HDI for year'], orient='v', color='darkorange', ax=ax[1])

fig.suptitle("'HDI for year' distribution")

plt.show()
df['HDI for year'].fillna((df['HDI for year'].mean()), inplace=True)
df[' gdp_for_year ($) '] = pd.to_numeric(df[' gdp_for_year ($) '].str.replace(',',''))
CONTINUOUS.append(' gdp_for_year ($) ')
plt.figure(figsize=(20,6))

sns.pointplot(x='year', y=' gdp_for_year ($) ', hue='country_group', 

              data=df.groupby(['country_group','year']).mean().reset_index())

plt.title('GDP per year by country group')

plt.show()
df[' gdp_for_year ($) '].isnull().any()
plt.figure(figsize=(15,6))

sns.pointplot(x='year', y='gdp_per_capita ($)', hue='country_group', 

              data=df.groupby(['country_group','year']).mean().reset_index())

plt.title("GPD per year by country group trends")

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(df[' gdp_for_year ($) '], df['suicides/100k pop'], hue=df['sex'], ax=ax[0])

ax[0].set_title("GPD per year / Suicides, by sex")

sns.scatterplot(df['gdp_per_capita ($)'], df['suicides/100k pop'], hue=df['sex'], ax=ax[1])

ax[1].set_title("GPD per capita / Suicides, by sex")

plt.show()
print(f"Generation number of unique values: {df['generation'].nunique()}")

print(df['generation'].value_counts())



plt.figure(figsize=(20,6))

sns.boxplot(df['generation'], df['suicides/100k pop'], hue=df['sex'])

plt.title("Generation per suicides distribution by sex")

plt.show()
CATEGORICALS.append('generation')
from scipy.stats import skew



for c in CONTINUOUS:

    skew_value = skew(df[c].dropna())

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    sns.distplot(df[c], ax=ax[0])

    ax[0].set_title(f'{c}, skew: {skew_value}')

    if abs(skew_value) >= 0.6:

        new_skew_value = skew(np.log1p(df[c]).dropna())

        new_serie = np.log1p(df[c])

        sns.distplot(new_serie, ax=ax[1])

        ax[1].set_title(f'{c}, skew: {new_skew_value}')

        df[c] = new_serie

    plt.show()
TO_DELETE.append('suicides_no')
TO_DELETE.append('country-year')
print("COLUMNS: ", list(df.columns))

print("CATEGORICALS: ", CATEGORICALS)

print("CONTINUOUS: ", CONTINUOUS)

print("TO_DELETE: ", TO_DELETE)
df.drop(TO_DELETE,1,inplace=True)
OBJECTIVE = ['suicides/100k pop']
dataset = df.merge(pd.get_dummies(df[CATEGORICALS], drop_first=True, columns=CATEGORICALS), 

                   left_index=True, right_index=True, how='inner').drop(CATEGORICALS,1)



print(dataset.shape)



dataset.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

from sklearn.metrics import r2_score, mean_squared_error
X = dataset.drop(OBJECTIVE,1).values

y = dataset[OBJECTIVE].values.ravel()



X.shape, y.shape
cv = KFold(n_splits=5, shuffle=True, random_state=33)

scaler = MinMaxScaler(feature_range=(0,1))

model = RandomForestRegressor(n_jobs=4, random_state=33, n_estimators=100)
predictions = pd.DataFrame(columns=['true','train','test'], index=dataset.index)

predictions['true'] = np.expm1(y)

for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):

    print(f"Cross-Validation iteration: {i+1}")

    X_train, y_train = X[train_idx], y[train_idx]

    X_test, y_test = X[test_idx], y[test_idx]

    

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    

    model.fit(X_train, y_train)

    

    predictions.iloc[train_idx, 1] = np.maximum(np.expm1(model.predict(X_train)), 0)

    predictions.iloc[test_idx, 2] = np.maximum(np.expm1(model.predict(X_test)), 0)



predictions = predictions.astype(np.float32)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

sns.regplot(x=predictions['true'], y=predictions['train'], color='b', scatter_kws={'alpha':0.2}, label='train', ax=ax[0])

sns.regplot(x=predictions['true'], y=predictions['test'], color='darkorange', scatter_kws={'alpha':0.2}, label='test', ax=ax[0])

ax[0].set_title("Regression plot")

sns.residplot(x=predictions['true'], y=predictions['train'], color='b', scatter_kws={'alpha':0.2}, label='train', ax=ax[1])

sns.residplot(x=predictions['true'], y=predictions['test'], color='darkorange', scatter_kws={'alpha':0.2}, label='test', ax=ax[1])

ax[1].set_title("Residuals plot")

plt.legend()

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(predictions['true']-predictions['train'], kde=False, label='train')

sns.distplot(predictions['true']-predictions['test'], kde=False, label='test')

plt.title("Residuals distribution")

plt.legend()

plt.show()
print(f'r2_score train: {r2_score(predictions["true"], predictions["train"]):.2f}')

print(f'r2_score test: {r2_score(predictions["true"], predictions["test"]):.2f}')

print(f'rmse train: {np.sqrt(mean_squared_error(predictions["true"], predictions["train"])):.2f}')

print(f'rmse test: {np.sqrt(mean_squared_error(predictions["true"], predictions["test"])):.2f}')