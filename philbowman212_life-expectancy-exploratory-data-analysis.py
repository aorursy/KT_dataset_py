import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

from scipy.stats.mstats import winsorize

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

import os

%matplotlib inline
df = pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')
df.head()
orig_cols = list(df.columns)

new_cols = []

for col in orig_cols:

    new_cols.append(col.strip().replace('  ', ' ').replace(' ', '_').lower())

df.columns = new_cols
df.rename(columns={'thinness_1-19_years':'thinness_10-19_years'}, inplace=True)
df.describe().iloc[:, 1:]
plt.figure(figsize=(15,10))

for i, col in enumerate(['adult_mortality', 'infant_deaths', 'bmi', 'under-five_deaths', 'gdp', 'population'], start=1):

    plt.subplot(2, 3, i)

    df.boxplot(col)
mort_5_percentile = np.percentile(df.adult_mortality.dropna(), 5)

df.adult_mortality = df.apply(lambda x: np.nan if x.adult_mortality < mort_5_percentile else x.adult_mortality, axis=1)

df.infant_deaths = df.infant_deaths.replace(0, np.nan)

df.bmi = df.apply(lambda x: np.nan if (x.bmi < 10 or x.bmi > 50) else x.bmi, axis=1)

df['under-five_deaths'] = df['under-five_deaths'].replace(0, np.nan)
df.info()
def nulls_breakdown(df=df):

    df_cols = list(df.columns)

    cols_total_count = len(list(df.columns))

    cols_count = 0

    for loc, col in enumerate(df_cols):

        null_count = df[col].isnull().sum()

        total_count = df[col].isnull().count()

        percent_null = round(null_count/total_count*100, 2)

        if null_count > 0:

            cols_count += 1

            print('[iloc = {}] {} has {} null values: {}% null'.format(loc, col, null_count, percent_null))

    cols_percent_null = round(cols_count/cols_total_count*100, 2)

    print('Out of {} total columns, {} contain null values; {}% columns contain null values.'.format(cols_total_count, cols_count, cols_percent_null))
nulls_breakdown()
df.drop(columns='bmi', inplace=True)
imputed_data = []

for year in list(df.year.unique()):

    year_data = df[df.year == year].copy()

    for col in list(year_data.columns)[3:]:

        year_data[col] = year_data[col].fillna(year_data[col].dropna().mean()).copy()

    imputed_data.append(year_data)

df = pd.concat(imputed_data).copy()
nulls_breakdown(df)
cont_vars = list(df.columns)[3:]

def outliers_visual(data):

    plt.figure(figsize=(15, 40))

    i = 0

    for col in cont_vars:

        i += 1

        plt.subplot(9, 4, i)

        plt.boxplot(data[col])

        plt.title('{} boxplot'.format(col))

        i += 1

        plt.subplot(9, 4, i)

        plt.hist(data[col])

        plt.title('{} histogram'.format(col))

    plt.show()

outliers_visual(df)
def outlier_count(col, data=df):

    print(15*'-' + col + 15*'-')

    q75, q25 = np.percentile(data[col], [75, 25])

    iqr = q75 - q25

    min_val = q25 - (iqr*1.5)

    max_val = q75 + (iqr*1.5)

    outlier_count = len(np.where((data[col] > max_val) | (data[col] < min_val))[0])

    outlier_percent = round(outlier_count/len(data[col])*100, 2)

    print('Number of outliers: {}'.format(outlier_count))

    print('Percent of data that is outlier: {}%'.format(outlier_percent))
for col in cont_vars:

    outlier_count(col)
def test_wins(col, lower_limit=0, upper_limit=0, show_plot=True):

    wins_data = winsorize(df[col], limits=(lower_limit, upper_limit))

    wins_dict[col] = wins_data

    if show_plot == True:

        plt.figure(figsize=(15,5))

        plt.subplot(121)

        plt.boxplot(df[col])

        plt.title('original {}'.format(col))

        plt.subplot(122)

        plt.boxplot(wins_data)

        plt.title('wins=({},{}) {}'.format(lower_limit, upper_limit, col))

        plt.show()
wins_dict = {}

test_wins(cont_vars[0], lower_limit=.01, show_plot=True)

test_wins(cont_vars[1], upper_limit=.04, show_plot=False)

test_wins(cont_vars[2], upper_limit=.05, show_plot=False)

test_wins(cont_vars[3], upper_limit=.0025, show_plot=False)

test_wins(cont_vars[4], upper_limit=.135, show_plot=False)

test_wins(cont_vars[5], lower_limit=.1, show_plot=False)

test_wins(cont_vars[6], upper_limit=.19, show_plot=False)

test_wins(cont_vars[7], upper_limit=.05, show_plot=False)

test_wins(cont_vars[8], lower_limit=.1, show_plot=False)

test_wins(cont_vars[9], upper_limit=.02, show_plot=False)

test_wins(cont_vars[10], lower_limit=.105, show_plot=False)

test_wins(cont_vars[11], upper_limit=.185, show_plot=False)

test_wins(cont_vars[12], upper_limit=.105, show_plot=False)

test_wins(cont_vars[13], upper_limit=.07, show_plot=False)

test_wins(cont_vars[14], upper_limit=.035, show_plot=False)

test_wins(cont_vars[15], upper_limit=.035, show_plot=False)

test_wins(cont_vars[16], lower_limit=.05, show_plot=False)

test_wins(cont_vars[17], lower_limit=.025, upper_limit=.005, show_plot=False)
plt.figure(figsize=(15,5))

for i, col in enumerate(cont_vars, 1):

    plt.subplot(2, 9, i)

    plt.boxplot(wins_dict[col])

plt.tight_layout()

plt.show()
wins_df = df.iloc[:, 0:3]

for col in cont_vars:

    wins_df[col] = wins_dict[col]
wins_df.describe()
wins_df.describe(include='O')
plt.figure(figsize=(15, 20))

for i, col in enumerate(cont_vars, 1):

    plt.subplot(5, 4, i)

    plt.hist(wins_df[col])

    plt.title(col)
plt.figure(figsize=(15, 25))

wins_df.country.value_counts(ascending=True).plot(kind='barh')

plt.title('Count of Rows by Country')

plt.xlabel('Count of Rows')

plt.ylabel('Country')

plt.tight_layout()

plt.show()
wins_df.year.value_counts().sort_index().plot(kind='barh')

plt.title('Count of Rows by Year')

plt.xlabel('Count of Rows')

plt.ylabel('Year')

plt.show()
plt.figure(figsize=(10, 5))

plt.subplot(121)

wins_df.status.value_counts().plot(kind='bar')

plt.title('Count of Rows by Country Status')

plt.xlabel('Country Status')

plt.ylabel('Count of Rows')

plt.xticks(rotation=0)



plt.subplot(122)

wins_df.status.value_counts().plot(kind='pie', autopct='%.2f')

plt.ylabel('')

plt.title('Country Status Pie Chart')



plt.show()
wins_df[cont_vars].corr()
mask = np.triu(wins_df[cont_vars].corr())

plt.figure(figsize=(15,6))

sns.heatmap(wins_df[cont_vars].corr(), annot=True, fmt='.2g', vmin=-1, vmax=1, center=0, cmap='coolwarm', mask=mask)

plt.ylim(18, 0)

plt.title('Correlation Matrix Heatmap')

plt.show()
sns.lineplot('year', 'life_expectancy', data=wins_df, marker='o')

plt.title('Life Expectancy by Year')

plt.show()
wins_df.year.corr(wins_df.life_expectancy)
years = list(wins_df.year.unique())

years.sort()
yearly_le = {}

for year in years:

    year_data = wins_df[wins_df.year == year].life_expectancy

    yearly_le[year] = year_data
for year in years[:-1]:

    print(10*'-' + str(year) + ' to ' + str(year+1) + 10*'-')

    print(stats.ttest_ind(yearly_le[year], yearly_le[year+1], equal_var=False))
wins_df.groupby('status').life_expectancy.agg(['mean'])
developed_le = wins_df[wins_df.status == 'Developed'].life_expectancy

developing_le = wins_df[wins_df.status == 'Developing'].life_expectancy

stats.ttest_ind(developed_le, developing_le, equal_var=False)
wins_df_cols = list(wins_df.columns)

interested_vars = [wins_df_cols[2]]

for col in wins_df_cols[4:]:

    interested_vars.append(col)
wins_df[interested_vars].groupby('status').agg('mean')
developed_df = wins_df[wins_df.status == 'Developed']

developing_df = wins_df[wins_df.status == 'Developing']

for col in interested_vars[1:]:

    print(5*'-' + str(col) + ' Developed/Developing t-test comparison' + 5*'-')

    print('p-value=' +str(stats.ttest_ind(developed_df[col], developing_df[col], equal_var=False)[1]))
feat_df = wins_df.join(pd.get_dummies(wins_df.status)).drop(columns='status').copy()
feat_df.iloc[:, 2:].corr().iloc[:, -2:].T
feat_df.drop(columns=['country', 'year'], inplace=True)
def feat_heatmap():

    mask = np.triu(feat_df.corr())

    plt.figure(figsize=(15,6))

    sns.heatmap(feat_df.corr(), annot=True, fmt='.2g', vmin=-1, vmax=1, center=0, cmap='coolwarm', mask=mask)

    plt.ylim(len(feat_df.columns), 0)

    plt.title('Features Correlation Matrix Heatmap')

    plt.show()

feat_heatmap()
feat_df.drop(columns=['infant_deaths', 'percentage_expenditure','polio','thinness_10-19_years','schooling','Developing'], inplace=True)
feat_df.drop(columns=['population'], inplace=True)
feat_heatmap()
pca_df = feat_df.drop(columns='Developed').copy()
pca_df.drop(columns='life_expectancy', inplace=True)
len(pca_df.columns)
X = scale(pca_df)

sklearn_pca = PCA()

Y = sklearn_pca.fit_transform(X)

print('Explained variance by Principal Components:', sklearn_pca.explained_variance_ratio_)

print('Eigenvalues:', sklearn_pca.explained_variance_)
plt.plot(sklearn_pca.explained_variance_)

plt.show()

print('PC1 Explained Variance:', str(round(sklearn_pca.explained_variance_ratio_[0]*100, 2))+'%')
feat_heatmap()