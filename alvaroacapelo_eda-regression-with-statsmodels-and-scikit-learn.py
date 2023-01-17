# basic libraries for data acquisition, handling and visualization

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



# libraries for modelling

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import explained_variance_score, r2_score

from sklearn.metrics import mean_absolute_error, mean_squared_error
os.listdir('../input')
diam = pd.read_csv('../input/diamonds.csv', index_col=0)
diam.head()
data = diam.copy()
diam.sample(10)
# check fror null values on target variable

diam.price.isnull().any()
print("We have {:.0f} priced diamonds.".format(diam.price.count()))
diam.price.describe()
f, ax = plt.subplots(ncols=2, figsize=(12,6))

sns.boxplot(y='price', data=diam, ax=ax[0])

sns.boxenplot(y='price', data=diam, ax=ax[1])

plt.show()
f, ax = plt.subplots(ncols=2, figsize=(10,5))

sns.kdeplot(diam.price, color='b', shade=True, ax=ax[0])

sns.kdeplot(diam.price, color='r', shade=True, bw=100, ax=ax[1])



ax[0].set_title('KDE')

ax[1].set_title('KDE, bandwidth = 100')



plt.show()
diam.columns
diam.info()
for col in ['cut', 'color', 'clarity']:

    print("Column : {}".format(col))

    print(diam[col].value_counts())

    print()
# turn to 'categorical' data type and order

cut_dtype = pd.api.types.CategoricalDtype(

    categories=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], 

    ordered=True)

data['cut'] = diam.cut.astype(cut_dtype)
data.cut.head()
# turn to 'categorical' data type and order

color_dtype = pd.api.types.CategoricalDtype(

    categories=['J', 'I', 'H', 'G', 'F', 'E', 'D'], 

    ordered=True)

data['color'] = diam.color.astype(color_dtype)
data.color.head()
clar_dtype = pd.api.types.CategoricalDtype(

    categories=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], 

    ordered=True)

data['clarity'] = diam.clarity.astype(clar_dtype)
data.clarity.head()
diam.carat.describe()
ax = sns.boxplot(y='carat', data=diam)
print(diam[diam.carat < 3].sample(5))

print()

print(diam[diam.carat > 3].sample(5))
print("At least one null entry? {}".format(diam.depth.isnull().any()))
diam.depth.describe()
ax = sns.boxplot(y='depth', data=diam)
print("# diamonds, depth > 65:", diam[diam.depth > 65].depth.count())

print()

print('Sample')

print(diam[diam.depth > 65].sample(10))
print("# diamonds, depth < 58:", diam[diam.depth < 58].depth.count())

print()

print('Sample')

print(diam[diam.depth < 58].sample(10))
print("At least one null entry? {}".format(diam.table.isnull().any()))
diam.table.describe()
ax = sns.boxplot(y='table', data=diam)
print("# diamonds, table% > 65:", diam[diam.table > 65].table.count())

print()

print("Sample:")

print(diam[diam.table > 65].sample(10))
print("# diamonds, table% < 50:", diam[diam.table < 50].table.count())

print()

print("Sample:")

print(diam[diam.table < 50].sample(4))
print("Any null entry for \'x\'? {}".format(diam.x.isnull().any()))

print("Any null entry for \'y\'? {}".format(diam.y.isnull().any()))

print("Any null entry for \'z\'? {}".format(diam.z.isnull().any()))
diam.loc[:, ['x', 'y', 'z']].describe()
f, ax = plt.subplots(ncols=2, figsize=(10,5))

sns.scatterplot(x=diam.x, y=diam.y, ax=ax[0])

ax[0].set_title("y vs x")



sns.scatterplot(x=diam.x, y=diam.y, ax=ax[1])

ax[1].set_xlim(0, 15)

ax[1].set_ylim(0, 15)

ax[1].set_title("y vs x - Zoomed in")



plt.show()
ax = sns.scatterplot(x='carat', y='y', data=diam)
diam[(diam.y > 10) | (diam.z > 10)]
diam[(diam.x < 1) | (diam.y < 1) | (diam.z < 1)].sample(10)
absurdx_i = ((diam.x == 0) | (diam.x > 15)) & (diam.y != 0)

data.loc[absurdx_i, 'x'] = diam.loc[absurdx_i, 'y']
# create Boolean mask to subset DataFrame

absurdxy_i = ((data.x == 0) | (data.y == 0))



# compute mean value of x

mean_x = np.mean(data.x)



# substitute on the dataFrame

data.loc[absurdxy_i, ['x', 'y']] = data.loc[absurdxy_i, ['x', 'y']].replace(0, mean_x)
absurdy_i = (((data.y > 15) | (data.y == 0)) & (data.x != 0))

data.loc[absurdy_i, 'y'] = data.loc[absurdy_i, 'x']
data[(data.z == 0) | (data.z > 15)]
# find rows where z is absurd

absurd_z = ((diam.z == 0) | (diam.z > 15))



# define function to calculate z

calc_z = lambda row: (row['depth']/100) * (row['x'] + row['y'])/2



# apply on dataframe

data.loc[absurd_z, 'z'] = data.loc[absurd_z, :].apply(calc_z, axis=1)
data[['x', 'y', 'z']].describe()
data.info()
data.describe()
data['depth_table_ratio'] = data['depth'] / data['table']
data.sample(10)
ax = sns.boxplot(y='carat', data=data)

print(data[['carat']].describe())
ax = sns.scatterplot(x='carat', y='price', data=data,

                     edgecolors='k', alpha=0.3)
ax = sns.violinplot(y='price', data=data)
f, ax = plt.subplots(ncols=2, figsize=(10, 5))

sns.regplot(x='carat', y='price', data=data, ax=ax[0],

            x_bins=10, x_estimator=np.mean, ci=None)

sns.regplot(x='carat', y='price', data=data, ax=ax[1],

            x_bins=10, x_estimator=np.mean, ci=None, order=3)



ax[0].set_title("Price vs Carat - 1st order linear")

ax[1].set_title("Price vs Carat - 3rd order polynomial")

plt.show()
data['log_price'] = data.price.apply(np.log)
f, ax = plt.subplots(ncols=3, figsize=(15,5))

sns.regplot(x='carat', y='log_price', data=data, x_bins=10, ax=ax[0],

                 x_estimator=np.mean, ci=None)

sns.regplot(x='carat', y='log_price', data=data, x_bins=10, ax=ax[1],

                 x_estimator=np.mean, ci=None, order=2)

sns.regplot(x='carat', y='log_price', data=data, x_bins=10, ax=ax[2],

                 x_estimator=np.mean, ci=None, order=3)



ax[0].set_title("Log(Price) vs Carat - 1st order polynom.")

ax[0].set_ylabel("Log (price)")



ax[1].set_title("Log(Price) vs Carat - 2nd order polynom.")

ax[1].set_ylabel("Log (price)")



ax[2].set_title("Log(Price) vs Carat - 3rd order polynom.")

ax[2].set_ylabel("Log (price)")



plt.show()
data['carat_bin'] = pd.cut(data.carat, range(6))
ax = sns.countplot(x='carat_bin', data=data)



print(data.carat_bin.value_counts(normalize=True, sort=True, ascending=False)*100)
f, ax = plt.subplots(ncols=2, figsize=(10,5))

sns.boxplot(x='carat_bin', y='log_price', data=data, palette='husl', ax=ax[0])

sns.pointplot(x='carat_bin', y='log_price', data=data, ax=ax[1])



ax[0].set_title("Distribution - log (Price) vs Carat category")

ax[1].set_title("Mean log(price) vs Carat category")

plt.show()



print("Mean price per carat_bin:")

print(data.groupby('carat_bin').price.mean())
data[(data.carat <= 1) & (data.price > 12000)]
f, ax = plt.subplots(ncols=3, figsize=(18, 5))

sns.countplot(x='cut', data=data, ax=ax[0])

sns.boxplot(x='cut', y='log_price', data=data, ax=ax[1])

sns.stripplot(x='cut', y='log_price', data=data, ax=ax[1],

              size=1, edgecolor='k', linewidth=.1)

sns.pointplot(x='cut', y='log_price', data=data, ax=ax[2])

sns.pointplot(x='cut', y='log_price', data=data, ax=ax[2],

              estimator=np.median, color='r')



ax[0].set_title("Count diamonds per carat")

ax[1].set_title("log(Price) vs Cut - Distribution")

ax[2].set_title("Mean log(price) vs Cut (blue)\nMedian log(price) vs Cut (red)")

plt.show()
print('% diamonds per cut grade:')

print(data.cut.value_counts(normalize=True, sort=True, ascending=False)*100)
print('Mean log(price) per cut grade:')

print(data.groupby('cut').log_price.mean())
f, ax = plt.subplots(ncols=3, figsize=(15, 5))

sns.countplot(x='color', data=data, ax=ax[0])

sns.boxplot(x='color', y='log_price', data=data, ax=ax[1])

sns.pointplot(x='color', y='log_price', data=data, ax=ax[2])

sns.pointplot(x='color', y='log_price', data=data, ax=ax[2],

              estimator=np.median, color='r')



ax[0].set_title("Count diamonds per color grade")

ax[1].set_title("log(Price) vs Color - Distribution")

ax[2].set_title("Mean log(price) vs Color (blue)\nMedian log(price) vs Color (red)")

plt.show()
print("# diamonds per color grade")

print(data.color.value_counts(normalize=True, sort=True, ascending=True) * 100)
print("Mean log(price) of diamonds per color grade")

print(data.groupby('color').log_price.mean())
f, ax = plt.subplots(ncols=3, figsize=(15, 5))

sns.countplot(x='clarity', data=data, ax=ax[0])

sns.boxplot(x='clarity', y='log_price', data=data, ax=ax[1])

sns.pointplot(x='clarity', y='log_price', data=data, ax=ax[2])

sns.pointplot(x='clarity', y='log_price', data=data, ax=ax[2],

              estimator=np.median, color='r')



ax[0].set_title("Count diamonds per clarity grade")

ax[1].set_title("log(Price) vs clarity - Distribution")

ax[2].set_title("Mean log(price) vs clarity (blue)\nMedian log(price) vs clarity (red)")

plt.show()
data[['depth']].describe()
f, ax = plt.subplots(ncols=2, figsize=(10,5))

sns.boxplot(y='depth', data=data, ax=ax[0])

ax[0].set_title("Depth distribution")



sns.boxplot(y='depth', data=data, ax=ax[1])

ax[1].set_ylim(55, 70)

ax[1].set_title("Depth distribution - Zoomed in")



plt.show()
ax = sns.scatterplot(x='depth', y='log_price', data=data,

                     alpha=0.3, edgecolor='k')
# create bins

depth_desc = data[['depth']].describe()

depth_bins = depth_desc['min':'max'].depth.tolist()



# create column for bins

data['depth_bin'] = pd.cut(data.depth, depth_bins)

data.depth_bin.value_counts()
f, ax = plt.subplots(ncols=3, figsize=(15, 5))

sns.countplot(x='depth_bin', data=data, ax=ax[0])

sns.boxplot(x='depth_bin', y='log_price', data=data, ax=ax[1])

sns.pointplot(x='depth_bin', y='log_price', data=data, ax=ax[2])

sns.pointplot(x='depth_bin', y='log_price', data=data, ax=ax[2],

              estimator=np.median, color='r')



ax[0].set_title("Count diamonds per depth bin")

ax[1].set_title("log(Price) vs depth_bin - Distribution")

ax[2].set_title("Mean log(price) vs depth_bin (blue)\nMedian log(price) vs depth_bin (red)")

plt.show()
data[['table']].describe()
f, ax = plt.subplots(ncols=2, figsize=(10,5))

sns.boxplot(y='table', data=data, ax=ax[0])

ax[0].set_title("Table distribution")



sns.boxplot(y='table', data=data, ax=ax[1])

ax[1].set_ylim(50, 65)

ax[1].set_title("Table distribution - Zoomed in")



plt.show()
ax = sns.scatterplot(x='table', y='log_price', data=data,

                     alpha=0.3, edgecolor='k')
# create bin list

table_desc = data[['table']].describe()

table_bins=table_desc['min':'max'].table.tolist()

table_bins.append(65)

table_bins.sort()



# create column for bins

data['table_bin'] = pd.cut(data.table, table_bins)

data.table_bin.value_counts()
f, ax = plt.subplots(ncols=3, figsize=(18, 5))

sns.countplot(x='table_bin', data=data, ax=ax[0])

sns.boxplot(x='table_bin', y='log_price', data=data, ax=ax[1])

sns.pointplot(x='table_bin', y='log_price', data=data, ax=ax[2])

sns.pointplot(x='table_bin', y='log_price', data=data, ax=ax[2],

              estimator=np.median, color='r')



ax[0].set_title("Count diamonds per table bin")

ax[1].set_title("log(Price) vs table_bin - Distribution")

ax[2].set_title("Mean log(price) vs table_bin (blue)\nMedian log(price) vs table_bin (red)")

plt.show()
data['depth_table_ratio'].describe()
ax = sns.boxplot(y='depth_table_ratio', data=data)
ax = sns.scatterplot(x='depth_table_ratio', y='log_price', data=data,

                     edgecolor='k', alpha=.3, s=10)
# create bins list

dt_ratio_desc = data[['depth_table_ratio']].describe()

dt_ratio_bins = dt_ratio_desc['min':'max'].depth_table_ratio.tolist()



# create columns for bins

data['dt_ratio_bin'] = pd.cut(data.depth_table_ratio, dt_ratio_bins)

data.dt_ratio_bin.value_counts()
f, ax = plt.subplots(nrows=3, figsize=(6, 18))

sns.countplot(x='dt_ratio_bin', data=data, ax=ax[0])

sns.boxplot(x='dt_ratio_bin', y='log_price', data=data, ax=ax[1])

sns.pointplot(x='dt_ratio_bin', y='log_price', data=data, ax=ax[2])

sns.pointplot(x='dt_ratio_bin', y='log_price', data=data, ax=ax[2],

              estimator=np.median, color='r')



ax[0].set_title("Count diamonds per dt_ratio_bin")

ax[1].set_title("log(Price) vs dt_ratio_bin - Distribution")

ax[2].set_title("Mean log(price) vs dt_ratio_bin (blue)\nMedian log(price) vs dt_ratio_bin (red)")

plt.show()
data.price.describe()
ax = sns.boxplot(y='price', data=data)
data['high_price'] = data.price.apply(lambda x: 1 if x >= 10000 else 0)
ax = sns.boxplot(x='high_price', y='carat', data=data)
ax = sns.countplot(x='cut', data=data, hue='high_price')
pricebin_cut_ct = pd.crosstab(data.high_price, data.cut, values=data.price, 

                              aggfunc='count', normalize='index')

pricebin_cut_ct.style.background_gradient(cmap='autumn', axis=1)
pricebin_cut_ct[['Very Good', 'Premium', 'Ideal']].sum(axis=1)
ax = sns.countplot(x='color', data=data, hue='high_price')
pricebin_color_ct = pd.crosstab(data.high_price, data.color, values=data.price, 

                                aggfunc='count', normalize='index')

pricebin_color_ct.style.background_gradient(cmap='autumn', axis=1)
ax = sns.countplot(x='clarity', data=data, hue='high_price')
pricebin_clarity_ct = pd.crosstab(data.high_price, data.clarity, values=data.price, 

                                  aggfunc='count', normalize='index')

pricebin_clarity_ct.style.background_gradient(cmap='autumn', axis=1)
f, ax = plt.subplots(ncols=2, figsize=(10,5))

sns.boxplot(x='high_price', y='depth', data=data, ax=ax[0])

sns.boxplot(x='high_price', y='depth', data=data, ax=ax[1])



ax[0].set_title("Depth distribution by high_price")

ax[1].set_ylim((58, 66))

ax[1].set_title("Depth distribution by high_price\nZoomed in")



plt.show()
f, ax = plt.subplots(ncols=2, figsize=(10,5))

sns.boxplot(x='high_price', y='table', data=data, ax=ax[0])

sns.boxplot(x='high_price', y='table', data=data, ax=ax[1])



ax[0].set_title("Table distribution by high_price")

ax[1].set_ylim((50, 65))

ax[1].set_title("Table distribution by high_price\nZoomed in")



plt.show()
data['cut_encod'] = LabelEncoder().fit_transform(np.asarray(data.cut))

data['color_encod'] = LabelEncoder().fit_transform(np.asarray(data.color))

data['clarity_encod'] = LabelEncoder().fit_transform(np.asarray(data.cut))
cor_mat = data.corr()
f, ax = plt.subplots(figsize=(10,10))

ax = sns.heatmap(cor_mat, cmap='autumn', annot=True)
g = sns.pairplot(data, vars=['log_price', 'price', 'carat', 'x', 'y', 'z'])
data.drop(['carat_bin', 'high_price', 'x', 'y', 'z'], axis=1, inplace=True)
g = sns.pairplot(data, vars=['log_price', 'price', 'depth', 'table', 'depth_table_ratio'])
data.drop(['depth_table_ratio', 'dt_ratio_bin', 'depth_bin', 'table_bin'], axis=1, inplace=True)
data.head()
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
ct = pd.crosstab(data.cut, data.clarity, data.log_price, aggfunc=np.mean)

print("Table 1. Mean log(price) map - cut vs clarity")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data.cut, data.clarity, data.carat, aggfunc=np.mean)

print("Table 2. Mean carat map - cut vs clarity")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data.cut, data.clarity)

print("Table 3. Count diamonds - cut vs clarity")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data.cut, data.color, data.log_price, aggfunc=np.mean)

print("Table 4. Mean log(price) - cut vs color")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data.cut, data.color, data.carat, aggfunc=np.mean)

print("Table 5. Mean carat - cut vs color")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data.cut, data.color)

print("Table 6. Count of diamonds - cut vs color")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data.clarity, data.color, data.log_price, aggfunc=np.mean)

print("Table 7. Mean log(price) - clarity vs color")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data.clarity, data.color, data.carat, aggfunc=np.mean)

print("Table 8. Mean carat - clarity vs color")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data.clarity, data.color)

print("Table 9. Count of diamonds - clarity vs color")

ct.style.background_gradient(cmap=cmap, axis=1)
g = sns.catplot(y='depth', kind='violin', hue='cut', data=data,

                col='cut', col_wrap=3)
g = sns.catplot(y='table', kind='violin', hue='cut', data=data,

                col='cut', col_wrap=3)
g = sns.relplot(x='depth', y='table', data=data, hue='cut',

                col='cut', col_wrap=3)
data.carat.value_counts(sort=True, ascending=False).head()
data_fixcarat = data[data.carat == 0.3]

data_fixcarat.carat.describe()
ct = pd.crosstab(data_fixcarat.cut, data_fixcarat.color, data_fixcarat.log_price, aggfunc=np.mean)

print("Table 10. Mean log(price) for carat = 0.3 - cut vs color")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data_fixcarat.cut, data_fixcarat.color, data_fixcarat.log_price, aggfunc=np.mean)

print("Table 11. Mean log(price) for carat = 0.3 - cut vs color")

ct.style.background_gradient(cmap=cmap, axis=0)
ct = pd.crosstab(data_fixcarat.cut, data_fixcarat.color, data_fixcarat.log_price, aggfunc='count')

print("Table 12. Count of diamonds for carat = 0.3 - cut vs color")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data_fixcarat.cut, data_fixcarat.clarity, data_fixcarat.log_price, aggfunc=np.mean)

print("Table 13. Mean log(price) for carat = 0.3 - cut vs clarity")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data_fixcarat.cut, data_fixcarat.clarity, data_fixcarat.log_price, aggfunc=np.mean)

print("Table 14. Mean log(price) for carat = 0.3 - cut vs clarity")

ct.style.background_gradient(cmap=cmap, axis=0)
ct = pd.crosstab(data_fixcarat.cut, data_fixcarat.clarity, data_fixcarat.log_price, aggfunc='count')

print("Table 15. Mean log(price) for carat = 0.3 - cut vs clarity")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data_fixcarat.color, data_fixcarat.clarity, data_fixcarat.log_price, aggfunc=np.mean)

print("Table 16. Mean log(price) for carat = 0.3 - color vs clarity")

ct.style.background_gradient(cmap=cmap, axis=1)
ct = pd.crosstab(data_fixcarat.color, data_fixcarat.clarity, data_fixcarat.log_price, aggfunc=np.mean)

print("Table 17. Mean log(price) for carat = 0.3 - color vs clarity")

ct.style.background_gradient(cmap=cmap, axis=0)
ct = pd.crosstab(data_fixcarat.color, data_fixcarat.clarity, data_fixcarat.log_price, aggfunc='count')

print("Table 18. Count of diamonds for carat = 0.3 - color vs clarity")

ct.style.background_gradient(cmap=cmap, axis=1)
data.head()
data.drop(['cut_encod', 'color_encod', 'clarity_encod'], axis=1, inplace=True)

data.head()
data = pd.get_dummies(data, drop_first=True)

data.head()
X = data.drop(['price', 'log_price'], axis=1).values

y = data.price.values



assert X.ndim == 2

assert y.ndim == 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, 

                                                    random_state=42)
train_X = sm.add_constant(X_train[:, 0])

train_y = y_train.copy()



lm1 = sm.OLS(train_y, train_X).fit()

lm1.summary()
fitted_y = lm1.fittedvalues

res = lm1.resid

res_student = lm1.get_influence().resid_studentized_internal
f = plt.figure(figsize=(8, 8))

plt.scatter(fitted_y, res, s=70, alpha=.2, edgecolors='k', linewidths=.1)

sns.regplot(fitted_y, res, ci=None, scatter=False, lowess=True,

            line_kws=dict(linewidth=1, color='r'))

plt.plot(fitted_y, np.zeros_like(fitted_y), linestyle='--', color='k')

plt.xlabel("Fitted values", fontdict=dict(fontsize=16))

plt.ylabel("Residuals", fontdict=dict(fontsize=16))

plt.show()
train_X = sm.add_constant(X_train[:, 0])

train_y = np.log(y_train)



lm2 = sm.OLS(train_y, train_X).fit()

lm2.summary()
fitted_y = lm2.fittedvalues

res = lm2.resid
f = plt.figure(figsize=(8, 8))

plt.scatter(fitted_y, res, s=70, alpha=.2, edgecolors='k', linewidths=.1)

sns.regplot(fitted_y, res, ci=None, scatter=False, lowess=True,

            line_kws=dict(linewidth=1, color='r'))

plt.plot(fitted_y, np.zeros_like(fitted_y), linestyle='--', color='k')

plt.xlabel("Fitted values", fontdict=dict(fontsize=16))

plt.ylabel("Residuals", fontdict=dict(fontsize=16))

plt.show()
train_X = X_train[:, 0].reshape(-1, 1)

train_X = PolynomialFeatures(degree=4).fit_transform(train_X)



train_X = sm.add_constant(train_X[:, 1:])

train_y = y_train.copy()



lm3 = sm.OLS(train_y, train_X).fit()

lm3.summary()
fitted_y = lm3.fittedvalues

res = lm3.resid

res_student = lm3.get_influence().resid_studentized_internal
f = plt.figure(figsize=(8, 8))

plt.scatter(fitted_y, res, s=70, alpha=.4)

sns.regplot(fitted_y, res, ci=None, scatter=False, lowess=True,

            line_kws=dict(linewidth=3, color='r'))

plt.plot(fitted_y, np.zeros_like(fitted_y), linestyle='--', color='k')

plt.xlabel("Fitted values", fontdict=dict(fontsize=16))

plt.ylabel("Residuals", fontdict=dict(fontsize=16))

plt.show()
train_X = X_train[:, 0].reshape(-1, 1)

train_X = PolynomialFeatures(degree=4).fit_transform(train_X)



train_X = sm.add_constant(train_X[:, 1:])

train_y = np.log(y_train)



lm4 = sm.OLS(train_y, train_X).fit()

lm4.summary()
fitted_y = lm4.fittedvalues

res = lm4.resid

res_student = lm4.get_influence().resid_studentized_internal
f = plt.figure(figsize=(8, 8))

plt.scatter(fitted_y, res, s=70, alpha=.4)

sns.regplot(fitted_y, res, ci=None, scatter=False, lowess=True,

            line_kws=dict(linewidth=3, color='r'))

plt.plot(fitted_y, np.zeros_like(fitted_y), linestyle='--', color='k')

plt.xlabel("Fitted values", fontdict=dict(fontsize=16))

plt.ylabel("Residuals", fontdict=dict(fontsize=16))

plt.show()
f = plt.figure(figsize=(8, 8))

plt.scatter(fitted_y, res_student, s=70, alpha=.4)

plt.plot(fitted_y, np.zeros_like(fitted_y), linestyle='--', color='k')

plt.xlabel("Fitted values", fontdict=dict(fontsize=16))

plt.ylabel("Studentized Residuals", fontdict=dict(fontsize=16))

plt.show()
data.head()
columns = data.drop(['carat', 'price', 'log_price'], axis=1).columns

carat_columns = ['carat', 'carat^2', 'carat^3', 'carat^4']



# build polynomial carats, exclude cons

carat_poly = X_train[:, 0].reshape(-1, 1)

carat_poly = PolynomialFeatures(degree=4, include_bias=False).fit_transform(carat_poly)

carat_poly_df = pd.DataFrame(data=carat_poly, columns=carat_columns)



# take quality features + concatenate carat and quality features 

train_X_df = pd.DataFrame(data=X_train[:, 1:], columns=columns)

train_X_df = pd.concat([carat_poly_df, train_X_df], axis=1)



# get responde DataFrame

train_y_df = pd.DataFrame(y_train, columns=['log_price']).apply(np.log)
# train model

train_X_df = sm.add_constant(train_X_df)

lm5 = sm.OLS(train_y_df, train_X_df).fit()

lm5.summary()
fitted_y = lm5.fittedvalues

res = lm5.resid

res_student = lm5.get_influence().resid_studentized_internal
f = plt.figure(figsize=(6, 6))

plt.scatter(fitted_y, res, s=70, alpha=.4)

plt.plot(fitted_y, np.zeros_like(fitted_y), linestyle='--', color='k')

plt.xlabel("Fitted values", fontdict=dict(fontsize=16))

plt.ylabel("Residuals", fontdict=dict(fontsize=16))

plt.show()
f = plt.figure(figsize=(6, 6))

plt.scatter(fitted_y, res_student, s=70, alpha=.4)

plt.plot(fitted_y, np.zeros_like(fitted_y), linestyle='--', color='k')

plt.xlabel("Fitted values", fontdict=dict(fontsize=16))

plt.ylabel("Studentized Residuals", fontdict=dict(fontsize=16))

plt.show()
train_X_df.head()
# take quality features + concatenate carat and quality features 

train_X_df = train_X_df.drop(['depth', 'table'], axis=1)



# train_y_df remains as used in the previous example
# train model

train_X_df = sm.add_constant(train_X_df)

lm6 = sm.OLS(train_y_df, train_X_df).fit()

lm6.summary()
fitted_y = lm6.fittedvalues

res = lm6.resid

res_student = lm6.get_influence().resid_studentized_internal
f = plt.figure(figsize=(6, 6))

plt.scatter(fitted_y, res, s=70, alpha=.4)

plt.plot(fitted_y, np.zeros_like(fitted_y), linestyle='--', color='k')

plt.xlabel("Fitted values", fontdict=dict(fontsize=16))

plt.ylabel("Residuals", fontdict=dict(fontsize=16))

plt.show()
f = plt.figure(figsize=(6, 6))

plt.scatter(fitted_y, res_student, s=70, alpha=.4)

plt.plot(fitted_y, np.zeros_like(fitted_y), linestyle='--', color='k')

plt.xlabel("Fitted values", fontdict=dict(fontsize=16))

plt.ylabel("Studentized Residuals", fontdict=dict(fontsize=16))

plt.show()
columns_cut = [col for col in train_X_df.columns if 'cut' in col]



# remove cut columns from DataFrame

train_X_df = train_X_df.drop(columns_cut, axis=1)



# train_y_df is the same as used for the previous model
# train model

train_X_df = sm.add_constant(train_X_df)

lm7 = sm.OLS(train_y_df, train_X_df).fit()

lm7.summary()
fitted_y = lm7.fittedvalues

res = lm7.resid

res_student = lm7.get_influence().resid_studentized_internal
f = plt.figure(figsize=(6, 6))

plt.scatter(fitted_y, res, s=70, alpha=.4)

plt.plot(fitted_y, np.zeros_like(fitted_y), linestyle='--', color='k')

plt.xlabel("Fitted values", fontdict=dict(fontsize=16))

plt.ylabel("Residuals", fontdict=dict(fontsize=16))

plt.show()
f = plt.figure(figsize=(6, 6))

plt.scatter(fitted_y, res_student, s=70, alpha=.4)

plt.plot(fitted_y, np.zeros_like(fitted_y), linestyle='--', color='k')

plt.xlabel("Fitted values", fontdict=dict(fontsize=16))

plt.ylabel("Studentized Residuals", fontdict=dict(fontsize=16))

plt.show()
data.head()
X = data.drop(['price', 'log_price'], axis=1)    # as DataFrame

y = data[['log_price']]    # as DataFrame



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)



print(type(X_train), type(X_test))

print(type(y_train), type(y_test))
# create identifiers for polynomial features and linear features

POLY_COLS = ['carat']

LIN_COLS = [col for col in data.columns if col not in ['carat', 'price', 'log_price']]
# create the functions to get each subset of the data

get_polyfeatures = FunctionTransformer(lambda x: x[POLY_COLS], validate=False)

get_linfeatures = FunctionTransformer(lambda x: x[LIN_COLS], validate=False)
poly_pl = Pipeline([

    ('selector', get_polyfeatures),

    ('polynomial', PolynomialFeatures(degree=4, include_bias=False))

])



# display polynomial features after transformation

poly_pl.fit_transform(X_train)[:5, :]
lin_pl = Pipeline([('selector', get_linfeatures)])



# display first few lines of linear features (in this case, no other operation is performed)

lin_pl.fit_transform(X_train).head()
# join both pipelines into one

prep_join = FeatureUnion([

    ('polynomial', poly_pl),

    ('linear', lin_pl)

])



# display final resulting array's first 5 rows

prep_join.fit_transform(X_train)
# create identifiers for polynomial features and linear features

POLY_COLS = ['carat']

LIN_COLS = [col for col in data.columns if col not in ['carat', 'price', 'log_price']]



# create the functions to get each subset of the data

get_polyfeatures = FunctionTransformer(lambda x: x[POLY_COLS], validate=False)

get_linfeatures = FunctionTransformer(lambda x: x[LIN_COLS], validate=False)
POLY_DEG=4



# join both pipelines into one

prep_join = FeatureUnion([

    ('polynomial', Pipeline([

        ('selector', get_polyfeatures),

        ('polynomial', PolynomialFeatures(degree=POLY_DEG, include_bias=False))

    ])),

    ('linear', Pipeline([('selector', get_linfeatures)]))

])
POLY_DEG = 4



# create the regressor pipeline

ridge_pl = Pipeline([

    ('union', FeatureUnion([

        ('polynomial', Pipeline([

            ('selector', get_polyfeatures),

            ('polynomial', PolynomialFeatures(degree=POLY_DEG, include_bias=False))

        ])),

        ('linear', Pipeline([

            ('selector', get_linfeatures)

        ]))

    ])),

    ('regressor', Ridge(alpha=0.1))

])
# perform first fit and use as starting point

ridge_pl.fit(X_train, y_train)

ridge_pl.score(X_test, y_test)
# set up grid of alphas to search

alphas = np.logspace(-4, 4, 9)



# set up GridSearch object to select best alpha and fit to data

CV_FOLDS = 5

PARAM_GRID = {'regressor__alpha': alphas}

SCORE = 'neg_mean_squared_error'

gs = GridSearchCV(ridge_pl, cv=CV_FOLDS, param_grid=PARAM_GRID, scoring=SCORE)
# fit to data and print best scores and parameters

gs.fit(X_train, y_train)



print("Best hyperparameters:", gs.best_params_)

print("Best RMSE:           ", (-gs.best_score_) ** 0.5)
ridge_pl.set_params(regressor__alpha=1.0)



ridge_pl.fit(X_train, y_train)
ridge_coef = np.squeeze(ridge_pl.named_steps['regressor'].coef_)
predictors = ['carat', 'carat^2', 'carat^3', 'carat^4']

predictors.extend(X_train.columns[1:].tolist())



plt.figure(figsize=(6, 8))

plt.barh(y=range(len(ridge_coef)), width=ridge_coef)

plt.yticks(range(len(ridge_coef)), predictors)

plt.xlabel("Coefficient estimate")

plt.ylabel("Predictors")

plt.title("Ridge Coefficient Estimates", fontdict=dict(fontsize=16))

plt.show()
# predictions and actual values as arrays

y_pred = ridge_pl.predict(X_test)

y_true = y_test.values
print('R^2: %.4f' % (r2_score(y_test, y_pred)))

print('Exp. Var.: %.4f' % (explained_variance_score(y_test, y_pred)))

print('RMSE: %.4f' % (mean_squared_error(y_test, y_pred) ** .5))

print('MAE: %.4f' % (mean_absolute_error(y_test, y_pred)))
resid = (y_true - y_pred)

sns.jointplot(x=y_pred, y=resid, kind='reg', 

              joint_kws=dict(fit_reg=False))

plt.xlabel("Fitted Values")

plt.ylabel("Residuals")

plt.show()
POLY_DEG = 4



# create the regressor pipeline

lasso_pl = Pipeline([

    ('union', FeatureUnion([

        ('polynomial', Pipeline([

            ('selector', get_polyfeatures),

            ('polynomial', PolynomialFeatures(degree=POLY_DEG, include_bias=False))

        ])),

        ('linear', Pipeline([

            ('selector', get_linfeatures)

        ]))

    ])),

    ('regressor', Lasso(alpha=0.1))

])
# perform first fit and use as starting point

lasso_pl.fit(X_train, y_train)

lasso_pl.score(X_test, y_test)
# set up grid of alphas to search

alphas = np.logspace(-4, 4, 9)



# set up GridSearch object to select best alpha and fit to data

CV_FOLDS = 5

PARAM_GRID = {'regressor__alpha': alphas}

SCORE = 'neg_mean_squared_error'

gs = GridSearchCV(lasso_pl, cv=CV_FOLDS, param_grid=PARAM_GRID, scoring=SCORE)
# fit to data and print best scores and parameters

gs.fit(X_train, y_train)



print("Best hyperparameters:", gs.best_params_)

print("Best RMSE:           ", (-gs.best_score_) ** 0.5)
lasso_pl.set_params(regressor__alpha=0.0001)



lasso_pl.fit(X_train, y_train)
lasso_coef = np.squeeze(lasso_pl.named_steps['regressor'].coef_)
predictors = ['carat', 'carat^2', 'carat^3', 'carat^4']

predictors.extend(X_train.columns[1:].tolist())



plt.figure(figsize=(6, 8))

plt.barh(y=range(len(lasso_coef)), width=lasso_coef)

plt.yticks(range(len(lasso_coef)), predictors)

plt.xlabel("Coefficient estimate")

plt.ylabel("Predictors")

plt.title("Lasso Coefficient Estimates, alpha = %.4f" % 0.0001, fontdict=dict(fontsize=16))

plt.show()
# get prediction and actual values as arrays

y_pred = lasso_pl.predict(X_test).reshape(-1, 1)

y_true = y_test.values
print('R^2: %.4f' % (r2_score(y_test, y_pred)))

print('Exp. Var.: %.4f' % (explained_variance_score(y_test, y_pred)))

print('RMSE: %.4f' % (mean_squared_error(y_test, y_pred) ** .5))

print('MAE: %.4f' % (mean_absolute_error(y_test, y_pred)))
resid = (y_true - y_pred)

sns.jointplot(x=y_pred, y=resid, kind='reg', 

              joint_kws=dict(fit_reg=False))

plt.xlabel("Fitted Values")

plt.ylabel("Residuals")

plt.show()
en_pl = Pipeline([

    ('union', FeatureUnion([

        ('polynomial', Pipeline([

            ('selector', get_polyfeatures),

            ('polynomial', PolynomialFeatures(degree=POLY_DEG, include_bias=False))

        ])),

        ('linear', Pipeline([

            ('selector', get_linfeatures)

        ]))

    ])),

    ('regressor', ElasticNet())

])
en_pl.fit(X_train, y_train)

en_pl.score(X_test, y_test)
# create alphas space for search

alphas = np.logspace(-4, 4, 9)

l1_ratios = np.linspace(0, 1, 6)
# prepare GridSearch arguments

CV_FOLDS = 5

PARAM_GRID = {'regressor__alpha': alphas, 'regressor__l1_ratio': l1_ratios}

SCORE = 'neg_mean_squared_error'



gs = RandomizedSearchCV(en_pl, cv=CV_FOLDS, param_distributions=PARAM_GRID, scoring=SCORE)
gs.fit(X_train, y_train)



print("Best hyperparameters:", gs.best_params_)

print("Best RMSE:           ", (-gs.best_score_) ** 0.5)
en_pl.set_params(regressor__alpha=0.0001, regressor__l1_ratio=0.8)



en_pl.fit(X_train, y_train)
en_coef = np.squeeze(en_pl.named_steps['regressor'].coef_)

en_coef
predictors = ['carat', 'carat^2', 'carat^3', 'carat^4']

predictors.extend(X_train.columns[1:].tolist())



plt.figure(figsize=(6, 8))

plt.barh(y=range(len(en_coef)), width=en_coef)

plt.yticks(range(len(en_coef)), predictors)

plt.xlabel("Coefficient estimate")

plt.ylabel("Predictors")

plt.title("ElasticNet Coefficient Estimates\nalpha = %.4f, l1_ratio = %.1f" % (0.0001,0.8), 

          fontdict=dict(fontsize=16))

plt.show()
# get prediction and actual values as arrays

y_pred = en_pl.predict(X_test).reshape(-1, 1)

y_true = y_test.values
print('R^2: %.4f' % (r2_score(y_test, y_pred)))

print('Exp. Var.: %.4f' % (explained_variance_score(y_test, y_pred)))

print('RMSE: %.4f' % (mean_squared_error(y_test, y_pred) ** .5))

print('MAE: %.4f' % (mean_absolute_error(y_test, y_pred)))
resid = (y_true - y_pred)

sns.jointplot(x=y_pred, y=resid, kind='reg', 

              joint_kws=dict(fit_reg=False))

plt.xlabel("Fitted Values")

plt.ylabel("Residuals")

plt.show()
plt.figure(figsize=(10, 10))

plt.barh(y=range(len(ridge_coef)), width=ridge_coef, color='r')

plt.barh(y=range(len(lasso_coef)), width=lasso_coef, color='b')

plt.yticks(range(len(ridge_coef)), predictors)

plt.xlabel("Coefficient estimate")

plt.ylabel("Predictors")

plt.title("Comparison of Coefficient Estimates\nRidge in red, Lasso in blue", 

          fontdict=dict(fontsize=16))

plt.show()