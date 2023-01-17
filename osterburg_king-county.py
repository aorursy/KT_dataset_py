import os

import math

import random



import pandas as pd

import numpy as np

import json



from scipy import stats

from scipy import linalg



import statsmodels.api as sm

import statsmodels.stats.stattools as sms

from statsmodels.formula.api import ols



from sklearn import metrics

from sklearn import linear_model

from sklearn import neighbors

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



import missingno as msno



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

# set style

sns.set_style('whitegrid')

# overriding font size and line width

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})



# map visualization

import folium

from folium.plugins import HeatMap



# don't print matching warnings

import warnings

warnings.filterwarnings('ignore') 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))
# local functions which are in a seperate python file

#

def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):

    """

    Perform a forward-backward feature selection based on p-value from statsmodels.api.OLS



    Arguments:

        X - pandas.DataFrame with candidate features

        y - list-like with the target

        initial_list - list of features to start with (column names of X)

        threshold_in - include a feature if its p-value < threshold_in

        threshold_out - exclude a feature if its p-value > threshold_out

        verbose - whether to print the sequence of inclusions and exclusions

    Returns: list of selected features



    Always set threshold_in < threshold_out to avoid infinite looping.

    See https://en.wikipedia.org/wiki/Stepwise_regression for the details

    """



    included = list(initial_list)

    while True:

        changed = False

        # forward step

        excluded = list(set(X.columns) - set(included))

        new_pval = pd.Series(index=excluded)



        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]



        best_pval = new_pval.min()



        if best_pval < threshold_in:

            best_feature = new_pval.idxmin()

            included.append(best_feature)

            changed = True



            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))



        # backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()



        # use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        # null if pvalues is empty

        worst_pval = pvalues.max()



        if worst_pval > threshold_out:

            changed = True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)



            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break



    return included





def display_heatmap(data):

    """

    Display a heatmap from a given dataset



    :param data: dataset

    :return: g (graph to display)

    """



    # Set the style of the visualization

    # sns.set(style = "white")

    sns.set_style("white")



    # Create a covariance matrix

    corr = data.corr()



    # Generate a mask the size of our covariance matrix

    mask = np.zeros_like(corr)

    mask[np.triu_indices_from(mask)] = None



    # Set up the matplotlib figure

    f, ax = plt.subplots(figsize=(15, 12))



    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(240, 10, sep=20, n=9, as_cmap=True)



    # Draw the heatmap with the mask and correct aspect ratio

    g = sns.heatmap(corr, cmap=cmap, mask=mask, square=True)



    return g





def display_jointplot(data, columns):

    """

    Display seaborn jointplot on given dataset and feature list



    :param data: dataset

    :param columns: feature list

    :return: g

    """



    sns.set_style('whitegrid')



    for column in columns:

        g = sns.jointplot(x=column, y="price", data=data, dropna=True,

                          kind='reg', joint_kws={'line_kws': {'color': 'red'}})



    return g





def display_plot(data, vars, target, plot_type='box'):

    """

    Generates a seaborn boxplot (default) or scatterplot



    :param data: dataset

    :param vars: feature list

    :param target: feature name

    :param plot_type: box (default), scatter, rel

    :return: g

    """



    # pick one dimension

    ncol = 3

    # make sure enough subplots

    nrow = math.floor((len(vars) + ncol - 1) / ncol)

    # create the axes

    fig, axarr = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, 20))



    # go over a linear list of data

    for i in range(len(vars)):

        # compute an appropriate index (1d or 2d)

        ix = np.unravel_index(i, axarr.shape)



        feature_name = vars[i]



        if plot_type == 'box':

            g = sns.boxplot(y=feature_name, x=target, data=data, width=0.8,

                            orient='h', showmeans=True, fliersize=3, ax=axarr[ix])



        # elif plot_type == 'scatter':

        else:

            g = sns.scatterplot(x=feature_name, y=target, data=data, ax=axarr[ix])



        # else:

        #     col_name = vars[i]

        #     g = sns.relplot(x=feature_name, y=target, hue=target, col=col_name,

        #                     size=target, sizes=(5, 500), col_wrap=3, data=data)



    return g





def map_feature_by_zipcode(zipcode_data, col):

    """

    Generates a folium map of Seattle

    :param zipcode_data: zipcode dataset

    :param col: feature to display

    :return: m

    """



    # read updated geo data

    king_geo = "cleaned_geodata.json"



    # Initialize Folium Map with Seattle latitude and longitude

    m = folium.Map(location=[47.35, -121.9], zoom_start=9,

                   detect_retina=True, control_scale=False)

    # tiles='stamentoner')



    # Create choropleth map

    m.choropleth(

        geo_data=king_geo,

        name='choropleth',

        data=zipcode_data,

        # col: feature of interest

        columns=['zipcode', col],

        key_on='feature.properties.ZIPCODE',

        fill_color='OrRd',

        fill_opacity=0.9,

        line_opacity=0.2,

        legend_name='house ' + col

    )



    folium.LayerControl().add_to(m)



    # Save map based on feature of interest

    m.save(col + '.html')



    return m





def measure_strength(data, feature_list, target):

    """

    Calculate a Pearson correlation coefficient and the p-value to test for non-correlation.



    :param data: dataset

    :param feature_list: feature list

    :param target: feature name

    :return:

    """



    print("Pearson correlation coefficient R and p-value \n\n")



    for k, v in enumerate(feature_list):

        r, p = stats.pearsonr(data[v], data[target])

        print("{0} <=> {1}\t\tR = {2} \t\t p = {3}".format(target, v, r, p))





def heatmap_features_by_loc(data, feature):

    """

    Generates a heatmap based on lat, long and a feature



    :param data: dataset

    :param feature: feature name

    :return:

    """

    max_value = data[feature].max()



    lat = np.array(data.lat, dtype=pd.Series)

    lon = np.array(data.long, dtype=pd.Series)

    mag = np.array(data[feature], dtype=pd.Series) / max_value



    d = np.dstack((lat, lon, mag))[0]

    heatmap_data = [i for i in d.tolist()]



    hmap = folium.Map(location=[47.55, -122.0], zoom_start=10, tiles='stamentoner')



    hm_wide = HeatMap(heatmap_data,

                      min_opacity=0.7,

                      max_val=max_value,

                      radius=1, blur=1,

                      max_zoom=1,

                      )



    hmap.add_child(hm_wide)



    return hmap

# We can import the above function from a seperate python file:

#

# import function_filename as f

#

# you can check out the the documentation for the rest of the autoreaload modes

# by apending a question mark to %autoreload, like this:

# %autoreload?

#

# %load_ext autoreload

# %autoreload 2
# read data and read date correctly

#

dataset = pd.read_csv("../input/kc_house_data.csv", parse_dates = ['date'])
dataset.shape
dataset.dtypes
# Display all missing data

#

msno.matrix(dataset);
# Handling Null values for view

#

dataset.view.fillna(0, inplace=True)
# Handling yr_renovated

# - create new column 'is_renovated' and 'yr_since_renovation'

# - if sqft_living15 > sqft_living set renovated

# - drop yr_renovated

#

import datetime

cur_year = datetime.datetime.now().year



def calc_years(row):

    return cur_year - row['yr_renovated'] if row['yr_renovated'] > 0 else 0



def set_renovated(row):

    return 1 if row['yr_since_renovation'] > 0 or row['sqft_living'] != row['sqft_living15'] else 0



# Set yr_renovated to int

dataset.yr_renovated.fillna(0, inplace = True)

# now I can convert yr_renovated to int

dataset.yr_renovated = dataset.yr_renovated.astype('int64')



dataset['yr_since_renovation'] = dataset.apply(calc_years, axis = 1)



# Create category 'is_renovated'

dataset['is_renovated'] = dataset.apply(set_renovated, axis=1)

# Binning

bins = [0., 1950., 1980., 1990., 2000., 2015.]

names = ['never', 'before 1980', '1980-1989', '1990-1999', '2000-2015']

dataset['yr_renov_bins'] = pd.cut(dataset['yr_renovated'], bins, labels=names, right=False)

dataset.yr_renov_bins.fillna('never', inplace=True)



dataset.drop(columns=['yr_renovated'], inplace=True)
print(cur_year)
dataset.yr_built.shape
dataset.yr_built.value_counts()
# While are at it, lets convert yr_built to house_age and drop yr_built

#

dataset['house_age'] = cur_year - dataset.yr_built

# dataset.drop(columns=['yr_built'], inplace=True)
dataset.house_age.value_counts()
# To answer this question, it's best to build a new variable (feature engineering) ...

dataset['yr_built_cat'] = dataset['house_age'].apply(lambda x: ('old' if x >= 50 else 'middle-aged') if x >= 15 else 'modern')



# ... and turn it into a category

dataset['yr_built_cat'] = pd.Categorical(dataset['yr_built_cat'], categories = ['old', 'middle-aged', 'modern'])

dataset.head(2)
dataset.yr_built_cat.value_counts()
# What is the percential of NaN in waterfront?

#

print(dataset.waterfront.isnull().sum() / dataset.shape[0])
# Because the percential is about 10% we set the NaN values to zero

#

dataset.waterfront.fillna(0, inplace=True)
msno.matrix(dataset);
dataset.shape
# Handling sqft_basement

#

def calc_basement(row):

    """

    Calculate basement sqft based on difference sqft_living and sqft_above

    Deals at the same time with the '?' string

    

    :param row: feature (column)

    :return: value (sqft)

    """

    return row['sqft_living'] - row['sqft_above'] if row['sqft_above'] < row['sqft_living']  else 0



dataset.sqft_basement = dataset.apply(calc_basement, axis = 1)
# sort dataset by date and reset index (Do I have a good reason for it? No.)

#

dataset = dataset.sort_values(by = ['date'])

dataset = dataset.reset_index(drop=True)
display_heatmap(dataset);
dataset.head()
dataset.columns
dataset['zipcode'] = dataset['zipcode'].astype(int)
cols = ['bedrooms', 'bathrooms', 'sqft_above', 'sqft_basement', 'sqft_living15', 

        'sqft_lot15', 'yr_since_renovation', 'house_age', 'zipcode']
ncol = 3 # pick one dimension

nrow = math.floor((len(cols)+ ncol-1) / ncol) # make sure enough subplots

fig, axarr = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, 20)) # create the axes



for i in range(len(cols)): # go over a linear list of data

    ix = np.unravel_index(i, axarr.shape) # compute an appropriate index (1d or 2d)



    name = cols[i]

    dataset.plot(kind='scatter', x=name, y='price', ax=axarr[ix], label=name) 



plt.tight_layout()

plt.show();

# plt.savefig('pics/scatter_plot_1.png', dpi = 320)
dataset.sqft_lot15.value_counts(bins=10, sort=False)
dataset.sqft_living15.value_counts(bins=10, sort=False)
dataset.bedrooms.value_counts(bins=10, sort=False)
dataset.price.value_counts(bins=10, sort=False)
# 'house_age', 'sqft_basement', 'sqft_above', 'sqft_living15',  'sqft_lot15', 'yr_since_renovation'

#

continous_vars = ['sqft_living15',  'sqft_lot15', 'house_age', 'yr_since_renovation']
display_jointplot(dataset, continous_vars)
measure_strength(dataset, continous_vars, 'price')
discrete_vars = ['grade', 'condition', 'view', 'floors', 'bedrooms', 'bathrooms']
# Display box-and-whisker plot

#

display_plot(dataset, discrete_vars, 'price')
measure_strength(dataset, discrete_vars, 'price')
# # Set zipcode type to string (folium)

# dataset['zipcode'] = dataset['zipcode'].astype('str')



# # get the mean value across all data points

# zipcode_data = dataset.groupby('zipcode').aggregate(np.mean)

# zipcode_data.reset_index(inplace = True)
# # count number of houses grouped by zipcode

# #

# dataset['count'] = 1

# t = dataset.groupby('zipcode').sum()

# t.reset_index(inplace = True)

# t = t[['zipcode', 'count']]

# zipcode_data = pd.merge(zipcode_data, t, on='zipcode')



# # drop count from org dataset

# dataset.drop(['count'], axis = 1, inplace = True)
# # Get geo data file path

# geo_data_file = os.path.join('data', '../input/king_county_wa_zipcode_area.geojson')



# # load GeoJSON

# with open(geo_data_file, 'r') as jsonFile:

#     geo_data = json.load(jsonFile)

    

# tmp = geo_data



# # remove ZIP codes not in geo data

# geozips = []

# for i in range(len(tmp['features'])):

#     if tmp['features'][i]['properties']['ZIPCODE'] in list(zipcode_data['zipcode'].unique()):

#         geozips.append(tmp['features'][i])



# # creating new JSON object

# new_json = dict.fromkeys(['type','features'])

# new_json['type'] = 'FeatureCollection'

# new_json['features'] = geozips



# # save uodated JSON object

# open("../input/cleaned_geodata.json", "w").write(json.dumps(new_json, sort_keys=True, indent=4, separators=(',', ': ')))
# map_feature_by_zipcode(zipcode_data, 'count')
# map_feature_by_zipcode(zipcode_data, 'price')
# Get the top 5 zipcode by price

#

# zipcode_data.nlargest(5, 'price')['zipcode']
# Initialize Folium Map with Seattle latitude and longitude



# from folium.plugins import HeatMap



# max_val = dataset.price.max()



# lat = np.array(dataset.lat, dtype=pd.Series)

# lon = np.array(dataset.long, dtype=pd.Series)

# mag = np.array(dataset.price, dtype=pd.Series)



# d = np.dstack((lat, lon, mag))[0]

# heatmap_data = [i for i in d.tolist()]



# m = folium.Map(location=[47.35, -121.9], zoom_start=9, detect_retina=True, control_scale=False)

# HeatMap(heatmap_data, radius=1, blur=1).add_to(m)

# m
dataset.plot(kind="scatter", x="long", y="lat", figsize=(16, 8), c="price", 

             cmap="gist_heat_r", colorbar=True, sharex=False);

plt.show();
sns.relplot(x="sqft_living15", y="price", hue="price", col="is_renovated", 

            size="price", sizes=(5, 500), col_wrap=3, data=dataset);
dataset.is_renovated.value_counts()
# get statistics for houses which are renovated

df_is_renovated = dataset[dataset['is_renovated'] == 1.0]



subset = ['price', 'bedrooms', 'floors', 'sqft_living15', 'sqft_lot15']

is_renovated_descriptives = round(df_is_renovated[subset].describe(), 2)

is_renovated_descriptives
df_not_renovated = dataset[dataset['is_renovated'] == 0.0]



subset = ['price', 'bedrooms', 'floors', 'sqft_living15', 'sqft_lot15']

not_renovated_descriptives = round(df_not_renovated[subset].describe(), 2)

not_renovated_descriptives
is_renovated_descriptives.price.median()
not_renovated_descriptives.price.median()
fig, ax = plt.subplots(figsize=(16, 8))

sns.barplot(x='is_renovated', y='price', hue='is_renovated', data=dataset, palette="PuBu_r")



# add title, legend and informative axis labels

ax.set_title('\nMedian Prices depending on if House is renovated\n', fontsize=14, fontweight='bold')

ax.set(ylabel='Price', xlabel='Is Renovated')

ax.legend(loc=2);
dataset['price'][dataset.is_renovated.max()] - dataset['price'][dataset.is_renovated.min()]
sns.relplot(x="sqft_living15", y="price", hue="price", col="condition",

            size="price", sizes=(5, 500), col_wrap=3, data=dataset);
# plot this dataframe with seaborn

fig, ax = plt.subplots(figsize=(16, 8))

sns.barplot(x='condition', y='price', hue='yr_built_cat', data=dataset, palette="PuBu_r")



# add title, legend and informative axis labels

ax.set_title('\nMedian Prices depending on Condition and Age of Houses\n', fontsize=14, fontweight='bold')

ax.set(ylabel='Price', xlabel='Condition')

ax.legend(loc=2);
sns.relplot(x="sqft_living15", y="price", hue="price", col="grade", 

            size="price", sizes=(5, 500), col_wrap=3, data=dataset);
# plot this dataframe with seaborn

fig, ax = plt.subplots(figsize=(16, 8))

sns.barplot(x='grade', y='price', hue='yr_built_cat', data=dataset, palette="PuBu_r")



# add title, legend and informative axis labels

ax.set_title('\nMedian Prices depending on Condition and Age of Houses\n', fontsize=14, fontweight='bold')

ax.set(ylabel='Price', xlabel='Grade')

ax.legend(loc=2);
dataset['condition'] = dataset['condition'].astype('category', ordered = True)

dataset['waterfront'] = dataset['waterfront'].astype('category', ordered = True)

dataset['is_renovated'] = dataset['is_renovated'].astype('category', ordered = False)

dataset['view'] = dataset['view'].astype('category', ordered = False)



# Create category 'has_basement'

dataset['has_basement'] = dataset.sqft_basement.apply(lambda x: 1 if x > 0 else 0)

dataset['has_basement'] = dataset.has_basement.astype('category', ordered = False)
# Set dummies (we may want to add zipcode as well)

cat_columns = ['floors', 'view', 'condition', 'waterfront', 'is_renovated', 'has_basement']



for col in cat_columns:

    dummies = pd.get_dummies(dataset[col])

    dummies = dummies.add_prefix("{}_".format(col))

    

    dataset.drop(col, axis=1, inplace=True)

    dataset = dataset.join(dummies)
# replace the '.' in the column name

for col in dataset.columns:

    if col.find('.') != -1: 

        dataset.rename(columns={col: col.replace('.', '_')}, inplace=True)
# dropping id and date

dataset.drop(['id', 'date', 'lat', 'long'], axis = 1, inplace = True)
dataset.head()
dataset.describe()
# Using MinMax

#

minmax_df = dataset[['house_age', 'yr_since_renovation', 'zipcode']]



scaler = preprocessing.MinMaxScaler()

minmax_scaled_df = scaler.fit_transform(minmax_df)

minmax_scaled_df = pd.DataFrame(minmax_scaled_df, columns=['house_age', 'yr_since_renovation', 'zipcode'])
# Using Robust for price and sqft

#

robust_df = dataset[['price', 'sqft_above', 'sqft_living15', 'sqft_lot15']]



scaler = preprocessing.RobustScaler()

robust_scaled_df = scaler.fit_transform(robust_df)

robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['price', 'sqft_above', 'sqft_living15', 'sqft_lot15'])
dataset_ols = pd.concat([dataset[['grade', 'bedrooms', 'bathrooms', 'condition_3', 'condition_4', 

                                  'condition_5']], minmax_scaled_df, robust_scaled_df], axis=1)
dataset_ols.head()
ols_results = []

if len(ols_results) != 1:

    ols_results = [['ind_var', 'r_squared', 'intercept', 'slope', 'p-value', 'normality (JB)']]
features = ['grade', 'bedrooms', 'bathrooms', 'house_age', 'yr_since_renovation', 'sqft_above',

            'sqft_living15', 'sqft_lot15', 'zipcode', 'condition_3', 'condition_4', 'condition_5']
def run_ols_regression(store_results, data, target, feature, show_plots=False):

    """

    Run ols model, prints model summary, displays plot_regress_exog and qqplot

    

    :param data: dataset

    :param target: target feature name

    :param feature: feature name

    :return:

    """

    

    formula = target + '~' + feature

    model = ols(formula=formula, data=data).fit()



    df = pd.DataFrame({feature: [data[feature].min(), data[feature].max()]})

    pred = model.predict(df)



    if show_plots:

        print('Regression Analysis and Diagnostics for formula: ', formula)

        print('\n')



        fig = plt.figure(figsize=(16, 8))

        fig = sm.graphics.plot_regress_exog(model, feature, fig=fig)

        plt.show();



        residuals = model.resid

        fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)

        fig.show();

    

    # append all information to results

    store_results.append([feature, model.rsquared, model.params[0], model.params[0],

                        model.pvalues[1], sms.jarque_bera(model.resid)[0]])

for feature in features:

    run_ols_regression(ols_results, dataset_ols, 'price', feature)
pd.DataFrame(ols_results)
y = dataset_ols['price']

X = dataset_ols.drop(['price'], axis=1)
result = stepwise_selection(X, y, verbose = True)

print('resulting features:')

print(result)
pred = '+'.join(features)

formula = 'price~' + pred
model = ols(formula=formula, data=dataset_ols).fit()

model.summary()
y = dataset_ols['price']

X = dataset_ols.drop(['price', 'yr_since_renovation', 'sqft_above', 'condition_3', 'condition_4'], axis=1)
X_int = sm.add_constant(X)

model = sm.OLS(y, X_int).fit()

model.summary()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))
# Fitting the model to the training data

linreg = LinearRegression().fit(X_train, y_train)



# Calc preditors on the train and test set

y_hat_train = linreg.predict(X_train)

y_hat_test = linreg.predict(X_test)
# Calc residuals

train_residuals = y_hat_train - y_train

test_residuals = y_hat_test - y_test
# Calc MSE (Mean Squared Error)

train_mse = mean_squared_error(y_train, y_hat_train)

test_mse = mean_squared_error(y_test, y_hat_test)

print('Train Mean Squarred Error:', train_mse)

print('Test Mean Squarred Error:', test_mse)
fig = plt.figure(figsize=(16, 8))

sns.distplot(y_test - y_hat_test, bins=100);

# sns.distplot(test_residuals, bins=50)
train_error = []

test_error = []



for t in range(5, 95):

    train_temp = []

    test_temp = []

    for i in range(100):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t/100)

        linreg.fit(X_train, y_train)



        y_hat_train = linreg.predict(X_train)

        y_hat_test = linreg.predict(X_test)



        train_temp.append(mean_squared_error(y_train, y_hat_train))

        test_temp.append(mean_squared_error(y_test, y_hat_test))

    

    # save average train/test errors

    train_error.append(np.mean(train_temp))

    test_error.append(np.mean(test_temp))



fig = plt.figure(figsize=(16, 12))

plt.scatter(range(5, 95), train_error, label='training error')

plt.scatter(range(5, 95), test_error, label='testing error')



plt.legend()

plt.show()
from sklearn.model_selection import cross_val_score



cv_5_results  = np.mean(cross_val_score(linreg, X, y, cv=5, scoring="neg_mean_squared_error"))

cv_10_results = np.mean(cross_val_score(linreg, X, y, cv=10, scoring="neg_mean_squared_error"))

cv_20_results = np.mean(cross_val_score(linreg, X, y, cv=20, scoring="neg_mean_squared_error"))
print(cv_5_results, cv_10_results, cv_20_results)
print('Measure of the quality of an estimator - values closer to zero are better\n\n')

print('MAE: ', metrics.mean_absolute_error(y_test, y_hat_test))

print('MSE: ', metrics.mean_squared_error(y_test, y_hat_test))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_hat_test)))