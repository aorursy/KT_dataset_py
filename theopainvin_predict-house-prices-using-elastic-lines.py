import pandas as pd

import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# Import the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Combine the two datasets

entire = pd.merge(train, test, how='outer')
print("Total number of rows: {0}\n\t- {1} in training set\n\t- {2} in testing set"

      "\nNumber of features: {3}".format(len(entire.index), len(train.index),

                                         len(test.index), len(train.columns)))
categ_feat = sorted(list(entire.dtypes[entire.dtypes == 'object'].index))

num_feat = sorted(list(entire.dtypes[(entire.dtypes == 'int64') |\

                              (entire.dtypes == 'float64')].index))

int_feat = sorted(list(entire.dtypes[entire.dtypes == 'int64'].index))

print("{} categorical features:\n".format(len(categ_feat)), categ_feat)

print("\n{} numerical features (float + int):\n".format(len(num_feat)), num_feat)

print("\n{} numerical features (int only):\n".format(len(int_feat)), int_feat)

features = categ_feat + num_feat
def plot_features(x_list, y_name, df, threshold=0.5):

    """

        Generate a figure with an axe per plot, each axe plotting y_name

        feature function of numerical or categorical features in x_list.

        Print the feature names and corresponding regression coefficients

        ordered by coefficient values

        

        Arguments:

            - x_list (list): list of feature names for x-axis

            - y_name (str): feature name for y-axis

            - df (pandas.DataFrame): dataset containing the data to plot

            - threshold (float): threshold value for regression coefficient

              printing

    """

    colors = sns.color_palette(n_colors=len(x_list))

    list_tup = []  # List of (feature_name, reg_coef)

    if not len(x_list)%2:

        row_nb = int(len(x_list)/2)

    else:

        row_nb = int(len(x_list)/2) + 1



    size_y = row_nb * 5

    fig = plt.figure(figsize=(15, size_y*1.05))

    i = 1

    for x_name in x_list:

        ax = fig.add_subplot(row_nb, 2, i)

        if df[x_name].dtypes == 'object':  # Categorical feature

            sns.boxplot(x=x_name, y=y_name, data=df, ax=ax)

        else:  # Numerical feature

            sns.regplot(x=x_name, y=y_name, data=df, ax=ax,

                        color=colors[i-1])

            corr = np.corrcoef(df[x_name], df[y_name])[0, 1]

            list_tup.append((x_name, corr))

            ax.set_title('Regression coef: {:.2f}'.format(corr))

        i += 1

    

    list_tup = [tup for tup in list_tup if tup[1] >= threshold]

    list_tup = sorted(list_tup, key=lambda x: x[1], reverse=True)

    

    for tup in list_tup:

        print("Regression coefficient for {0}:\t{1:.2f}"

              .format(tup[0], tup[1]))
plot_features(categ_feat, 'SalePrice', entire[entire.SalePrice.notnull()])
entire.Utilities.value_counts()
train.groupby('Utilities')['SalePrice'].agg(['median', 'mean', 'std'])
plot_features(num_feat, 'SalePrice', entire[entire.SalePrice.notnull()])
entire['DateSold'] = entire['YrSold'].astype(str)+"/"+entire['MoSold'].astype(str)

entire['DateSold'] = entire['DateSold'].apply(lambda x: datetime.strptime(x, "%Y/%m"))

title = 'Evolution of SalePrice with DateSold'

gpby = entire[entire.SalePrice.notnull()].groupby('DateSold')

gpby['SalePrice'].agg(['median', 'mean', 'std']).plot(grid=True,

                                                          title=title,

                                                          figsize=(8, 6))
title = 'Evolution of SalePrice with YearBuilt'

gpbyDate = entire.groupby('YearBuilt')

gpbyDate['SalePrice'].agg(['median', 'mean', 'std']).plot(grid=True,

                                                          title=title,

                                                          figsize=(8, 6))
def show_nan(df, features, exclude='SalePrice'):

    """

        Analyze a DataFrame and show if it contains missing values.

        Print the names of features containing missing values sorted

        by number of missing values

        

        Arguments:

            - df (DataFrame): dataset to analyse

            - features (list): feature names to be analysed

            - exclude (str): feature name not to be analysed

    """

    features = [f for f in features if f != exclude]

    if True not in df[features].isnull().values:

        print("This dataset does not contain missing value.")

    else:

        list_feat = []

        for feat in features:

            if (True in df[feat].isnull().values):

                miss_val = df[feat].isnull().value_counts()[True]

                prop = round((miss_val / df.Id.count()), 5)

                list_feat.append((feat, miss_val, prop))

        # Sort `list_feat`

        list_feat = sorted(list_feat, key=lambda x: x[-1], reverse=True)

        

        print("This dataset contains {} feature(s) with missing values:"

              .format(len(list_feat)))

        for feat_nb_prop in list_feat:

            print("\t- {1} missing values for '{0}' "

                  "({2:.2%})".format(*feat_nb_prop))
show_nan(entire, features)
entire[(entire.PoolArea > 0) & (entire.PoolQC.isnull())][['OverallQual',

                                                          'OverallCond',

                                                          'PoolQC',

                                                          'PoolArea',

                                                          'SalePrice']]
for i in entire[(entire.PoolArea > 0) & (entire.PoolQC.isnull())].index:

    print("Replacement of PoolQC at index {}".format(i))

    entire.set_value(i, 'PoolQC', 'Fa')
entire[(entire.FireplaceQu.isnull()) & (entire.Fireplaces != 0)]
feat_to_print = ['Id', 'GarageType', 'GarageCars', 'GarageArea',

                 'GarageCond', 'GarageQual']

entire[(entire.GarageQual.isnull()) & (entire.GarageCond.isnull())&\

       (entire.GarageCars != 0) & (entire.GarageArea != 0)][feat_to_print]
gpby = entire.groupby('GarageType')

for gtype, gr in gpby:

    if gtype == 'Detchd':

        print(gr['GarageQual'].value_counts())

        print(gr['GarageCond'].value_counts())
for i in entire[(entire.GarageQual.isnull()) & (entire.GarageCond.isnull())&\

                 (entire.GarageCars != 0) & (entire.GarageArea != 0)].index:

    for feat in ['GarageCond', 'GarageQual']:

        entire.set_value(i, feat, gpby[feat].describe()['top']['Detchd'])



# Fill in the two garage surface related values

for feat in ['GarageArea', 'GarageCars']:

    entire[feat] = entire[feat].fillna(gpby[feat].median()['Detchd'])



# Fill in the rest with 'ABS' value ('absence')

for feat in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',

             'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']:

    entire[feat] = entire[feat].fillna('ABS')



# For GarageYrBlt, it's complicated: a house without garage has no garage

# construction date! We can fill these missing value with the median value

# of the dataset

entire['GarageYrBlt'] = entire['GarageYrBlt'].fillna(entire['GarageYrBlt'].median())
show_nan(entire, features)
# Transform the int value into a string

for feat in ['MSSubClass', 'MoSold']:

    entire[feat] = entire[feat].apply(lambda x: str(x))



# Transfer these features from num_feat list to categ_feat

categ_feat.extend(['MSSubClass', 'MoSold'])

num_feat = [feat for feat in num_feat if feat not in ['MSSubClass', 'MoSold']]
gpby = entire.groupby('MSSubClass')

gpby['LotFrontage'].agg(['median', 'mean'])
dict_subclass = {}



# Iterate on `grby` to get mean `LotFrontage` values

for subclass, gr in gpby:

    dict_subclass[subclass] = gr['LotFrontage'].describe()['mean']



# Warining! NaN value for `MSSubClass` = '150': set it to `entire` mean value

dict_subclass['150'] = entire.LotFrontage.mean()



# Create a function to be applied on `grby`

fill_with_dict = lambda g: g.fillna(dict_subclass[g.name])



# Apply it

entire['LotFrontage'] = gpby['LotFrontage'].apply(fill_with_dict)
show_nan(entire, features)
feat_bsmt = [feat for feat in features if 'Bsmt' in feat]

df_bsmt = entire[(entire.TotalBsmtSF != 0) & ((entire.BsmtCond.isnull()) |\

                                              (entire.BsmtExposure.isnull()) |\

                                              (entire.BsmtQual.isnull()) |\

                                              (entire.BsmtFinType1.isnull()) |\

                                              (entire.BsmtFinType2.isnull()) |\

                                              (entire.BsmtFullBath.isnull()) |\

                                              (entire.BsmtHalfBath.isnull()) |\

                                              (entire.BsmtFinSF1.isnull()) |\

                                              (entire.BsmtFinSF2.isnull()) |\

                                              (entire.BsmtUnfSF.isnull())

                                             )][feat_bsmt]

df_bsmt
# Link between index and features containing missing values

dict_ind_feat = {

    332: ['BsmtFinType2'],

    948: ['BsmtExposure'],

    1487: ['BsmtExposure'],

    2040: ['BsmtCond'],

    2120: df_bsmt.columns,

    2185: ['BsmtCond'],

    2217: ['BsmtQual'],

    2218: ['BsmtQual'],

    2348: ['BsmtExposure'],

    2524: ['BsmtCond'],

}



# Link between feature names and values to replace missing ones

# For houses with basement

values_with_bsmt = {

    'BsmtCond': 'TA',

    'BsmtExposure': 'Av',

    'BsmtFinType1': 'ALQ',

    'BsmtFinType2': 'ALQ',

    'BsmtQual': 'TA',

    'BsmtFinSF1': entire.BsmtFinSF1.median(),

    'BsmtFinSF2': entire.BsmtFinSF2.median(),

    'BsmtFullBath': entire.BsmtFullBath.median(),

    'BsmtHalfBath': entire.BsmtHalfBath.median(),

    'BsmtUnfSF': entire.BsmtUnfSF.median(),

    'TotalBsmtSF': entire.TotalBsmtSF.median(),

}



for ind in df_bsmt.index:

    for feat in dict_ind_feat[ind]:

        entire.set_value(ind, feat, values_with_bsmt[feat])



# Link between feature names and values to replace missing ones

# For houses without basement

values_without_bsmt = {

    'BsmtCond': 'ABS',

    'BsmtExposure': 'ABS',

    'BsmtFinType1': 'ABS',

    'BsmtFinType2': 'ABS',

    'BsmtQual': 'ABS',

    'BsmtFinSF1': 0.,

    'BsmtFinSF2': 0.,

    'BsmtFullBath': 0.,

    'BsmtHalfBath': 0.,

    'BsmtUnfSF': 0.,

    'TotalBsmtSF': 0.,

}

entire = entire.fillna(value=values_without_bsmt)
show_nan(entire, features)
entire.MasVnrType.value_counts()
entire[(entire.MasVnrArea != 0) &\

       (entire.MasVnrType.isnull())][['MasVnrType', 'MasVnrArea']]
entire.loc[(entire.MasVnrType == 'None') & (entire.MasVnrArea != 0),

           ['MasVnrType', 'MasVnrArea']]
entire.loc[((entire.MasVnrType == 'None') | (entire.MasVnrType.isnull())) &\

           ((entire.MasVnrArea != 0) & (entire.MasVnrArea.notnull())),

           'MasVnrType'] = 'BrkFace'

entire = entire.fillna(value={'MasVnrType': 'None', 'MasVnrArea': 0.})

show_nan(entire, features)
entire[entire.MSZoning.isnull()][['MSZoning', 'Neighborhood']]
gpby = entire.groupby('Neighborhood')

dict_neigh_mszone = {}

for neigh, gr in gpby:

    if neigh == 'IDOTRR' or neigh == 'Mitchel':

        print("\nNeighborhood: {}".format(neigh))

        print(gr['MSZoning'].describe())

        dict_neigh_mszone[neigh] = gr['MSZoning'].describe()['top']

    else:

        dict_neigh_mszone[neigh] = np.nan



fill_with_dict = lambda g: g.fillna(dict_neigh_mszone[g.name])

entire['MSZoning'] = entire.groupby('Neighborhood')['MSZoning'].apply(fill_with_dict)
# Fill in the rest with most common values

most_common = {}

for feat in ['Functional', 'Utilities', 'Electrical', 'Exterior1st',

             'Exterior2nd', 'KitchenQual', 'SaleType']:

    most_common[feat] = entire[feat].describe().top



entire = entire.fillna(value=most_common)
show_nan(entire, features)
def plot_skewness(feat_list, df, threshold=0.5):

    """

        Generate a figure plotting numerical feature distribution and its

        skewness, before and after logarithmic transformation

        

        Arguments:

            - feat_list (list): list of feature names

            - df (pandas.DataFrame): dataset containing the data to plot

            - threshold (float): threshold value for regression coefficient

              printing

        

        Returns:

            reduced_skew (list): feature names for which skewness reduction

            is observed after log transformation

    """

    row_nb = len(feat_list)

    colors = sns.color_palette(n_colors=row_nb)

    

    size_y = row_nb * 5

    fig = plt.figure(figsize=(15, size_y*1.05))

    reduced_skew = []

    i, ind_color = 1, 0

    

    for feat in feat_list:

        # Before log transformation

        ax = fig.add_subplot(row_nb, 2, i)

        sns.distplot(df[feat], ax=ax, color=colors[ind_color])

        before = df[feat].skew()

        ax.set_title('Skewness before log: {:.2f}'.format(before))



        # After log transformation

        ax = fig.add_subplot(row_nb, 2, i+1)

        sns.distplot(np.log1p(df[feat]), ax=ax, color=colors[ind_color])

        after = np.log1p(df[feat]).skew()

        ax.set_title('Skewness after log: {:.2f}'.format(after))

        i += 2

        ind_color += 1

        

        if abs(after) < abs(before):

            reduced_skew.append(feat)

    

    return reduced_skew
to_log = plot_skewness([f for f in num_feat if f not in ['Id', 'SalePrice']], entire)
to_log.extend(plot_skewness(['SalePrice'], entire[entire.SalePrice.notnull()]))
for feat in to_log:

    entire[feat] = np.log1p(entire[feat])
ordered = {

    'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

    'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

    'BsmtQual': ['ABS', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],

    'BsmtCond': ['ABS', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],

    'BsmtExposure': ['ABS', 'No', 'Mn', 'Av', 'Gd'],

    'BsmtFinType1': ['ABS', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],

    'BsmtFinType2': ['ABS', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],

    'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

    'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

    'FireplaceQu': ['ABS', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],

    'GarageFinish': ['ABS', 'Unf', 'RFn', 'Fin'],

    'GarageQual': ['ABS', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],

    'GarageCond': ['ABS', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],

    'PoolQC': ['ABS', 'Fa', 'TA', 'Gd', 'Ex'],

    'Fence': ['ABS', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],

}



non_ordered = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',

               'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

               'Neighborhood','Condition1', 'Condition2', 'BldgType',

               'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

               'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',

               'CentralAir', 'Electrical', 'Functional','GarageType',

               'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition']
# Ordered categorical features

for feat, val in ordered.items():

    entire[feat] = entire[feat].astype("category",

                                       categories=val,

                                       ordered=True).cat.codes



# Non ordered categorical features

entire = pd.get_dummies(entire, columns=non_ordered, drop_first=True)



# Drop `Id` and `DateSold` features

entire = entire.drop(['Id', 'DateSold'], axis=1)
# Create the (X, y) training vectors to be injected in our models

X_train = entire[entire.SalePrice.notnull()].drop(['SalePrice'], axis=1)

y_train = entire[entire.SalePrice.notnull()]['SalePrice']



# And the test set

X_test = entire[entire.SalePrice.isnull()].drop(['SalePrice'], axis=1)
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

scaler.fit(entire[[feat for feat in entire.columns if feat != 'SalePrice']])



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, ElasticNetCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score, learning_curve

import warnings

warnings.filterwarnings('ignore')
models = [LinearRegression(), LassoCV(), Ridge(), ElasticNetCV(),

          RandomForestRegressor(), GradientBoostingRegressor()]



for model in models:

    score = cross_val_score(model, X_train, y_train, cv=4,

                            scoring="neg_mean_squared_error")

    print("{0}:\n\tRMSE\t= {1:.5f} "

          .format(model.__class__.__name__, np.sqrt(-score.mean())))
# RandomForestRegressor



for n_estimators in [500, 1500, 2500]:

    model = RandomForestRegressor(n_estimators=n_estimators)

    score = cross_val_score(model, X_train, y_train, cv=4,

                            scoring="neg_mean_squared_error")

    print("{0} with {2} estimators:\n\tRMSE\t= {1:.5f} "

          .format(model.__class__.__name__, np.sqrt(-score.mean()),

                  n_estimators))
# GradientBoostingRegressor



for n_estimators in [500, 1500, 2500]:

    model = GradientBoostingRegressor(n_estimators=n_estimators)

    score = cross_val_score(model, X_train, y_train, cv=4,

                            scoring="neg_mean_squared_error")

    print("{0} with {2} estimators:\n\tRMSE\t= {1:.5f} "

          .format(model.__class__.__name__, np.sqrt(-score.mean()),

                  n_estimators))
def plot_learning_curve(models, X_train, y_train, cv=5,

                        tr_sizes=np.linspace(0.1, 1.0, 5)):

    """

        Generate a figure plotting learning curves for different models

        

        Arguments:

            - models (list): list of model instances

            - X_train (numpy.ndarray): training set learning values

            - y_train (numpy.ndarray): training set target values

            - cv (int): number of splits of the training set

            - tr_sizes (numpy.ndarray): number of training examples used

              to generate the learning curves

    """

    

    scoring="neg_mean_squared_error"

    row_nb = len(models)

    

    size_y = row_nb * 5

    fig = plt.figure(figsize=(15, size_y*1.05))

    

    for i, model in enumerate(models):

        train_size, train_scores, test_scores = learning_curve(model,

                                                               X_train,

                                                               y_train,

                                                               cv=cv,

                                                               train_sizes=tr_sizes,

                                                               scoring=scoring)

        train_scores = np.sqrt(-train_scores)

        train_scores_mean = np.mean(train_scores, axis=1)

        train_scores_std = np.std(train_scores, axis=1)

        test_scores = np.sqrt(-test_scores)

        test_scores_mean = np.mean(test_scores, axis=1)

        test_scores_std = np.std(test_scores, axis=1)



        ax = fig.add_subplot(row_nb, 2, i+1)

        ax.set_title(model.__class__.__name__)

        ax.set_xlabel("Training examples")

        ax.set_ylabel("RMSE value")



        ax.plot(tr_sizes, train_scores_mean, 'o-', color='r',

                label="Training score")

        ax.fill_between(tr_sizes, train_scores_mean - train_scores_std,

                        train_scores_mean + train_scores_std, alpha=0.1,

                        color='r')

        ax.plot(tr_sizes, test_scores_mean, 'o-', color='g',

                label="Cross-validation score")

        ax.fill_between(tr_sizes, test_scores_mean - test_scores_std,

                        test_scores_mean + test_scores_std, alpha=0.1,

                        color='g')

        ax.legend(loc='best')
models = [LassoCV(), Ridge(), ElasticNetCV(),

          RandomForestRegressor(n_estimators=1500),

          GradientBoostingRegressor(n_estimators=1500)]

plot_learning_curve(models, X_train, y_train)
final_regressor = ElasticNetCV()

final_regressor.fit(X_train, y_train)



# Predict on the test set with the chosen model

pred = np.expm1(final_regressor.predict(X_test))



# Build the requested DataFrame

ids = test['Id']



dict_pred = {'Id': ids, 'SalePrice': pred}

df_pred = pd.DataFrame(dict_pred).set_index(['Id'])



# Get the date to append to the 'Predictions_*.csv' file

from datetime import datetime

date_str = datetime.now().strftime('%Y-%m-%d_%Hh%M')



# The prediction file's name will contain the date of creation

df_pred.to_csv('Predictions_' + date_str + '.csv')