import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

import time

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Configs

pd.options.display.float_format = '{:,.3f}'.format

sns.set(style="whitegrid")

plt.style.use('seaborn')

seed = 42

np.random.seed(seed)
sns.set(style="whitegrid")

plt.style.use('seaborn')
file_path = '/kaggle/input/used-cars-price-prediction/train-data.csv'

df_train = pd.read_csv(file_path)

print("Train DataSet = {} rows and {} columns".format(df_train.shape[0], df_train.shape[1]), '\n')



quantitative = [f for f in df_train.columns if df_train.dtypes[f] != 'object']



qualitative = [f for f in df_train.columns if df_train.dtypes[f] == 'object']



print("Qualitative Variables: (Numerics)", "\n=>", qualitative,

      "\n\nQuantitative Variable: (Strings)\n=>", quantitative)



df_train.head()
file_path = '/kaggle/input/used-cars-price-prediction/test-data.csv'

df_test = pd.read_csv(file_path)

print("Test DataSet = {} rows and {} columns".format(df_test.shape[0], df_test.shape[1]))
df_all = pd.concat([df_train, df_test]).reset_index(drop=True)

print(df_all.shape)

df_all.head()
def eda_categ_feat_desc_df(series_categorical):

    """Generate DataFrame with quantity and percentage of categorical series

    @series_categorical = categorical series

    """

    series_name = series_categorical.name

    val_counts = series_categorical.value_counts()

    val_counts.name = 'quantity'

    val_percentage = series_categorical.value_counts(normalize=True)

    val_percentage.name = "percentage"

    val_concat = pd.concat([val_counts, val_percentage], axis = 1)

    val_concat.reset_index(level=0, inplace=True)

    val_concat = val_concat.rename( columns = {'index': series_name} )

    return val_concat
def eda_categ_feat_desc_plot(series_categorical, title = ""):

    """Generate 2 plots: barplot with quantity and pieplot with percentage. 

       @series_categorical: categorical series

       @title: optional

    """

    series_name = series_categorical.name

    val_counts = series_categorical.value_counts()

    val_counts.name = 'quantity'

    val_percentage = series_categorical.value_counts(normalize=True)

    val_percentage.name = "percentage"

    val_concat = pd.concat([val_counts, val_percentage], axis = 1)

    val_concat.reset_index(level=0, inplace=True)

    val_concat = val_concat.rename( columns = {'index': series_name} )

    

    fig, ax = plt.subplots(figsize = (15,4), ncols=2, nrows=1) # figsize = (width, height)

    if(title != ""):

        fig.suptitle(title, fontsize=18)

        fig.subplots_adjust(top=0.8)



    s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])

    for index, row in val_concat.iterrows():

        s.text(row.name, row['quantity'], row['quantity'], color='black', ha="center")



    s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),

                             labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],

                             title="Percentage Plot")



    ax[1].set_ylabel('')

    ax[0].set_title('Quantity Plot')



    plt.show()
def eda_numerical_feat(series, title="", number_format="", with_label=True):

    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 4), sharex=False)

#     print(series.describe())

    if(title != ""):

        f.suptitle(title, fontsize=18)

    sns.distplot(series, ax=ax1, rug=True)

    sns.boxplot(series, ax=ax2)

    ax1.set_title("distplot")

    ax2.set_title("boxplot")

    if(with_label):

        describe = series.describe()

        labels = { 'min': describe.loc['min'], 'max': describe.loc['max'], 

              'Q1': describe.loc['25%'], 'Q2': describe.loc['50%'],

              'Q3': describe.loc['75%']}

        if(number_format != ""):

            for k, v in labels.items():

                ax2.text(v, 0.3, k + "\n" + number_format.format(v), ha='center', va='center', fontweight='bold',

                         size=10, color='white', bbox=dict(facecolor='#445A64'))

        else:

            for k, v in labels.items():

                ax2.text(v, 0.3, k + "\n" + str(v), ha='center', va='center', fontweight='bold',

                     size=10, color='white', bbox=dict(facecolor='#445A64'))

    plt.show()
def plot_model_score_regression(models_name_list, model_score_list, title=''):

    fig = plt.figure(figsize=(15, 6))

    ax = sns.pointplot( x = models_name_list, y = model_score_list, 

        markers=['o'], linestyles=['-'])

    for i, score in enumerate(model_score_list):

        ax.text(i, score + 0.002, '{:.4f}'.format(score),

                horizontalalignment='left', size='large', 

                color='black', weight='semibold')

    plt.ylabel('Score', size=20, labelpad=12)

    plt.xlabel('Model', size=20, labelpad=12)

    plt.tick_params(axis='x', labelsize=12)

    plt.tick_params(axis='y', labelsize=12)

    plt.xticks(rotation=70)

    plt.title(title, size=20)

    plt.show()
def describe_y_by_x_cat_boxplot(dtf, x_feat, y_target, title='', figsize=(15,5), rotatioon_degree=0):

    the_title = title if title != '' else '{} by {}'.format(y_target, x_feat)

    fig, ax1 = plt.subplots(figsize = figsize)

    sns.boxplot(x=x_feat, y=y_target, data=df_train, ax=ax1)

    ax1.set_title(the_title, fontsize=18)

    plt.xticks(rotation=rotatioon_degree) # recomend 70

    plt.show()
df_missing = pd.concat([df_train.isnull().sum(), df_test.isnull().sum()], axis=1)

df_missing = df_missing.rename({0: 'train', 1: 'test'}, axis = 1).fillna(0)

df_missing['test'] = df_missing['test'].astype(int)

df_missing.T
df_train.drop(['Unnamed: 0'], axis=1, inplace=True)

df_train.drop(['New_Price'],  axis=1, inplace=True)
# get numeric part

for i in range(df_train.shape[0]):

    try:

        df_train.at[i, 'Name']        = df_train['Name'][i].split()[0] # Get First Name

        df_train.at[i, 'Mileage'] = df_train['Mileage'][i].split()[0] # Get Number

        df_train.at[i, 'Engine']     = df_train['Engine'][i].split()[0] # Get Number

        df_train.at[i, 'Power']     = df_train['Power'][i].split()[0] # Get Number

    except:

        pass # is nan or string



# # Engine :: Example: 998 CC

replace_engine = {'1798 CC': '1798', '72 CC': '72'}

df_train['Engine'] = df_train['Engine'].replace(replace_engine).astype(float)

median_engine = df_train['Engine'].median()

df_train['Engine'] = df_train['Engine'].fillna(median_engine)



# # Mileage :: Example: 26.6 km/kg || 19.67 kmpl

df_train['Mileage'] = df_train['Mileage'].astype(float)

df_train['Mileage'] = df_train['Mileage'].fillna( df_train['Mileage'].median() )



# # Power ::  Example: 58.16 bhp

replace_power = {'nan': '-1', 'null': '-1', '73 bhp': '-1', '41 bhp': '-1'}

df_train['Power'] = df_train['Power'].replace(replace_power)

median_power = df_train['Power'].median()

df_train['Power'] = df_train['Power'].replace({-1: median_power}).fillna(median_power).astype(float)

df_train['Power'] = df_train['Power'].replace({-1: median_power})



# # Seats :: float WARNING: There Are Seat=0 (Error)

median_seats = df_train['Seats'].median()

df_train['Seats'] = df_train['Seats'].replace({0.0: median_seats}).fillna(median_seats).astype(float)



df_train.isnull().sum()
# After data cleaning

df_train.head()
# % by Name in DataSet, is 30 Names: Maruti: 20%, Hyundai: 10%, Toyota: 6% .....

eda_categ_feat_desc_df(df_train['Name']).T
# well distributed among locations

eda_categ_feat_desc_plot(df_train['Location'], 'Location')
# Seats: 84% is 5; 11% is 7; the others together are 5%

eda_categ_feat_desc_df(df_train['Seats'])   
eda_categ_feat_desc_plot(df_train['Owner_Type'], 'Owner_Type')
eda_numerical_feat(df_train['Kilometers_Driven'], 'Kilometers_Driven distribution')
# remove row with outiliers

df_train.drop(  df_train[df_train['Kilometers_Driven'] > 300000 ].index, inplace = True)
eda_numerical_feat(df_train['Kilometers_Driven'], 'Kilometers_Driven distribution')
eda_numerical_feat(df_train['Year'], 'Year distribution')
eda_categ_feat_desc_plot(df_train['Fuel_Type'], 'Fuel Type')
eda_categ_feat_desc_plot(df_train['Transmission'], 'Transmission')
eda_numerical_feat(df_train['Mileage'], 'Mileage distribution')
eda_numerical_feat(df_train['Engine'], 'Engine distribution')
eda_numerical_feat(df_train['Power'], 'Power distribution')
# Remove Outiliers Rows with Price > 100

df_train.drop(  df_train[df_train['Price'] > 100 ].index, inplace = True)

eda_numerical_feat(df_train['Price'], 'Price distribution')
# The bigger the year tends to be the higher the price

describe_y_by_x_cat_boxplot(df_train, 'Year', 'Price', figsize=(18,5))
describe_y_by_x_cat_boxplot(df_train, 'Name', 'Price', figsize=(20,8), rotatioon_degree=65)
fig, (ax1) = plt.subplots(figsize = (20,7))



sns.boxplot(x="Location", y="Price", data=df_train, ax=ax1)

ax1.set_title('Price by Location')

plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize = (15,10), ncols=2, nrows=2, sharex=False, sharey=False)

font_size = 14

fig.suptitle('Price by Fuel_Type', fontsize=18)



sns.boxplot(x="Fuel_Type", y="Price", data=df_train, ax=ax1)

sns.distplot(df_train[(df_train.Fuel_Type == 'CNG')]["Price"],ax=ax2, hist=False, label='CNG')

sns.distplot(df_train[(df_train.Fuel_Type == 'Diesel')]['Price'],ax=ax2, hist=False, label='Diesel')

sns.distplot(df_train[(df_train.Fuel_Type == 'Petrol')]['Price'],ax=ax2, hist=False, label='Petrol')

sns.distplot(df_train[(df_train.Fuel_Type == 'LPG')]['Price'],ax=ax2, hist=False, label='LPG')

sns.distplot(df_train[(df_train.Fuel_Type == 'Electric')]['Price'],ax=ax2, hist=False, label='Electric')



# Only Diesel and Petrol, cuz LPG, Eletric and CNG has few rows

df_real_fuel = df_train[ (df_train.Fuel_Type == 'Diesel') | (df_train.Fuel_Type == 'Petrol') ]

sns.boxplot(x="Fuel_Type", y="Price", data=df_real_fuel, ax=ax3)

sns.distplot(df_train[(df_train.Fuel_Type == 'Diesel')]['Price'],ax=ax4,hist=False, label='Diesel')

sns.distplot(df_train[(df_train.Fuel_Type == 'Petrol')]['Price'],ax=ax4,hist=False, label='Petrol')



ax1.set_title('Price by Fuel_Type: All Fuel Type', fontsize=font_size)

ax2.set_title('Distribution of Price for Fuel_Type: All', fontsize=font_size)

ax3.set_title('Price by Fuel_Type: Only Diesel and Petrol', fontsize=font_size)

ax3.set_title('Distribution of Price for Fuel_Type: : Only Diesel and Petrol', fontsize=font_size)



plt.show()
ax = sns.scatterplot(x="Kilometers_Driven", y="Price", data=df_train)

ax.set_title("Price by Kilometers_Driven")

plt.show()
ax = sns.scatterplot(x="Mileage", y="Price", data=df_train)

ax.set_title("Price by Mileage")

plt.show()
ax = sns.scatterplot(x="Engine", y="Price", data=df_train)

ax.set_title("Price by Engine")

plt.show()
ax = sns.scatterplot(x="Power", y="Price", data=df_train)

ax.set_title("Price by Power")

plt.show()
fig, (ax1) = plt.subplots(figsize = (10,7))



sns.boxplot(x="Seats", y="Price", data=df_train, ax=ax1)

ax1.set_title('Price by Seats')

plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize = (17,11), ncols=2, nrows=2, sharex=False, sharey=False)



sns.scatterplot(x="Engine",  y="Price", hue="Transmission", data=df_train, ax=ax1)

sns.scatterplot(x="Mileage", y="Price", hue="Transmission", data=df_train, ax=ax2)

sns.scatterplot(x="Power",   y="Price", hue="Transmission", data=df_train, ax=ax3)

sns.scatterplot(x="Kilometers_Driven", y="Price", hue="Transmission", data=df_train, ax=ax4)



fig.suptitle('Price by some Numerical Features and Transmission', fontsize=18)

plt.show()
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(figsize = (13,11), ncols=2, nrows=3)



sns.scatterplot(x="Power", y="Price", hue="Transmission", data=df_train, ax=ax1)

ax2.remove()



sns.scatterplot(x="Power", y="Price", hue="Owner_Type", data=df_train, ax=ax3)

df_train_re = df_train[ (df_train['Owner_Type'] == 'Fourth & Above') | (df_train['Owner_Type'] == 'Third') ]

sns.scatterplot(x="Power", y="Price", hue="Owner_Type", data=df_train_re, ax=ax4, palette='Set2')



sns.scatterplot(x="Power", y="Price", hue="Fuel_Type", data=df_train, ax=ax5)

df_train_re = df_train[ (df_train['Fuel_Type'] == 'Electric') | (df_train['Fuel_Type'] == 'LPG') | (df_train['Fuel_Type'] == 'CNG')]

sns.scatterplot(x="Power", y="Price", hue="Fuel_Type", data=df_train_re, palette='Set2', ax=ax6)



# Config Titles

fig.suptitle('Price by Power with others features', fontsize=18)

ax1.set_title("Price by Power and Transmission")

ax3.set_title("Price by Power and Ower Type: All")

ax4.set_title("Price by Power and Ower Type: 3°, 4° and others")

ax5.set_title("Price by Power and Fuel Type: All")

ax6.set_title("Price by Power and Fuel Type: CNG, PLG and ELetric")



# padding betwen axis

fig.tight_layout(pad=0.4)



# Correct height of suptitle

plt.subplots_adjust(top=0.92)



plt.show()
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(figsize = (13,11), ncols=2, nrows=3)

ax2.remove()



sns.scatterplot(x="Kilometers_Driven", y="Price", hue="Transmission", data=df_train, ax=ax1)



sns.scatterplot(x="Kilometers_Driven", y="Price", hue="Owner_Type", data=df_train, ax=ax3)

df_train_re = df_train[ (df_train['Owner_Type'] == 'Fourth & Above') | (df_train['Owner_Type'] == 'Third') ]

sns.scatterplot(x="Kilometers_Driven", y="Price", hue="Owner_Type", data=df_train_re, ax=ax4, palette='Set2')



sns.scatterplot(x="Kilometers_Driven", y="Price", hue="Fuel_Type", data=df_train, ax=ax5)

df_train_re = df_train[ (df_train['Fuel_Type'] == 'Electric') | (df_train['Fuel_Type'] == 'LPG') | (df_train['Fuel_Type'] == 'CNG')]

sns.scatterplot(x="Kilometers_Driven", y="Price", hue="Fuel_Type", data=df_train_re, palette='Set2', ax=ax6)



# Config Titles

fig.suptitle('Price by Kilometers Driven with others features', fontsize=18)

ax3.set_title("Price by Kilometers Driven and Ower Type: All")

ax4.set_title("Price by Kilometers Driven and Ower Type: 3°, 4° and others")

ax5.set_title("Price by Kilometers Driven and Fuel Type: All")

ax6.set_title("Price by Kilometers Driven and Fuel Type: CNG, PLG and ELetric")



# padding betwen axis

fig.tight_layout(pad=0.4)



# Correct height of suptitle

plt.subplots_adjust(top=0.92)



plt.show()
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(figsize = (13,11), ncols=2, nrows=3)



sns.scatterplot(x="Mileage", y="Price", hue="Transmission", data=df_train, ax=ax1)



sns.scatterplot(x="Mileage", y="Price", hue="Owner_Type", data=df_train, ax=ax3)

df_train_re = df_train[ (df_train['Owner_Type'] == 'Fourth & Above') | (df_train['Owner_Type'] == 'Third') ]

sns.scatterplot(x="Mileage", y="Price", hue="Owner_Type", data=df_train_re, ax=ax4, palette='Set2')



sns.scatterplot(x="Mileage", y="Price", hue="Fuel_Type", data=df_train, ax=ax5)

df_train_re = df_train[ (df_train['Fuel_Type'] == 'Electric') | (df_train['Fuel_Type'] == 'LPG') | (df_train['Fuel_Type'] == 'CNG')]

sns.scatterplot(x="Mileage", y="Price", hue="Fuel_Type", data=df_train_re, palette='Set2', ax=ax6)



# Config Titles

fig.suptitle('Price by Mileage with others features', fontsize=18)

ax3.set_title("Price by Mileage and Ower Type: All")

ax4.set_title("Price by Mileage and Ower Type: 3°, 4° and others")

ax5.set_title("Price by Mileage and Fuel Type: All")

ax6.set_title("Price by Mileage and Fuel Type: CNG, PLG and ELetric")



# padding betwen axis

fig.tight_layout(pad=0.4)



# Correct height of suptitle

plt.subplots_adjust(top=0.92)



plt.show()
df_train.head()
# Name

df_final_train = df_train.drop(['Name'],axis=1)



# Location

df_location = pd.get_dummies(df_train['Location'],drop_first=True)



# Fuel Type

df_fuel_typ = pd.get_dummies(df_train['Fuel_Type'],drop_first=True)



# Transmission

df_final_train['Transmission'].replace( {'Manual': 0, 'Automatic': 1}, inplace=True) 



# Owener_Type

df_final_train['Owner_Type'].replace( {'First': 1, 'Second': 2, "Third": 3,"Fourth & Above":4}, inplace=True) 



df_final_train = pd.concat([df_final_train, df_location, df_fuel_typ],axis=1).drop(['Location', 'Fuel_Type'],axis=1)

df_final_train.head()
for i in range(df_test.shape[0]):

    try:

        df_test.at[i, 'Name']        = df_test['Name'][i].split()[0] # Get First Name

        df_test.at[i, 'Mileage'] = df_test['Mileage'][i].split()[0] # Get Number

        df_test.at[i, 'Engine']     = df_test['Engine'][i].split()[0] # Get Number

        df_test.at[i, 'Power']     = df_test['Power'][i].split()[0] # Get Number

    except:

        pass # is nan or string



# # Engine :: 998 CC



replace_engine = {'1798 CC': '1798', '72 CC': '72'}

df_test['Engine'] = df_test['Engine'].replace(replace_engine).astype(float)

median_engine = df_test['Engine'].median()

df_test['Engine'] = df_test['Engine'].fillna(median_engine)



# # Mileage :: 26.6 km/kg || 19.67 kmpl

df_test['Mileage'] = df_test['Mileage'].astype(float)

df_test['Mileage'] = df_test['Mileage'].fillna( df_test['Mileage'].median() )



# # Power :: 58.16 bhp

replace_power = {'nan': '-1', 'null': '-1', '73 bhp': '-1', '41 bhp': '-1'}

df_test['Power'] = df_test['Power'].replace(replace_power)

median_power = df_test['Power'].median()

df_test['Power'] = df_test['Power'].replace({-1: median_power}).fillna(median_power).astype(float)

df_test['Power'] = df_test['Power'].replace({-1: median_power})



# # Seats :: float WARNING: There Are Seat=0 (Error)

median_seats = df_test['Seats'].median()

df_test['Seats'] = df_test['Seats'].replace({0.0: median_seats}).fillna(median_seats).astype(float)



df_test.isnull().sum()
# Name

df_final_test = df_test.drop(['Name','New_Price','Unnamed: 0'],axis=1)



# Location

df_location = pd.get_dummies(df_test['Location'],drop_first=True)



# Fuel Type

df_fuel_typ = pd.get_dummies(df_test['Fuel_Type'],drop_first=True)



# Transmission

df_final_test['Transmission'].replace( {'Manual': 0, 'Automatic': 1}, inplace=True) 



# Owener_Type

df_final_test['Owner_Type'].replace( {'First': 1, 'Second': 2, "Third": 3,"Fourth & Above":4}, inplace=True) 



df_final_test = pd.concat([df_final_test, df_location, df_fuel_typ],axis=1).drop(['Location', 'Fuel_Type'],axis=1)

df_final_test.head()
df_final_train.head()
corr_matrix = df_final_train.corr()

f, ax1 = plt.subplots(figsize=(18, 14), sharex=False)



ax1.set_title('Top Corr to {}'.format('"Price"'))

cols_top = corr_matrix.sort_values(by="Price", ascending=False)['Price'].index



cm = np.corrcoef(df_final_train[cols_top].values.T)

mask = np.zeros_like(cm)

mask[np.triu_indices_from(mask)] = True

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',

                 annot_kws={'size': 14}, yticklabels=cols_top.values,

                 xticklabels=cols_top.values, mask=mask, ax=ax1)



# see the leftmost column, it's the correlations ordered from highest to lowest for Price
from sklearn.model_selection import train_test_split





X = df_final_train.loc[:, df_final_train.columns != 'Price'].values

Y = df_final_train['Price'].values



x_features = df_final_train.loc[:, df_final_train.columns != 'Price'].columns.tolist()



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler, scale

from sklearn.decomposition import PCA

from sklearn.linear_model import ElasticNet, LassoCV, BayesianRidge, LassoLarsIC

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV, LinearRegression

from sklearn.kernel_ridge import KernelRidge

from mlxtend.regressor import StackingCVRegressor

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.svm import SVR



from lightgbm import LGBMRegressor

from xgboost import XGBRegressor
# Setup cross validation folds

kf = KFold(n_splits=4, random_state=42, shuffle=True)



# Define error metrics

def rmse(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=x_train):

    rmse = np.sqrt(-cross_val_score(model, X, y_train, scoring="neg_mean_squared_error", cv=kf))

    return (rmse)
# Create ML Models



# Light Gradient Boosting Regressor

lightgb_model = LGBMRegressor(objective='regression',  num_leaves=6, learning_rate=0.01,  n_estimators=7000,

                       max_bin=200,  bagging_fraction=0.8, bagging_freq=4,  bagging_seed=8,

                       feature_fraction=0.2, feature_fraction_seed=8, min_sum_hessian_in_leaf = 11,

                       verbose=-1, random_state=42)



# XGBoost Regressor

xgboost_model = XGBRegressor(learning_rate=0.01, n_estimators=6000, max_depth=4, min_child_weight=0,

                       gamma=0.6, subsample=0.7, colsample_bytree=0.7, objective='reg:squarederror',

                       nthread=-1, scale_pos_weight=1, seed=42, reg_alpha=0.00006, random_state=42)



# Linear Regressor

linear_model = LinearRegression()



# Ridge Regressor

ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 

                1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]

ridge_model = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))



# Lasso Regressor

lasso_alphas2 = [5e-05, 0.0001, 0.0008, 0.01, 0.1, 1]

lasso_model = make_pipeline(RobustScaler(),

                      LassoCV(max_iter=1e7, alphas=lasso_alphas2,

                              random_state=42, cv=kf))



# Elastic Net Regressor

elastic_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

elastic_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

elasticnet_model = make_pipeline(RobustScaler(),  

                           ElasticNetCV(max_iter=1e7, alphas=elastic_alphas,

                                        cv=kf, l1_ratio=elastic_l1ratio))



# Kernel Ridge

keridge_model = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



# Support Vector Regressor

svm_model = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))



# Gradient Boosting Regressor

gboost_model = GradientBoostingRegressor(n_estimators=6000, learning_rate=0.01, max_depth=4, max_features='sqrt', 

                                min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)  



# Random Forest Regressor

randomforest_model = RandomForestRegressor(n_estimators=1200, max_depth=15, min_samples_split=5, min_samples_leaf=5,

                          max_features=None, oob_score=True, random_state=42)



# Neural Net

neuralnet_model = MLPRegressor()



# Extra Tree Regressor

extratree_model = ExtraTreesRegressor()
regressor_models = {

    'Linear': linear_model,

    'Ridge': ridge_model,

    'Lasso': lasso_model,

    'KernelRidge': keridge_model,

    'ElasticNet': elasticnet_model,

    'SVM': svm_model,

    'RandomForest': randomforest_model,

    'ExtraTree': extratree_model,

    'NeuralNet': neuralnet_model,

    'GBoost': gboost_model,

    'LightGB': lightgb_model,

    'XGBoost': xgboost_model,

}
## Cross Validation



cv_scores = {}

t_start = time.time()



for model_name, model in regressor_models.items():

    print('{:17}'.format(model_name), end='')

    t0 = time.time()

    score = cv_rmse(model)

    m, s = score.mean(), score.std()

    cv_scores[model_name] = [m,s]

    print('| MSE in CV | rmse_mean: {:11,.3f}, | rmse_std: {:9,.3f}  | took: {:9,.3f} s |'.format(m,s, time.time() - t0))

    

print('\nTime total to CrossValidation: took {:9,.3f} s'.format(time.time() - t_start)) # 200s



# Show Sorted Train Scores DataFrame

df_cv = pd.DataFrame(data = cv_scores.values(), columns=['rmse_cv', 'std_cv'], index=cv_scores.keys())

df_cv = df_cv.sort_values(by='rmse_cv').reset_index().rename({'index': 'model'}, axis=1)

df_cv
# Def Stack Model: Stack up some the models above, optimized using one ml model

stack_regressors = (regressor_models['XGBoost'],

                    regressor_models['GBoost'],

                    regressor_models['LightGB'],

                    regressor_models['RandomForest'],

                    regressor_models['ExtraTree']

                   )



stack_model = StackingCVRegressor(regressors = stack_regressors,

                                meta_regressor = regressor_models['XGBoost'],

                                use_features_in_secondary=True)



regressor_models['Stack'] = stack_model
train_scores = {}

t_start = time.time()



for model_name, model in regressor_models.items():

    print('{:14}'.format(model_name), end='')

    t0 = time.time()

    if(model_name == 'Stack'):

        model  = model.fit(np.array(x_train), np.array(y_train))

        y_pred = model.predict(np.array(x_train))

    else:

        model  = model.fit( x_train, y_train )

        y_pred = model.predict(x_train)

    r2, mse = r2_score(y_train, y_pred), mean_squared_error(y_train, y_pred)

    train_scores[model_name] = [r2, mse, np.sqrt(mse)]

    text_print = '| Train | r2: {:5,.3f} | mse: {:7,.3f}  | took: {:9,.3f} s |'

    print(text_print.format(r2, mse, time.time() - t0))

    regressor_models[model_name] = model

    

print('\nTime total to Fit Models: took {:9,.3f} s'.format(time.time() - t_start)) # UsedCars: 686 s = 11min
# Blend Model is use a porcentage of some models mixing

class BlendModel:

    

    @classmethod

    def predict(self, X):

        return ((0.10 * regressor_models['LightGB'].predict(X)) + \

            (0.10 * regressor_models['GBoost'].predict(X)) + \

            (0.15 * regressor_models['XGBoost'].predict(X)) + \

            (0.10 * regressor_models['LightGB'].predict(X)) + \

            (0.20 * regressor_models['RandomForest'].predict(X)) + \

            (0.35 * regressor_models['Stack'].predict(np.array(X))))



regressor_models['BlendModel'] = BlendModel()

y_pred = BlendModel.predict(x_train)

r2, mse = r2_score(y_train, y_pred), mean_squared_error(y_train, y_pred)

train_scores['BlendModel'] = [r2, mse, np.sqrt(mse)]

print('RMSE score on train data to Blend Model:\n\t=>', np.sqrt(mse))
from mlens.ensemble import SuperLearner



# create a list of base-models

def get_models_to_super_leaner():

    models = list()

    models.append(regressor_models['LightGB'])

    models.append(regressor_models['GBoost'])

    models.append(regressor_models['XGBoost'])

    models.append(regressor_models['LightGB'])

    models.append(regressor_models['ExtraTree'])

    models.append(regressor_models['RandomForest'])

    models.append(regressor_models['KernelRidge'])

    return models



# create the super learner

def get_super_learner(X):

    ensemble = SuperLearner(scorer=rmse, folds=5, shuffle=True, sample_size=len(X))

    # add base models

    models = get_models_to_super_leaner()

    ensemble.add(models)

    # add the meta model

    ensemble.add_meta(LinearRegression())

    return ensemble



# key to regressros models

model_name = 'SuperLeaner'



# create the super learner

ensemble = get_super_learner(x_train)

# fit the super learner

t0 = time.time()

ensemble.fit(x_train, np.array(y_train)) # UsedCars: took 565s = 10min

# pred and evaluate in train dataset

y_pred = ensemble.predict(x_train)

r2, mse = r2_score(y_train, y_pred), mean_squared_error(y_train, y_pred)

train_scores[model_name] = [r2, mse, np.sqrt(mse)]

# show results

text_print = '| Super Leaner in Train | r2: {:6,.3f}, | mse: {:9,.3f}  | took: {:9,.3f} s |\n'

print(text_print.format(r2, mse, time.time() - t0))

# set in dict regressors

regressor_models[model_name] = ensemble

# summarize base learners

print(ensemble.data)

# evaluate meta model
# Show train_scores dataframe

df_train_scores = pd.DataFrame(data = train_scores.values(),index=train_scores.keys(), columns=['r2_train', 'mse_train', 'rmse_train'])

df_train_scores = df_train_scores.sort_values(by='r2_train', ascending=False).reset_index().rename({'index': 'model'}, axis=1)

df_train_scores
test_scores = {}



# predcit x_test to y_test and compare

for model_name, model in regressor_models.items():

    if(model_name == 'Stack'):

        y_pred = model.predict(np.array(x_test))

    else:

        y_pred = model.predict(x_test)

    r2, mse = r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)

    test_scores[model_name] = [r2, mse, np.sqrt(mse)]

    

# Sort DF test scores

df_test_scores = pd.DataFrame(data = test_scores.values(), columns=['r2_test', 'mse_test', 'rmse_test'], index=test_scores.keys())

df_test_scores = df_test_scores.sort_values(by='r2_test', ascending=False).reset_index().rename({'index': 'model'}, axis=1)

df_test_scores
# Include Blend in Train Scores

df_train_scores = pd.DataFrame(data = train_scores.values(),index=train_scores.keys(), columns=['r2_train', 'mse_train', 'rmse_train'])

df_train_scores = df_train_scores.sort_values(by='r2_train', ascending=False).reset_index().rename({'index': 'model'}, axis=1)



# df_test_scores

df_cv2 = df_cv.merge(df_train_scores, how='right'  ,left_on='model', right_on='model')

df_final_scores = df_cv2.merge(df_test_scores, how='right' ,left_on='model', right_on='model')



df_final_scores.sort_values(by='mse_test')
plot_model_score_regression(list(test_scores.keys()), [r2 for r2, mse, rmse in test_scores.values()], 'Evaluate Models in Test: R2')
# To one of best models: GBoost



plt.figure(figsize = (12,4))

feat_importances = pd.Series(regressor_models['XGBoost'].feature_importances_)#, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()
from yellowbrick.model_selection import FeatureImportances, RFECV



# FeatureImportances and RFECV to a good model, if put GBoost(the best) take a long time



fig, (ax3,ax4) = plt.subplots(figsize = (15,5), ncols=2, sharex=False, sharey=False)



the_model = 'Linear'

t_start = time.time()



viz3 = FeatureImportances(regressor_models[the_model], ax=ax3, relative=False)

viz3.fit(x_train, y_train)

viz3.finalize()



viz4 = RFECV(regressor_models[the_model], ax=ax4)

viz4.fit(x_train, y_train)

viz4.finalize()



print('Time total to RFECV to {} : took {:9,.3f} s'.format(the_model, time.time() - t_start))



plt.show()
from yellowbrick.regressor import ResidualsPlot, PredictionError

from yellowbrick.model_selection import FeatureImportances, RFECV



# Can't use 'SuperLeaner' than, use the second place: GBoost



fig, (ax1, ax2) = plt.subplots(figsize = (15,5), ncols=2)



viz1 = ResidualsPlot(regressor_models['XGBoost'], ax=ax1)

viz1.score(x_test, y_test)

viz1.finalize()



viz2 = PredictionError(regressor_models['XGBoost'], ax=ax2)

viz2.score(x_test, y_test)  

viz2.finalize()



plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_log_error



y_pred = regressor_models['XGBoost'].predict(x_test)

print('The best Regressor Model to Test DataSet:')

print('MAE : {:14,.3f}'.format(mean_absolute_error(y_pred, y_test)))

print('MSE : {:14,.3f}'.format(mean_squared_error(y_pred, y_test)))

print('RMSE: {:14,.3f}'.format(np.sqrt(mean_squared_error(y_pred, y_test))))

print('R2  : {:14,.3f}'.format(r2_score(y_pred, y_test)))