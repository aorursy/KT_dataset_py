import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import xgboost as xgb



from scipy.stats import kurtosis, skew # to explore statistics on Sale Price



# Importing plotting libraries

from plotly.offline import init_notebook_mode, iplot, plot 

import plotly.graph_objs as go 

init_notebook_mode(connected=True)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
datafile = "../input/ameshousing/Ames_Housing_Data.tsv"

df=pd.read_csv(datafile, sep='\t')
df.head()
# Let's have a look at the features

df.info()
# Quick peek at the data

def expandHead(x, nrow = 6, ncol = 4):

    # https://stackoverflow.com/a/53873661/1578274

    pd.set_option('display.expand_frame_repr', False)

    seq = np.arange(0, len(x.columns), ncol)

    for i in seq:

        print(x.loc[range(0,nrow), x.columns[range(i,min(i+ncol, len(x.columns)))]])

    pd.set_option('display.expand_frame_repr', True)

    

expandHead(df, 3, 8)
# Exclude nominal and ordinal as well (per data dict)

exc_cols = ['Bedroom AbvGr', 'HalfBath', 'Kitchen AbvGr','Bsmt Full Bath', 'Bsmt Half Bath', 'MS SubClass']

numerical_cols = [c for c in df.columns if df[c].dtype != 'object' and c not in exc_cols]



# expandHead(df.loc[:4, numerical_cols], 4, 8)
#Lets start by plotting a heatmap to determine if any variables are correlated

# Correlation Heatmap

# ref: https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57

f, ax = plt.subplots(figsize=(25, 15))

corr = df[numerical_cols].corr()

hm = sns.heatmap(round(corr,2), annot=False, ax=ax, cmap="coolwarm",fmt='.2f',

                 linewidths=.05) # Set annot=True for pearson labels.

f.subplots_adjust(top=0.93)

t= f.suptitle('Housing Attributes Correlation Heatmap', fontsize=18)
df['N_priceLbl'] = df.SalePrice.apply(lambda p: 

                                    'low' if p < 129500 else

                                   'medium' if p < 213500 else

                                   'high')
corr_features = ['Overall Qual', 'Year Built', 'Total Bsmt SF', 'Enclosed Porch', 'SalePrice']



# Density and scatter pair plots for highly correlated features

sns.pairplot(df, vars=corr_features, hue='N_priceLbl', diag_kind = 'kde',

             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             height = 4)
missing_df = df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,15))

rects = ax.barh(ind, missing_df.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")



# Add value labels

for i, v in enumerate(missing_df.missing_count.values):

    ax.text(v + 10, i, str(v), color='b')

plt.show()
# GarageYrBlt has missing values compared to YearBuilt. So let's drop it

df = df.drop('Garage Yr Blt', axis=1)
mask = df['Garage Area'] == 0

df.loc[mask, 'Garage Area'].count()
df.loc[df['Bsmt Half Bath'].isnull(), 'Bsmt Half Bath'] = 0.0

df.loc[df['Bsmt Full Bath'].isnull(), 'Bsmt Full Bath'] = 0.0

df.loc[df['Garage Cars'].isnull(), 'Garage Cars'] = 0.0

df.loc[df['BsmtFin SF 1'].isnull(), 'BsmtFin SF 1'] = 0.0

df.loc[df['BsmtFin SF 2'].isnull(), 'BsmtFin SF 2'] = 0.0

df.loc[df['Bsmt Unf SF'].isnull(), 'Bsmt Unf SF'] = 0.0

df.loc[df['Total Bsmt SF'].isnull(), 'Total Bsmt SF'] = 0.0

df.loc[df['Garage Area'].isnull(), 'Garage Area'] = 0.0

df.loc[df['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
missing_df = df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count', ascending=False)

missing_df
missing_prop = (df.isnull().sum()/df.shape[0]).reset_index()

missing_prop.columns = ['field', 'proportion']

missing_prop = missing_prop.sort_values(by='proportion', ascending=False)

missing_prop.head()
df.loc[df.nunique().values == 1]

# There are no constant variables
plt.figure(figsize=(14,6))



sns.countplot(x='Neighborhood', data=df, order = df['Neighborhood'].value_counts()[:10].index)

plt.title("Top 10 Most Frequent Neighborhoods", fontsize=20) # Adding Title and seting the size

plt.xlabel("Neighborhood", fontsize=16) # Adding x label and seting the size

plt.ylabel("Sale Counts", fontsize=16) # Adding y label and seting the size

plt.xticks(rotation=45) # Adjust the xticks, rotating the labels



plt.show()
plt.figure(figsize=(16,6))

sns.set_style("whitegrid")

g1 = sns.boxenplot(x='Neighborhood', y='SalePrice', 

                   data=df[df['SalePrice'] > 0])

g1.set_title('Neighborhoods by SalePrice', fontsize=20)

g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

g1.set_xlabel('Neighborhood', fontsize=18) # Xlabel

g1.set_ylabel('SalePrice', fontsize=18) #Ylabel



plt.show()
# Setting the first trace

trace1 = go.Histogram(x=df["Yr Sold"],

                      name='Year Count')



# Setting the second trace

trace2 = go.Histogram(x=df["Mo Sold"],

                name='Month Count')



data = [trace1, trace2]



# Creating menu options

updatemenus = list([

    dict(active=-1,

         x=-0.15,

         buttons=list([  

             dict(

                 label = 'Years Count',

                 method = 'update',

                 args = [{'visible': [True, False]}, # This trace visible flag

                         {'title': 'Count of Year'}]),

             dict(

                 label = 'Months Count',

                 method = 'update',

                 args = [{'visible': [False, True]},

                         {'title': 'Count of Months'}])

         ]))

])



layout = dict(title='Number of Sales by Year/Month (Select from Dropdown)',

              showlegend=False,

              updatemenus=updatemenus,

#              xaxis = dict(

#                  type="category"

#                      ),

              barmode="group"

             )

fig = dict(data=data, layout=layout)

print("SELECT OPTION BELOW: ")

iplot(fig)

df.SalePrice.describe()
df.SalePrice.plot.hist()
print('Excess kurtosis of normal distribution (should be 0): {}'.format(

    kurtosis(df[df['SalePrice'] > 0]['SalePrice'])))

print( 'Skewness of normal distribution (should be < abs 0.5): {}'.format(

    skew((df[df['SalePrice'] > 0]['SalePrice']))))

def explore_outliers(df_num, num_sd = 3, verbose = False): 

    '''

    Set a numerical value and it will calculate the upper, lower and total number of outliers.

    It will print a lot of statistics of the numerical feature that you set on input.

    Adapted from: https://www.kaggle.com/kabure/exploring-the-consumer-patterns-ml-pipeline

    '''

    

    data_mean, data_std = np.mean(df_num), np.std(df_num)



    # Outlier SD

    cut = data_std * num_sd



    # IQR thresholds

    lower, upper = data_mean - cut, data_mean + cut



    # creating an array of lower, higher and total outlier values 

    outliers_lower = [x for x in df_num if x < lower]

    outliers_higher = [x for x in df_num if x > upper]

    outliers_total = [x for x in df_num if x < lower or x > upper]



    # array without outlier values

    outliers_removed = [x for x in df_num if x > lower and x < upper]

    

    print('Identified lower outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers

    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers

    print('Total outliers: %d' % len(outliers_total)) # printing total number of values outliers of both sides

    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values

    print("Total percentage of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points

    

    if verbose:

        print('\nVerbose: Printing outliers')

        if len(outliers_lower) > 0:

            print(f'Lower outliers: {outliers_lower}')

            

        if len(outliers_higher) > 0:

            print(f'Upper outliers: {outliers_higher}')

    

    return



explore_outliers(df.SalePrice, 5, True)
# But first, let's remove the feature created using target variable and split train/test

df.drop('N_priceLbl', axis=1, inplace=True)



# Also, perform logarithmic transformation on target. Why? because we're interested in 

#..relative differences in prices and this normalizes the skew. 

# Read more here: https://stats.stackexchange.com/a/48465/236332

# and here: https://towardsdatascience.com/why-take-the-log-of-a-continuous-target-variable-1ca0069ee935

Y = np.log(df.loc[:, 'SalePrice'] + 1)#.apply(lambda y: )

# Y.fillna(-1)

df.drop('SalePrice', axis=1, inplace=True)



# Also drop 'order' and 'PID' as they're just record identifiers

df.drop(['Order', 'PID'], axis=1, inplace=True)
Y.plot.hist()
# Let's Label encode all categorical variables

for c in df.columns:

    df[c]=df[c].fillna(-1) # Imp. for both encoder and regressor. They don't like NaNs.

    if df[c].dtype == 'object':

        le = preprocessing.LabelEncoder()

        df[c] = le.fit_transform(df[c].astype('str')) # https://stackoverflow.com/a/46406995/1578274
# Split into train/test (80% training, 20% testing)

X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.20)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
# Linear Regression

from sklearn import linear_model

from sklearn.metrics import mean_squared_error as mse

reg = linear_model.LinearRegression()

reg.fit(X_train, Y_train)

lr_pred = reg.predict(X_test)
mse(Y_test, lr_pred)
# Ridge regression using CV

reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)

reg.fit(X_train, Y_train)

lr_pred = reg.predict(X_test)

mse(Y_test, lr_pred)
# ref: https://www.kaggle.com/nikunjm88/creating-additional-features/data

xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'silent': 1,

    'seed' : 0

}



dtrain = xgb.DMatrix(X_train, Y_train, feature_names=X_train.columns.values)

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=150)



# plot the important features

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=30, height=0.8, ax=ax)

plt.show()
from sklearn.feature_selection import SelectPercentile, f_classif



X_indices = np.arange(X_train.shape[-1])

selector = SelectPercentile(f_classif, percentile=10)

selector.fit(X_train, Y_train)

scores = -np.log10(selector.pvalues_)

scores /= scores.max()
top_k = 30

top_cols = df.columns[np.argsort(scores)][-top_k:]

top_scores = np.sort(scores)[-top_k:]



ind = np.arange(top_cols.shape[0])

width = 0.2

fig, ax = plt.subplots(figsize=(12,15))

rects = ax.barh(ind, top_scores, color='darkorange',

        edgecolor='black')

ax.set_yticks(ind)

ax.set_yticklabels(top_cols, rotation='horizontal')

ax.set_xlabel(r'Univariate score ($-Log(p_{value})$)')

ax.set_title("Feature importance using SelectPercentile")



# Add value labels

for i, v in enumerate(top_scores):

    ax.text(v+0.01, i, str(round(v, 4)), color='k')

plt.show()
# Split into train/test

X_train, X_test, Y_train, Y_test = train_test_split(df.loc[:,top_cols], Y, test_size=0.20)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor(n_neighbors=5)

reg.fit(X_train, Y_train)

lr_pred = reg.predict(X_test)

mse(Y_test, lr_pred)
reg = linear_model.LinearRegression()

reg.fit(X_train, Y_train)

lr_pred = reg.predict(X_test)

mse(Y_test, lr_pred)
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)

reg.fit(X_train, Y_train)

lr_pred = reg.predict(X_test)

mse(Y_test, lr_pred)