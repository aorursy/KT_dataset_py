import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
%matplotlib inline

# some stuff left over here but I'm leaving it because I will come back to expand on this
# to include calculation of the mean across the results of different feature ranking methods
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
# supress warnings because we are working with placeholder dataframes, features, and parameters
# DO NOT use this in your actual implementation of this code

#warnings.simplefilter("ignore", NameError)

## doesn't work, just letting it throw the errors for now
## I don't know how to do this without using try/except for every 
## block where an error is thrown but appreciate any suggestions

# I'm making the assumption that we are working on a training set df_train
# First select only numeric data and extract the target variable
# If you have categorical data that you want to include, OneHot encode it
df_train_rfe = df_train.select_dtypes(include=[np.number])
y = df_train_rfe[TARGET].values
x = df_train_rfe.loc[:, df_train_rfe.columns != TARGET]

# Store the column/feature names into a list "colnames"
colnames = x.columns
# Define dictionary to store rankings
ranks = {}

# Create a function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))
# Construct a Linear Regression model
lr = LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)
lr.fit(x, y)

#stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=1, verbose =3 )
rfe.fit(x, y)
ranks["RFE"] = [ranking(list(map(float, rfe.ranking_)), list(colnames), order=-1)]
rfe_ranks = pd.DataFrame(ranks["RFE"]).transpose()
rfe_ranks.columns = ['rfe_rank']
# Create a feature ranking matrix
r_df = pd.DataFrame(rfe_ranks.index)
r_df['Feature'] = pd.DataFrame(rfe_ranks.index)
r_df['rfe_rank'] = rfe_ranks['rfe_rank'].values
r_df = r_df.sort_values('rfe_rank', ascending=False)
r_df
# Plot the contents of the feature ranking matrix

rankplot = pd.DataFrame(rfe_ranks.index)
rankplot['rfe_rank'] = r['rfe_rank']
rankplot['Feature'] = r['Feature']
rankplot = rankplot.sort_values('rfe_rank', ascending=False)


plt.clf()
sns.set_style("darkgrid")
sns.factorplot(x='rfe_rank', y='Feature', data = rankplot, kind="bar", 
               height=14, aspect=1.9, palette='Reds_d')
# for selectng only the features above a certain degree of 
# importance to use in the next step of your model development

threshold = THRESHOLD
keep = r_df.loc[r_df['rfe_rank'] > threshold].Feature
keep.head(10)
