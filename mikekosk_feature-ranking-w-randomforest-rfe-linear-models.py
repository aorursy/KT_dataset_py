import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.feature_selection import RFE, f_regression

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
# Load Data

train = pd.read_csv('../input/train_2016_v2.csv')

prop = pd.read_csv('../input/properties_2016.csv')



# Convert to float32

for c, dtype in zip(prop.columns, prop.dtypes):

    if dtype == np.float64:

        prop[c] = prop[c].astype(np.float32)



# Merge training dataset with properties dataset

df_train = train.merge(prop, how='left', on='parcelid')



# Remove ID columns

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
x_train.head()
print(x_train.dtypes)
x_train['taxdelinquencyflag'].value_counts()
x_train['fireplaceflag'].value_counts()
x_train['hashottuborspa'].value_counts()
x_train['taxdelinquencyflag'] = x_train['taxdelinquencyflag'] == 'Y'
for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)
# Looking for nulls

print(x_train.isnull().sum())
x_train = x_train.fillna(x_train.mean())
# First extract the target variable which is our Log Error

Y = df_train['logerror'].values

X = x_train.as_matrix()



# Store the column/feature names into a list "colnames"

colnames = x_train.columns
# Define dictionary to store our rankings

ranks = {}

# Create our function which stores the feature rankings to the ranks dictionary

def ranking(ranks, names, order=1):

    minmax = MinMaxScaler()

    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]

    ranks = map(lambda x: round(x,2), ranks)

    return dict(zip(names, ranks))
# Finally let's run our Selection Stability method with Randomized Lasso

rlasso = RandomizedLasso(alpha=0.04)

rlasso.fit(X, Y)

ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)

print('finished')
# Construct our Linear Regression model

lr = LinearRegression(normalize=True)

lr.fit(X,Y)

#stop the search when only the last feature is left

rfe = RFE(lr, n_features_to_select=1, verbose =3 )

rfe.fit(X,Y)

ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
# Using Linear Regression

lr = LinearRegression(normalize=True)

lr.fit(X,Y)

ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)



# Using Ridge 

ridge = Ridge(alpha = 7)

ridge.fit(X,Y)

ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)



# Using Lasso

lasso = Lasso(alpha=.05)

lasso.fit(X, Y)

ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)
rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=2)

rf.fit(X,Y)

ranks["RF"] = ranking(rf.feature_importances_, colnames)
# Create empty dictionary to store the mean value calculated from all the scores

r = {}

for name in colnames:

    r[name] = round(np.mean([ranks[method][name] 

                             for method in ranks.keys()]), 2)

 

methods = sorted(ranks.keys())

ranks["Mean"] = r

methods.append("Mean")



print("\t%s" % "\t".join(methods))

for name in colnames:

    print("%s\t%s" % (name, "\t".join(map(str, 

                         [ranks[method][name] for method in methods]))))
# Put the mean scores into a Pandas dataframe

meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])



# Sort the dataframe

meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
# Let's plot the ranking of the features

sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", size=10, aspect=1, palette='coolwarm')