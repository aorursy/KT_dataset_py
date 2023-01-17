#Import Modules to use



import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from IPython.core.debugger import Tracer

from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.naive_bayes import GaussianNB

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
# Import and Fix Data

#I also created two new columns, Cost per Discharge and Total Costs



df = pd.read_csv('../input/inpatientCharges.csv')

df.columns = df.columns.str.strip()

df['Average Covered Charges'] = df['Average Covered Charges'].str.replace('$', '').astype(float)

df['Average Total Payments'] = df['Average Total Payments'].str.replace('$', '').astype(float)

df['Average Medicare Payments'] = df['Average Medicare Payments'].str.replace('$', '').astype(float)

df['Total Cost'] = df[['Average Covered Charges', 'Average Total Payments', 'Average Medicare Payments']].sum(axis=1)

df['Insurance Covered Ratio'] = (df['Average Covered Charges'] / df['Total Cost'])

df['Cost per Discharge'] = (df['Total Cost'] / df['Total Discharges'])
# I picked these Features to look into

ax = sns.pairplot(df[['Insurance Covered Ratio', 'Cost per Discharge', 'Total Cost', 'DRG Definition']], 

             hue='DRG Definition', 

             size=5)
sns.distplot(df['Total Cost'], bins=60, rug=False, color='red')
sns.distplot(df['Insurance Covered Ratio'], bins=60, rug=False, color='green')
sns.distplot(df['Cost per Discharge'], bins=60, rug=False, color='blue')
# Eliminate Rare Insurance Companies and Illness Conditions

provider_count = df.groupby(['Provider Id']).apply(lambda x: len(x) > 50)

df = df[df['Provider Id'].isin(provider_count[provider_count].index)]



drg_count = df.groupby(['DRG Definition']).apply(lambda x: len(x) > 200)

df = df[df['DRG Definition'].isin(drg_count[drg_count].index)]
# drop outliers for cost for each individual DRG Definition (I Used 2 STD, but could do this many ways)



def drop_by_std_dev(df):

    list_of_DRG = df['DRG Definition'].unique().tolist()

    out = []

    for drg in list_of_DRG:

        data = df[df['DRG Definition'] == drg]

        mean, std = data['Total Cost'].mean(), data['Total Cost'].std()

        inliers = data[(data['Total Cost'] <= (mean + (std*2))) & (data['Total Cost'] >= (mean - (std*2)))]

        out.append(inliers)

    return pd.concat(out)

df = drop_by_std_dev(df)
#looking at my features after outlier removal (I just wanted to play with seaborn)

ax = sns.PairGrid(df[['Insurance Covered Ratio', 'Cost per Discharge', 'Total Cost', 'DRG Definition']], 

             hue='DRG Definition',

             size = 5)

ax.map_diag(plt.hist, bins = 40)

ax.map_offdiag(plt.scatter, s=0.5)
sns.distplot(df['Total Cost'], bins=60, rug=False, color='red')
sns.distplot(df['Insurance Covered Ratio'], bins=60, rug=False, color='green')
sns.distplot(df['Cost per Discharge'], bins=60, rug=False, color='blue')
# Account for illness type in getting an insurance indicator (1, if in upper quartile, 0 if not)

def good_insurance(df):

    list_of_DRG = df['DRG Definition'].unique().tolist()

    out = []

    for drg in list_of_DRG:

        data = df[df['DRG Definition'] == drg]

        quantile = data['Insurance Covered Ratio'].quantile(.75)

        data.loc[(data['Insurance Covered Ratio'] >= quantile), 'insurance indicator'] = 1

        data.loc[(data['Insurance Covered Ratio'] < quantile), 'insurance indicator'] = 0

        out.append(data)

    return pd.concat(out)

df = good_insurance(df)

    
df.columns
# Split the data into training and testing data



data = df[['Provider Id', 'insurance indicator']].dropna(how='any')



X = df['Provider Id'].values

y = df['insurance indicator'].values



sss = StratifiedShuffleSplit(y, 3, test_size=0.5, random_state=0)

for train_index, test_index in sss:

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

# Features need to be a Verticle Matrix for single feature models

X_train = X_train.reshape(len(X_train), 1)

X_test = X_test.reshape(len(X_test), 1)
# Model Data



clf = GaussianNB()

clf.fit(X_train, y_train)



score = clf.score(X_test, y_test)



score