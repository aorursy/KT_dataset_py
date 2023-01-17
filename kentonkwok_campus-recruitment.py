# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read csv file

path = '/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'



df = pd.read_csv(path)

print('The shape of our data is:' , df.shape)

df.head()

print('Numeric columns')

df.describe(include = np.number)
print('Categorical columns')

df.describe(include = np.object)
# processing

column_names = list(df.columns)



# extract features and the target

data = df.iloc[:, 1 :-2]

target = df.iloc[:,-2:]



# separate between categorical and numeric columns

numeric_columns = data.select_dtypes(include=['int64' , 'float64'])

categorical_columns = data.select_dtypes(exclude=['int64' , 'float64'])
# data visualisation

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('darkgrid')



# seaborn pairplot

sns.pairplot(df.iloc[:, 1:], hue="status")

plt.show()



#from pandas.plotting import corr_matrix

corr_matrix = df.corr()

print('Correlation matrix:')

corr_matrix["salary"].sort_values()
# Bar charts

fig, ax = plt.subplots(4,2,figsize=(15,15))



for i in range(len(categorical_columns.columns)): 

    sns.countplot(x = categorical_columns[categorical_columns.columns[i]], 

                  hue = target['status'], 

                  ax = ax[i//2,i%2])
# label encode the binary data, as this would increase the prediction accuracy

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



# get column names with binary data i.e. two classes

c = categorical_columns.nunique() == 2

binary_cats = c[c].index



label_categorical_columns = categorical_columns.copy()

label_data = data.copy()



for c in binary_cats: 

    label_categorical_columns[c] = label_encoder.fit_transform(categorical_columns[c])

    label_data[c] = label_encoder.fit_transform(categorical_columns[c])



# make sure to keep track which are the positive and negative classes

print(label_categorical_columns.head())

print('---')

print(categorical_columns)
# label encode the status column from target data

label_target = target.copy()

label_placed = label_encoder.fit_transform(target['status'])

label_target['status'] = label_placed



# one hot encode the categorical columns, except those with binary values

categorical_columns_onehot = pd.get_dummies(label_categorical_columns)



#t = label_encoder.fit_transform(data['gender'])

data_onehot = pd.get_dummies(label_data)

data_onehot
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



X = data_onehot

y = label_placed



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



forest_model = RandomForestClassifier(random_state = 1)

forest_model.fit(train_X, train_y)



# evaluate model

print('F1 Score:', forest_model.score(val_X, val_y))

# evaluate each feature's importance

importances = forest_model.feature_importances_



# sort by descending order of importances

indices = np.argsort(importances)[::-1]



#create sorted dictionary

sorted_importances = {}



print("Feature ranking:")

for f in range(X.shape[1]):

    sorted_importances[X.columns[indices[f]]] = importances[indices[f]]

    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
# Obtain average percentages for the DataFrame

percent_df = pd.DataFrame(numeric_columns.agg('mean', axis = 1)).join(df['status'])



percent = percent_df.groupby(['status']).agg(['mean','var', 'count'])

percent.columns = percent.columns.droplevel()

percent
# one tailed t-test

# H0: mu_placed - mu_notplaced = 0

# H1: mu_placed - mu_notplaced > 0



# unequal sample sizes, similar variance



# extract variables from table

placed_mu, notplaced_mu = percent['mean']['Placed'], percent['mean']['Not Placed']

placed_var, notplaced_var = percent['var']['Placed'], percent['var']['Not Placed']

n1, n2 = percent['count']['Placed'], percent['count']['Not Placed']



# calculate the t statistic

sp = np.sqrt(((n1 - 1) * placed_var + (n2 - 1) * notplaced_var)/ (n1 + n2 - 2))

t_stat = (placed_mu - notplaced_mu) / (sp * np.sqrt(1/ n1 + 1/ n2))



print('The t statistic is', t_stat)

from scipy.stats import t

print('The p-value is,', t.cdf(-np.abs(t_stat), df = n1 + n2- 2))

data_joined = data.join(df['status'])



# obtain table for placed students in each specialisation

p = data_joined.groupby(['specialisation'])['status'].agg([lambda z: np.mean(z=='Placed'), "size"])

p.columns = ["Placed", 'Total']

print(p)

# We want to test whether it is finance students find it easier to get placed 

# H0 is pfin - phr > 0, as we want to do a one-tailed test 

# H1: pfin - phr <= 0 



# calculate pool proportion

p_us = len(df[df['status']=='Placed']) / len(df)



# obtaining individual proportions and total counts from table above

pfin, phr = p['Placed']['Mkt&Fin'], p['Placed']['Mkt&HR']

n1, n2 = p['Total']['Mkt&Fin'], p['Total']['Mkt&HR']



# calculate standard error

se = np.sqrt(p_us*(1- p_us)*(1/n1 + 1/n2))



# Calculate the best estimate of the proportion distribution

be = pfin - phr



# Calculate the hypothesized estimate, which is no difference

he = 0



#Calculate the test statistic

test_statistic = (be - he)/se



# Obtain one tailed p-value

from scipy.stats import norm

pvalue = norm.cdf(-np.abs(test_statistic))



print('The p-value is {0:.6f}'.format(pvalue))
onehot_target = data_onehot.join(label_target)

corr_matrix = onehot_target.corr()

fig = plt.figure(figsize = (10,10))

sns.heatmap(corr_matrix, square=True, cmap="YlGnBu")