# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.model_selection import train_test_split



# Any results you write to the current directory are saved as output.
path = "../input/restaurant-revenue-prediction/train.csv"

res_data = pd.read_csv(path, index_col ="Open Date", parse_dates = True)

# print the columns of Training data

res_data.columns







#print the first 10 colums of Training data

res_data.head(10)
#print the last 10 colums of Training data

res_data.tail(10)
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



#Add the width and height  of the figure

plt.figure(figsize=(20,10))

#Add label for the horizintal axis

plt.xlabel("Open Data")

#Add title of the figure

plt.title("The Revenue of the Restaurant in a Given Year")



# Line chart showing yearly revenue of the restaurant 

sns.lineplot(data = res_data['revenue'], label = 'revenue')

#Add the width and height  of the figure

plt.figure(figsize=(10,5))

#Add label for the horizintal axis

plt.xlabel("City Group")

#Add title of the figure

plt.title("The Revenue of the Restaurant in a Given Year classified by  the \"City Group\"")

sns.swarmplot(x=res_data['City Group'],

              y=res_data['revenue'])
#Add the width and height  of the figure

plt.figure(figsize=(6,3))



#Add title of the figure

plt.title("The Restaurant Revenue when the Type of restaurant changes")



# Bar chart showing yearly revenue of the restaurant 

sns.barplot(x=res_data['Type'], y=res_data['revenue'])
res_data.columns
res_data2 = pd.read_csv(path)

#Add the width and height  of the figure



sns.lmplot(x="Id", y="revenue", hue="Type", data = res_data2)

plt.figure(figsize=(15,6))



# #Add label for the horizintal axis

# plt.xlabel("Open Data")

# #Add title of the figure

# plt.title("The Restaurant Revenue when the Type of restaurant changes")



missing_val_count_by_column = (res_data.isnull().sum())

print(missing_val_count_by_column)

# Get list of categorical variables

s = (res_data.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
res_data.columns

# Separate target from predictors

y = res_data.revenue

X = res_data.drop(['revenue'], axis=1)



# Divide data into training and validation subsets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# # Drop columns with missing values (simplest approach)

# cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 

# X_train_full.drop(cols_with_missing, axis=1, inplace=True)

# X_valid_full.drop(cols_with_missing, axis=1, inplace=True)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() <  10 and 

                        X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_train.head(10)