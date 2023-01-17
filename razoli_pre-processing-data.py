# Packages 

import os

import numpy as np 

import pandas as pd 

from matplotlib import pyplot as plt
# List files

print(os.listdir("../input/kdd-cup-2009-customer-relationship-prediction"))
features = pd.read_table('../input/kdd-cup-2009-customer-relationship-prediction/orange_small_train.data') ; print('There are %s observations and %s variables' %(features.shape))

outcomes = pd.read_table('../input/kdd-cup-2009-customer-relationship-prediction/orange_small_train_churn.labels')  
features.head(5)
num_var = features.dtypes[features.dtypes == 'float64'].index # We need check it

cat_var = features.dtypes[features.dtypes == 'object'].index 
def check_data(check_df, var_type):

    

    # For all elements on list num_var:

    check_df = pd.DataFrame()

    

    # As a result we will have the total counts per element of each column inside of dataframe.

    for col in var_type:

        col_count = features.groupby(col).size()

        check_df = check_df.append(col_count, ignore_index = True)

    

    # We need transpose de matrix

    check_df = check_df.T

        

    check_df.index = check_df.index.astype('str')

    check_df.sort_index(inplace = True)

    

    print('Top 10: ', list(check_df.index[:10]))

    print('Tail 10: ', list(check_df.index[(len(check_df.index)-10):]))
#Checking the numerical variables

check_data(features ,num_var)
#Checking the categorical variables

check_data(features ,cat_var)
features_empty_by_var = features.isna().sum(axis = 0) 

features_empty_by_var.head(5)
features_empty_by_var.hist(bins = 12)

plt.show() # The most of variables have many NA's values.
# We can see that 50% of the values are below the value of 48513 by our median, so looking at the histogram together is preferable to using a cutoff line of 10,000 missing values.

print('The median is: %s' %(features_empty_by_var.median())) 
num_entries = len(features)

threshold = 0.25 # cut-off

keep_vars = np.array(features.columns[(features_empty_by_var) <= threshold*num_entries]) ; keep_vars
# We need just take the variables that contain 12500 NA's

num_var = [elem for elem in num_var if elem in keep_vars]

cat_vars = [elem for elem in cat_var if elem in keep_vars]
# Numerical variables

for col in num_var:

    col_mean = features[col].mean()

    features[col] = features[col].fillna(col_mean)



# Category variables

for col in cat_vars:

    features[col] = features[col].astype('category')



for col in cat_vars:

    features[col] = features[col].cat.add_categories('missing')

    features[col] = features[col].fillna('missing')
### Treatment categorical variables

n_categories_per_feature = features[cat_vars].apply(lambda x: len(set(x)))

plt.hist(n_categories_per_feature)

plt.show()
cat_vars = np.array(n_categories_per_feature[n_categories_per_feature < 1400].index)

print('After processing, we have only %s remaining categorical variables.' % len(cat_vars))
# After all this analysis, we can group only the selected variables.

features = features[list(num_var) + list(cat_vars)] ; features.head(10)
onehot_features = pd.get_dummies(features) ; onehot_features.head(10)
# You can see our dataset.

print('We have %s observations and %s new variables' % onehot_features.shape)