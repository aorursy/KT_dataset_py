# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns
# Import Data

train = pd.read_csv('../input/train_2016.csv', index_col='parcelid')

properties = pd.read_csv('../input/properties_2016.csv', index_col='parcelid', low_memory=False)

#data_dictionary = pd.read_excel('../data/zillow_data_dictionary.xlsx')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='ParcelId')
# Join train and details

joined_train = train.join(properties)
train.head()
train['logerror'].hist(figsize = [10,5], bins = 50)
# Data Shapes

print(sample_submission.shape)

print(properties.shape)

print(train.shape)

print(joined_train.shape)
joined_train.head()
avg_error = train.mean()

avg_error = avg_error[0]
# Replace with average error

baseline_preds = sample_submission



sample_submission = sample_submission.replace(to_replace=0, value=avg_error)
# sample_submission.to_csv('../submissions/baseline_submission_052717.csv')
# Describe the data

joined_train.describe()
var_counts = pd.DataFrame(joined_train.count(), columns=['Count'])

var_counts.loc[var_counts['Count'] > 80000].sort_values(by=['Count'], ascending = False)
from sklearn import model_selection
main_features = ['roomcnt','bedroomcnt','bathroomcnt',

                 'fips','landtaxvaluedollarcnt','taxvaluedollarcnt',

                 'taxamount','regionidzip','yearbuilt','finishedsquarefeet12','lotsizesquarefeet']

output_vars = ['transactiondate','logerror']
df = joined_train.loc[:,main_features+output_vars]
df.describe()
sns.pairplot(df.loc[:,['roomcnt','bedroomcnt','bathroomcnt','yearbuilt','logerror','fips']].dropna(),

            hue='fips')
properties['fips'].unique()
fips_means = df.groupby('fips', as_index=False)['logerror'].mean()

fips_means
properties.groupby('fips', as_index=False).count()
# fips_means['logerror'].loc(:,[fips_means['fips'] == 6037.0])

fips_means.loc[fips_means['fips'] == 6037.0, 'logerror'].iloc[0]
def fips_mean (row):    

    '''Function that goes through each fips code and assigns to the average

    logerror for that fips. If a NaN fips code then it returns the average of 

    all properties'''

    if row['fips'] == 6037 :

        return fips_means.loc[fips_means['fips'] == 6037.0, 'logerror'].iloc[0]

    if row['fips'] == 6059 :

        return fips_means.loc[fips_means['fips'] == 6059.0, 'logerror'].iloc[0]

    if row['fips'] == 6111 :

        return fips_means.loc[fips_means['fips'] == 6111.0, 'logerror'].iloc[0]

    else:

        return avg_error
### THIS TAKES A LONG TIME ##

# Creat a guess erorr vector which is just the average of each fips

#guess_error = properties.apply (lambda row: fips_mean (row),axis=1)
#guess_error.shape

#sample_submission['201610']=guess_error

#sample_submission['201611']=guess_error

#sample_submission['201612']=guess_error

#sample_submission['201710']=guess_error

#sample_submission['201711']=guess_error

#sample_submission['201712']=guess_error

#sample_submission.to_csv('../submissions/baseline_submission_fipsaverages_052817.csv')
