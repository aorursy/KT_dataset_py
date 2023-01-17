# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load the Titanic Dataset , which could use some cleanup



data = pd.read_csv('/kaggle/input/titanic-dataset-from-kaggle/train.csv')

data.head(3)
# Look for attributes which has lot of nulls-cabin and age 

# are two potential candidate columns for our test



data.isnull().mean()
# Add column with median age



data['new_age_median'] = data['Age'].fillna(data.Age.median())



# Add column with mean age



data['new_age_mean'] = data['Age'].fillna(data.Age.mean())



# Display newly added columns, which shows two different approaches to 

# filling null values in the age column



data.loc[:, ['Name', 'Age','new_age_median', 'new_age_mean']].loc[data['Age'].isnull()].head(4)
# we can see that the distribution has changed, as seen in the density values centered 

# around the middle of the graph below



fig_new = plt.figure()

x = fig_new.add_subplot(111)



# original distribution of variables (before replacing NULL values)



data['Age'].plot(kind='kde', x=x, color='red')



# replaced with mean

data['new_age_mean'].plot(kind='kde', x=x, color='purple')



# replaced with median

data['new_age_median'].plot(kind='kde', x=x, color='grey')



# Details

lines, labels = x.get_legend_handles_labels()

x.legend(lines, labels, loc='best')
# we also see that mean / median imputation may affect the relationship 

# with the other variables in the dataset; in otherwords, the Covariance is impacted



data[['Fare','Survived', 'Age', 'new_age_median', 'new_age_mean']].cov()
data[['Age', 'new_age_median', 'new_age_mean']].boxplot()