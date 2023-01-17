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
import pandas as pd



# Numpy

import numpy as np



# Matplotlib for additional customization

from matplotlib import pyplot as plt

import matplotlib

# matplotlib.style.use('ggplot')

matplotlib.style.use('fivethirtyeight')

%matplotlib inline



# Seaborn for plotting and styling

import seaborn as sns
sui_df = pd.read_csv('../input/Suicides in India 2001-2012.csv')
all_india_df = sui_df[sui_df['State'] == 'Total (All India)']

education_status_df = all_india_df[all_india_df['Type_code'] == 'Education_Status']

social_status_df = all_india_df[all_india_df['Type_code'] == 'Social_Status']
gender_count_ds = education_status_df.groupby(['Gender'])['Total'].sum()

gender_count_ds.plot.bar()
gender_count_ds.plot.pie()
causes_df = sui_df[sui_df['Type_code'] == 'Causes']

age_grp_ds = causes_df.groupby(['Age_group'])['Total'].sum()
age_grp_ds.plot.bar()
age_grp_ds.plot.pie()
male_ds = causes_df[causes_df['Gender'] == 'Male'].groupby(['Age_group'])['Total'].sum()

female_ds = causes_df[causes_df['Gender'] == 'Female'].groupby(['Age_group'])['Total'].sum()

gender_df = pd.DataFrame({'Male': male_ds, 'Female': female_ds})

gender_df.plot.bar()
male_ds = causes_df[causes_df['Gender'] == 'Male'].groupby(['State'])['Total'].sum()

female_ds = causes_df[causes_df['Gender'] == 'Female'].groupby(['State'])['Total'].sum()

gender_df = pd.DataFrame({'Male': male_ds, 'Female': female_ds})

gender_df.plot.bar()
male_ds = causes_df[causes_df['Gender'] == 'Male'].groupby(['Year'])['Total'].sum()

female_ds = causes_df[causes_df['Gender'] == 'Female'].groupby(['Year'])['Total'].sum()

gender_df = pd.DataFrame({'Male': male_ds, 'Female': female_ds})

gender_df.plot.bar()
print(causes_df.groupby(['State']).sum()['Total'].sort_values(ascending=False).head(3))
print(causes_df[causes_df['Gender'] == 'Male'].groupby(['State']).sum()['Total'].sort_values(ascending=False).head(3))
print(causes_df[causes_df['Gender'] == 'Female'].groupby(['State']).sum()['Total'].sort_values(ascending=False).head(3))
print(causes_df.groupby(['Year']).sum()['Total'].sort_values(ascending=False).head(3))
print(causes_df.groupby(['Age_group']).sum()['Total'].sort_values(ascending=False).head(3))