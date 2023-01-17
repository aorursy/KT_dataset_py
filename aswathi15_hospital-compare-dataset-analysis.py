# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="darkgrid")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/HospInfo.csv')

dataset.head()
dataset.info()
dataset.drop(['Hospital overall rating footnote','Mortality national comparison footnote','Safety of care national comparison footnote','Readmission national comparison footnote','Patient experience national comparison footnote','Effectiveness of care national comparison footnote','Timeliness of care national comparison footnote','Efficient use of medical imaging national comparison footnote'],axis=1,inplace=True)
dataset.info()
Total_state = pd.value_counts(dataset['State'])

Total_state = pd.DataFrame(Total_state)

Total_state = Total_state.reset_index()

Total_state.columns = ['State', 'Number of Hospitals']
dims = (10, 10)

fig, ax = plt.subplots(figsize=dims)

ax = sns.barplot(x = 'Number of Hospitals', y = 'State', data = Total_state)

ax.set(xlabel = 'Number of Hospitals', ylabel = 'States')

ax.set_title('Number of Hospitals per State')
Hospital_Type_df = pd.value_counts(dataset['Hospital Type'])

Hospital_Type_df = pd.DataFrame(Hospital_Type_df)

Hospital_Type_df = Hospital_Type_df.reset_index()

Hospital_Type_df.columns = ['Hospital Type', 'Number of Hospitals']
ax = sns.barplot(x = 'Hospital Type', y= 'Number of Hospitals', data = Hospital_Type_df)

ax.set(xlabel = 'Type of Hospital', ylabel = 'Number of Hospitals')

ax.set_title('Count of the different Types of Hospitals(Acute/Critical/Childrens)')
Hospital_owner = pd.value_counts(dataset['Hospital Ownership'])

Hospital_owner = pd.DataFrame(Hospital_owner)

Hospital_owner = Hospital_owner.reset_index()

Hospital_owner.columns = ['Hospital Ownership', 'Number of Hospitals']
dims = (10, 10)

fig, ax = plt.subplots(figsize=dims)

ax = sns.barplot(y = 'Hospital Ownership', x= 'Number of Hospitals', data = Hospital_owner)

ax.set(xlabel = 'Hospital Ownership', ylabel = 'Number of Hospitals')

ax.set_title('Count of the different Types of Hospital Ownership')
dataset['Hospital overall rating'].unique()
Hospital_rating = dataset.drop(dataset[dataset['Hospital overall rating']=='Not Available'].index)

Hospital_rating['Hospital overall rating'].unique()
Hospital_rating = pd.value_counts(Hospital_rating['Hospital overall rating'])

Hospital_rating = pd.DataFrame(Hospital_rating)

Hospital_rating = Hospital_rating.reset_index()

Hospital_rating.columns = ['Hospital Rating', 'Number of Hospitals']
dims = (10, 5)

fig, ax = plt.subplots(figsize=dims)

ax = sns.barplot(x = 'Hospital Rating', y = 'Number of Hospitals', palette="BuGn_d",data = Hospital_rating)

ax.set(xlabel = 'Hospital Rating', ylabel = 'Number of Hospitals')
