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



#For plotting

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
vg_df = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
print('This dataset has ' + str(vg_df.shape[0]) + ' rows, and ' + str(vg_df.shape[1]) + ' columns')
vg_df.head(7)
vg_df.describe()
sns.distplot(vg_df.Year_of_Release.dropna(), kde=False, bins = 39);
Global_Salesfig, axs = plt.subplots(ncols = 4, figsize=(13, 4))



sns.distplot(vg_df.Global_Sales.dropna(), kde=False, ax=axs[0])

second_plt = sns.distplot(vg_df.Global_Sales.dropna()[vg_df.Global_Sales > 2], kde=False, ax=axs[1])

sns.boxplot(vg_df.Global_Sales, ax=axs[2], orient = 'v')

sns.boxplot(vg_df.Global_Sales, ax=axs[3], orient = 'v', showfliers=False)



second_plt.set_yscale('log')
Global_Salesfig, axs = plt.subplots(ncols = 4, figsize=(13, 4))



sns.distplot(vg_df.NA_Sales.dropna(), kde=False, ax=axs[0])

second_plt = sns.distplot(vg_df.NA_Sales.dropna()[vg_df.NA_Sales > 1], kde=False, ax=axs[1])

sns.boxplot(vg_df.NA_Sales, ax=axs[2], orient = 'v')

sns.boxplot(vg_df.NA_Sales, ax=axs[3], orient = 'v', showfliers=False)



second_plt.set_yscale('log')
Global_Salesfig, axs = plt.subplots(ncols = 4, figsize=(13, 4))



sns.distplot(vg_df.EU_Sales.dropna(), kde=False, ax=axs[0])

second_plt = sns.distplot(vg_df.EU_Sales.dropna()[vg_df.EU_Sales > 1], kde=False, ax=axs[1])

sns.boxplot(vg_df.EU_Sales, ax=axs[2], orient = 'v')

sns.boxplot(vg_df.EU_Sales, ax=axs[3], orient = 'v', showfliers=False)



second_plt.set_yscale('log')
Global_Salesfig, axs = plt.subplots(ncols = 4, figsize=(13, 4))



sns.distplot(vg_df.JP_Sales.dropna(), kde=False, ax=axs[0])

second_plt = sns.distplot(vg_df.JP_Sales.dropna()[vg_df.JP_Sales > 1], kde=False, ax=axs[1])

sns.boxplot(vg_df.JP_Sales, ax=axs[2], orient = 'v')

sns.boxplot(vg_df.JP_Sales, ax=axs[3], orient = 'v', showfliers=False)



second_plt.set_yscale('log')
Global_Salesfig, axs = plt.subplots(ncols = 4, figsize=(13, 4))



sns.distplot(vg_df.Other_Sales.dropna(), kde=False, ax=axs[0])

second_plt = sns.distplot(vg_df.Other_Sales.dropna()[vg_df.Other_Sales > 1], kde=False, ax=axs[1])

sns.boxplot(vg_df.Other_Sales, ax=axs[2], orient = 'v')

sns.boxplot(vg_df.Other_Sales, ax=axs[3], orient = 'v', showfliers=False)



second_plt.set_yscale('log')
sns.distplot(vg_df.Critic_Score.dropna());
sns.distplot(vg_df.Critic_Count.dropna());
vg_df.User_Score = vg_df.User_Score.convert_objects(convert_numeric=True)

sns.distplot(vg_df.User_Score.dropna());
sns.distplot(vg_df.User_Count.dropna(), kde=False);
plt.figure(figsize=(13, 4))

#http://stackoverflow.com/questions/32891211/limit-the-number-of-groups-shown-in-seaborn-countplot for odering

sns.countplot(vg_df.Platform.dropna(), order = vg_df.Platform.value_counts().index);
plt.figure(figsize=(13, 4))

sns.countplot(vg_df.Genre.dropna(), order = vg_df.Genre.value_counts().index);
plt.figure(figsize=(13, 4))

sns.countplot(vg_df.Developer.dropna(), order = vg_df.Developer.value_counts().index);
plt.figure(figsize=(13, 4))

sns.countplot(vg_df.Developer.dropna(), order = vg_df.Developer.value_counts().iloc[:40].index)

plt.xticks(rotation=90);