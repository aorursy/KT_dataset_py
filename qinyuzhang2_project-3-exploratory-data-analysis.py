# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1 = pd.read_excel('/kaggle/input/concrete-compressive-strength-and-slump-data/Concrete_Data.xls')

df1.rename(columns = {'Cement (component 1)(kg in a m^3 mixture)':'Cement',

                     'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'Blast Furnace Slag',

                     'Fly Ash (component 3)(kg in a m^3 mixture)':'Fly Ash',

                     'Water  (component 4)(kg in a m^3 mixture)':'Water',

                     'Superplasticizer (component 5)(kg in a m^3 mixture)':'Superplasticizer',

                     'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'Coarse Aggregate',

                     'Fine Aggregate (component 7)(kg in a m^3 mixture)':'Fine Aggregate',

                     'Age (day)':'Age',

                     'Concrete compressive strength(MPa, megapascals) ':'Concrete Compressive Strength'},

                      inplace = True)

df1.head()
print(df1.dtypes)
print(df1.isna().sum())

print(df1.isnull().sum())
plt.figure(figsize=(14,6))

sns.heatmap(round(df1.describe()[0:].transpose(),2),linewidth=2,annot=True,fmt="f",cmap="YlGnBu")

plt.xticks(fontsize=20)

plt.yticks(fontsize=12)

plt.title("Figure 1. Variables Summary",fontsize = 16)

plt.show()
fig, axis = plt.subplots(3, 3, figsize=(16, 16))



sns.distplot(df1['Cement'],ax=axis[0][0])

sns.distplot(df1['Blast Furnace Slag'],ax=axis[0][1])

sns.distplot(df1['Fly Ash'],ax=axis[0][2])

sns.distplot(df1['Water'],ax=axis[1][0])

sns.distplot(df1['Superplasticizer'],ax=axis[1][1])

sns.distplot(df1['Coarse Aggregate'],ax=axis[1][2])

sns.distplot(df1['Fine Aggregate'],ax=axis[2][0])

sns.distplot(df1['Age'],ax=axis[2][1])

sns.distplot(df1['Concrete Compressive Strength'],ax=axis[2][2])

sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})  

fig.suptitle('Figure 2. Data Distribution', fontsize=16)

plt.figure(figsize=(35,15))



f = sns.boxplot(data=df1)

f.set_title("Figure 3. Outliers",fontsize = 40)

f.tick_params(labelsize=24)
fig, axis_2 = plt.subplots(2,4, figsize=(20, 10))

sns.regplot(x = 'Cement',y = 'Concrete Compressive Strength',data=df1,ax = axis_2[0][0])

sns.regplot(x = 'Blast Furnace Slag',y = 'Concrete Compressive Strength',data=df1,ax = axis_2[0][1])

sns.regplot(x = 'Fly Ash',y = 'Concrete Compressive Strength',data=df1,ax = axis_2[0][2])

sns.regplot(x = 'Water',y = 'Concrete Compressive Strength',data=df1,ax = axis_2[0][3])

sns.regplot(x = 'Superplasticizer',y = 'Concrete Compressive Strength',data=df1,ax = axis_2[1][0])

sns.regplot(x = 'Coarse Aggregate',y = 'Concrete Compressive Strength',data=df1,ax = axis_2[1][1])

sns.regplot(x = 'Fine Aggregate',y = 'Concrete Compressive Strength',data=df1,ax = axis_2[1][2])

sns.regplot(x = 'Age',y = 'Concrete Compressive Strength',data=df1,ax = axis_2[1][3])



plt.suptitle('Figure 4. Single-Variable Analysis',fontsize = 16)
df1[

    ['Cement','Blast Furnace Slag','Fly Ash', 'Water','Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate','Age']

   ].corrwith(df1['Concrete Compressive Strength'])
df1['Cement/Water'] = df1['Cement'] / df1['Water']

df1.head()
f5 = sns.regplot(x = 'Cement/Water',y = 'Concrete Compressive Strength',data=df1)



f5.set_title("Figure 5. Cement/Water vs Concrete Compressive Strength",fontsize = 16)
df1['Coarse Aggregate / Fine Aggregate'] = df1['Coarse Aggregate'] / df1['Fine Aggregate']

df1.head()
f6 = sns.regplot(x = 'Coarse Aggregate / Fine Aggregate',y = 'Concrete Compressive Strength',data=df1)



f6.set_title("Figure 6. Coarse Aggregate / Fine Aggregate vs Concrete Compressive Strength",fontsize = 16)
df2 = df1[['Cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age']]
f7 = sns.pairplot(df2)

f7.fig.suptitle('Figure 7. Relations Between Independent Variables',fontsize = 16)
df3 = df1[['Cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age','Concrete Compressive Strength']]
corr = df3.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(8, 8))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(230, 20, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.title('Figure 8. Correlation Between Independent Variables',fontsize = 16)
df4 = df1[['Cement/Water','Blast Furnace Slag','Fly Ash','Superplasticizer','Coarse Aggregate / Fine Aggregate','Age','Concrete Compressive Strength']]
corr2 = df4.corr()



# Generate a mask for the upper triangle

mask2 = np.triu(np.ones_like(corr2, dtype=bool))



# Set up the matplotlib figure

f2, ax2 = plt.subplots(figsize=(8, 8))



# Generate a custom diverging colormap

cmap2 = sns.diverging_palette(230, 20, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr2, mask=mask2, cmap=cmap2, vmax=.3, center=0,annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.title('Figure 8. Correlation Between Independent Variables',fontsize = 16)