# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
penguin_df = pd.read_csv("/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_lter.csv")
penguin_df.head()
penguin_df.tail()
penguin_df.shape
penguin_df.info()
penguin_df.describe()
import pandas_profiling as pp

pp.ProfileReport(penguin_df)


penguin_df.isnull().values.any()
penguin_df.isnull().sum()
# Handling missing values



from sklearn.impute import SimpleImputer

#setting strategy to 'most frequent' to impute by the mean

imputer = SimpleImputer(strategy='most_frequent')# strategy can also be mean or median 

penguin_df.iloc[:,:] = imputer.fit_transform(penguin_df)
penguin_df.isnull().sum()


penguin_df.groupby('Species').size()
# countplot----Plot the frequency of the Outcome



fig1, ax1 = plt.subplots(1,2,figsize=(13,10))



#It shows the count of observations in each categorical bin using bars



sns.countplot(penguin_df['Species'],ax=ax1[0])



#Find the % of diabetic and Healthy person



labels = 'Adelie Penguin', 'Chinstrap penguin' , 'Gentoo penguin'



penguin_df.Species.value_counts().plot.pie(labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
penguin_df.hist(figsize=(15,10))

# Distplot





fig, ax2 = plt.subplots(3, 2, figsize=(16, 16))

sns.distplot(penguin_df['Body Mass (g)'],ax=ax2[0][0])

sns.distplot(penguin_df['Culmen Depth (mm)'],ax=ax2[0][1])

sns.distplot(penguin_df['Culmen Length (mm)'],ax=ax2[1][0])

sns.distplot(penguin_df['Delta 13 C (o/oo)'],ax=ax2[1][1])

sns.distplot(penguin_df['Delta 15 N (o/oo)'],ax=ax2[2][0])

sns.distplot(penguin_df['Flipper Length (mm)'],ax=ax2[2][1])



# pairplot--Multiple relationship of scatterplot



sns.pairplot(penguin_df,hue='Species')
# corrlation matrix 



cor=penguin_df.corr()

cor
# correlation plot---heatmap



sns.heatmap(cor,annot=True)
# Categorical vs Continuous ----Species vs Culmen Length



fig, ax2 = plt.subplots(3, 2, figsize=(20, 20))

sns.boxplot(x="Species", y="Culmen Length (mm)", data=penguin_df,ax=ax2[0][0])

sns.barplot(penguin_df['Species'], penguin_df['Culmen Length (mm)'],ax=ax2[0][1])

sns.stripplot(penguin_df['Species'], penguin_df['Culmen Length (mm)'], jitter=True,ax=ax2[1][0])

sns.swarmplot(penguin_df['Species'], penguin_df['Culmen Length (mm)'], ax=ax2[1][1])

sns.violinplot(penguin_df['Species'], penguin_df['Culmen Length (mm)'], ax=ax2[2][0])

sns.countplot(x='Culmen Length (mm)',hue='Species',data=penguin_df,ax=ax2[2][1])
# Categorical vs Continuous ----Species vs Body Mass (g)



fig, ax2 = plt.subplots(3, 2, figsize=(20, 20))

sns.boxplot(x="Species", y="Body Mass (g)", data=penguin_df,ax=ax2[0][0])

sns.barplot(penguin_df['Species'], penguin_df['Body Mass (g)'],ax=ax2[0][1])

sns.stripplot(penguin_df['Species'], penguin_df['Body Mass (g)'], jitter=True,ax=ax2[1][0])

sns.swarmplot(penguin_df['Species'], penguin_df['Body Mass (g)'], ax=ax2[1][1])

sns.violinplot(penguin_df['Species'], penguin_df['Body Mass (g)'], ax=ax2[2][0])

sns.countplot(x='Body Mass (g)',hue='Species',data=penguin_df,ax=ax2[2][1])
# Categorical vs Continuous ----Species vs Culmen Depth (mm)



fig, ax2 = plt.subplots(3, 2, figsize=(20, 20))

sns.boxplot(x="Species", y="Culmen Depth (mm)", data=penguin_df,ax=ax2[0][0])

sns.barplot(penguin_df['Species'], penguin_df['Culmen Depth (mm)'],ax=ax2[0][1])

sns.stripplot(penguin_df['Species'], penguin_df['Culmen Depth (mm)'], jitter=True,ax=ax2[1][0])

sns.swarmplot(penguin_df['Species'], penguin_df['Culmen Depth (mm)'], ax=ax2[1][1])

sns.violinplot(penguin_df['Species'], penguin_df['Culmen Depth (mm)'], ax=ax2[2][0])

sns.countplot(x='Culmen Depth (mm)',hue='Species',data=penguin_df,ax=ax2[2][1])
# Categorical vs Continuous ----Species vs Delta 13 C (o/oo)



fig, ax2 = plt.subplots(3, 2, figsize=(20, 20))

sns.boxplot(x="Species", y="Delta 13 C (o/oo)", data=penguin_df,ax=ax2[0][0])

sns.barplot(penguin_df['Species'], penguin_df['Delta 13 C (o/oo)'],ax=ax2[0][1])

sns.stripplot(penguin_df['Species'], penguin_df['Delta 13 C (o/oo)'], jitter=True,ax=ax2[1][0])

sns.swarmplot(penguin_df['Species'], penguin_df['Delta 13 C (o/oo)'], ax=ax2[1][1])

sns.violinplot(penguin_df['Species'], penguin_df['Delta 13 C (o/oo)'], ax=ax2[2][0])

sns.countplot(x='Delta 13 C (o/oo)',hue='Species',data=penguin_df,ax=ax2[2][1])
# Categorical vs Continuous ----Species vs Delta 15 N (o/oo)



fig, ax2 = plt.subplots(3, 2, figsize=(20, 20))

sns.boxplot(x="Species", y="Delta 15 N (o/oo)", data=penguin_df,ax=ax2[0][0])

sns.barplot(penguin_df['Species'], penguin_df['Delta 15 N (o/oo)'],ax=ax2[0][1])

sns.stripplot(penguin_df['Species'], penguin_df['Delta 15 N (o/oo)'], jitter=True,ax=ax2[1][0])

sns.swarmplot(penguin_df['Species'], penguin_df['Delta 15 N (o/oo)'], ax=ax2[1][1])

sns.violinplot(penguin_df['Species'], penguin_df['Delta 15 N (o/oo)'], ax=ax2[2][0])

sns.countplot(x='Delta 15 N (o/oo)',hue='Species',data=penguin_df,ax=ax2[2][1])
# Categorical vs Continuous ----Species vs Delta 15 N (o/oo)



fig, ax2 = plt.subplots(3, 2, figsize=(20, 20))

sns.boxplot(x="Species", y="Flipper Length (mm)", data=penguin_df,ax=ax2[0][0])

sns.barplot(penguin_df['Species'], penguin_df['Flipper Length (mm)'],ax=ax2[0][1])

sns.stripplot(penguin_df['Species'], penguin_df['Flipper Length (mm)'], jitter=True,ax=ax2[1][0])

sns.swarmplot(penguin_df['Species'], penguin_df['Flipper Length (mm)'], ax=ax2[1][1])

sns.violinplot(penguin_df['Species'], penguin_df['Flipper Length (mm)'], ax=ax2[2][0])

sns.countplot(x='Flipper Length (mm)',hue='Species',data=penguin_df,ax=ax2[2][1])
#Body Mass (g) vs Flipper Length (mm)

sns.pointplot(penguin_df['Body Mass (g)'], penguin_df['Flipper Length (mm)'], hue=penguin_df['Species'])

sns.jointplot(penguin_df['Body Mass (g)'], penguin_df['Flipper Length (mm)'], kind='hex')

sns.lmplot(x='Body Mass (g)',y='Flipper Length (mm)',data=penguin_df,hue='Species')
#Body Mass (g) vs Flipper Length (mm)

sns.pointplot(penguin_df['Culmen Length (mm)'], penguin_df['Flipper Length (mm)'], hue=penguin_df['Species'])

sns.jointplot(penguin_df['Culmen Length (mm)'], penguin_df['Flipper Length (mm)'], kind='hex')

sns.lmplot(x='Culmen Length (mm)',y='Flipper Length (mm)',data=penguin_df,hue='Species')