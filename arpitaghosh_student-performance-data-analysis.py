# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# Any results you write to the current directory are saved as output.
import os

print(os.listdir("../input/students-performance-in-exams"))
df = pd.read_csv("..//input//students-performance-in-exams//StudentsPerformance.csv")
df.head()
df.shape #Size of Data Frame
df.describe()
#Analysis the data correlation

corr=df.corr()

corr
#Checking for missing values

df.isnull().sum()
#Total Students By Gender

df['gender'].value_counts()
# Set theme

sns.set_style('whitegrid')
#Univariate Analysis for math score

sns.distplot(df['math score'],  bins=10);



#Univariate Analysis for reading score

sns.distplot(df['reading score'],  bins=10,color='green' );
#Univariate Analysis for writing score

sns.distplot(df['writing score'],  bins=10, color='purple' );
# Violin plot

sns.violinplot(x='gender', y='math score', data=df)
pkmn_type_colors = ['#78C850',  # Grass

                    '#F08030',  # Fire

                    '#6890F0',  # Water

                    '#A8B820',  # Bug

                    '#A8A878',  # Normal

                    '#A040A0',  # Poison

                    '#F8D030',  # Electric

                    '#E0C068',  # Ground

                    '#EE99AC',  # Fairy

                    '#C03028',  # Fighting

                    '#F85888',  # Psychic

                    '#B8A038',  # Rock

                    '#705898',  # Ghost

                    '#98D8D8',  # Ice

                    '#7038F8',  # Dragon

                   ]
# Set figure size with matplotlib

plt.figure(figsize=(10,6))

 

# Create plot

sns.violinplot(x='gender',

               y='math score', 

               data=df, 

               inner=None, # Remove the bars inside the violins

               palette=pkmn_type_colors)

 

sns.swarmplot(x='gender',

               y='math score', 

              data=df, 

              color='k', # Make points black

              alpha=0.7) # and slightly transparent

 

# Set title with matplotlib

plt.title('Math Score by Gender')
sns.heatmap(corr)
#  Bar Plot

sns.countplot(x='race/ethnicity', data=df, palette=pkmn_type_colors)

 

# Rotate x-labels

plt.xticks(rotation=-45)

plt.title('Data Analysis by Race/Ethnicity')
g = sns.FacetGrid(df, col="race/ethnicity", height=4, aspect=.5)

g.map(sns.barplot, "gender", "math score");
sns.catplot(x="gender", y="math score", hue="race/ethnicity", kind="bar", data=df);
sns.pairplot(df, hue="gender", height=2.5);
df_mean=df.groupby(

   ['gender'],as_index=True

).agg(

    {

         'math score':"mean",   

         'reading score': "mean", 

         'writing score': 'mean'  

    }

)

df_mean