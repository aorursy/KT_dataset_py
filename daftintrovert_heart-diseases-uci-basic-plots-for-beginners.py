# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import regex as re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
file = '../input/heart.csv'

df = pd.read_csv(file)

df
df.dropna()
label = 'Men','Women'

sizes = [0.6831, 0.3168]

colors = ['lightblue','pink']

explode = (0.1, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=label, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('% of men and women in our dataset\n\n\n')

plt.axis('equal')

plt.show()
label = 'risk of Men of having heartattack','risk of women having heartattack'

sizes = [(0.449), (0.75)]

colors = ['lightblue','pink']

explode = (0.08, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=label, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=0)

 

plt.axis('equal')

plt.show()
slices_hours = [207-93, 93]

activities = ['% Men not at risk', '% Men at risk']

colors = ['lightgreen', 'orange','b','y']

plt.pie(slices_hours, labels=activities, colors=colors,shadow = True, startangle=90, autopct='%.1f%%')

plt.title('Analysis of positive heart attack in men out of total men')

plt.show()



#percentage of men at risk
slices_hours = [96-72, 72]

activities = ['% Women not at risk', '% Women at risk']

colors = ['yellow','teal']

plt.pie(slices_hours, labels=activities, colors=colors, startangle=0, autopct='%.1f%%')

plt.title('Aanlysis of positive heart attack in women out of total women')

plt.show()



#percentage of women at risk
# age distribution of people affected



import seaborn as sns



# seaborn histogram

sns.distplot(df['age'], hist=True, kde=False, 

             bins=int(180/5), color = 'yellow',

             hist_kws={'edgecolor':'black'})

# Add labels

plt.title('distribution of people count vs age')

plt.xlabel('age(in years)')

plt.ylabel('number of person')

plt.show()

sns.distplot(df['age'])
# distribution of cholesterol level in affected people



sns.distplot(df['chol'], hist=True, kde=False, 

             bins=int(200/5), color = 'green',

             hist_kws={'edgecolor':'black'})



plt.title('distribution serum cholesterol in mg/dl')

plt.xlabel('serum cholesterol in mg/dl')

plt.ylabel('count')

plt.show()

sns.distplot(df['chol'])
# max heart rate distribution 



sns.distplot(df['thalach'], hist=True, kde=False, 

             bins=int(200/5), color = 'red',

             hist_kws={'edgecolor':'black'})



plt.title('distribution of maximum heart rate achieved')

plt.xlabel('maximum heart rate achieved')

plt.ylabel('count')

plt.show()

sns.distplot(df['thalach'])
# resting heart rate distribution



sns.distplot(df['trestbps'], hist=True, kde=False, 

             bins=int(100/5), color = 'brown',

             hist_kws={'edgecolor':'red'})



plt.title('distribution of resting blood pressure (in mm Hg on admission to the hospital)')

plt.xlabel('resting blood pressure (in mm Hg on admission to the hospital)')

plt.ylabel('count')

plt.show()

sns.distplot(df['trestbps'],rug = True,hist = False)
