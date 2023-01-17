# Import necessary libraries:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import *

%matplotlib inline
data=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

data.head()
data.info()
for feature in data.columns:

    uniq = np.unique(data[feature])

    print('{}: {} distinct values -  {}'.format(feature,len(uniq),uniq))
corr = data.corr()

print(corr)
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

plt.title('Correlation Analysis with Original Data')

# Draw the heatmap with the mask and correct aspect ratio

ca = sns.heatmap(corr, cmap='coolwarm',center=0, vmin = -1,

            square=True, linewidths=1, cbar_kws={"shrink": .8}, annot = True)
sns.set(style="ticks", color_codes=True)

g = sns.pairplot(data, palette="coolwarm")

title = g.fig.suptitle("Scores Pair Plot", y = 1.05)
data_label_encoding = data.copy()
# converting type of columns to 'category'

data_label_encoding['gender']= data_label_encoding['gender'].astype('category')

data_label_encoding['race/ethnicity']= data_label_encoding['race/ethnicity'].astype('category')

data_label_encoding['parental level of education']= data_label_encoding['parental level of education'].astype('category')

data_label_encoding['lunch']= data_label_encoding['lunch'].astype('category')

data_label_encoding['test preparation course']= data_label_encoding['test preparation course'].astype('category')
# Assigning numerical values and storing in another column

data_label_encoding['gender_cat']= data_label_encoding['gender'].cat.codes

data_label_encoding['race/ethnicity_cat']= data_label_encoding['race/ethnicity'].cat.codes

data_label_encoding['parental level of education_cat']= data_label_encoding['parental level of education'].cat.codes

data_label_encoding['lunch_cat']= data_label_encoding['lunch'].cat.codes

data_label_encoding['test preparation course_cat']= data_label_encoding['test preparation course'].cat.codes
data_label_encoding.info()
corr_label_encoding = data_label_encoding.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

plt.title('Correlation Analysis with Label Encoding')

# Draw the heatmap with the mask and correct aspect ratio

ca = sns.heatmap(corr_label_encoding, cmap='coolwarm',center=0, vmin = -1,

            square=True, linewidths=1, cbar_kws={"shrink": .8}, annot = True)
dt_tmp = data_label_encoding[['math score', 'reading score', 'writing score', 'gender']]

dt_tmp = dt_tmp.melt(id_vars = ['gender'])
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 7))

plt.title('Gender influence in math, reading and writing scores')

violin_gender = sns.violinplot(x="variable", y="value", hue="gender",

                     data=dt_tmp, palette="coolwarm", split=True,

                     scale="count", inner="quartile", bw=.1)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
dt_tmp = data_label_encoding[['math score', 'reading score', 'writing score', 'race/ethnicity']]

dt_tmp = dt_tmp.melt(id_vars = ['race/ethnicity'])
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 7))

plt.title('Race / Ethnicity influence in math, reading and writing scores')

violin_gender = sns.violinplot(x="variable", y="value", hue="race/ethnicity",

                     data=dt_tmp, palette="coolwarm", 

                     scale="count", inner="quartile", bw=.1)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
dt_tmp = data_label_encoding[['math score', 'reading score', 'writing score', 'test preparation course']]

dt_tmp = dt_tmp.melt(id_vars = ['test preparation course'])
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 7))

plt.title('Test preparation course influence in math, reading and writing scores')

violin_gender = sns.violinplot(x="variable", y="value", hue="test preparation course",

                     data=dt_tmp, palette="coolwarm", split=True,

                     scale="count", inner="quartile", bw=.1)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
dt_tmp['test preparation course'].value_counts()
dt_tmp = data_label_encoding[['math score', 'reading score', 'writing score', 'lunch']]

dt_tmp = dt_tmp.melt(id_vars = ['lunch'])
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 7))

plt.title('Test preparation course influence in math, reading and writing scores')

violin_gender = sns.violinplot(x="variable", y="value", hue="lunch",

                     data=dt_tmp, palette="coolwarm", split=True,

                     scale="count", inner="quartile", bw=.1)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
dt_tmp['lunch'].value_counts()
data_onehotencoding = data.copy()
data_onehotencoding = pd.get_dummies(data_onehotencoding, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch',

       'test preparation course'])
corr_label_encoding = data_onehotencoding.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(14, 14))

plt.title('Correlation Analysis with One-Hot Encoding')

# Draw the heatmap with the mask and correct aspect ratio

ca = sns.heatmap(corr_label_encoding, cmap='coolwarm',center=0, vmin = -1,

            square=True, linewidths=1, cbar_kws={"shrink": .8}, annot = True)
dt_tmp = data_onehotencoding[['math score', 'reading score', 'writing score', 'race/ethnicity_group E']].copy()
# Set up the matplotlib figure

sns.set(style="ticks", color_codes=True)

pairplot_group_E = sns.pairplot(hue='race/ethnicity_group E',data=dt_tmp, palette="coolwarm")

title = pairplot_group_E.fig.suptitle("Race/Ethnicity Group E influence on scores", y = 1.05)
dt_tmp = data_onehotencoding[['math score', 'reading score', 'writing score', "parental level of education_high school"]].copy()
# Set up the matplotlib figure

sns.set(style="ticks", color_codes=True)

pairplot_bachelor = sns.pairplot(hue="parental level of education_high school",data=dt_tmp, palette="coolwarm")

title = pairplot_group_E.fig.suptitle("Parental level of education_high_school", y = 1.05)
dt_tmp = data_onehotencoding[['math score', 'reading score', 'writing score', "parental level of education_master's degree"]].copy()
# Set up the matplotlib figure

sns.set(style="ticks", color_codes=True)

pairplot_bachelor = sns.pairplot(hue="parental level of education_master's degree",data=dt_tmp, palette="coolwarm")

title = pairplot_group_E.fig.suptitle("Parental level of education_master's degree influence on scores", y = 1.05)