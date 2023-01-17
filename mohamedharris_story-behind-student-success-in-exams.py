# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as stats

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

sns.set_context('notebook')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/student-alcohol-consumption/student-por.csv')
print('No. of students : ', df.shape[0])

print('No. of attributes : ', df.shape[1])
df.info()
df.describe()
df.head()
df.columns
ax = sns.catplot(x = 'school', data = df, kind = 'count',hue = 'sex', palette = 'husl')

plt.title('Student Distribution in School')

plt.xlabel('School Name')

plt.ylabel('# Students')

ax.set(xticklabels = ["Gabriel Pereira", "Mousinho da Silveira"])

plt.show()
fig, ax = plt.subplots()

ax.hist(df.loc[(df['sex'] == 'F'), 'age'], color = 'k', histtype = 'step', label = 'Female')

ax.hist(df.loc[(df['sex'] == 'M'), 'age'], color = 'r', histtype = 'step', label = 'Male')

plt.title('Student Age by Gender')

plt.xlabel('Age')

plt.ylabel('# Students')

plt.legend()

plt.show()
fig, ax = plt.subplots()

ax.hist(df.loc[(df['sex'] == 'F'), 'G3'], color = 'r', histtype = 'step', label = 'Female')

ax.hist(df.loc[(df['sex'] == 'M'), 'G3'], color = 'b', histtype = 'step', label = 'Male')

plt.title('Student Grade by Gender')

plt.xlabel('Grade')

plt.ylabel('# Students')

plt.legend()

plt.show()
ax = df.groupby('sex')['G1', 'G2', 'G3'].mean().plot(kind = 'bar')

plt.title('Mean Score by Gender')

plt.xlabel('Gender')

plt.ylabel('Avg Grade')

plt.legend(loc = 'upper right')

ax.set_xticklabels(['Female', 'Male'], rotation = 360)

plt.show()
ax = sns.FacetGrid(df,  col = 'Medu', hue = 'sex').map(plt.hist, 'G3').add_legend()

ax.fig.suptitle("Student Grade Analysis by Mother's Education")

plt.subplots_adjust(top = 0.7)

plt.show()
df.groupby('Medu')['G3'].mean()
ax = sns.FacetGrid(df, col = 'Fedu', hue = 'sex').map(plt.hist, 'G3').add_legend()

ax.fig.suptitle("Student Grade Analysis by Father's Education")

plt.subplots_adjust(top = 0.7)

plt.show()
df.groupby('Fedu')['G3'].mean()
ax = sns.FacetGrid(df, col = 'Mjob', hue = 'sex').map(plt.hist, 'G3').add_legend()

ax.fig.suptitle("Student Grade Analysis by Mother's Occupation")

plt.subplots_adjust(top = 0.7)

plt.show()
df.groupby('Mjob')['G3'].mean()
ax = sns.FacetGrid(df, col = 'Fjob', hue = 'sex').map(plt.hist, 'G3').add_legend()

ax.fig.suptitle("Student Grade Analysis by Father's Occupation")

plt.subplots_adjust(top = 0.7)

plt.show()
df.groupby('Fjob')['G3'].mean()
ax = sns.FacetGrid(df, col = 'studytime', row = 'traveltime', hue = 'studytime').map(plt.hist, 'G3').add_legend()

ax.fig.suptitle("Student Grade Analysis by Study Time & Travel Time")

plt.subplots_adjust(top = 0.9)

plt.show()
ax = sns.lmplot(x = 'studytime',y = 'G3', hue = 'sex', data = df, palette = 'Set1')

ax.fig.suptitle('Correlation b/w Study Time and Grade')

plt.subplots_adjust(top = 0.9)
sns.catplot(x = 'schoolsup',y = 'G3', hue = 'sex', data = df, kind = 'bar', ci = None)

plt.title('Effect of Extra Educational Support on Grade')

plt.ylabel('Avg. Grade')

plt.show()
sns.catplot(x = 'famsup',y = 'G3', hue = 'sex', data = df, kind = 'bar', ci = None)

plt.title('Effect of Family Educational Support on Grade')

plt.ylabel('Avg. Grade')

plt.show()
sns.catplot(x = 'paid',y = 'G3', hue = 'sex', data = df, kind = 'bar', ci = None)

plt.title('Effect of Tuitions on Grade')

plt.ylabel('Avg. Grade')

plt.show()
sns.catplot(x = 'activities', y = 'G3', kind = 'bar', data = df)

plt.title('Effect of Extra Activities on Grade')

plt.ylabel('Avg. Grade')

plt.show()
sns.catplot(x = 'internet', y = 'G3', kind = 'bar', data = df)

plt.title('Effect of Internet on Grade')

plt.ylabel('Avg. Grade')

plt.show()
sns.catplot(x = 'romantic', y = 'G3', kind = 'bar', data = df)

plt.title('Effect of Romantic Relationships on Grade')

plt.ylabel('Avg. Grade')

plt.show()
ax = sns.FacetGrid(df, col = 'goout', row = 'freetime', hue = 'sex').map(plt.hist, 'G3').add_legend()

ax.fig.suptitle("Effects of Personal Preferences on Grades")

plt.subplots_adjust(top = 0.95)

plt.show()
ax = sns.lmplot(x = 'goout',y = 'freetime', hue = 'sex', data = df, palette = 'Set1')

ax.fig.suptitle('Correlation b/w Hanging out and Leisure Time')

plt.subplots_adjust(top = 0.9)
ax = sns.FacetGrid(df, col = 'Dalc', row = 'goout', hue = 'Dalc').map(plt.hist, 'G3').add_legend()

ax.fig.suptitle("Effects of Personal Preferences on Grades")

plt.subplots_adjust(top = 0.95)

plt.show()
ax = sns.lmplot(x = 'goout',y = 'Dalc', hue = 'sex', data = df, palette = 'Set1')

ax.fig.suptitle('Correlation b/w Hanging out and Alcohol Consumption')

plt.subplots_adjust(top = 0.9)
ax = sns.FacetGrid(df, col = 'Walc', row = 'goout', hue = 'Walc').map(plt.hist, 'G3').add_legend()

ax.fig.suptitle("Effects of Personal Preferences on Grades")

plt.subplots_adjust(top = 0.95)

plt.show()
ax = sns.lmplot(x = 'goout',y = 'Walc', hue = 'sex', data = df, palette = 'Set1')

ax.fig.suptitle('Correlation b/w Hanging out and Alcohol Consumption')

plt.subplots_adjust(top = 0.9)
ax = sns.lmplot(x = 'failures',y = 'absences', hue = 'sex', data = df, palette = 'Set1')

ax.fig.suptitle('Correlation b/w Absences and Failures in Subjects')

plt.subplots_adjust(top = 0.9)
fig, ax = plt.subplots(figsize = (15, 10))

sns.heatmap(df.corr(), annot = True, cmap = 'Blues', linewidths = .5)