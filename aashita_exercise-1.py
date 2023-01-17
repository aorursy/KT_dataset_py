import numpy as np

import pandas as pd



# The following two modules matplotlib and seaborn are for plots

import matplotlib.pyplot as plt

import seaborn as sns # Comment this if seaborn is not installed

%matplotlib inline



# The module re is for regular expressions

import re
# Uncomment the below two lines only if using Google Colab

# from google.colab import files

# uploaded = files.upload()

df = pd.read_csv('../input/train.csv')

df.head()
df.describe()
df.isnull().sum()
df.head()
plt.axis('equal')

plt.pie(df['Survived'].value_counts(), labels=('Died', "Survived"));
sns.barplot(x = 'Sex', y = 'Survived', data = df);
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df);
sns.pointplot(x='Sex', y='Survived', hue='Pclass', data=df);
df.head()
df.isnull().sum()
df.loc[:20, 'Name'].values
re.findall("\w\w[.]", 'Braund, Mr. Owen Harris')
re.findall("\w\w[.]", 'Heikkinen, Miss. Laina')[0]
# Fill in below:

re.findall("FILL IN HERE", 'Heikkinen, Miss. Laina')[0]
get_title('Futrelle, Mrs. Jacques Heath (Lily May Peel)')
get_title('Simonius-Blumer, Col. Oberst Alfons')
df.head()
# We first make a copy of the dataframe in case we want 

# to use it later before we fill in missing values

df2 = df.copy() 

correlation_matrix = df.corr();

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(correlation_matrix);
plt.figure(figsize=(14, 10))

sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df);