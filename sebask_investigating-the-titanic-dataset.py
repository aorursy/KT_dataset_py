import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

from pylab import savefig

from sklearn.preprocessing import Imputer





# Import data into pandas data frame

df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )



#Convert Sex into workable feature, 1 for female and 0 for male passenger

df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})



# Probing missing values

print(df['Age'].isnull().sum()/ df['Age'].size)

print(df['Sex'].isnull().sum()/ df['Sex'].size)

print(df['Pclass'].isnull().sum()/ df['Pclass'].size)
# Describe the age distribution of data without missing values

dfage = df.Age.dropna()

print(dfage.describe())

print(df.groupby(['Sex'])['Age'].mean())



# Create a visualization of the distribution

sns.distplot(dfage)

plt.title('Age Distribution for Titanic Training Data')

plt.ylabel('Proportion of Dataset')

sns.plt.show()



# Draw a nested violinplot, to see the distribution of various categories

sns.set(style="whitegrid", palette="pastel", color_codes=True)

df['Gender'] = df['Sex'].map({1 : 'Female', 0 : 'Male'})

plt.title('Violin Plot for Titanic Training Data')

sns.violinplot(x="Pclass", y="Age", hue="Gender", data=df, split=True,

               inner="quartile")

sns.despine(left=True)

sns.plt.xlabel('Passenger Class')

sns.plt.show()



'''The missing values were mostly male and had a lower survival rate'''

missing_values = df[df['Age'].isnull()]

print("% Female missing data: ")

print(missing_values['Sex'].mean())

print('% survival missing data: ') 

print(missing_values['Survived'].mean())

print('mean passenger class: ')

print(missing_values['Pclass'].mean())
# Imputation code

df['Age'].fillna(df.groupby(['Pclass','Sex'])['Age'].transform("mean"), inplace=True)

print(df['Age'][df['Age'].isnull()])



# Create new variable for whether the passenger was a minor

df['Child'] = df['Age']<16

df['Child'] = df['Child'].astype(int)

# View correlation matrix with survival

corr = df.corr()

print(corr['Survived'])



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(10, 220, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

plt.title('Correlation Matrix for Titanic Training Data')

sns.plt.show()
print('Overall Survival Rate')

print(df['Survived'].mean())

print('')

print('Differing survival rates')

dfsummary= df.groupby(['Sex','Pclass','Child'])['Survived'].mean()

print(dfsummary)



print(df.groupby(['Sex'],as_index=False)['Survived'].mean())

print(stats.ttest_ind(df['Survived'][df['Sex']==1],df['Survived'][df['Sex']==0],equal_var=False))



print(df.groupby(['Child'],as_index=False)['Survived'].mean())

print(stats.ttest_ind(df['Survived'][df['Child']==1],df['Survived'][df['Child']==0],equal_var=False))



print(df.groupby(['Pclass'],as_index=False)['Survived'].mean())

print(stats.ttest_ind(df['Survived'][df['Pclass']==1],df['Survived'][df['Pclass']!=0],equal_var=False))