%matplotlib inline
import numpy as np
import pandas as pd
import pylab as p

# load csv data
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, header=0)
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, header=0)

# review first 3 lines of file
print("\n\nTop of the training data:")
print(train.head(3))
print("\n\nSummary statistics of training data")
print(train.describe())
#---
# Age is missing data - Count is at 714 rather than 891

# save data to additional csv
train.to_csv('copy_of_the_training_data.csv', index=False)
print('Male mean age:')
print(train[train['Sex'] == 'male']['Age'].mean())

print('\n\nfemale mean age:')
print(train[train['Sex'] == 'female']['Age'].mean())
print('\n\nMale Age Distribution')
train[train['Sex'] == 'male']['Age'].hist()
p.show()
print('\n\nFemale Age Distribution')
train[train['Sex'] == 'female']['Age'].hist()
p.show()
male_survived = train[train['Survived'] == 1]['Sex'].value_counts()['male']
male_perished = train[train['Survived'] == 0]['Sex'].value_counts()['male']

female_survived = train[train['Survived'] == 1]['Sex'].value_counts()['female']
female_perished = train[train['Survived'] == 0]['Sex'].value_counts()['female']

print('male survived', male_survived)
print('male perished', male_perished)
print('Survival Ratio', male_survived / (male_survived + male_perished))
print()
print('female survived', female_survived)
print('female perished', female_perished)
print('Survival Ratio', female_perished / (female_survived + female_perished))