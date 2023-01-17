

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

 

iowa_file_path = '../input/titanic123/ChanDarren_RaiTaran_Lab2a.csv'

home_data = pd.read_csv(iowa_file_path)

home_data.describe(include = "all")



home_data.info()
home_data.head()
home_data.corr()
print(pd.isnull(home_data).sum())
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.barplot(x="Sex", y="Survived", data=home_data)



#print("Percentage of females who survived:", home_data["Survived"][home_data["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



#print("Percentage of males who survived:", home_data["Survived"][home_data["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
sns.barplot(x="Pclass", y="Survived", data=home_data)
sns.barplot(x="SibSp", y="Survived", data=home_data)
sns.barplot(x="Age", y="Survived", data=home_data)

# the output contain alot of bins, so I have to devide the x into range
home_data["Age"] = home_data["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

home_data['AgeGroup'] = pd.cut(home_data["Age"], bins, labels = labels)

sns.barplot(x="AgeGroup", y="Survived", data=home_data)

home_data.corr()