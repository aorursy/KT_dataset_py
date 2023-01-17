# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



dana_file_path = '../input/titanic123/ChanDarren_RaiTaran_Lab2a.csv'

home_data = pd.read_csv(dana_file_path)

home_data.describe(include="all")





# Any results you write to the current directory are saved as output.
home_data.head()
home_data.info()
home_data.corr()

#pclass is high correlation 



print(home_data.columns)
print(pd.isnull(home_data).sum())
sns.barplot(x="Sex", y="Survived", data=home_data)

print("Percentage of females who survived:", home_data["Survived"][home_data["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", home_data["Survived"][home_data["Sex"] == 'male'].value_counts(normalize = True)[1]*100)

sns.barplot(x="Pclass", y="Survived", data=home_data)
sns.barplot(x="SibSp", y="Survived", data=home_data)
sns.barplot(x="Parch", y="Survived", data=home_data)

plt.show()
home_data["Age"] = home_data["Age"].fillna(-0.10)

bins = [-1, 0, 7, 9, 15, 29, 40, 65, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

home_data['AgeGroup'] = pd.cut(home_data["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=home_data)

plt.show()