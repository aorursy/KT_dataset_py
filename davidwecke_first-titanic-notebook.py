# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# Question 7
train_data.describe()
# Question 8
train_data_categorial = train_data.loc[:,["Sex", "Ticket","Cabin", "Embarked"]]
categorial_stats = train_data_categorial.describe()
categorial_stats
# Question 9
pclass1 = train_data.loc[train_data.Pclass == 1]["Survived"]
pclass1_survived = sum(pclass1)/len(pclass1)
print(pclass1_survived)
# Question 10
women = train_data.loc[train_data.Sex == 'female']["Survived"]
women_survived = sum(women)/len(women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
men_survived = sum(men)/len(men)

print(women_survived)
print(men_survived)
# Question 11
age_survived = train_data[train_data.Survived == 1]["Age"]

age_survived.plot.hist(title="Survived", bins=20).set_ylim(0,60)
# Question 11 Continued
age_not_survived = train_data[train_data.Survived == 0]["Age"]
age_not_survived.plot.hist(title="Not Survived", bins=20, label="Age").set_ylim(0,60)
# Question 12
import numpy as np
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,15))
ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()
x0 = train_data.loc[(train_data.Pclass == 1) & (train_data.Survived == 0)]["Age"]
ax0.hist(x0, 20)
ax0.set_title('Pclass = 1 & Survived = 0')
ax0.set_ylim(0,50)

x1 = train_data.loc[(train_data.Pclass == 1) & (train_data.Survived == 1)]["Age"]
ax1.hist(x1, 20)
ax1.set_title('Pclass = 1 & Survived = 1')
ax1.set_ylim(0,50)

x2 = train_data.loc[(train_data.Pclass == 2) & (train_data.Survived == 0)]["Age"]
ax2.hist(x2, 20)
ax2.set_title('Pclass = 2 & Survived = 0')
ax2.set_ylim(0,50)

x3 = train_data.loc[(train_data.Pclass == 2) & (train_data.Survived == 1)]["Age"]
ax3.hist(x3, 20)
ax3.set_title('Pclass = 2 & Survived = 1')
ax3.set_ylim(0,50)

x4 = train_data.loc[(train_data.Pclass == 3) & (train_data.Survived == 0)]["Age"]
ax4.hist(x4, 20)
ax4.set_title('Pclass = 3 & Survived = 0')
ax4.set_ylim(0,50)

x5 = train_data.loc[(train_data.Pclass == 3) & (train_data.Survived == 1)]["Age"]
ax5.hist(x5, 20)
ax5.set_title('Pclass = 3 & Survived = 1')
ax5.set_ylim(0,50)
# Question 13
import numpy as np
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,15))
ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()
x0 = train_data.loc[(train_data.Embarked == "S") & (train_data.Survived == 0)]["Fare"]
ax0.hist(x0, 20)
ax0.set_title('Emarked = S & Survived = 0')
ax0.set_ylim(0,250)

x1 = train_data.loc[(train_data.Embarked == "S") & (train_data.Survived == 1)]["Fare"]
ax1.hist(x1, 20)
ax1.set_title('Emarked = S & Survived = 1')
ax1.set_ylim(0,250)

x2 = train_data.loc[(train_data.Embarked == "Q") & (train_data.Survived == 0)]["Fare"]
ax2.hist(x2, 20)
ax2.set_title('Emarked = Q & Survived = 0')
ax2.set_ylim(0,250)

x3 = train_data.loc[(train_data.Embarked == "Q") & (train_data.Survived == 1)]["Fare"]
ax3.hist(x3, 20)
ax3.set_title('Emarked = Q & Survived = 1')
ax3.set_ylim(0,250)

x4 = train_data.loc[(train_data.Embarked == "C") & (train_data.Survived == 0)]["Fare"]
ax4.hist(x4, 20)
ax4.set_title('Emarked = C & Survived = 0')
ax4.set_ylim(0,250)

x5 = train_data.loc[(train_data.Embarked == "C") & (train_data.Survived == 1)]["Fare"]
ax5.hist(x5, 20)
ax5.set_title('Emarked = C & Survived = 1')
ax5.set_ylim(0,250)
# Question 13

sns.set(style="ticks", color_codes=True)

titanic_survived = train_data.loc[train_data.Survived == 1]
titanic_not_survived = train_data.loc[train_data.Survived == 0]
sns.catplot(x="Sex", y="Fare", hue="Embarked", kind="bar", data= titanic_not_survived)
sns.catplot(x="Sex", y="Fare", hue="Embarked", kind="bar", data= titanic_survived)
# Question 14
duplicate_tickets = train_data[train_data.duplicated(['Ticket'])]
print(len(duplicate_tickets))

tickets_not_survived = train_data.loc[train_data.Survived==0]['Ticket']
tickets_not_survived.hist(bins=100).set_title("Not survived")
tickets_survived = train_data.loc[train_data.Survived==1]['Ticket']
tickets_survived.hist(bins=100).set_title("Survived")
# Question 15
cabin_na = train_data['Cabin'].isna()
print(sum(cabin_na))
cabin_na2 = test_data['Cabin'].isna()
print(sum(cabin_na2))
# Question 16
train_data['Gender'] = train_data['Sex'].map(dict(male=0, female=1))
train_data.head()
# Question 17
# Using random numbers between mean and std
age_na = train_data['Age'].isna()
print(sum(age_na))
train_data_stats = train_data.describe()
age_mean = train_data_stats['Age']['mean']
age_std = train_data_stats['Age']['std']
rand_age = pd.Series(np.random.uniform((age_mean-age_std),(age_mean+age_std),size=len(age_na)))
rand_age.head()
train_data['Age'].fillna(value=rand_age, inplace=True)
age_na = train_data['Age'].isna()
print(sum(age_na))
# Question 18
most_embarked = categorial_stats['Embarked']['top']
print(sum(train_data['Embarked'].isna()))
train_data['Embarked'].fillna(value=most_embarked, inplace=True)
print(sum(train_data['Embarked'].isna()))
# Question 19
fare_mode = train_data_stats['Fare']['max']
print(sum(test_data['Fare'].isna()))
test_data['Fare'].fillna(value=fare_mode, inplace=True)
print(sum(test_data['Fare'].isna()))
train_data['Fare'] = pd.cut(x=train_data['Fare'], bins=[-0.001, 7.91, 14.454, 31.0, 512.329], labels=[0,1,2,3])
train_data.head()
selected_features = train_data[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Gender']]
selected_features.head()
selected_features.to_csv('processed_train_data.csv')
