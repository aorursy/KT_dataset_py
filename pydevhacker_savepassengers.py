# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
genderclassmodel_df = pd.read_csv('../input/genderclassmodel.csv')
gendermodel_df = pd.read_csv('../input/gendermodel.csv')
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.describe()
train_df.isnull().sum()
#Cabin has 687 null values we can drop this column
train_df.drop('Cabin', axis=1, inplace=True)
#fill Age by mean value
mean_age = train_df['Age'].mean()
mean_age
train_df['Age'].fillna(mean_age, inplace=True)
#fill two null value in Embarked columns
train_df['Embarked'].value_counts()
train_df['Embarked'].fillna('S', inplace=True)

#We can drop passanger Id
train_df.drop('PassengerId', inplace=True
)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")
sns.barplot(train_df['Sex'], train_df['Survived'])
sns.set(style="whitegrid")

# Load the example Titanic dataset

# Draw a nested barplot to show survival for class and sex
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train_df,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
sns.set(style="whitegrid")

# Load the example Titanic dataset

# Draw a nested barplot to show survival for class and sex
g = sns.factorplot(x="Age", y="Survived", hue="Sex", data=train_df,
                   size=10, palette="muted")
g.despine(left=True)

