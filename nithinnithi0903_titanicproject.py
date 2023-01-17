# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
training = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")



training['train_test'] = 1

test['train_test'] = 0

test['Survived'] = np.NaN



all_data = pd.concat([training, test])

all_data.columns
#quick look at our data types & null counts 

training.info() #Age, Cabin and Embarked has Null values
training.describe()
print(training.head()) #Visualise the values for analysis
# look at numeric and categorical values separately

df_num = training[['Age', 'SibSp', 'Parch', 'Fare']]

df_cat = training[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]
# Potting the distributions of all numeric data

for col in df_num.columns:

    plt.hist(df_num[col])

    plt.title(col)

    plt.show()
#Let's plot the correlation matrix for this numeric data and analyse

print(df_num.corr())

sns.heatmap(df_num.corr())
#This gives us the avg in each column to Survived column

pd.pivot_table(training, index = 'Survived', values = ['Age', 'SibSp', 'Parch', 'Fare'])
for col in df_cat.columns:

    sns.barplot(df_cat[col].value_counts().index,df_cat[col].value_counts()).set_title(col)

    plt.show()