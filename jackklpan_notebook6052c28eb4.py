# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.info()
train_df.describe()
sns.distplot(train_df['Survived'])
survived_df = train_df[train_df['Survived']==1]

not_survived_df = train_df[train_df['Survived']==0]
g = sns.FacetGrid(train_df, col="Pclass")

g.map(plt.hist, "Survived");
g = sns.FacetGrid(train_df, col="Sex")

g.map(plt.hist, "Survived");
sns.swarmplot(x="Survived", y="Age", data=train_df);
sns.boxplot(x="Survived", y="Age", data=train_df);
sns.violinplot(x="Survived", y="Age", data=train_df);

#sns.swarmplot(x="Survived", y="Age", data=train_df, color="w", alpha=.5);
sns.barplot(x="Survived", y="Age", data=train_df);
sns.pointplot(x="Survived", y="Age", data=train_df);
sns.violinplot(x="Survived", y="SibSp", data=train_df);
sns.pointplot(x="Survived", y="SibSp", data=train_df);
sns.violinplot(x="Survived", y="Parch", data=train_df);
sns.pointplot(x="Survived", y="Parch", data=train_df);
sns.violinplot(x="Pclass", y="Fare", data=train_df);
sns.barplot(x="Embarked", y="Survived", data=train_df);
train_df.info()
test_df.info()
sns.violinplot(x="Pclass", y="Fare", data=train_df);
sns.violinplot(x="Embarked", y="Fare", data=train_df);
train_df['Embarked'].dropna().mode()
#Get the values that we want to assign to the missing values

all_df = pd.concat([train_df, test_df])

age_median = all_df['Age'].dropna().median()

fare_median = np.zeros(3)

for f in range(0,3):

    fare_median[f] = all_df[ all_df['Pclass'] == f+1 ]['Fare'].dropna().median()



def clean_df(df):    

    result_df = df.copy()

    

    result_df.loc[ (result_df['Age'].isnull()), 'Age'] = age_median

    #Assign missing Fare based on Pclass

    for f in range(0,3):

        result_df.loc[ (result_df['Fare'].isnull())&(result_df['Pclass'] == f+1 ), 'Fare'] = fare_median[f]

    #Assign missing Embarked based on Fare

    result_df.loc[ (result_df['Embarked'].isnull())&(result_df['Fare'] > 300 ), 'Embarked'] = 'C'

    result_df.loc[ (result_df['Embarked'].isnull())&(result_df['Fare'] <= 25 ), 'Embarked'] = 'Q'

    result_df.loc[ (result_df['Embarked'].isnull()), 'Embarked'] = 'S'

    

    result_df = pd.concat( [ result_df, pd.get_dummies( result_df['Pclass'], 'Pclass' ) ], axis=1 ); #Add columns Pclass_1 .. Pclass_3

    result_df = pd.concat( [ result_df, pd.get_dummies( result_df['Sex'] ) ], axis=1 ); #Add columns male, female

    result_df['Age<10'] = result_df['Age'] < 10

    result_df['Age_10_40'] = (result_df['Age'] > 10)&(result_df['Age']<40)

    result_df['SibSp_0'] = result_df['SibSp'] == 0

    result_df['SibSp_1'] = result_df['SibSp'] == 1

    result_df['Parch_0'] = result_df['Parch'] == 0

    result_df = result_df.drop(['PassengerId', 'Pclass', 'Pclass_2', 'Name', 'Sex', 'male', 'Ticket', 'Cabin'], axis=1)

    return result_df
train_clean_df = clean_df(train_df)

print(train_df.describe())

print(train_clean_df.describe())