# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')



import seaborn as sns

import plotly.graph_objs as go



from collections import Counter



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerID = test_df['PassengerId']
train_df.columns
train_df.head()



train_df.describe()
train_df.info()
def bar_plot(variable):

    """

        input : variable ex:"Sex"

        output: bar plot & value count

    """

    # get feature

    var = train_df[variable]

    # count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))
category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
category2 = ["Cabin", "Name", "Ticket"]

for c in category2:

    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar = ["Fare", "Age", "PassengerId"]

for n in numericVar:

    plot_hist(n)
list1 = ['SibSp','Parch','Age','Fare','Survived']

sns.heatmap(train_df[list1].corr(), annot = True, fmt = '.2f', linewidths=.5)
g = sns.factorplot(x = 'Pclass', y = 'Survived', data = train_df, kind = 'bar', size = 6)

g.set_ylabels('Survived Probability')

plt.show()

train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)

g = sns.factorplot(x = 'Sex', y = 'Survived', data = train_df, kind = 'bar', size = 6)

g.set_ylabels('Survived Probability')

plt.show()



train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending = False)
g = sns.factorplot(x = 'SibSp', y = 'Survived', data = train_df, kind = 'bar', size = 6)

g.set_ylabels('Survived Probability')

plt.show()



train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending = False)
g = sns.factorplot(x = 'Parch', y = 'Survived', data = train_df, kind = 'bar', size = 6)

g.set_ylabels('Survived Probability')

plt.show()



train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False)
g = sns.FacetGrid(train_df, col = 'Survived', size = 6)

g.map(sns.distplot, 'Age', bins = 25)

plt.show()
g = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass', size = 3)

g.map(plt.hist, 'Age', bins = 25)

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = 'Embarked', size = 4)

g.map(sns.pointplot, 'Pclass','Survived','Sex')

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = 'Embarked', col = 'Survived', size = 3)

g.map(sns.barplot, 'Sex','Fare')

g.add_legend()

plt.show()
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier Step

        outlier_step = IQR * 1.5

        # Detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # Store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
# Drop outliers

train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop = True)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df['Embarked'].isnull()]
train_df.boxplot(column='Fare', by = 'Embarked')

plt.show()
train_df['Embarked'] = train_df['Embarked'].fillna('C')

train_df[train_df['Embarked'].isnull()]
train_df[train_df['Fare'].isnull()]
np.mean(train_df[train_df['Pclass']==3]['Fare'])
train_df['Fare'] = train_df['Fare'].fillna(np.mean(train_df[train_df['Pclass']==3]['Fare']))

train_df[train_df['Fare'].isnull()]
train_df[train_df['Age'].isnull()]
sns.factorplot(x= 'Sex', y = 'Age', data = train_df, kind = 'box')

plt.show()
sns.factorplot(x= 'Sex', y = 'Age', hue = 'Pclass', data = train_df, kind = 'box')

plt.show()
sns.factorplot(x= 'Parch', y = 'Age', data = train_df, kind = 'box')

sns.factorplot(x= 'SibSp', y = 'Age', data = train_df, kind = 'box')

plt.show()
# In order to see Age feature in heatmap, we need to convert it to numerical value.

train_df['Sex'] = [1 if i == 'male' else 0 for i in train_df['Sex']]
sns.heatmap(train_df[['Age','Sex','SibSp','Parch','Pclass']].corr(), annot = True,linewidths=.5)

plt.show()
index_nan_age = list(train_df['Age'][train_df['Age'].isnull()].index)

for each in index_nan_age:

    age_prediction = train_df['Age'][((train_df['SibSp'] == train_df.iloc[each]['SibSp']) &(train_df['Parch'] == train_df.iloc[each]['Parch'])& (train_df['Pclass'] == train_df.iloc[each]['Pclass']))].median()

    age_median = train_df['Age'].median()

    if not np.isnan(age_prediction):

        train_df['Age'].iloc[each] = age_prediction

    else:

        train_df['Age'].iloc[each] = age_median
train_df[train_df['Age'].isnull()]