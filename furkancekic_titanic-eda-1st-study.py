import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
def bar_plot(variable):
    """
    input: variable, ex: "sex"
    output: bar plot & value count
    
    """
    # get feature 
    var = train_df[variable]
    
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9, 3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title("Variable")
    plt.show()
    print(" {}: \n {} " .format(variable, varValue))
category = ['Survived', 'Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']

for i in category:
    bar_plot(i)
category2 = ['Name', 'Cabin', 'Ticket']
for i in category2:
    print("{} \n " .format(train_df[i].value_counts()))
def plot_hist(variable):
    plt.figure(figsize = (10, 4))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist" .format(variable))
    plt.show()
numericVar = [ "Fare", "Age", "PassengerId"]
for i in numericVar:
    plot_hist(i)
# Pclass - Survived
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
# Sex - Survived
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
# SibSp - Survived
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
# Sex - Survived
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
train_df.loc[detect_outliers(train_df, ['Age', 'SibSp', 'Parch', 'Fare'])]
train_df = train_df.drop(detect_outliers(train_df, ['Age', 'SibSp', 'Parch', 'Fare']), axis = 0).reset_index(drop = True)
train_df_len = len(train_df)
train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)
train_df.head()
# check if any column has null variable
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
## We can delete the missing values but we do not want to lose any value, so that they will be filled.

train_df[train_df['Embarked'].isnull()]
train_df.boxplot(column = 'Fare', by = 'Embarked')
plt.show()
train_df['Embarked'] = train_df['Embarked'].fillna('C')
# CHECK
train_df[train_df['Embarked'].isnull()]
train_df[train_df['Fare'].isnull()]
train_df.groupby('Pclass').Fare.mean()
# train_df.groupby('Embarked').Fare.mean()
# train_df['Fare'] = train_df['Fare'].fillna(12.741220)
# train_df[train_df.PassengerId == 1044]

## Or

train_df['Fare'] = train_df['Fare'].fillna(np.mean(train_df[train_df['Pclass'] == 3]['Fare']))
train_df[train_df.PassengerId == 1044]
train_df[train_df['Fare'].isnull()]