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
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')

matplotlib.rcParams['figure.figsize'] = (12,8)



from plotly.offline import download_plotlyjs,init_notebook_mode, iplot

init_notebook_mode(connected=True)

import cufflinks as cf

cf.go_offline()



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn import metrics
df1 = pd.read_csv('/kaggle/input/titanic/train.csv')

df2 = pd.read_csv('/kaggle/input/titanic/test.csv')

df3 = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df1.info()
# Get numeric columns

df1_numeric = df1.select_dtypes(include=[np.number])

numeric_cols1 = df1_numeric.columns.values

print(numeric_cols1)
# Get Non-numeric columns

df1_non_numeric = df1.select_dtypes(exclude=[np.number])

non_numeric_cols1 = df1_non_numeric.columns.values

print(non_numeric_cols1)
df1.describe(include='all')
# Check for missing data

sns.heatmap(df1.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# Percentage of missing values

print((df1.isnull().mean()*100).sort_values(ascending=False))
df1['Age'].dropna().hist()
sns.countplot(x='Survived', hue='Sex',data=df1)
sns.countplot(x='Pclass', hue='Survived',data=df1)
sns.heatmap(df1.corr(), annot=True, cmap='coolwarm')
# Pclass and count of Missing Age

df1[df1['Age'].isna()]['Pclass'].value_counts()
# Interactive Box plot

df1[['Pclass', 'Age']].pivot(columns='Pclass', values='Age').iplot(kind='box')
# Mean values of Age w.r.t Pclass

df1[['Pclass', 'Age']].groupby('Pclass', as_index=False).mean().round()
# N/A values of age are replaced by Median Age according to Pclass using below function

def imputeAge1(x):

  Pclass = x[0]

  Age = x[1]

  if pd.isnull(Age):

    if Pclass == 1:

      return 38

    elif Pclass == 2:

      return 30

    else:

      return 25

  else:

    return Age



df1['Age'] = df1[['Pclass','Age']].apply(imputeAge1, axis=1)

df1
sns.heatmap(df1.isnull(), yticklabels=False, cbar=False, cmap='viridis')
training_df = df1
sns.heatmap(training_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
training_df.drop('Cabin', axis=1, inplace=True) # Lot of data in Cabin is missing hence Cabin is not useful, so let's drop
training_df.drop(['Name', 'Ticket'], axis=1, inplace=True)
training_df
Sex1 = pd.get_dummies(training_df['Sex'], drop_first=True)

Sex1
Embarked1 = pd.get_dummies(training_df['Embarked'], drop_first=True)

Embarked1
training_df.drop(['Sex', 'Embarked'], axis=1, inplace=True)
training_df = pd.concat([training_df, Sex1, Embarked1], axis=1)
training_df
df2.info()
# Get numeric columns

df2_numeric = df2.select_dtypes(include=[np.number])

numeric_cols2 = df2_numeric.columns.values

print(numeric_cols2)
# Get Non-numeric columns

df2_non_numeric = df2.select_dtypes(exclude=[np.number])

non_numeric_cols2 = df2_non_numeric.columns.values

print(non_numeric_cols2)
df2.describe(include='all')
# Check for missing data

sns.heatmap(df2.isnull(), yticklabels=False, cbar=False, cmap='viridis')
print((df2.isnull().sum()).sort_values(ascending=False))
df2['Age'].dropna().plot.hist()
sns.countplot(x='Pclass', hue='Sex',data=df2)
sns.heatmap(df2.corr(), annot=True, cmap='coolwarm')
# Pclass and count of Missing Age

df2[df2['Age'].isna()]['Pclass'].value_counts()
# Interactive Box plot

df2[['Pclass', 'Age']].pivot(columns='Pclass', values='Age').iplot(kind='box')
# Mean values of Age w.r.t Pclass

df2[['Pclass', 'Age']].groupby('Pclass', as_index=False).mean().round()
# N/A values of age are replaced by Median Age according to Pclass using below function

def imputeAge2(x):

  Pclass = x[0]

  Age = x[1]

  if pd.isnull(Age):

    if Pclass == 1:

      return 41

    elif Pclass == 2:

      return 29

    else:

      return 24

  else:

    return Age



df2['Age'] = df2[['Pclass','Age']].apply(imputeAge2, axis=1)

df2
sns.heatmap(df2.isnull(), yticklabels=False, cbar=False, cmap='viridis') # Let's check again
testing_df = df2
sns.heatmap(testing_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
testing_df.drop('Cabin', axis=1, inplace=True) #Cabin is not useful, so let's drop
sns.heatmap(testing_df.isnull(), yticklabels=False, cbar=False, cmap='viridis') #Let's check again if any data is missing
testing_df.loc[testing_df['Fare'].isnull(), 'Fare'] = 0  # Missing Fare value is set to 0
sns.heatmap(testing_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
testing_df.drop(['Name', 'Ticket'], axis=1, inplace=True)
testing_df
Sex2 = pd.get_dummies(testing_df['Sex'], drop_first=True)

Sex2
Embarked2 = pd.get_dummies(testing_df['Embarked'], drop_first=True)

Embarked2
testing_df.drop(['Sex', 'Embarked'], axis=1, inplace=True)
testing_df = pd.concat([testing_df, Sex2, Embarked2], axis=1)
testing_df
logmod = LinearRegression()

X_train = training_df.drop('Survived', axis=1)

y_train = training_df['Survived'] 

X_test = testing_df

logmod.fit(X_train, y_train)
predict = logmod.predict(X_test)

predict = predict.round().astype(int)

predict
predicted = pd.DataFrame({'PassengerId':testing_df['PassengerId'], 'Survived': predict})

predicted
# Output data file obtained from Kaggle 'gender_submission.csv'

df3
y_test = df3['Survived']
# Plotting the Actual and Predicted data

fig, ax = plt.subplots(1,2, figsize=(12,8))

sns.countplot(y_test, ax=ax[0])

sns.countplot(predict, ax=ax[1])

ax[0].set(xlabel='Survived', title='Actual')

ax[1].set(xlabel='Survived', title='Predicted')

plt.tight_layout()
# Classification Report

print(metrics.classification_report(y_test,predict))
# Confusion Matrix

print(metrics.confusion_matrix(y_test,predict))
# Accuracy

print(metrics.accuracy_score(y_test,predict)*100)