# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# EDA on train data set
raw_data = pd.read_csv('/kaggle/input/titanic/train.csv')
df = raw_data.copy()
df.head(5)
# Check types of each columns and non-Null values
df.info()
# Remove column Cabin and missing value
df = df.drop(['Cabin'], axis=1)
df_no_na = df.dropna(axis=0, how='any')
df_no_na.describe()
# Reorder columns to get our label at the end
df_no_na.columns.values
cols = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked', 'Survived']
df_no_na = df_no_na[cols] 
df_no_na.head()
# Remove unwanted features and check Parch values
df_rm_feat = df_no_na.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
pd.unique(df_rm_feat['Parch'])
# Count the number of Parch values
np.unique(df_rm_feat['Parch'], return_counts=True)
# Map categories
np.unique(df_rm_feat['Parch'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}),
          return_counts=True)
df_rm_feat['Parch'] = df_rm_feat['Parch'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
df_rm_feat.head()
# Embarked categorical value
np.unique(df_rm_feat['Embarked'], return_counts=True)
df_rm_feat['Embarked'] = df_rm_feat['Embarked'].map({'S': 0, 'C': 1, 'Q': 1})
df_rm_feat.head()
# SibSp categorical value
np.unique(df_rm_feat['SibSp'], return_counts=True)
df_rm_feat['SibSp'] = df_rm_feat['SibSp'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
df_rm_feat.head()
# Binarize gender
df_rm_feat['Sex'] = df_rm_feat['Sex'].map({'male': 0, 'female': 1})
df_rm_feat.head()
# Pclass categories
np.unique(df_rm_feat['Pclass'], return_counts=True)
dummies = pd.get_dummies(df_rm_feat['Pclass'], drop_first=True)
dummies
# Drop Pclass and replace by dummies
df_rm_feat[['Pclass_2', 'Pclass_3']] = dummies
df_rm_feat = df_rm_feat.drop(['Pclass'], axis=1)
df_rm_feat.columns.values
df_rm_feat = df_rm_feat[['Pclass_2', 'Pclass_3', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                               'Embarked', 'Survived']]
df_rm_feat.head()
# Checkpoint categories cols
df_categories = df_rm_feat.copy()
df_categories.describe(include='all')
# Import matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Histogram distributions Age
plt.hist(df_categories["Age"])
plt.xlabel('Age')
# Histo Fare
plt.hist(df_categories["Fare"])
plt.xlabel('Fare')
# Boxplot Fare
plt.boxplot(df_categories["Fare"])
plt.xlabel('Fare')
# Remove outliers and keep 95% of data
df_no_outliers = df_categories[df_categories["Fare"] <= df_categories["Fare"]
                               .quantile(0.95)].reset_index(drop=True)
df_no_outliers
# Boxplot Fare
plt.boxplot(df_no_outliers["Fare"])
plt.xlabel('Fare')
# Histo Fare
plt.hist(df_no_outliers["Fare"])
plt.xlabel('Fare')
# Last check of preprocess
df_no_outliers.describe()
# We need to make a function to preprocess the test dataset
def preprocess(df):
    # Remove columns and NA
    df = df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1)
    df = df.fillna(0) # NA values in test but we can not drop them
    
    # Map values
    df['Parch'] = df['Parch'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 9: 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 1})
    df['SibSp'] = df['SibSp'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 8: 1})
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Dummies Pclas
    dummies = pd.get_dummies(df['Pclass'], drop_first=True)
    df[['Pclass_2', 'Pclass_3']] = dummies
    df = df.drop(['Pclass'], axis=1)
    
    # Reorder cols
    df = df[['Pclass_2', 'Pclass_3', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                               'Embarked']]
    
    
    return df
# Import libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
# Init data
train_data = df_no_outliers.copy()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.info()
np.unique(test_data['Parch'], return_counts=True)
# Preprocess test data
test_data_processed = preprocess(test_data)
test_data_processed.head()
#Train data
scaler = StandardScaler()
unscaled_data_train = train_data[['Age','Fare']]
unscaled_data_test = test_data_processed[['Age','Fare']]
df_train = train_data.drop(['Age','Fare'], axis=1)
df_test = test_data_processed.drop(['Age','Fare'], axis=1)
# Fit scaler on train data
scaler.fit(unscaled_data_train)
scaled_train = pd.DataFrame(data = scaler.transform(unscaled_data_train), columns=['Age', 'Fare'])
scaled_test = pd.DataFrame(data = scaler.transform(unscaled_data_test), columns=['Age', 'Fare'])
scaled_test
df_train_scaled = pd.concat([df_train,scaled_train], axis=1) 
df_test_scaled = pd.concat([df_test,scaled_test], axis=1) 
df_test_scaled.info()
# Train the model
logR = LogisticRegression()
x_train = df_train_scaled.drop(['Survived'], axis=1)
y_train = df_train_scaled['Survived']
logR.fit(x_train, y_train)
result = logR.predict(df_test_scaled)
test_data['Survived'] = result.tolist()
response = test_data[['PassengerId', 'Survived']]
response
# Save result
response.to_csv('submission.csv', index=False)