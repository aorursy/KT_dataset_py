import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load and peek training data
train_data = pd.read_csv("/kaggle/input/titanicdata/train.csv")
train_data.head()
# Get some information about training data
train_data.info()
test_data = pd.read_csv("/kaggle/input/titanicdata/test.csv")
test_data.head()
test_data.info()
# Concatenate the two data sets to create a big dataset. This will be useful to process data
dataset =  pd.concat(objs=[train_data, test_data], axis=0).reset_index(drop=True)
dataset.info()
# Looking for missing data on train dataset
plt.pcolor(train_data.isnull())
plt.xticks(np.arange(0.0, len(train_data.columns), 1), train_data.columns, rotation='vertical')
plt.show()
# Looking for missing data on train dataset
plt.pcolor(dataset.isnull())
plt.xticks(np.arange(0.0, len(dataset.columns), 1), dataset.columns, rotation='vertical')
plt.show()
# Correlation between survived and other data
numerical_features = ["Survived","SibSp","Parch","Age","Fare"]
plt.pcolor(dataset[numerical_features].corr(), cmap='plasma')
plt.xticks(np.arange(0, 5), numerical_features, rotation='vertical')
plt.yticks(np.arange(0, 5), numerical_features)
plt.colorbar()
plt.show()
# Split the data of people who survived and not survived to further analysis
survived_data = dataset[dataset['Survived'] == 1.0]
not_survived_data = dataset[dataset['Survived'] == 0.0]
# Plot histogram with the age distributions of people who survived or not
fig, axes = plt.subplots(1,2, figsize=(15,5))
axes[0].hist(not_survived_data['Age'], bins=10)
axes[0].set(title='Age of people who NOT survived');
axes[0].set_ylim(0,120);

axes[1].hist(survived_data['Age'], bins=10)
axes[1].set(title='Age of people who survived');
axes[1].set_ylim(0,120);

plt.subplots_adjust(wspace=0.2, hspace=0)
# Explore the Age vs Survived features in a grid view
fig = sns.FacetGrid(train_data, hue = 'Survived', aspect = 4)
fig.map(sns.kdeplot, 'Age' , shade = True)
fig.set(xlim = (0, dataset['Age'].max()))
fig.add_legend()
# Explore the Age vs Passengers class features
facet = sns.FacetGrid(dataset, hue="Pclass", aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, dataset['Age'].max()))
facet.add_legend()
plt.show()
# Explore the Age vs Passengers class features using boxplot 
PA = sns.catplot(data = dataset , x = 'Pclass' , y = 'Age', kind = 'box')
# Filling missing data on age with average of PClass
# using a custom function for age imputation
def AgeImpute(df):
    Age = df[0]
    Pclass = df[1]
    
    if pd.isnull(Age):
        if Pclass == 1: return 37
        elif Pclass == 2: return 29
        else: return 24
    else:
        return Age

# Age Impute
dataset['Age'] = dataset[['Age' , 'Pclass']].apply(AgeImpute, axis = 1)
# Looking for missing data on train dataset
plt.pcolor(dataset.isnull())
plt.xticks(np.arange(0.0, len(dataset.columns), 1), dataset.columns, rotation='vertical')
plt.show()
dataset["Fare"].isnull().sum()
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
# Reviewing missing data on full dataset
plt.pcolor(dataset.isnull())
plt.xticks(np.arange(0.0, len(dataset.columns), 1), dataset.columns, rotation='vertical')
plt.show()
dataset.head()
# 'Embarked' vs 'Survived'
sns.barplot(dataset['Embarked'], dataset['Survived']);

dataset["Embarked"].value_counts()
dataset["Embarked"].isnull().sum()
# Count missing values
print(dataset["Embarked"].isnull().sum()) # 2

# Fill Embarked nan values of dataset set with 'S' most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna("S")
# Use one hot encoding to transform "Embarked" into values
embarked = pd.get_dummies(dataset['Embarked'], drop_first = True)
dataset = pd.concat([dataset,embarked], axis = 1)

# We can drop the non-numerical feature now
dataset.drop(['Embarked'] , axis = 1 , inplace = True)
# Convert Sex column into categorical value 0 for male and 1 for female
sex = pd.get_dummies(dataset['Sex'], drop_first = True)
dataset = pd.concat([dataset,sex], axis = 1)

# After now, we really don't need to Sex features, we can drop it.
dataset.drop(['Sex'] , axis = 1 , inplace = True)
dataset.head()
# using Countplot to estimate amount of people who survived related to the sex
sns.countplot(data = train_data , x = 'Survived' , hue = 'Sex')

# Let's see the percentage
train_data[["Sex","Survived"]].groupby('Sex').mean()

# Reviewing missing data on full dataset
plt.pcolor(dataset.isnull())
plt.xticks(np.arange(0.0, len(dataset.columns), 1), dataset.columns, rotation='vertical')
plt.show()
dataset.head()
# Split train and test data again
train_data = dataset[0:891][:]
train_data['Survived'] = train_data['Survived'].astype(int)
test_data.tail()
test_data = dataset[891:][:]
test_data = test_data.drop('Survived', axis=1)
from sklearn.ensemble import RandomForestClassifier
y = train_data["Survived"]

features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "male", "Q", "S"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.head()
output.to_csv('titanic_submission.csv', index=False)
print("Your submission was successfully saved!")