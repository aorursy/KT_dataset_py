import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
%pylab inline
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
full_data = pd.concat([train_data, test_data]).reset_index(drop=True)
full_data['Pclass'] = full_data['Pclass'].astype('category')
full_data['Embarked'] = full_data['Embarked'].astype('category')
full_data['Sex'] = full_data['Sex'].astype('category')
full_data.tail()
# Statistical description of the data
full_data.describe()
# Count how many NaN values there are in each column
len(full_data) - full_data.count()
# Passengers with missing values for Embarked and Fare.
full_data[full_data.drop(['Age','Cabin','Survived'], axis=1).isnull().any(axis=1)]
sns.barplot(x='Sex', y='Survived', data=full_data)
age_df = full_data[['Age','Survived', 'Sex']].copy()
age_df.loc[age_df.Age<15,'AgeGroup'] = 'Children'
age_df.loc[age_df.Age>=15,'AgeGroup'] = 'Adult'
sns.barplot(x='AgeGroup', y='Survived', hue='Sex', data=age_df)
sns.swarmplot(x='Age',y='Sex',hue='Survived',data=full_data)
p = plt.hist([full_data[(full_data.Survived==1)&(full_data.Fare<30)].Fare, 
              full_data[(full_data.Survived==0)&(full_data.Fare<30)].Fare], histtype='bar', stacked=True, bins=10)
p = plt.hist([full_data[(full_data.Survived==1)&(full_data.Fare>30)].Fare, 
              full_data[(full_data.Survived==0)&(full_data.Fare>30)].Fare], histtype='bar', stacked=True, bins=10)
money_df = full_data[['Fare','Survived', 'Sex','Pclass']].copy()
money_df.loc[money_df.Fare>30,'FareLabel'] = 'Expensive'
money_df.loc[money_df.Fare<30,'FareLabel'] = 'Cheap'
sns.barplot(x='FareLabel', y='Survived', hue='Sex', data=money_df)
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=money_df)
family_df = full_data[['SibSp','Parch','Survived', 'Sex']].copy()
family_df.loc[:,'FamilySize'] =  family_df['SibSp'] + family_df['Parch'] +1
sns.barplot(x='FamilySize', y='Survived', hue='Sex', data=family_df)
family_df.loc[family_df.FamilySize==1,'FamilyLabel'] = 'Single'
family_df.loc[family_df.FamilySize==2,'FamilyLabel'] = 'Couple'
family_df.loc[(family_df.FamilySize>2)&(family_df.FamilySize<=4),'FamilyLabel'] = 'Small'
family_df.loc[family_df.FamilySize>4,'FamilyLabel'] = 'Big'
sns.barplot(x='FamilyLabel', y='Survived', hue='Sex', data=family_df, order=['Single', 'Couple', 'Small', 'Big'])