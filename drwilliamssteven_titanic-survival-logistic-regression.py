import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
combined = pd.concat([ train, test ])
combined.describe()
sns.heatmap(combined.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived', data=combined, palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived', hue='Sex', data=combined, palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived', hue='Pclass', data=combined, palette='rainbow')
sns.distplot(combined['Age'].dropna(), kde=False, color='darkred', bins=30)
sns.countplot(x='SibSp',data=combined)
combined['Fare'].hist(color='green',bins=40,figsize=(8,4))
sns.countplot(x='Embarked',data=combined)
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=combined,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]   

          

    if pd.isnull(Age): 

        

        if Pclass == 1:            

            return np.mean(combined[combined['Pclass'] == 1 ]['Age'])



        elif Pclass == 2:

            return np.mean(combined[combined['Pclass'] == 2 ]['Age'])



        else:

            return np.mean(combined[combined['Pclass'] == 3 ]['Age'])



    else:

        return Age
combined['Age'] = combined[['Age','Pclass']].apply(impute_age, axis=1)
combined.drop('Cabin', axis=1, inplace=True)
combined.fillna(value={'Embarked': 'S', 'Fare': np.mean(combined['Fare'])}, inplace=True)
combined.info()
sex = pd.get_dummies(combined['Sex'], drop_first=True)

embark = pd.get_dummies(combined['Embarked'], drop_first=True)
combined.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)
combined = pd.concat([combined,sex,embark], axis=1)
combined.head()
sns.heatmap(combined.isnull(),yticklabels=False,cbar=False,cmap='viridis')
from sklearn.model_selection import train_test_split
train = combined[combined['Survived'].notnull()]

test = combined[combined['Survived'].isnull()]

test = test.drop('Survived', axis=1)
X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'],axis=1), 

                                                    train['Survived'], test_size=0.30)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test).astype(int)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
# Check for any remaining missing values

print("Remaining NaN?", np.any(np.isnan(test)) )

#np.all(np.isfinite(test))
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = logmodel.predict(test.drop('PassengerId', axis=1)).astype(int)



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)