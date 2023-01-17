import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')

print('Training Data')

print(df_train.info(),'\n')

print('Test Data')

print(df_test.info())
# Passenger ID

print('PassengerId is a unique ID given to every passenger on board\n')

# Passenger Class

print('Pclass is the class of the purchased ticket')

print(sorted(df_train['Pclass'].unique()),'\n')

# Name

print('Name is the name of the passenger\n')

# Sex

print('Sex is the gender of the passenger\n')

# Age

print('Age is the age of the passenger\n')

# Siblings and Spouses

print('SibSp is the number of siblings and spouses of the passenger on board')

print(sorted(df_train['SibSp'].unique()),'\n')

# Parents and Children

print('Parch is the number of parents and children of the passenger on board')

print(sorted(df_train['Parch'].unique()),'\n')

# Ticket Number

print('Ticket is the ticket number of the passenger\n')

# Fare

print('Fare is the price of the ticket of the passenger\n')

# Cabin

print('Cabin is the cabin number of the passenger\n')

# Embarked

print('Embarked is the port of embarkation of the passenger')

print(sorted(df_train['Embarked'].dropna().unique()))
df_miss_train = pd.DataFrame(((len(df_train)-df_train.count())*100/len(df_train)).round(1), columns = ['Missing Training Data%'])

df_miss_train = df_miss_train.drop('Survived')

df_miss_test = pd.DataFrame(((len(df_test)-df_test.count())*100/len(df_test)).round(1), columns = ['Missing Test Data%'])

df_miss_train_test = pd.concat([df_miss_train, df_miss_test], axis = 1)

df_miss_train_test
# Drop Cabin from both train and test data

df_train = df_train.drop('Cabin', axis=1)

df_test = df_test.drop('Cabin', axis=1)
# Plot the correlation of Age with other features

df_train.corr()['Age'].sort_values()[:-1].plot(kind='bar')

plt.show()
df_mean_age_pclass = df_train.groupby('Pclass').mean()['Age']



# Create function to calculate mean Age per Pclass

def mean_age_pclass(features):

    age = features[0]

    pclass = features[1]

    if pd.isnull(age):

        return df_mean_age_pclass.loc[pclass]

    else:

        return age



# Apply mean age to missing rows in training data

df_train['Age'] = df_train[['Age','Pclass']].apply(mean_age_pclass, axis=1)
# Apply mean age(from training set) to missing rows in test data

df_test['Age'] = df_test[['Age','Pclass']].apply(mean_age_pclass, axis=1)
# Find Pclass of passenger with missing fare info

mrow_pclass = df_test[df_test['Fare'].isnull()]['Pclass'].iloc[0]

# Find mean fare for mrow_pclass

mrow_fare = df_test.groupby('Pclass').mean()['Fare'].loc[mrow_pclass]

# Fill missing row with mrow_fare

df_test['Fare'].fillna(value=mrow_fare, inplace=True)
plt.figure(figsize=(10,6))

sns.countplot(df_train['Survived'],palette='pastel')

plt.title('Total count of passengers who survived')

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(df_train['Survived'],palette='pastel',hue=df_train['Sex'])

plt.title('Total count of men and women who survived')

plt.show()
plt.figure(figsize=(6,6))

colors = ['royalblue','orange','darkgreen']

plt.pie(df_train[df_train['Survived'] == 1]['Pclass'].value_counts()/df_train['Pclass'].value_counts(),

        autopct = '%.1f%%',

        colors = colors,

        shadow = True

       )

# labels

pclass_labels = (df_train[df_train['Survived'] == 1]['Pclass'].value_counts()/df_train['Pclass'].value_counts()).index

pclass_label_dict = {"1":"1st class",

                     "2":"2nd class",

                     "3":"3rd class"

                    }

pclass_labels = [pclass_label_dict[str(value)] for value in pclass_labels]

plt.legend(pclass_labels, bbox_to_anchor=(-0.1, 1.),

           fontsize=10)

plt.title('Percentage of passengers per class who survived')

plt.tight_layout()

plt.show()
# Combining two features into one

df_train['Relatives'] = df_train['SibSp'] + df_train['Parch']

df_test['Relatives'] = df_test['SibSp'] + df_test['Parch']



# Count plot of total relatives on board of passenges who survived

plt.figure(figsize=(14,6))

sns.countplot(df_train['Relatives'], palette='viridis', alpha=0.75, hue = df_train['Survived'])

# sns.countplot(df_train[df_train['Survived']==1]['Relatives'], palette='viridis')

plt.title('Total Relatives on board of passengers who survived')

plt.show()
# Drop SibSp and Parch column from Training Data

df_train = df_train.drop(['SibSp','Parch'],axis=1)



# Drop SibSp and Parch column from Test Data

df_test = df_test.drop(['SibSp','Parch'],axis=1)
plt.figure(figsize=(12,6))

sns.boxplot(x='Survived', y='Fare', data=df_train, linewidth=1.5, palette='Set3')

plt.title('Box plot of fares of passengers who survived')

plt.show()
plt.figure(figsize=(14,6))

sns.countplot(df_train['Embarked'], palette='GnBu', hue=df_train['Survived'])

plt.title('Count of port of embarkation of passengers who survived')

plt.show()
# Drop Embarked columns from Training Data

df_train = df_train.drop('Embarked', axis=1)



# Drop Embarked columns from Test Data

df_test = df_test.drop('Embarked', axis=1)
# Drop from training data

df_train = df_train.drop(['Name','PassengerId','Ticket'],axis=1)



# Drop from test data

df_test_PassId = df_test['PassengerId'] # Needed for submission

df_test = df_test.drop(['Name','PassengerId','Ticket'],axis=1)
print('Training Data')

print(df_train.columns)

print('Number of Observations: ',len(df_train),'\n')

print('Test Data')

print(df_test.columns)

print('Number of Observations: ',len(df_test))
X_train = df_train.drop('Survived',axis=1)

y_train = df_train['Survived']

X_test = df_test
# Convert Sex to dummy variable

X_train = pd.concat([X_train, pd.get_dummies(X_train['Sex'], drop_first=True)], axis=1)

X_train = X_train.drop('Sex',axis=1)



X_test = pd.concat([X_test, pd.get_dummies(X_test['Sex'], drop_first=True)], axis=1)

X_test = X_test.drop('Sex',axis=1)



# Convert Embarked to dummy variable

# X_train = pd.concat([X_train, pd.get_dummies(X_train['Embarked'], drop_first=True)], axis=1)

# X_train = X_train.drop('Embarked',axis=1)



# X_test = pd.concat([X_test, pd.get_dummies(X_test['Embarked'], drop_first=True)], axis=1)

# X_test = X_test.drop('Embarked',axis=1)
# Apply StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Import Grid Search and Random Forest Classifiers

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

# Grid Search

parameters = [{'n_estimators': [10,50,100,150,200,250,300] , 'criterion': ['entropy', 'gini'], 'max_features': ['sqrt','log2'] ,'n_jobs': [-1]}]

classifier = RandomForestClassifier()

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy')

# Fit on training set

grid_search.fit(X_train, y_train)
grid_search.best_params_
rf_classifier = grid_search.best_estimator_

# Fit on training data

rf_classifier.fit(X_train,y_train)
# Import k-fold cross validation

from sklearn.model_selection import cross_validate

score = cross_validate(estimator=rf_classifier,

                       X=X_train,

                       y=y_train,

                       cv=10,

                       scoring = ['accuracy','f1'],

                       n_jobs=-1

                      )
print('Mean Accuracy: ',(score['test_accuracy'].mean()*100).round(2),'%\n')

print('Standard Deviation Accuracy: ',(score['test_accuracy'].std()*100).round(1),'%')
predictions = rf_classifier.predict(X_test)
output = pd.DataFrame({'PassengerId': df_test_PassId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print('Your submission was successfully saved!')