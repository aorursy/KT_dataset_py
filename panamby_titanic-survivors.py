# Import the main libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

% matplotlib inline
# Let's read the 'train' file which countais all the informations that we use to predict the result.

train = pd.read_csv('../input/train.csv')
train.head()
# I will adjuste the chart only with the essencial informations to our model.
# We don't need the following columns: 'PassengerId', 'Name', 'Ticket', 'Cabin'

new_train = train.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
new_train.head()
# I will see if there is some 'Nan' informations.

plt.figure(figsize=(12,8))
sns.heatmap(data=pd.isnull(new_train))

# I relized that in 'Age' column there is a lot of missing informations. 
# For this reason is not a good idea delete these inforations.
# On the other hand, in 'Embarked' column there are only two spaces without information.
# So it's a good idea delete these info.
# First, let's complete the missing informations in "Age" column.
# Let's build some graphs to predict how the "Age" can be fill.

sns.barplot(x='Pclass', y='Age', data=new_train)
sns.barplot(x='Sex', y='Age', data=new_train)
sns.barplot(x='SibSp', y='Age', data=new_train)
sns.barplot(x='Parch', y='Age', data=new_train)
sns.barplot(x='Embarked', y='Age', data=new_train)
# Looking these charts I can realize that "SibSp" could give me a good informations to fill the missing info in "Age" columns, because I have in this graph a lot of variables.
# Let's see the age means.

plt.figure(figsize=(12,8))
sns.boxplot(x='SibSp', y='Age', data=new_train)

# Now I'm realizing that the mean in many parameters are almost the same.
# I will try the second better graph: "Parch"

plt.figure(figsize=(12,8))
sns.boxplot(x='Parch', y='Age', data=new_train)

# I can see here that in four of them has good mean, but others one are very bad.
# Let's try now the third better: "Pclass"

plt.figure(figsize=(12,8))
sns.boxplot(x='Pclass', y='Age', data=new_train)

# Now I can see pretty avarages to put in "Age" columns.
# I will code a function to complete the missing informations in "Age" column with the age mean in this last chart.

def fill_age(col):
    age = col[0]
    pclass = col[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age

# Now put this function inside of the data.

new_train['Age'] = new_train[['Age', 'Pclass']].apply(fill_age, axis=1)
# Let's check if it worked!

plt.figure(figsize=(12,8))
sns.heatmap(data=pd.isnull(new_train))

# Great!
# Now I will delete the two missing informations in "Embarked" column.

new_train.dropna(inplace=True)
# Let's check again!

plt.figure(figsize=(12,8))
sns.heatmap(data=pd.isnull(new_train), cmap='inferno')

# Good job!
# Now I will change categorical values to number values.
# It will happen to help fitting correctly.

# They will be the "Sex" and "Embarked" columns.
sex = pd.get_dummies(new_train['Sex'], drop_first=True)
embarked = pd.get_dummies(new_train['Embarked'], drop_first=True)
# Now I will delete the old columns and add the new columns.

new_train.drop(['Sex', 'Embarked'], axis=1, inplace=True)

new_train = pd.concat([new_train, sex, embarked], axis=1)
# Let's check!

new_train.head()
# Now it's time to see how all the informations correlate between themselves.

sns.pairplot(data=new_train, hue='Survived', kind='scatter', palette='inferno')
# The first thing that we can realize with this pairplot is that it's a Classification Algorithm.
# The first Classification Algorithm that I will use will be the Logistic

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
# Now I will set "X" and "y" within the "train" file.

X_train = new_train.iloc[:,1:].values
y_train = new_train.iloc[:,0].values
# Let's check!

X_train
y_train
# Fitting...

classifier.fit(X_train, y_train)
# Now that I already train the data, I will test him.
# First I will read the "test" file and delete the same columns than before.

test = pd.read_csv('../input/test.csv')
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Let's see about the missing info.

plt.figure(figsize=(12,8))
sns.heatmap(test.isnull())
# Again we have to fill the missing info in "Age" column with the function.

test['Age'] = test[['Age', 'Pclass']].apply(fill_age, axis=1)

# Let's Check!

plt.figure(figsize=(12,8))
sns.heatmap(test.isnull())

# Great!
# The numbers of "test" file row and "gender_submission" file are the same. 
# For this reason, We can't delete any row.

# Let's fill the missing info in the "Fare" column.
# First I will see the informations that can help me to predict a fare value.

test[test.Fare.isnull()==True]
# Let me see if I can found other passenger who had the same main features and almost the same age.

test[(test.Pclass==3) & (test.Sex=='male') & (test.Age>45) & (test.Embarked=='S')]

# I got it!
# Now I will fill the "Fare" info with the same value from other passenger.

test.Fare.fillna(value=14.5, inplace=True)

# Check!

test.iloc[152]

# Done!
# Now go ahead and tap dummie code.

sex_test = pd.get_dummies(test['Sex'], drop_first=True)
embarked_test = pd.get_dummies(test['Embarked'], drop_first=True)

test.drop(['Sex', 'Embarked'], axis=1, inplace=True)
new_test = pd.concat([test, sex_test, embarked_test], axis=1)

X_test = new_test.iloc[:, 0:8].values

# Let's check!

X_test

# It's time to predict.

y_pred = classifier.predict(X_test)
y_pred
# Let's compare the results!

compare = pd.read_csv('../input/gender_submission.csv')
y_test = compare.iloc[:,1].values
y_test
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print('\n')
print(confusion_matrix(y_test, y_pred))