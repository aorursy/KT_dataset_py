from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from sklearn.linear_model import LinearRegression

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()
sex_gb = train.groupby('Sex').agg({'Survived':['count','sum']}).reset_index()

sex_gb.columns = ['sex', 'total', 'survived']

sex_gb['survival_rate'] = sex_gb.survived*100 / sex_gb.total

plt.figure(figsize=(15,8))

ax1 = plt.subplot(1,2,1)

# Survival Rate by gender

bars1 = plt.bar(sex_gb.sex, sex_gb.survival_rate)

plt.title('Survival Rate by Gender', fontweight='bold')

plt.ylabel('Survival Rate', fontweight='bold')

plt.xlabel('Gender', fontweight='bold')

plt.xticks(rotation=0)

plt.ylim(0,80)

plt.legend(loc='best')



for p in bars1:

    height = p.get_height()

    ax1.annotate(f'{height:.1f}',

                xy=((p.get_x() + p.get_width() / 2.), height),

                xytext=(0,3),  # 3 points vertical offset

                textcoords="offset points",

                ha='center', va='bottom')

# Survival Rate by class

train['class_string'] = 'Third'

train.loc[train.Pclass == 1, 'class_string'] = 'First'

train.loc[train.Pclass == 2, 'class_string'] = 'Second'

class_gb = train.groupby('class_string').agg({'Survived':['count','sum']}).reset_index()

class_gb.columns = ['class_string', 'total', 'survived']

class_gb['survival_rate'] = class_gb.survived*100 / class_gb.total

ax2 = plt.subplot(1,2,2)

bars2 = plt.bar(class_gb.class_string, class_gb.survival_rate)

plt.title('Survival Rate by Class', fontweight='bold')

plt.ylabel('Survival Rate', fontweight='bold')

plt.xlabel('Class', fontweight='bold')

plt.xticks(rotation=0)

plt.ylim(0,80)

plt.legend(loc='best')



for p in bars2:

    height = p.get_height()

    ax2.annotate(f'{height:.1f}',

                xy=((p.get_x() + p.get_width() / 2.), height),

                xytext=(0,3),  # 3 points vertical offset

                textcoords="offset points",

                ha='center', va='bottom')



    

plt.show()
bins = [x*5 for x in range(int(80/5))]



plt.figure(figsize=(20,12))

# All passengers

plt.subplot(2,3,1)

survived_ages = train.loc[train.Survived==1, 'Age']

perished_ages = train.loc[train.Survived==0, 'Age']

plt.hist(survived_ages, bins, label='Survived', density=True, alpha=0.5)

plt.hist(perished_ages, bins, label='Perished', density=True, alpha=0.5)

plt.legend(loc='best')

plt.title('Historgram of Ages by Survival Flag', fontweight='bold')

plt.xlabel('Age', fontweight='bold')

plt.ylabel('density')

# Men

plt.subplot(2,3,2)

survived_ages = train.loc[(train.Survived==1) & (train.Sex=='male'), 'Age']

perished_ages = train.loc[(train.Survived==0) & (train.Sex=='male'), 'Age']

plt.hist(survived_ages, bins, label='Survived', density=True, alpha=0.5)

plt.hist(perished_ages, bins, label='Perished', density=True, alpha=0.5)

plt.legend(loc='best')

plt.title('Historgram of Ages by Survival Flag\n(male passengers only)', fontweight='bold')

plt.xlabel('Age', fontweight='bold')

plt.ylabel('density')

#Women

plt.subplot(2,3,3)

survived_ages = train.loc[(train.Survived==1) & (train.Sex=='female'), 'Age']

perished_ages = train.loc[(train.Survived==0) & (train.Sex=='female'), 'Age']

plt.hist(survived_ages, bins, label='Survived', density=True, alpha=0.5)

plt.hist(perished_ages, bins, label='Perished', density=True, alpha=0.5)

plt.legend(loc='best')

plt.title('Historgram of Ages by Survival Flag\n(female passengers only)', fontweight='bold')

plt.xlabel('Age', fontweight='bold')

plt.ylabel('density')

# First Class

plt.subplot(2,3,4)

survived_ages = train.loc[(train.Survived==1) & (train.Pclass==1), 'Age']

perished_ages = train.loc[(train.Survived==0) & (train.Pclass==1), 'Age']

plt.hist(survived_ages, bins, label='Survived', density=True, alpha=0.5)

plt.hist(perished_ages, bins, label='Perished', density=True, alpha=0.5)

plt.legend(loc='best')

plt.title('Historgram of Ages by Survival Flag\n(first class passengers only)', fontweight='bold')

plt.xlabel('Age', fontweight='bold')

plt.ylabel('density')

# Second Class

plt.subplot(2,3,5)

survived_ages = train.loc[(train.Survived==1) & (train.Pclass==2), 'Age']

perished_ages = train.loc[(train.Survived==0) & (train.Pclass==2), 'Age']

plt.hist(survived_ages, bins, label='Survived', density=True, alpha=0.5)

plt.hist(perished_ages, bins, label='Perished', density=True, alpha=0.5)

plt.legend(loc='best')

plt.title('Historgram of Ages by Survival Flag\n(second class passengers only)', fontweight='bold')

plt.xlabel('Age', fontweight='bold')

plt.ylabel('density')

#Thrid Class

plt.subplot(2,3,6)

survived_ages = train.loc[(train.Survived==1) & (train.Pclass==3), 'Age']

perished_ages = train.loc[(train.Survived==0) & (train.Pclass==3), 'Age']

plt.hist(survived_ages, bins, label='Survived', density=True, alpha=0.5)

plt.hist(perished_ages, bins, label='Perished', density=True, alpha=0.5)

plt.legend(loc='best')

plt.title('Historgram of Ages by Survival Flag\n(third class passengers only)', fontweight='bold')

plt.xlabel('Age', fontweight='bold')

plt.ylabel('density')



plt.show()
clean_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']].copy()

# One-hot encode gender

for g in ['male', 'female']:

    clean_train[g] = 0

    clean_train.loc[clean_train.Sex==g, g]=1

clean_train = clean_train.drop(columns='Sex')

# One-hot encode class

for c,s in zip(range(1,4), ['first_class', 'second_class', 'third_class']):

    clean_train[s] = 0

    clean_train.loc[clean_train.Pclass==c, s]=1

clean_train = clean_train.drop(columns='Pclass')   





# Impute null age values

temp_train = clean_train.loc[~clean_train.Age.isna()].copy()

temp_test = clean_train.loc[clean_train.Age.isna()].copy().drop(columns='Age')



X_train = temp_train.drop(columns='Age')

y_train = temp_train.Age

X_test = temp_test.copy()



lr1 = LinearRegression()

lr1.fit(X_train, y_train)

temp_test['Age'] = lr1.predict(X_test)



no_nulls_train = pd.concat([temp_train, temp_test])

no_nulls_train.sample(10)
# # Create the parameter grid based on the results of random search 

# param_grid = {

#     'max_depth': [10, 20, 30],

#     #'max_features': [2, 3],

#     'min_samples_leaf': [5, 10, 15],

#     #'min_samples_split': [5, 8, 12],

#     'n_estimators': [300,400]

# }

# # Create a based model

# rf = RandomForestClassifier()

# X_train = no_nulls_train.drop(columns='Survived')

# y_train = no_nulls_train.Survived

# # Get model scoring



# clf = GridSearchCV(

#     rf, param_grid, cv=5, scoring='roc_auc', verbose=1

# )

# clf.fit(X_train, y_train)



# print("Best parameters set found on development set:")

# print()

# print(clf.best_params_)

# print("Best Score:")

# print()

# print(clf.best_score_)
clean_test = test[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].copy()

# One-hot encode gender

for g in ['male', 'female']:

    clean_test[g] = 0

    clean_test.loc[clean_test.Sex==g, g]=1

clean_test = clean_test.drop(columns='Sex')

# One-hot encode class

for c,s in zip(range(1,4), ['first_class', 'second_class', 'third_class']):

    clean_test[s] = 0

    clean_test.loc[clean_test.Pclass==c, s]=1

clean_test = clean_test.drop(columns='Pclass')   





X_test = clean_test.drop(columns=['PassengerId', 'Age']).copy()

X_train = temp_train.drop(columns=['Age', 'Survived'])

y_train = temp_train.Age

lr2 = LinearRegression()

lr2.fit(X_train, y_train)

no_nulls_test = clean_test.copy()

no_nulls_test['Pred_Age'] = lr2.predict(X_test)



no_nulls_test['Age'] = clean_test.Age.combine_first(no_nulls_test['Pred_Age']).drop(columns='Pred_Age')

no_nulls_test = no_nulls_test.drop(columns='Pred_Age')

no_nulls_test
rf = RandomForestClassifier(max_depth=20,min_samples_leaf=5, n_estimators=400)

rf.fit(no_nulls_train.drop(columns='Survived'), no_nulls_train.Survived)

no_nulls_test['Survived']=rf.predict(no_nulls_test.drop(columns='PassengerId'))

no_nulls_test.sample(10)
output = no_nulls_test[['PassengerId', 'Survived']]

output.to_csv('output.csv', index=False)

print("Your submission was successfully saved!")