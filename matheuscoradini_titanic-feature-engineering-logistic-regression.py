import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedKFold

import numpy as np

import re
df = pd.DataFrame(pd.read_csv('../input/titanic/train.csv'))

df_test = pd.DataFrame(pd.read_csv('../input/titanic/test.csv'))
df.head()
sns.lmplot(data = df, x = 'Age', y = 'Fare', hue = 'Survived', fit_reg = False, size = 4, aspect = 3)

plt.show()
def countplot(var):

    sns.countplot(data = df, x = var, hue = 'Survived', palette = ('R', 'b'))

    plt.show()

countplot('Sex')
countplot('Pclass')
countplot('Embarked')
df.isnull().sum()
print(df.groupby(['Pclass']).mean()['Age'])

print('\n')

print(df.groupby(['Sex']).mean()['Age'])
# fill NaN Age values with the means, separating by Sex and Pclass



def age_nan(df):

    for i in df.Sex.unique():

        for j in df.Pclass.unique():

            x = df.loc[((df.Sex == i) & (df.Pclass == j)), 'Age'].mean()

            df.loc[((df.Sex == i) & (df.Pclass == j)), 'Age'] = df.loc[((df.Sex == i) & (df.Pclass == j)), 'Age'].fillna(x)



age_nan(df)

age_nan(df_test)
# fill NaN values of Embarked with 'S', because it's the most commom value for it



df['Embarked'] = df['Embarked'].fillna('S')

df_test['Embarked'] = df_test['Embarked'].fillna('S')
#creating Cabin_NaN to test and train dataset and analysing it



df['Cabin_NaN'] = df['Cabin'].isnull().astype(int)

df_test['Cabin_NaN'] = df_test['Cabin'].isnull().astype(int)

countplot('Cabin_NaN')
df_test.isnull().sum()
df_test.Fare = df_test.Fare.fillna(-1)
# logistic regression - cross validation function



def reg_cross_val(variables):

    

    X = df[variables]

    y = df['Survived']

    

    rkfold = RepeatedKFold(n_splits = 2, n_repeats = 10, random_state = 10)

    result = []

    for treino, teste in rkfold.split(X):

        X_train, X_test = X.iloc[treino], X.iloc[teste]

        y_train, y_test = y.iloc[treino], y.iloc[teste]

        

        reg = LogisticRegression(max_iter = 500)

        reg.fit(X_train, y_train)

        result.append(reg.score(X_test, y_test))

        

    return np.mean(result)
#creating feature: Sex_bin



def is_female(x):

    if x == 'female':

        return 1

    else:

        return 0



df['Sex_bin'] = df['Sex'].map(is_female)

df_test['Sex_bin'] = df_test['Sex'].map(is_female)
# creating features: Embarked_S and Embarked_C



def embarked_s(x):

    if x == 'S':

        return 1

    else:

        return 0



df['Embarked_S'] = df['Embarked'].map(embarked_s)

df_test['Embarked_S'] = df_test['Embarked'].map(embarked_s)



def embarked_c(x):

    if x == 'C':

        return 1

    else:

        return 0

    

df['Embarked_C'] = df['Embarked'].map(embarked_c)

df_test['Embarked_C'] = df_test['Embarked'].map(embarked_c)
variables_before = ['Age', 'Pclass', 'Fare', 'SibSp', 'Parch']

print('Before the new features:', reg_cross_val(variables_before))



variables = ['Age', 'Sex_bin', 'Pclass', 'Fare', 'SibSp', 'Parch', 'Embarked_S',\

             'Embarked_C', 'Cabin_NaN']



print('With the new features:', reg_cross_val(variables))
fig, ax =plt.subplots(1,2)

sns.countplot(data = df, x = 'SibSp', hue = 'Survived', palette = ('R', 'b'), ax = ax[0])

sns.countplot(data = df, x = 'Parch', hue = 'Survived', palette = ('R', 'b'), ax = ax[1])

plt.show()
# creating 'Family':



df['Family'] = df.SibSp + df.Parch

df_test['Family'] = df_test.SibSp + df_test.Parch
variables = ['Age', 'Sex_bin', 'Pclass', 'Fare', 'Embarked_S',\

             'Embarked_C', 'Cabin_NaN', 'Family']



reg_cross_val(variables)
text_ticket = ''

for i in df.Ticket:

    text_ticket += i

    

lista = re.findall('[a-zA-Z]+', text_ticket)

print('Most repeated terms in Tickets: \n')

print(pd.Series(lista).value_counts().head(10))
# creating features based on some commom words in Ticket feature



df['CA'] = df['Ticket'].str.contains('CA|C.A.').astype(int)

df['SOTON'] = df['Ticket'].str.contains('SOTON|STON').astype(int)

df['PC'] = df['Ticket'].str.contains('PC').astype(int)

df['SC'] = df['Ticket'].str.contains('SC|S.C').astype(int)

df['C'] = df['Ticket'].str.contains('C').astype(int)



# same with the df_test



df_test['CA'] = df_test['Ticket'].str.contains('CA|C.A.').astype(int)

df_test['SOTON'] = df_test['Ticket'].str.contains('SOTON|STON').astype(int)

df_test['PC'] = df_test['Ticket'].str.contains('PC').astype(int)

df_test['SC'] = df_test['Ticket'].str.contains('SC|S.C').astype(int)

df_test['C'] = df_test['Ticket'].str.contains('C').astype(int)
text_name = ''

for i in df.Name:

    text_name += i

    

lista = re.findall('[a-zA-Z]+', text_name)

print('Most repeated words in Name column: \n')

print(pd.Series(lista).value_counts().head(10))
# creating features based on some commom words in Name feature



df['Master'] = df['Name'].str.contains('Master').astype(int)

df['Mr'] = df['Name'].str.contains('Mr').astype(int)

df['Miss'] = df['Name'].str.contains('Miss').astype(int)

df['Mrs'] = df['Name'].str.contains('Mrs').astype(int)



#same with df_teste



df_test['Master'] = df_test['Name'].str.contains('Master').astype(int)

df_test['Mr'] = df_test['Name'].str.contains('Mr').astype(int)

df_test['Miss'] = df_test['Name'].str.contains('Miss').astype(int)

df_test['Mrs'] = df_test['Name'].str.contains('Mrs').astype(int)
variables = ['Age', 'Sex_bin', 'Pclass', 'Fare', 'Embarked_S','Embarked_C',\

             'CA', 'SOTON', 'PC', 'SC','C', 'Mr', 'Miss', 'Master', 'Mrs', 'Family']



print(reg_cross_val(variables))
variables = ['Age', 'Sex_bin', 'Pclass', 'Fare','Family', 'Embarked_S','Embarked_C','Cabin_NaN',\

             'CA', 'SOTON', 'PC', 'SC', 'Master', 'Mr', 'Miss', 'C', 'Mrs']



X = df[variables]

y = df['Survived']



reg = LogisticRegression(max_iter = 500)

reg.fit(X,y)

resp = reg.predict(df_test[variables])



submit = pd.Series(resp, index=df_test['PassengerId'], name='Survived')

submit.to_csv("model.csv", header=True)