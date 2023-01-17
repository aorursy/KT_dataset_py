# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model, preprocessing, tree, model_selection

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df
fig = plt.figure(figsize= (18,16))



plt.subplot2grid((2,3), (0,0))

df.Survived.value_counts().plot(kind='bar', alpha=0.5)

plt.title('Survival Count')

plt.xlabel('Survived')

plt.ylabel('Count')



plt.subplot2grid((2,3),(0,1))

plt.scatter(df.Survived, df.Age, alpha=0.1)

plt.title('Survived v Age')

plt.xlabel('Survived')

plt.ylabel('Age')



plt.subplot2grid((2,3),(0,2))

df.Pclass.value_counts(normalize=True).plot(kind='bar', alpha=0.3)

plt.title('Class')

plt.xlabel('Class')

plt.ylabel('Count')



plt.subplot2grid((2,3),(1,0), colspan=2)

for x in range(1,4):

    df.Age[df.Pclass==x].plot(kind='kde')

plt.title('Class v Age')

plt.legend(['First', 'Second', 'Third'])



plt.subplot2grid((2,3),(1,2))

df.Embarked.value_counts(normalize=True).plot(kind='bar', alpha=0.3)

plt.title('Embarked')

plt.xlabel('Port')

plt.ylabel('Count')



plt.show()
fig = plt.figure(figsize= (18,16))



plt.subplot2grid((3,4), (0,0))

df.Survived.value_counts().plot(kind='bar', alpha=0.5)

plt.title('Survival Count')

plt.xlabel('Survived')

plt.ylabel('Count')



male_color = '#0000FA'

plt.subplot2grid((3,4),(0,1))

df.Survived[df.Sex=='male'].value_counts(normalize=True).plot(kind='bar', alpha=0.3, color = male_color)

plt.title('Men Survived')

plt.xlabel('Class')

plt.ylabel('Count')



female_color = '#FA0000'

plt.subplot2grid((3,4),(0,2))

df.Survived[df.Sex=='female'].value_counts(normalize=True).plot(kind='bar', alpha=0.3, color = female_color)

plt.title('Women Survived')

plt.xlabel('Class')

plt.ylabel('Count')



plt.subplot2grid((3,4),(0,3))

df.Sex[df.Survived==1].value_counts(normalize=True).plot(kind='bar', alpha=0.3, color = '#00FA00')

plt.title('Gender of Survivors')

plt.xlabel('Gender')

plt.ylabel('Count')



plt.subplot2grid((3,4),(1,0), colspan=1)

for x in range(1,4):

    df.Age[df.Survived==1].plot(kind='box')

plt.title('Survived v Age')



plt.subplot2grid((3,4),(1,1), colspan=1)

for x in range(1,4):

    df.Survived[df.Pclass==x].plot(kind='kde')

plt.title('Class v Survived')

plt.legend(['First', 'Second', 'Third'])



plt.subplot2grid((3,4),(1,2), colspan=1)

for x in ['male', 'female']:

    df.Age[(df.Sex==x) & (df.Survived==1)].plot(kind='kde')

plt.title('Sex v Age (for Survived)')

plt.xlim(0,100)

plt.legend(['male', 'female'])



plt.show()
df['pred'] = 0

df.loc[df.Sex=='female', 'pred'] = 1

df['result'] = 0

df.loc[df.pred==df.Survived, 'result'] = 1
def clean_data(df):

    df.Fare = df.Fare.fillna(df.Fare.dropna().median())

    df.Age = df.Age.fillna(df.Age.dropna().median())

    df.loc[df.Sex=='male', 'Sex'] = 0

    df.loc[df.Sex=='female', 'Sex'] = 1

    df.Embarked = df.Embarked.fillna('S')

    df.loc[df.Embarked=='S', 'Embarked'] = 0

    df.loc[df.Embarked=='C', 'Embarked'] = 2

    df.loc[df.Embarked=='U', 'Embarked'] = 3
clean_data(df)

target = df['Survived'].values

features = df[['Pclass', 'Age', 'Sex', 'Parch', 'Fare']].values



classifier = linear_model.LogisticRegression()

classifier_ = classifier.fit(features, target)

print(classifier_.score(features, target))
poly = preprocessing.PolynomialFeatures(degree=2)

poly_features = poly.fit_transform(features)

classifier_ = classifier.fit(poly_features, target)

print(classifier_.score(poly_features, target))
poly = preprocessing.PolynomialFeatures(degree=2)

poly_features = poly.fit_transform(features)

dtree = tree.DecisionTreeClassifier(random_state = 8, max_depth = 6)

dtree_ = dtree.fit(features, target)

scores = model_selection.cross_val_score(dtree, features, target, scoring='accuracy', cv = 10)

print(dtree_.score(features, target), scores, scores.mean())
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test
clean_data(test)

res = dtree_.predict(test[['Pclass', 'Age', 'Sex', 'Parch', 'Fare']])

result_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

result_df.PassengerId = test.PassengerId

result_df.Survived = res

result_df
output_file = 'results.csv'

result_df.to_csv(output_file, header=True)