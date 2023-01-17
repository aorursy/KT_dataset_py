# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


pd.set_option('display.expand_frame_repr', False)





import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn
# Grafik için gerekli kütüphaneler

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#Train verisini kontrol edelim.

print(train.columns.values)

train.describe(include='all')
# Veri tiplerine bakalım.

print(train.dtypes)

print()

# Null değerleri kontrol edelim.

print(train.isna().sum())
# Test verini kontrol edelim.

print(test.columns.values)

test.describe(include='all')
print(test.dtypes)

print()

print(test.isna().sum())
# Korelasyon değerleri

print(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr())

sns.heatmap(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), annot=True, fmt = ".2f", cmap = "coolwarm")


print(train.columns.values)
# Pclass ve Survived arasındaki ilişkiyi görelim



print(train[['Pclass', 'Survived']].groupby(['Pclass']).mean())

sns.catplot(x='Pclass', y='Survived',  kind='bar', data=train)
print(train[['Sex', 'Survived']].groupby(['Sex']).mean())

sns.catplot(x='Sex', y='Survived',  kind='bar', data=train)
sns.catplot(x='Sex', y='Survived',  kind='bar', data=train, hue='Pclass')
g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Fare")
group = pd.cut(train.Fare, [0,50,100,150,200,550])

piv_fare = train.pivot_table(index=group, columns='Survived', values = 'Fare', aggfunc='count')

piv_fare.plot(kind='bar')
g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Age")
group = pd.cut(train.Age, [0,14,30,60,100])

piv_fare = train.pivot_table(index=group, columns='Survived', values = 'Age', aggfunc='count')

piv_fare.plot(kind='bar')
print(train[['Embarked', 'Survived']].groupby(['Embarked']).mean())

sns.catplot(x='Embarked', y='Survived',  kind='bar', data=train)
sns.catplot('Pclass', kind='count', col='Embarked', data=train)
print(train[['SibSp', 'Survived']].groupby(['SibSp']).mean())

sns.catplot(x='SibSp', y='Survived', data=train, kind='bar')
print(train[['Parch', 'Survived']].groupby(['Parch']).mean())

sns.catplot(x='Parch', y='Survived', data=train, kind='bar')


print(train.Name.head(1))

print()



print(train.Name.head(1).str)

print()



print(train.Name.head(1).str.split(','))

print()



print(train.Name.head(1).str.split(',').str[1])

print()


for dataset in [train, test]:

    #  Sadece isimdeki başlıkları almak için split kullandım

    dataset['Title'] = dataset['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()

    print(dataset['Title'].value_counts())

    print()
sns.catplot(x='Survived', y='Title', data=train, kind ='bar')
for df in [train, test]:

    print(df.shape)

    print()

    print(df.isna().sum())
for df in [train, test]:

    df.dropna(subset = ['Embarked'], inplace = True)
print(train[train['Fare'].isnull()])

print() 



print(test[test['Fare'].isnull()])

sns.catplot(x='Pclass', y='Fare', data=test, kind='point')
# Pclass ve Fare arasında net bir ilişki var. Bu bilgiyi eksik ücret değerini etkilemek için kullanabiliriz.

# Yolcunun Pclass 3'ten geldiğini görüyoruz. Bu yüzden tüm Pclass 3 ücretleri için ortanca değer alıyoruz.

test['Fare'].fillna(test[test['Pclass'] == 3].Fare.median(), inplace = True)
print(train[['Age','Title']].groupby('Title').mean())

sns.catplot(x='Age', y='Title', data=train, kind ='bar')


def getTitle(series):

    return series.str.split(',').str[1].str.split('.').str[0].str.strip()



print(getTitle(train[train.Age.isnull()].Name).value_counts())

#  Başlığa göre yaş doldurma medyanı

mr_mask = train['Title'] == 'Mr'

miss_mask = train['Title'] == 'Miss'

mrs_mask = train['Title'] == 'Mrs'

master_mask = train['Title'] == 'Master'

dr_mask = train['Title'] == 'Dr'

train.loc[mr_mask, 'Age'] = train.loc[mr_mask, 'Age'].fillna(train[train.Title == 'Mr'].Age.mean())

train.loc[miss_mask, 'Age'] = train.loc[miss_mask, 'Age'].fillna(train[train.Title == 'Miss'].Age.mean())

train.loc[mrs_mask, 'Age'] = train.loc[mrs_mask, 'Age'].fillna(train[train.Title == 'Mrs'].Age.mean())

train.loc[master_mask, 'Age'] = train.loc[master_mask, 'Age'].fillna(train[train.Title == 'Master'].Age.mean())

train.loc[dr_mask, 'Age'] = train.loc[dr_mask, 'Age'].fillna(train[train.Title == 'Dr'].Age.mean())



print()

print(getTitle(train[train.Age.isnull()].Name).value_counts())


print(getTitle(test[test.Age.isnull()].Name).value_counts())



mr_mask = test['Title'] == 'Mr'

miss_mask = test['Title'] == 'Miss'

mrs_mask = test['Title'] == 'Mrs'

master_mask = test['Title'] == 'Master'

ms_mask = test['Title'] == 'Ms'

test.loc[mr_mask, 'Age'] = test.loc[mr_mask, 'Age'].fillna(test[test.Title == 'Mr'].Age.mean())

test.loc[miss_mask, 'Age'] = test.loc[miss_mask, 'Age'].fillna(test[test.Title == 'Miss'].Age.mean())

test.loc[mrs_mask, 'Age'] = test.loc[mrs_mask, 'Age'].fillna(test[test.Title == 'Mrs'].Age.mean())

test.loc[master_mask, 'Age'] = test.loc[master_mask, 'Age'].fillna(test[test.Title == 'Master'].Age.mean())

test.loc[ms_mask, 'Age'] = test.loc[ms_mask, 'Age'].fillna(test[test.Title == 'Miss'].Age.mean())



print(getTitle(test[test.Age.isnull()].Name).value_counts())
# train.Age.fillna(train.Age.median(), inplace=True)

# validation.Age.fillna(validation.Age.median(), inplace=True)

print(train.isna().sum())

print(test.isna().sum())
train.drop(columns=['PassengerId'], inplace = True)

[df.drop(columns=['Ticket'], inplace = True) for df in [train, test]]
[train, test] = [pd.get_dummies(data = df, columns = ['Pclass', 'Sex', 'Embarked']) for df in [train, test]]
for df in [train, test]:

    df['HasCabin'] = df['Cabin'].notna().astype(int)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['IsAlone'] = (df['FamilySize'] > 1).astype(int)
[df.drop(columns=['Cabin', 'SibSp', 'Parch'], inplace = True) for df in [train, test]]
# Birkaç standart olmayan başlık olduğunu görüyoruz. Bazıları sadece Fransız başlıklarıdır.

# ingilizcede olduğu gibi aynı anlama gelirken, diğerleri muhtemelen

# Daha fazla imtiyaz veya askeri eğitim vs. var ve ayrı bir kategoriye yerleştirilebilirler.

# Fransız başlıklar - https://en.wikipedia.org/wiki/French_honorifics

# Mlle - https://en.wikipedia.org/wiki/Mademoiselle_(title)

# Mme - https://en.wikipedia.org/wiki/Madam

# Mme, Wikipedia’nın yetişkin kadınlar için kullanıldığını söylediği gibi anlaşılması biraz zordu.

# ama medeni durumlarına dair hiçbir işaretçi vermedi.

# Google’da arama yapma ve başlığın yetişkin kadınlar için kullanıldığını dikkate alarak

# Bu unvanın genellikle evli kadınlara atandığını varsayabiliriz.

# https://www.frenchtoday.com/blog/french-culture/madame-or-mademoiselle-a-delicate-question

# Ms - Miss için alternatif bir kısaltma

train['Title'] = train['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')

test['Title'] = test['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')

[df.drop(columns=['Name'], inplace = True) for df in [train, test]]

[train, test] = [pd.get_dummies(data = df, columns = ['Title']) for df in [train, test]]
print(train.columns.values)

print(test.columns.values)
# Güncellenen veri kümeleri ile korelasyonu kontrol edelim

train.corr()
# Veri setini train ve test setlerine ayır

from sklearn.model_selection import train_test_split

# Yalnızca 0,3'ten büyük katsayılı özellikleri kullanın

X = train[['Age', 'Fare', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Embarked_C',

       'Embarked_S', 'HasCabin', 'FamilySize', 'Title_Master', 'Title_Mr',

       'Title_Mrs', 'Title_Special']]

y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

print(X_train.shape, X_test.shape)
# Modelimizin iyiliğini kontrol etmek için bir temel model de oluşturacağız.

# Önce gerçek hayatta kalanları görüyoruz

print(y.value_counts())
# Daha büyük sayıyı seçeceğiz ve herkesin temel oluşturmak için öldüğünü düşüneceğiz.

y_default = pd.Series([0] * train['Survived'].shape[0], name = 'Survived')

print(y_default.value_counts())


from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score

print(confusion_matrix(y, y_default))

print()

print(accuracy_score(y, y_default))

# Eğer herkesin öldüğünü varsayarsak, zamanın% 61'i doğru oluruz.

# Yani bu, beklentimizin iyileştirilmesi gereken en küçük minimum doğruluk seviyesidir.
# LinearSVC ile ilk deneme

from sklearn.svm import LinearSVC



classifier = LinearSVC(dual=False)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')

print(scores.mean())
# KNN deneyelim

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')

print(scores.mean())
# KNN bizim için kullanışlı değil, şimdi birkaç popüler sınıflandırıcı seçelim.

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

print("AdaBoostClassifier")

ada_boost_classifier = AdaBoostClassifier()

ada_boost_classifier.fit(X_train, y_train)

y_pred = ada_boost_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

scores = cross_val_score(ada_boost_classifier, X_train, y_train, cv=10, scoring='accuracy')

print(scores.mean())

print("BaggingClassifier")

bagging_classifier = BaggingClassifier()

bagging_classifier.fit(X_train, y_train)

y_pred = bagging_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

scores = cross_val_score(bagging_classifier, X_train, y_train, cv=10, scoring='accuracy')

print(scores.mean())

print("ExtraTreesClassifier")

extra_trees_classifier = ExtraTreesClassifier(n_estimators=100)

extra_trees_classifier.fit(X_train, y_train)

y_pred = extra_trees_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

scores = cross_val_score(extra_trees_classifier, X_train, y_train, cv=10, scoring='accuracy')

print(scores.mean())

print("GradientBoostingClassifier")

gradient_boosting_classifier = GradientBoostingClassifier()

gradient_boosting_classifier.fit(X_train, y_train)

y_pred = gradient_boosting_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

scores = cross_val_score(gradient_boosting_classifier, X_train, y_train, cv=10, scoring='accuracy')

print(scores.mean())

print("RandomForestClassifier")

random_forest_classifier = RandomForestClassifier(n_estimators=100)

random_forest_classifier.fit(X_train, y_train)

y_pred = random_forest_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

scores = cross_val_score(random_forest_classifier, X_train, y_train, cv=10, scoring='accuracy')

print(scores.mean())
# XGB

from xgboost import XGBClassifier

xgboost_classifier = XGBClassifier()

xgboost_classifier.fit(X_train, y_train)

y_pred = xgboost_classifier.predict(X_test)



print(confusion_matrix(y_test, y_pred))



print(accuracy_score(y_test, y_pred))

scores = cross_val_score(xgboost_classifier, X_train, y_train, cv=10, scoring='accuracy')

print(scores.mean())