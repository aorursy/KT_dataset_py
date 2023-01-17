# -*- coding: utf-8 -*-



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, f1_score

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import train_test_split, GridSearchCV
df = pd.read_csv('../input/train.csv', index_col='PassengerId')

df.head()
df.describe()
alpha_level = 0.65

    

# Sex

df_sex_survived = df.Sex[df.Survived == 1].value_counts()

df_sex_survived.plot(kind='bar', alpha=alpha_level)

plt.title("Распределение выживания (в зависимости от пола)")

plt.show()
data = df[df.Survived == 1]

n_data = data.groupby(['Sex']).agg({'Survived': 'count'})

new_data = n_data.groupby(level=0).apply(lambda x: round(100 * x / float(n_data['Survived'].sum()), 2))
plt.figure()

figure = new_data.plot(kind='bar')

figure.set_ylabel('Percent of people')

figure.set_xlabel('Sex')

plt.suptitle('Процент выживших (в зависимости от класса)', fontsize = 12)

plt.show()
# Pclass

df_pclass_survived = df.Pclass[df.Survived == 1].value_counts()

df_pclass_survived.plot(kind='bar', alpha=alpha_level)

plt.title("Распределение выживания (в зависимости от класса)")

plt.show()
data = df[df.Survived == 1]

n_data = data.groupby(['Pclass']).agg({'Survived': 'count'})

new_data = n_data.groupby(level=0).apply(lambda x: round(100 * x / float(n_data['Survived'].sum()), 2))
plt.figure()

figure = new_data.plot(kind='bar')

figure.set_ylabel('Percent of people')

figure.set_xlabel('Pclass')

plt.suptitle('Процент выживших (в зависимости от класса)', fontsize = 12)

plt.show()
Pclass1 = df.Fare[df.Pclass == 1]

Pclass2 = df.Fare[df.Pclass == 2]

Pclass3 = df.Fare[df.Pclass == 3]



data_to_boxplot = [Pclass1, Pclass2, Pclass3]



plt.figure(1, figsize=(10, 7))

plt.boxplot([Pclass1, Pclass2, Pclass3], showfliers=False)  # убираем выбросы

plt.ylabel('Цена билета')

plt.xticks([1, 2, 3], ['Pclass 1', 'Pclass 2', 'Pclass 3'])

plt.suptitle('Зависимость стоимости билета от класса', fontsize = 12)

plt.ylim(0, 170)



plt.show()

df_male = df.Pclass[df.Sex == 'male'][df.Survived == 1].value_counts()       # выживших мужчин

df_female = df.Pclass[df.Sex == 'female'][df.Survived == 1].value_counts()   # выживших женщин



df_male.plot(kind='bar', label='Male', alpha=0.70)

df_female.plot(kind='bar', color='#FA2379', label='Female', alpha=0.70)

plt.title("Кто выжил? (с учетом пола)")

plt.legend(('Male', 'Female'), loc='best')

plt.show()
train = pd.read_csv('../input/train.csv', index_col='PassengerId')

train.head(8)
def missing(data):

    

    # Заменим пропуски в Age на медиану...

    median_age = data['Age'].median()

    data['Age'].fillna(value=median_age, inplace=True)

    

    # Заменим пропуски в Fare на медиану...

    median_fare = data['Fare'].median()

    data['Fare'].fillna(value=median_fare, inplace=True)



    # Заменим пропуски в Embarked

    mode_embarked = data['Embarked'].mode()[0]

    data['Embarked'].fillna(value=mode_embarked, inplace=True)

    

    # Создадим две новые категории. Можно предположить, что то, является ли человек одиночкой, 

    # и есть ли у него какие-то родственники может сыграть какую-нибудь роль

    data['Relatives'] = data["Parch"] + data["SibSp"]  # сколько всего родственников у человека

    data['Single'] = data["Relatives"].apply(lambda r: 1 if r == 0 else 0)   # является ли человек одиночкой

    

    # Дропнем ненужное

    data = data.drop(['Name','Cabin','Ticket'], 1) 



    data.head(8)

    return data
train = missing(train)

train.head(8)
# Переводим categorical в numerical

train = pd.get_dummies(train, drop_first=True)

train.head(8)
x_labels = ['Pclass', 'Age', 'Fare', 'Single', 'Sex_male', 'Embarked_Q', 'Embarked_S'] 
X, y = train[x_labels], train['Survived']
selector = SelectKBest(f_classif, k=5) # f_classif - дисперсионный анализ F-value между лейблом / функцией

selector.fit(X, y)
scores = -np.log10(selector.pvalues_)  # p-values of feature scores
plt.bar(range(len(x_labels)), scores)

plt.xticks(range(len(x_labels)), x_labels, rotation='vertical')

plt.show()
# Удалим параметры, играющие небольшую роль

x_labels = ['Pclass', 'Age', 'Fare', 'Relatives', 'Sex_male', 'Embarked_S'] 

X_train, y_train = train[x_labels], train['Survived']
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
def grid_search_function(clf, parameters, X_train, X_test, y_train, y_test):

    gs = GridSearchCV(clf, parameters)

    gs.fit(X_train, y_train)

    print (gs.best_params_)

    y_pred = gs.predict(X_test)

    print(classification_report(y_test, y_pred))

    return gs
dt_params = {

              'max_depth':[None, 3, 4, 5, 6, 10],

              'max_leaf_nodes':[None, 6, 8, 10, 12, 14],

              'min_samples_leaf': [3, 5, 7, 10]

             }
dt_gs = grid_search_function(DecisionTreeClassifier(random_state=42), dt_params, X_train, X_test, y_train, y_test)