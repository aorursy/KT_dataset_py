import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
titanic_data = pd.read_csv('/kaggle/input/titanic/train.csv')
print(titanic_data.head())
titanic_data.info()# <получение общей информации о данных в таблице titanic_data>
titanic_data.isnull().sum()# <суммарное количество пропусков, выявленных методом isnull() в таблице titanic_data>
titanic_data.duplicated().sum()# <получение суммарного количества дубликатов в таблице titanic_data>
titanic_data.describe()#описательная статистика (five-summary statistics)
# Смотрим на взаимодействие всех переменных
pair_plot = sns.PairGrid(titanic_data)
pair_plot.map(plt.scatter)
#можем убрать слишком "уникальные" данные
titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
# Смотрим на взаимодействие отобранных на данный момент переменных
pair_plot_2 = sns.PairGrid(titanic_data)
pair_plot_2.map(plt.scatter)
#Посмотрим на распределения поближе
sns.jointplot(x = 'Age', y = 'Fare', data = titanic_data)
sns.jointplot(x = 'Age', y = 'Survived', data = titanic_data)
sns.jointplot(x = 'Fare', y = 'Survived', data = titanic_data)
#смотрим, сколько среди выживших мужчин и женщин
sns.catplot(x = 'Sex', col = 'Survived', kind = 'count', data = titanic_data)
#Просмотрим сводные таблицы (таблицы сопряженности)
pd.crosstab(titanic_data['Embarked'], titanic_data['Survived'], margins = True).style.background_gradient()
#Посмотрим на распределение того, как менялась доля виживших в зависимости от пола
sns.catplot('Sex', 'Survived', kind = 'point', data = titanic_data)
pd.crosstab([titanic_data.Sex, titanic_data.Survived], titanic_data.Age, margins = True).style.background_gradient()
#смотрим на форму распределения с помощью violinplot
sns.violinplot(titanic_data['Fare'])
sns.violinplot(titanic_data['Age'])
# Строим корреляционную матрицу (SibSp Parch Age and Fare values) and Survived 
titanic_data_new = titanic_data.copy()
titanic_data_new['Sex'] = titanic_data_new['Sex'].map({'female': 0, 'male': 1})
titanic_data_new['Embarked'] = titanic_data_new['Embarked'].map({'S': 0, 'C': 1, 'Q' : 2})
sns.heatmap(titanic_data_new[["Survived", "Sex","SibSp","Parch","Age","Fare", 'Embarked']].corr(),annot=True, cmap = "coolwarm")
#Полезная вещь!!!
from pandas_profiling import ProfileReport
profile = ProfileReport(titanic_data, title = 'Pandas Profiling Report')
profile
print(titanic_data[titanic_data['Embarked'].isnull()])
#Stone, Mrs. George Nelson (Martha Evelyn) 
#Mrs Stone boarded the Titanic in Southampton on 10 April 1912 
#and was travelling in first class with her maid Amelie Icard. 
#She occupied cabin B-28. https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html
#Miss Rose Amélie Icard, 38, was born in Vaucluse, France on 31 October 1872, her father Marc Icard lived at Mafs á Murs (?).
#She boarded the Titanic at Southampton as maid to Mrs George Nelson Stone.
#She travelled on Mrs Stone's ticket (#113572).
titanic_data[titanic_data['PassengerId'] == 62] = titanic_data[titanic_data['PassengerId'] == 62].fillna('S')
titanic_data[titanic_data['PassengerId'] == 830] = titanic_data[titanic_data['PassengerId'] == 830].fillna('S')
#удаляем лишние данные(столбцы)
X_train = titanic_data.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1)#пока не совсем X_train
#проверка
print(X_train.isnull().sum())
# <получение суммарного количества дубликатов в таблице X_train>
Y_train = X_train['Survived']
X_train = X_train.drop('Survived', axis = 1)
#удостоверимся, что лишних данных нет (3 не дано, по крайней мере, в тот момент времени так было)
print(X_train['Sex'].unique())
#Переводим в числовой признак для удобства "скармливания" моделям, пусть male = 1, female = 0
def find_sex_and_replace(table):
    table['Sex'] = table['Sex'].map({'female': 0, 'male': 1})
    return table[(table['Sex'] == 'male') | (table['Sex'] == 'female')]['Sex'].count()
print(find_sex_and_replace(X_train))
X_train.head()
#то же самое делаем для Embarked
X_train['Embarked'].unique()
def find_Embarked_and_replace(table):
    '''# one-hot encoding
    table = pd.concat([table, pd.get_dummies(data = table['Embarked'], prefix="Embarked")], axis=1)
    table = table.drop(['Embarked'], axis = 1)'''
    table.loc[table['Embarked'] == 'S','Embarked'] = 0
    table.loc[table['Embarked'] == 'C','Embarked'] = 1
    table.loc[table['Embarked'] == 'Q','Embarked'] = 2
    return table
X_train = find_Embarked_and_replace(X_train)
X_train.head()
#осталось разобраться с Age, для начала визуализируем эти данные
sns.distplot(X_train['Age'], kde=False, bins=20)
#построим boxplot, более информативный график
sns.boxplot(data=X_train['Age'])
#таким образом, можно заполнить значения пропущенные на значения примерно 
#в интервале std +- sigma (примерно так), чтобы распределение не сильно сместилось
'''X_median = X_train['Age'].median()
X_mean = int(X_train['Age'].mean(axis=0))
X_std = int(X_train['Age'].std(axis=0))
p = np.linspace(0, 1, num = int(2*X_std))
p /= p.sum() # из-за машинной единицы особенность
X_train['Age'] = X_train['Age'].fillna(np.random.choice(np.arange(X_mean - X_std, X_mean + X_std, 1), p = p))
sns.boxplot(data = X_train['Age'])
print('Старое значение медианы: {},\nНовое: {}'.format(X_median, X_train['Age'].median()))'''
#заметим, что распределение теперь за пределами 1.5 межквантильного размаха имеет больше выбросов
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())
#проверяем, что получилось
X_train.info()
# переводим возраст в целое для того, чтобы алгоритмы работали быстрее
X_train = X_train.astype({"Age": int, "Fare": int})
#рассмотрим другие характеристики
X_train.head()
sns.distplot(X_train['Fare'], kde=False, bins=100)
#видны подозрительно большие значения, которые могут помешать классификации
#возможно, необходимо разделить по группам Fare и Age

def convert_to_fare_category(table): 
    table.loc[table['Fare'] <= 10,'Fare'] = 1
    table.loc[((table['Fare'] > 10) & (table['Fare'] <= 25)),'Fare'] = 2
    table.loc[((table['Fare'] > 25) & (table['Fare'] <= 50)),'Fare'] = 3
    table.loc[((table['Fare'] > 50) & (table['Fare'] <= 75)),'Fare'] = 4
    table.loc[((table['Fare'] > 75) & (table['Fare'] <= 100)),'Fare'] = 5
    table.loc[(table['Fare'] > 100),'Fare'] = 6
    
    return table

def convert_to_age_category(table):
    table.loc[table['Age'] <= 15, 'Age'] = 0
    table.loc[((table['Age'] > 15) & (table['Age'] <= 20)), 'Age'] = 1
    table.loc[((table['Age'] > 20) & (table['Age'] <= 26)), 'Age'] = 2
    table.loc[((table['Age'] > 26) & (table['Age'] <= 28)), 'Age'] = 3
    table.loc[((table['Age'] > 28) & (table['Age'] <= 35)), 'Age'] = 4
    table.loc[((table['Age'] > 35) & (table['Age'] <= 45)),'Age'] = 5
    table.loc[(table['Age'] > 45),'Age'] = 6

    return table

X_train = convert_to_fare_category(X_train)
X_train = convert_to_age_category(X_train)
sns.distplot(X_train['Fare'], kde=False, bins=6)
sns.distplot(X_train['Age'], kde=False, bins=6)
X_train['familySize'] = X_train['SibSp'] + X_train['Parch'] + 1
X_train.drop(['SibSp', 'Parch', 'Fare', 'Embarked'], axis=1, inplace = True)
#X_train['isAlone'] = np.where((X_train['familySize'] > 1),0,X_train['familySize'])
#распределение немного смещено в сторону умерших, можно сбалансировать выборку
X_train.head()
sns.countplot(Y_train)
#нужно выполнить сэмлирование
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
#применяем алгоритм SMOTE(дополняем меньшую выборку)
os = SMOTE(random_state=0)
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=2)
#дополняем только тренировочную выборку 
os_x_train, os_y_train=os.fit_sample(x_train, y_train)

os_x_train = pd.DataFrame(data = os_x_train, columns=x_train.columns)
os_y_train = pd.Series(data = os_y_train)
sns.countplot(os_y_train)
# Мы можем проверить числа наших данных
print("Длина данных с оверсемплингом составляет: {}".format(len(os_y_train)))
print("Число умерших в датасете с оверсемплингом: {}".format(len(os_y_train[os_y_train==0])))
print("Число выживших в датасете с оверсемплингом: {}".format(len(os_y_train[os_y_train==1])))
#данные сбалансированы
X_train.head()
# 1 модель - дерево
from sklearn import tree
from sklearn.model_selection import cross_val_score
def use_DecisionTreeClassifier(x_train, y_train, x_test, y_test):
    #делим на train и test выборки
    scores_data = pd.DataFrame()
    for max_depth in range(1, 50): # 50 - максимальная глубина
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)#критерий - энтропия
        clf.fit(x_train, y_train)                                                  #обучаем модель
        train_score = clf.score(x_train, y_train)                                  #точность на train
        test_score = clf.score(x_test, y_test)                                     #точность на test
        mean_cross_val_score = cross_val_score(clf, x_train, y_train, cv=10).mean() #точность на кросс-валидации
        scores_data = scores_data.append(
                          pd.DataFrame({'max_depth': [max_depth],
                                        'train_score': [train_score],
                                        'test_score': [test_score],
                                        'cross_val_score': [mean_cross_val_score]}))

    scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score', 'test_score', 'cross_val_score'],
                              var_name='set_type', value_name='score')
    sns.lineplot(x="max_depth", y="score", hue="set_type", data=scores_data_long)
    
    best_score_on_cross_validation = scores_data_long[scores_data_long['set_type'] == 'cross_val_score'].groupby(by = 'set_type')['score'].max().values[0]
    print("Лучший результат на кросс - валидации: {:.5f}".format(best_score_on_cross_validation))
    print(scores_data_long[scores_data_long['set_type'] == 'cross_val_score'].head(20))
#без семплирования
use_DecisionTreeClassifier(x_train, y_train, x_test, y_test)
#лучший результат - глубина 5
#строим матрицу ошибок
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def default_metrics(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    print(classification_report(y_test, y_pred))
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 4)
clf.fit(x_train, y_train)
default_metrics(clf, x_test, y_test)
#с семплированием
use_DecisionTreeClassifier(os_x_train, os_y_train, x_test, y_test)
#лучший результат - глубина 5
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 8)
clf.fit(os_x_train, os_y_train)
default_metrics(clf, x_test, y_test)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)# добавим min_samples_leaf
clf.fit(x_train, y_train)
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import sklearn
graph = Source(sklearn.tree.export_graphviz(clf, out_file=None,
                                   feature_names=list(x_train),
                                   class_names=['Negative','Positive'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))
#2. применим логистическую регрессию
from sklearn.linear_model import LogisticRegression
scores_data = pd.DataFrame()
for C in np.power(10, np.array(range(-3, 3)), dtype=float):
    logreg  = LogisticRegression(C = C, n_jobs = -1)
    logreg.fit(os_x_train, os_y_train)
    train_score = logreg.score(os_x_train, os_y_train)                                   #точность на train
    test_score = logreg.score(x_test, y_test)                                            #точность на test
    mean_cross_val_score = cross_val_score(logreg, os_x_train, os_y_train , cv=10).mean() #точность на кросс-валидации
    scores_data = scores_data.append(
                      pd.DataFrame({'C': [C],
                                    'train_score': [train_score],
                                    'test_score': [test_score],
                                    'cross_val_score': [mean_cross_val_score]}))

scores_data_long = pd.melt(scores_data, id_vars=['C'], value_vars=['train_score', 'test_score', 'cross_val_score'],
                          var_name='set_type', value_name='score')
sns.lineplot(x="C", y="score", hue="set_type", data=scores_data_long)
#ищем лучшую модель
best_score_on_cross_validation = scores_data_long[scores_data_long['set_type'] == 'cross_val_score'].groupby(by = 'set_type')['score'].max().values[0]
print("Лучший результат на кросс - валидации: {:.5f}".format(best_score_on_cross_validation))
scores_data_long[scores_data_long['set_type'] == 'cross_val_score']
# С = 0.100 - лучший параметр регуляризации
logit = LogisticRegression(C = 10.0, n_jobs = -1)
logit.fit(x_train, y_train)
default_metrics(logit, x_test, y_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def findBestParamsRandomForestClassifier(x_train, y_train):
    clf = RandomForestClassifier()
    parameters = ({'n_estimators': [5, 10, 20, 30, 50, 75, 100], 'criterion': ['gini', 'entropy']})
    grid_search_CV = GridSearchCV(clf, parameters, cv = 10, n_jobs = -1)
    grid_search_CV.fit(x_train, y_train)
    print("Best params: ", grid_search_CV.best_params_)
    
findBestParamsRandomForestClassifier(x_train, y_train)
forest_clf = RandomForestClassifier(n_estimators= 100)
forest_clf.fit(x_train, y_train)
default_metrics(forest_clf, x_test, y_test)
example = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
print(example.head())

test = pd.read_csv('/kaggle/input/titanic/test.csv')
print(test.head())
print(test.info())
#лучший классификатор
'''clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 4)
clf.fit(x_train, y_train)'''

forest_clf = RandomForestClassifier(n_estimators= 100)
forest_clf.fit(x_train, y_train)
#разбираемся с Age
'''X_mean = int(test['Age'].mean(axis=0))
X_std = int(test['Age'].std(axis=0))
p = np.linspace(0, 1, num = int(2*X_std))
p /= p.sum() # из-за машинной единицы особенность
test['Age'] = test['Age'].fillna(np.random.choice(np.arange(X_mean - X_std, X_mean + X_std, 1), p = p))'''
test['Age'] = test['Age'].fillna(X_train['Age'].median())
#преобразуем данные
#Survived Pclass Sex Age SibSp Parch Fare Embarked 
find_sex_and_replace(test)
test = find_Embarked_and_replace(test)
#пропущенное значение

#fare_old_woman = test[(test['Age'] == 5) & (test['Embarked_S'] == 1)]['Fare'].mode().min()
#можно было бы отдельно поработать с ней
#test['Fare'] = test['Fare'].fillna(fare_old_woman)

test = convert_to_fare_category(test) 
print(test.head())
test = convert_to_age_category(test)
test = test.astype({"Age": int, "Fare": int})

test['familySize'] = test['SibSp'] + test['Parch'] + 1
test.drop(['SibSp', 'Parch', 'Name','Ticket','Cabin'], axis=1, inplace = True)
test.info()
'''result = forest_clf.predict(test.drop(['PassengerId'], axis = 1))
'''
result = clf.predict(test.drop(['PassengerId'], axis = 1))
submission = pd.DataFrame({'PassengerId': test.PassengerId,'Survived':result})
submission.Survived = submission.Survived.astype(int)
filename = 'Titanic Predictions.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)