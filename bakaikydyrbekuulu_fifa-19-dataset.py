import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Подключаем наш дата-сет

df=pd.read_csv("../input/fifa19/data.csv")
#Показываем первые 10 записей

df.head(10)
#Показываем первые 10 записей

df.head(10)
#Удаляем все не нужные колонки кроме:

df.drop(df.columns.difference(['ID','Name','Age','Photo','Nationality', 'Position','Overall','Potential','Club','Composure','Dribbling', 'GKDiving', 'GKHandling','GKKicking','GKPositioning','GKReflexes','Joined','Wage','Preferred Foot', 'International Reputation']), 1, inplace=True)
#Выводим количество строк и столбцов.

df.shape
#Выводим информацию наших данных.

df.info()
#Выводим все наши колонки:

df.columns
#Выводим всех игроков по национальности:

def country(x):

    return df[df['Nationality'] == x][['Name','Overall','Potential','Position']]





country('Russia')
#Выводим всех игроков клуба Tottehnam Hostpur:

def club(x):

    return df[df['Club'] == x][['Name','Position','Overall','Nationality','Age']]



club('Tottenham Hotspur')
#Выводим количество строк по клубу "Tottenham Hotspur":

x = club('Tottenham Hotspur')

x.shape
#Выводим сумму пустых записей по колонкам

df.isnull().sum()
#Заменяем пустые записи на определенные или средние значения.

df['Club'].fillna('No Club', inplace = True)

df['Preferred Foot'].fillna('Right', inplace = True)

df['Position'].fillna('ST', inplace = True)

df['Dribbling'].fillna(df['Dribbling'].mean(), inplace = True)

df['Composure'].fillna(df['Composure'].mean(), inplace = True)

df['GKDiving'].fillna(df['GKDiving'].mean(), inplace = True)

df['GKHandling'].fillna(df['GKHandling'].mean(), inplace = True)

df['GKKicking'].fillna(df['GKKicking'].mean(), inplace = True)

df['GKPositioning'].fillna(df['GKPositioning'].mean(), inplace = True)

df['GKReflexes'].fillna(df['GKReflexes'].mean(), inplace = True)

df['Joined'].fillna('Jul 1, 2018', inplace = True)

df['Wage'].fillna('€200K', inplace = True)
#Используем диаграмму countplot и выводим данные

plt.rcParams['figure.figsize'] = (10, 5)

sns.countplot(df['Preferred Foot'], palette = 'Reds')

plt.title('Предпочитаемая нога игрока', fontsize = 20)

plt.show()
#Построение круговой диаграммы для представления доли международной репутации



labels = ['1', '2', '3', '4', '5']

sizes = df['International Reputation'].value_counts()

colors = plt.cm.copper(np.linspace(0, 1, 5))

explode = [0.1, 0.1, 0.2, 0.5, 0.9]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)

plt.title('Международная репутация футболистов', fontsize = 20)

plt.legend()

plt.show()

#Используем диаграмму countplot и выводим данные

p = sns.countplot(x='Position', data=df)

_ = plt.setp(p.get_xticklabels(), rotation=90)
#Используем диаграмму countplot и выводим данные

fig = plt.figure(figsize=(25, 10))

p = sns.countplot(x='Nationality', data=df)

_ = plt.setp(p.get_xticklabels(), rotation=90)
# To show that there are people having same age

# Histogram: number of players's age



x = df.Age

plt.figure(figsize = (15,8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'g')

ax.set_xlabel(xlabel = "Возраст футболистов", fontsize = 16)

ax.set_ylabel(ylabel = 'Количество футболистов', fontsize = 16)

ax.set_title(label = 'Гистограмма возраста футболистов', fontsize = 20)

plt.show()
# To show Different potential scores of the players participating in the FIFA 2019



x = df.Potential

plt.figure(figsize=(12,8))

plt.style.use('seaborn-paper')



ax = sns.distplot(x, bins = 58, kde = False, color = 'y')

ax.set_xlabel(xlabel = "Очки потенциала футболиста", fontsize = 16)

ax.set_ylabel(ylabel = 'Количество игроков', fontsize = 16)

ax.set_title(label = 'Гистограмма очков потенциала футболиста', fontsize = 20)

plt.show()
# Различные общие результаты игроков, участвующих в FIFA 2019



sns.set(style = "dark", palette = "deep", color_codes = True)

x = df.Overall

plt.figure(figsize = (12,8))

plt.style.use('ggplot')



ax = sns.distplot(x, bins = 52, kde = False, color = 'r')

ax.set_xlabel(xlabel = "Очки футолиста", fontsize = 16)

ax.set_ylabel(ylabel = 'Количество футболистов', fontsize = 16)

ax.set_title(label = 'Гистограмма: Общие очки футболистов', fontsize = 20)

plt.show()
plt.style.use('dark_background')

df['Nationality'].value_counts().head(80).plot.bar(color = 'orange', figsize = (20, 7))

plt.title('Разновидность национальностей в FIFA 2019', fontsize = 30, fontweight = 20)

plt.xlabel('Название страны')

plt.ylabel('Количество')

plt.show()
# Это HeatMap показывает корреляцию между различными переменными

plt.rcParams['figure.figsize']=(25,16)

hm=sns.heatmap(df[['ID', 'Name', 'Age', 'Photo', 'Nationality', 'Overall', 'Potential',

       'Club', 'Wage', 'Preferred Foot', 'Position', 'Joined', 'Dribbling',

       'Composure', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning',

       'GKReflexes']].corr(), annot = True, linewidths=.5, cmap='Blues')

hm.set_title(label='Heatmap of dataset', fontsize=20)

hm;
# Градиентный спуск

from xgboost import XGBClassifier

from xgboost import plot_importance



# Обучение модели

model = XGBClassifier()

model_importance = model.fit(X_train, y_train)



# Построение гистограммы важности функций

plt.rcParams['figure.figsize'] = [14,12]

sns.set(style = 'darkgrid')

plot_importance(model_importance);
#Препроцессинг

#Все строковые значения перевожу в числовые

from sklearn.preprocessing import LabelEncoder

x=['ID', 'Name', 'Age', 'Photo', 'Nationality', 'Overall', 'Potential',

       'Club', 'Wage', 'Preferred Foot', 'Position', 'Joined','Dribbling', 'Composure', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

for i in x:

    if (df[i].dtype == np.string_ or df[i].dtype == np.object):

        a=LabelEncoder()

        df[i]=a.fit_transform(df[i])

    

df.head()
# Импортируем функции из библиотеки sclarn

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier



import warnings; warnings.simplefilter('ignore')
#Разбиваем данные на случайные тестовые и тренировочные наборы

#Также указываем зависимый столбец

X_train, X_test, y_train, y_test = train_test_split(df[['ID', 'Name', 'Photo', 'Nationality', 'Position', 'Age',

       'Overall', 'Wage', 'Potential', 'Club', 'Joined', 'Composure', 'Dribbling', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']],df['Preferred Foot'], test_size=0.3)
#BerNoulliNB Model

# Обучение модели

NB = BernoulliNB(alpha = 0.3)

model_1 = NB.fit(X_train, y_train)



# Predictions

pred_1 = model_1.predict(X_test)



print("Accuracy для BerNoulliNB Model: %.2f" % (accuracy_score(y_test, pred_1) * 100))
#ANN Model

# Обучение модели

clf5 = MLPClassifier()

model_2 = clf5.fit(X_train, y_train)



# Predictions

pred_2 = model_2.predict(X_test)

print("Accuracy для ANN Model: %.2f" % (accuracy_score(y_test, pred_2) * 100))
#Logistic Regression Model

# Обучение модели

logistic = LogisticRegression(C = 0.5, max_iter = 500)

model_3 = logistic.fit(X_train, y_train)



# Predictions

pred_3 = model_3.predict(X_test)

print("Accuracy для Logistic Regression Model: %.2f" % (accuracy_score(y_test, pred_3) * 100))
#Decision Tree Model

# Обучение модели

drugTree = DecisionTreeClassifier(criterion="gini")

model_4 = drugTree.fit(X_train, y_train)



# Predictions

pred_4 = model_4.predict(X_test)

print("Accuracy для Decision Tree Model: %.2f" % (accuracy_score(y_test, pred_4) * 100))
#Random Forest Model

# Обучение модели

R_forest = RandomForestClassifier(n_estimators = 200)

model_5 = R_forest.fit(X_train, y_train)



# Predictions

pred_5 = model_5.predict(X_test)

print("Accuracy для Random Forest Model: %.2f" % (accuracy_score(y_test, pred_5) * 100))
#Classification Report of Models

list_pred = [pred_1, pred_2, pred_3, pred_4, pred_5]

model_names = [ "Bernoulli NB","ANN","Logistic Regression","Decision Tree" ,"Random Forest Classifier"]



for i, predictions in enumerate(list_pred) :

    print ("Classification Report of", model_names[i])

    print ()

    print (classification_report(y_test, predictions, target_names = ["Left Foot", "Right Foot"]))
#Confusion Matrix of Models

for i, pred in enumerate(list_pred) :

    print ("The Confusion Matrix of : ", model_names[i])

    print (pd.DataFrame(confusion_matrix(y_test, pred)))

    print ()
from sklearn.metrics import roc_auc_score, roc_curve

models = [model_1, model_2, model_3, model_4, model_5]



# Setting the parameters for the ROC Curve

plt.rcParams['figure.figsize'] = [10,8]

plt.style.use("bmh")



color = ['red', 'blue', 'green', 'fuchsia', 'cyan']

plt.title("ROC CURVE", fontsize = 15)

plt.xlabel("Specificity", fontsize = 15)

plt.ylabel("Sensitivity", fontsize = 15)

i = 1



for i, model in enumerate(models) :

    prob = model.predict_proba(X_test)

    prob_positive = prob[:,1]

    fpr, tpr, threshold = roc_curve(y_test, prob_positive)

    plt.plot(fpr, tpr, color = color[i])

    plt.gca().legend(model_names, loc = 'lower right', frameon = True)



plt.plot([0,1],[0,1], linestyle = '--', color = 'black')

plt.show()