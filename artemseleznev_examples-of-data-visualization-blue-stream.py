import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#ошибки отключим
import warnings
warnings.simplefilter('ignore')
world_cups = pd.DataFrame(pd.read_csv('../input/WorldCups.csv'), columns = ['Year',
                                   'Country',
                                   'Winner',
                                   'Runners-Up',
                                   'Third',
                                   'Fourth',
                                   'GoalsScored',
                                   'QualifiedTeams',
                                   'MatchesPlayed',
                                   'Attendance'])
world_cups.name = 'world_cups'
world_cups.head()
world_cups.info()
players = pd.DataFrame(pd.read_csv('../input/WorldCupPlayers.csv'))
players.name = 'players'
players.head()
players.info()
matches = pd.DataFrame(pd.read_csv('../input/WorldCupMatches.csv'))
matches.name = 'matches'
matches.head()
matches.info()
#соберем все даты в один список
datas = [world_cups, players, matches]
#проверим на пустые ячейки
for d in datas:
    print('Таблица - ', d.name, '\nДлина - ', d.count().sum())
    print(d.isnull().sum())
    
    #если есть пустые, то удалим их
    d.dropna(inplace=True)
#Узнаем, кто больше всего побеждал в чемпионатах
world_cups.Winner.value_counts().plot.bar();

#такой же результат
#world_cups['Winner'].value_counts().plot.bar()
#Выведем количество очков команд c сортировкой
world_cups.GoalsScored.value_counts().sort_index().plot.bar();
#интересно посмотреть количство людей на трибунах
#world_cups.Attendance.sort_index().plot.line();

#!НО! это выдасть следующую ошибку:
#TypeError: Empty 'DataFrame': no numeric data to plot
#Если есть проблема с типом объекта. Проблема, но почему?
world_cups.info()
#объект : Attendance        20 non-null object
#нужны int или float
#сменим тип всей колоник
#world_cups.Attendance.astype('int64'), но получим ошибку,
#так как объект записан в виде №№№(точка)№№№
#пойдем на хитрость, перезапишем слобец, но сначала сделам перевод в строку - уберем точку - замене
world_cups.Attendance = world_cups.Attendance.str.replace('.', '').astype(int)

#повторим желанное действие
world_cups.plot.line(x = 'Year', y = 'Attendance');
#другой тим, заолняем область внутри
#путь будет количество голов во время матча, команда которая принимала матч
matches['Home Team Goals'].value_counts().sort_index().plot.area();
#путь будет количество голов во время матча, команда которая была гостем
matches['Away Team Goals'].value_counts().sort_index().plot.area();
world_cups.plot.scatter(x = 'MatchesPlayed', y = 'QualifiedTeams');
#но не много данных, это жалко
#плохо понятно, что хотелось показать, надо больше данных.
#попробуем эти
#сделаме выбор данных
#собмрем новый дата фрайм
#players['Team Initials']
#Создаем цыферное обозначение Team Initials
#самый простой и быстрый способ находиться в библиотеки sklearn
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
#добавим новый столбец, где будет столбец Team Initials замаскирован цифрам
players['TI'] = lb_make.fit_transform(players['Team Initials'])
players.plot.scatter(x = 'Shirt Number', y = 'TI');
#существуют отдельные способы кодирования в стиле dummpy
#интересные способы dummy кодирования 
#
#for elem in data['column'].unique():
#    dfrm[str(elem)] = data['column'] == elem
#Результатом будет: добавлены колонки с True/False
#
#Для самостоятельного определения категорий, создаем словарь
#
#cat_names = {1:'Name1', 2:'Name2', 3:'Name3'}
#for elem in data['column'].unique():
#    data[cat_names[elem]] = data['column'] == elem


#есть возможность выделить с "тепловым" элементом, по насыщености
players.plot.hexbin(x = 'Shirt Number', y = 'TI',cmap='Blues');

#где взять все параметры, которые записываются в cmap:
#на офф сайте matplotlib https://matplotlib.org/users/colormaps.html
#seaborn
import seaborn as sns
sns.distplot(world_cups.Winner.value_counts(), bins = len(world_cups.Winner.value_counts()), kde= False);
#ящик с усами на таблицу players
sns.boxplot(
    x = 'Shirt Number', 
    y = 'TI',
    data = players);

#hue - можно добавить описание каждого ящика
#order - установить последовательность
#похож на ящик с усами, но постоит их точек
sns.swarmplot(
    x = 'Shirt Number', 
    y = 'TI',
    data = players);
#не понятное распределение, наложим этот график на boxplot
#изменим размер, что бы было понятнее
plt.subplots(figsize = (10,10))
#добавим ящик с усами
sns.boxplot(
    x = 'Shirt Number', 
    y = 'TI',
    data = players);
#отразим на нем точки черного цвета
sns.swarmplot(
    x = 'Shirt Number', 
    y = 'TI',
    data = players,
    color = '.25'
);
#сделам график и распределение номеров на футболках в каждой стране
s = sns.FacetGrid(players, col = 'Team Initials', col_wrap=2)
s.map(sns.boxplot,'Shirt Number');
#тепловая карта в seaborn
sns.heatmap(players[['RoundID', 'MatchID', 'Shirt Number', 'TI']],cmap='Blues');
#не понятно, используем другу таблицу
sns.heatmap(world_cups[['Year', 'GoalsScored', 'QualifiedTeams', 'MatchesPlayed', 'Attendance']],cmap='Blues');
#изобразим это более красиво
#не понятно, используем другу таблицу
sns.heatmap(world_cups[['Year', 'GoalsScored', 'QualifiedTeams', 'MatchesPlayed', 'Attendance']].corr(), annot = True,cmap='Blues');
#нужна библиотек
import missingno as msno
#в таблице players были пустые ячейки
players_2 = pd.DataFrame(pd.read_csv('../input/WorldCupPlayers.csv'))
msno.matrix(players_2, color=(0.1,0.5,0.75));
#сравним с таблицей, с которой работали
msno.matrix(players, color=(0.1,0.5,0.75));
msno.bar(players_2,  color=(0.1,0.5,0.75));
msno.bar(players, color=(0.1,0.5,0.75));
