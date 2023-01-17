#1.1 Объединение датасетов в один общий датасет

import numpy as np                                           # Импорт библиотеки NumPy
import pandas as pd                                          # Импорт библиотеки Pandas
from datetime import datetime as dt                          # Импорт библиотеки Datetime
import matplotlib.pyplot as plt                              # Импорт библиотеки Matplolib
from pandas.plotting import register_matplotlib_converters   # Импорт модуля для нормального отображения индексов pandas на графике
import re                                                    # Импорт модуля Re для работы с регулярными выражениями


courses_df = pd.read_csv('../input/data-1/courses.csv')                      # Создание датафреймов
contents_df = pd.read_csv('../input/data-1/course_contents.csv')
progresses_df = pd.read_csv('../input/data-1/progresses.csv')
phases_df = pd.read_csv('../input/data-1/progress_phases.csv')
students_df = pd.read_csv('../input/data-1/students.csv')

progresses_df.rename(columns={'id':'progress_id'}, inplace = True)   # Переименование столбцов в датафреймах для объединения
courses_df.rename(columns={'id':'course_id'}, inplace = True)        # в один общий датафрейм

del courses_df['Unnamed: 0']                                         # Удаление столбца без имени, в котором содержатся
                                                                     # порядковые номера курсов.
main_df = phases_df.merge(progresses_df)                             # Объединение датафреймов в один общий датафрейм
main_df = main_df.merge(courses_df, how = 'left')
main_df = main_df.merge(contents_df, how = 'left')


main_df[main_df.student_id == '768c2987a744c51ce64a5993a2a94eaf']    # Проверка основного датасета
# Общее количество курсов в датасете

print(f'Количество курсов в датасете равно {main_df.course_id.nunique()}')
# Количество модулей на каждом курсе

main_df.groupby('title').agg(module_count = ('module_number','max'))
# Количество уроков в каждом модуле на каждом курсе

main_df.groupby(['title','module_title']).agg(lesson_count = ('lesson_number','max'))
# Медианное количество уроков в модуле на каждом курсе

main_df.groupby(['title']).agg(lesson_count_median = ('lesson_number','median'))
# Количество учеников на каждом курсе

main_df.groupby(['title']).agg(students_count = ('student_id', 'nunique'))
# Минимальный, максимальный, средний, медианный возраст студентов

students_df.dropna(inplace = True)                                                # Удаление строк с Nan-значениями
def age(x):                                                                       # Функция для расчета возраста студентов
    Age = (dt.now() - dt.strptime(str(x).replace('-','/'),"%Y/%m/%d")).days//365
    if 10 < Age < 80:
        return Age
    else:
        return None
students_df['age'] = students_df.loc[:,'birthday'].map(lambda x: age(x))          # Создание нового столбца 'age' содержащего



print(students_df.age.min())      # Минимальный возраст студентов
print(students_df.age.max())      # Максимальный возраст студентов
print(students_df.age.mean())     # Средний возраст студентов
print(students_df.age.median())   # Медианный возраст студентов
# Минимальный, максимальный, средний, медианный возраст студентов на каждом курсе.

students_df = students_df.rename(columns = {'id':'student_id'})     # Переименование столбцов в датафреймах для объединения
                                                                    # в один датафрейм
progresses_df.merge(courses_df).merge(students_df).groupby('title').agg(age_max = ('age', 'max'), # объединение в один датафрейм
                                                          age_min = ('age', 'min'),
                                                          age_mean = ('age', 'mean'),
                                                          age_median = ('age', 'median'))
#1.2 Bar-chart, отражающий количество студентов на каждом курсе.

%matplotlib inline 
# Строка для отображения графиков в текущем ноутбуке. 

graph_df = main_df.groupby('title').agg(students_count = ('student_id', 'nunique'))  # Создание нового датафрейма который будет
                                                                                     # использоваться для построения графика.
                                                                                     # Используется агрегирование для подсчета суммы уникальных id студентов
fig, subplot = plt.subplots()                                                        # Получение доступа к Figure и Subplot.
x_labels = [i for i in graph_df.index]                                 # Создание списка подписей для оси X.
subplot.bar(graph_df.index.values, graph_df['students_count'].values)  # Формирование столбчатой диаграммы.
plt.xticks(graph_df.index, rotation = 'vertical', labels = x_labels)   # Установка расположения подписей на оси Х.
plt.show()                                                             # Команда для отображения графика в ноутбуке.
#1.3 Горизонтальный bar-chart, отражающий количество студентов на каждом курсе. 

%matplotlib inline
# Строка для отображения графиков в текущем ноутбуке. 

graph_df.sort_values('students_count', inplace = True)                # Сортировка значений в датафрейме по количеству студентов
fig, subplot = plt.subplots()                                         # Получение доступа к Figure и Subplot.     
wedges = subplot.barh(graph_df.index.values,                          # Формирование горизинтальной стобчатой диаграммы
                      graph_df['students_count'].values,              
                      color = np.random.rand(15, 3),                  # Создание условия для окраски столбцов в рандомные цвета
                      label=graph_df.index.values.all(), alpha = 0.1) # Обозначение подписей по оси У и установка прозрачности графика
subplot.set_title('Количество студентов на каждом курсе')             # Установка заголовка графика
plt.box(on=None)                                                      # Команда для скрытия рамки графика
x = np.median(graph_df['students_count'])                             # Вычисление медианного значения количества студентов на каждом курсе
line = subplot.vlines(x,                                  
                      0,
                      1,
                      transform=subplot.get_xaxis_transform(),
                      color='red',
                      label = 'медиана')                              # Формирование линии медианы на графике 
plt.legend(wedges, graph_df.index,                                    # Команда для расположения легенды на графике
          loc="best",
          bbox_to_anchor=(1, 0.1, 0, 1))
plt.show()                                                            # Команда для отображения графика в ноутбуке.
#2.1 Рассчёт прироста студентов на каждом курсе в каждом месяце за всю историю 

# Форматирование строк даты в основном датафрейме
main_df['start_date'] = main_df.loc[:,'start_date'].map(lambda x: dt.strptime(re.split('[.+]',
                                                                                       str(x))[0],'%Y-%m-%d %H:%M:%S'))

# Создание нового датафрейма в котором только первые модули, т.к. студенты могут начать заниматься не с 1 урока.
graph_gr = main_df.copy()

# Форматирование даты в столбце дат начала выполнения заданий
graph_gr['start_date'] = main_df.loc[:,'start_date'].map(lambda x: dt.strftime(x,'%Y-%m'))

# Датафрейм содержащий все даты активностей на каждом курсе
dates_df = graph_gr.loc[:,['title',"start_date"]]
# Группировка дат по каждому курсу в датафрейме с датами
dates_df = dates_df.groupby(['title', 'start_date']).sum()
# Сброс индексов в названия столбцов для возможности дальнейшего объединения с датафреймом прироста
dates_df.reset_index(level=[0,1], inplace = True)

# Фильтр оставляющий в датафрейме прироста только первые модули и только домашние задания
graph_gr = graph_gr[(main_df['module_number'] == 1) & (main_df['is_homework'] == True)]

# Группировка с агрегированием выполняющим подсчет уникальных студентов
graph_gr = graph_gr.groupby(['title', 'start_date']).agg(growth = ('student_id', 'nunique'))

# Сброс индексов в названия столбцов для возможности дальнейшего объединения
graph_gr.reset_index(level=[0,1], inplace = True)

# Объединение датафреймов с приростом и с общим списком дат
graph_gr = dates_df.merge(graph_gr, how = 'left').fillna(0)

# Установка иерархических индексов
graph_gr.set_index(['title', 'start_date'], inplace = True)
graph_gr
#2.2 line-graph с приростом студентов в каждом месяце для каждого курса. 

%matplotlib inline
# Строка для отображения графиков в текущем ноутбуке.

fig = plt.figure(figsize = (10,150)) # Получение доступа к Figure и Subplot. Установка размеров Figure для удобного отображения
s = 0                                # Создание переменной для счетчика порядкового номера координатной оси
for i in list(graph_gr.index.levels[0]):  # Запуск цикла по каждому значению в списке курсов
    s+=1
    subplot = fig.add_subplot(15,1,s)               # Добавление координатной оси
    x_ticks = graph_gr.loc[i].index                 # Задание значений по оси Х
    y_ticks = graph_gr.loc[i,'growth']              # Задание значений по оси У
    x_labels = [i for i in graph_gr.loc[i].index]   # Создание списка подписей для оси X.
    subplot.plot(x_ticks, y_ticks)                  # Формирование линейного графика по заданным значениям
    subplot.set_title(i)                            # Установка подписи графика соглсно текущему значению из списка курсов
    plt.xticks(graph_gr.loc[i].index, rotation = 'vertical', labels = x_labels)  # Установка расположения подписей на оси Х.
plt.show()  # Команда для отображения графика в ноутбуке.
#2.3 На основании первого пункта построить line-graph с несколькими линиями, 
#отражающими прирост студентов в каждом месяце для каждого курса. 
%matplotlib inline
# Строка для отображения графиков в текущем ноутбуке.
# Для корректного отображения я сначала построил "пустой" график с разметкой по датам на всем диапазоне значений.
# затем на него уже "добавлялись" графики со значениями по каждому курсу.

fig,subplot = plt.subplots(figsize = (10,6))                   # Получение доступа к Figure и Subplot. Установка размеров Figure для удобного отображения
x_ticks = graph_gr.index.levels[1]                             # Установка значений по оси Х.
y_ticks = [0 for i in range(len(graph_gr.index.levels[1]))]    # Установка значений по оси У (линия вдоль оси Х)   
subplot.plot(x_ticks, y_ticks)                                 # Построение "пустого" графика

for i in list(graph_gr.index.levels[0]):             # Запуск цикла по каждому значению в списке курсов
    x_ticks = graph_gr.loc[i].index                  # Присвоение занчений по оси Х
    y_ticks = graph_gr.loc[i,'growth']               # Присвоение значений по оси У
    subplot.plot(x_ticks, y_ticks, label = i)                   # Формирование графика по данным соотв. курса
    x_labels = [i for i in graph_gr.index.levels[1]] # Список подписей для оси Х
    plt.xticks(graph_gr.index.levels[1],
               rotation = 'vertical',
               labels = x_labels)        # Нанесение подписей на график с соотв. параметрами
plt.box(on = None)                       # Команда для удаления рамки на графике
plt.grid(color='grey',
         linestyle='-',
         linewidth=1,
         alpha = 0.2)                    # Установка сетки на график для удобства чтения данных
plt.legend(loc="best",
          bbox_to_anchor=(1, 0.1, 0, 1))           # Команда для расположения легенды на графике
subplot.set_title('Прирост учащихся за все время') # Название графика
subplot.set_ylabel('Прирост учащихся, чел')        # Подпись оси У
plt.show()  # Команда для отображения графика в ноутбуке
#2.4 Рассчёт количества прогрессов по выполнению домашних работ в каждом месяце за всю историю 
# Создание нового датасета на основе общего
graph_done = main_df.copy()

# Фильтрация нового датасета в котором все задания являются домашними работами
graph_done = graph_done[(graph_done['is_homework']==True)]

# Форматирование столбца с датами начала работ в удобную форму
graph_done['start_date'] = graph_done.loc[:,'start_date'].map(lambda x: 
                                                                dt.strftime(dt.strptime(re.split('[.+]',str(x))[0],
                                                                                        '%Y-%m-%d %H:%M:%S'),"%Y-%m"))
# Функция форматирования дат окончаний работ. В случае Nan-значений фозвращает конечную дату в датасете
def dates_format(x):
    if isinstance(x,str) == True:
        date = dt.strftime(dt.strptime(re.split('[.+]',str(x))[0],'%Y-%m-%d %H:%M:%S'),"%Y-%m")
    else:
        date = '2019-07'
    return date

# Функция форматирования дат окончаний работ. В случае Nan-значений фозвращает конечную дату в датасете
graph_done['finish_date'] = graph_done.loc[:,'finish_date'].map(lambda x: dates_format(x))

graph_done = graph_done.loc[:,['title','start_date','finish_date']]
#graph_done.set_index(['title','start_date'], inplace = True)


graph_done.set_index(['title','start_date'], inplace = True)
graph_done.reset_index(inplace = True)
graph_done
for i in graph_done.index:
    date_range = pd.date_range(graph_done.loc[i,'start_date'],
                               graph_done.loc[i,'finish_date'], freq='M')
    date_range = list(map(lambda x: dt.strftime(dt.strptime(str(x),'%Y-%m-%d %H:%M:%S'),'%Y-%m'), date_range))[1:]
    for date in date_range:
        graph_done.loc[len(graph_done)] = [graph_done.loc[i,'title'],
                                            date,
                                            graph_done.loc[i,'finish_date']]
# Создание нового столбца-счётчика в датасете
graph_done['counter'] = 1
# Группировка датасета в котором счетчик подсчитывает сумму заданий законченных в текущем месяце по каждому курсу
graph_done = graph_done.groupby(['title', 'start_date']).sum()
graph_done
#2.5 line-graph по четвертому пункту.

%matplotlib inline
# Строка для отображения графиков в текущем ноутбуке
fig = plt.figure(figsize = (10,150))        # Получение доступа к Figure и Subplot. Установка размеров Figure для удобного отображения
s = 0                                       # Создание переменной для счетчика порядкового номера координатной оси
for i in list(graph_done.index.levels[0]):  # Запуск цикла по каждому значению в списке курсов
    s+=1
    subplot = fig.add_subplot(15,1,s)       # Добавление координатной оси
    x_ticks = graph_done.loc[i].index       # Задание значений по оси Х
    y_ticks = graph_done.loc[i,'counter']  # Задание значений по оси У
    x_labels = [i for i in graph_done.loc[i].index]  # Создание списка подписей для оси X.
    subplot.plot(x_ticks, y_ticks)                   # Формирование линейного графика по заданным значениям
    subplot.set_title(i)                             # Установка подписи графика соглсно текущему значению из списка курсов
    plt.xticks(graph_done.loc[i].index, rotation = 'vertical', labels = x_labels) # Установка расположения подписей на оси Х.
plt.show()  # Команда для отображения графика в ноутбуке
#2.6 line-graph для всех курсов по четвертому пункту.

%matplotlib inline
# Строка для отображения графиков в текущем ноутбуке.
# Для корректного отображения я сначала построил "пустой" график с разметкой по датам на всем диапазоне.
# затем на него уже "добавлялись" графики по каждому курсу.

fig,subplot = plt.subplots(figsize = (10,10))  # Получение доступа к Figure и Subplot. Установка размеров Figure для удобного отображения
x_ticks = graph_done.index.levels[1]           # Присвоение значений по оси Х.
y_ticks = [0 for i in range(len(x_ticks))]     # Присвоение значений по оси У (линия вдоль оси Х)   
subplot.plot(x_ticks, y_ticks)                 # Построение "пустого" графика

for i in graph_done.index.levels[0]:           # Запуск цикла по каждому значению в списке курсов
    y_ticks = graph_done.loc[i,'counter']     # Присвоение занчений по оси Х
    x_ticks = graph_done.loc[i].index          # Присвоение занчений по оси У
    graph = subplot.plot(x_ticks,
                         y_ticks, label = i) 
   # Формирование графика по данным соотв. курса

x_labels = [i for i in graph_done.index.levels[1]]  # Создание списка подписей для оси X.
plt.xticks(graph_done.index.levels[1],
           rotation = 'vertical',
           labels = x_labels)                       # Нанесение подписей на график с соотв. параметрами
subplot.set_title('График количества прогрессов по выполнению домашних работ в каждом месяце за всю историю') #Установка заголовка
subplot.set_ylabel('Число прогрессов, шт') # Подпись оси У
plt.legend()   # Команда для расположения легенды на графике 
plt.grid(color='grey',
         linestyle='-',
         linewidth=1,
         alpha = 0.2)                    # Установка сетки на график для удобства чтения данных
plt.box(on = None)                       # Команда для удаления рамки на графике
plt.show()                               # Команда для отображения графика в ноутбуке
#3.1 Рассчет минимального, максимального, среднего и медианного времени прохождения каждого модуля
problem_df = main_df.dropna() # Создание нового датасета без пропущенных значений

problem_df['finish_date'] = problem_df.loc[:,'finish_date'].map(lambda x: 
                                                                dt.strptime(re.split('[.+]',
                                                                                     str(x))[0],
                                                                            '%Y-%m-%d %H:%M:%S')) # Форматирование столбца дат
# Группировка датасета с агрегацией выполняющей создание новых столбцов с датами начала и окончания прохождения модулей
problem_df = problem_df.groupby(['title',
                                 'module_title',
                                 'module_number',
                                 'student_id']).agg(module_start = ('start_date','min'),
                                                 module_finish = ('finish_date','max'))
# Добавление столбца содержащего разность между датами начала и окончания прохождения модулей
problem_df['delta'] = problem_df['module_finish'] - problem_df['module_start']
# Форматирование столбца содержащего время прохождения модулей в количество секунд
problem_df['delta'] = problem_df.loc[:,'delta'].map(lambda x: x.days)
# Группировка датасета с агрегацией выполняющей создание новых столбцов с соответствующими функциями
problem_df_graph = problem_df.groupby(['title','module_title']).agg(min_time = ('delta', 'min'),
                                                             max_time = ('delta', 'max'),
                                                             mean_time = ('delta', 'mean'),
                                                             median_time = ('delta', 'median'))
problem_df_graph
#3.2 line-graph с медианным временем прохождения каждого модуля для каждого курса
%matplotlib inline
# Строка для отображения графика в текущем ноутбуке
fig = plt.figure(figsize = (10,150))                        # Получение доступа к Figure и Subplot. Установка размеров Figure для удобного отображения
s = 0                                                       # Создание переменной для счетчика порядкового номера координатной оси
for i in list(problem_df_graph.index.levels[0]):            # Запуск цикла по каждому значению в списке курсов
    s+=1                                                    
    subplot = fig.add_subplot(15,1,s)                       # Добавление координатной оси
    x_ticks = problem_df_graph.loc[i].index                 # Присвоение значений по оси Х
    y_ticks = problem_df_graph.loc[i,'median_time']         # Присвоение значений по оси У
    x_labels = [i for i in problem_df_graph.loc[i].index]   # Создание списка подписей для оси X
    subplot.plot(x_ticks, y_ticks)                          # Формирование линейного графика по заданным значениям
    subplot.set_title(i)                                    # становка подписи графика соглсно текущему значению из списка курсо
    plt.xticks(problem_df_graph.loc[i].index,
               rotation = 'vertical',
               labels = x_labels)        # Установка расположения подписей на оси Х
plt.subplots_adjust(wspace=1, hspace=3)  # Установка расстояния между графиками
plt.show()                               # Команда для отображения графика в ноутбуке
#3.3 Медианное время выполнения домашней работы по месяцам
# Форматирование столбца с датой выполнения задания в порядковый номер месяца
problem_df['module_finish'] = problem_df.loc[:,'module_finish'].map(lambda x: dt.strftime(x,'%m'))
# Группировка датасета по месяцам с агрегацией выполняющей создание новых столбцов с соответствующими функциями
problem_df_sea = problem_df.groupby(['title','module_finish']).agg(min_time = ('delta', 'min'),
                                                             max_time = ('delta', 'max'),
                                                             mean_time = ('delta', 'mean'),
                                                             median_time = ('delta', 'median'))
problem_df_sea
#3.4 line-graph, на который будут нанесены линии для каждого курса с медианным временем
%matplotlib inline
# Строка для отображения графиков в текущем ноутбуке
fig, subplot = plt.subplots(figsize = (10,10))     # Получение доступа к Figure и Subplot. Установка размеров Figure для удобного отображения
for i in problem_df_sea.index.levels[0]:           # Запуск цикла по каждому значению в списке курсов
    y_ticks = problem_df_sea.loc[i,'median_time']  # Присвоение значений по оси Х
    x_ticks = problem_df_sea.loc[i].index          # Присвоение значений по оси У
    subplot.plot(x_ticks, y_ticks, label = i)      # Формирование линейного графика по заданным значениям
plt.legend(loc = 'best',          
          bbox_to_anchor=(1, 0.1, 0, 1))           # Команда для расположения легенды на график
subplot.set_ylabel('Время выполнения, сек')        # Подпись оси У
subplot.set_title('Сезонность выполнения')         # Установка заголовка
plt.grid(color='grey',
         linestyle='-',
         linewidth=1,
         alpha = 0.2)   # Установка сетки на график 
plt.box(on = None)      # Команда для удаления рамки на графике
plt.show()              # Команда для отображения графика в ноутбуке
#4.1 Расчет конверсии перехода студентов из одного модуля в другой на каждом курсе

# Создание нового датасета из основного без пропущенных значений для подсчета студентов выполнивших задания
conv_df_done = main_df.copy()

# Выполнение условия по датасету в котором все уроки являются домашними заданиями
conv_df_done = conv_df_done[conv_df_done.loc[:,'is_homework'] == True]

# Выполнение условия по датасету в котором все домашние задания являются выполненными
conv_df_done = conv_df_done[conv_df_done['status'] == 'done']

# Группировка датасета по номеру урока в каждом модуле каждого курса с агрегацией подсчета суммы уникальных студентов
conv_df_done = conv_df_done.groupby(['title', 'module_number', 'module_title', 'lesson_number']).agg(count_stud = ('student_id',
                                                                                                        'nunique'))
# Конвертация индексов 4 уровней в названия столбцов
conv_df_done.reset_index(level=[0,1,2,3], inplace = True)

# Создание датасета в котором содержатся номера последних уроков каждого модуля
conv_df_last = conv_df_done.groupby(['title', 'module_number', 'module_title']).agg(lesson_number = ('lesson_number', 'max'))

# Создание нового датасета из основного для подсчета всех студентов с заданиями в любом статусе
conv_df_start = main_df.copy()

# Выполнение условия по датасету в котором все уроки являются домашними заданиями
conv_df_start = conv_df_start[conv_df_start.loc[:,'is_homework'] == True]

# Группировка датасета по номеру урока в каждом модуле каждого курса с агрегацией подсчета суммы уникальных студентов
conv_df_start = conv_df_start.groupby(['title', 'module_number', 'module_title', 'lesson_number']).agg(count_stud = ('student_id',
                                                                                                           'nunique'))
# Конвертация индексов 4 уровней в названия столбцов
conv_df_start.reset_index(level=[0,1,2,3], inplace = True)

# Создание датасета в котором содержатся номера первых домашних заданий каждого модуля
conv_df_first = conv_df_start.groupby(['title', 'module_number', 'module_title']).agg(lesson_number = ('lesson_number', 'min'))

# Конвертация индексов 3 уровней в названия столбцов
conv_df_first.reset_index(level=[0,1,2], inplace = True)

# Конвертация индексов 3 уровней в названия столбцов
conv_df_last.reset_index(level=[0,1,2], inplace = True)

# Объединение датасета содержащего последние уроки с датасетом студентов закончивших модуль
conv_df_done = conv_df_last.merge(conv_df_done, how = 'left')

# Переименование столбцов для корректного объединения
conv_df_done.rename(columns={'lesson_number': 'lesson_last', 'count_stud': 'count_stud_done'}, inplace=True)

# Объединение датасета содержащего первые первые домашние задания с датасетом студентов 
conv_df_start = conv_df_first.merge(conv_df_start, how = 'left')

# Переименование столбцов для корректного объединения
conv_df_start.rename(columns={'lesson_number': 'lesson_first', 'count_stud': 'count_stud_start'}, inplace=True)

# Объединение в один общий датасет
conv_df = conv_df_start.merge(conv_df_done, how = 'left')

# Удалиение столбцов с уроками
del conv_df['lesson_first']
del conv_df['lesson_last']

# Перемещение столбца с количеством студентов закончивших последний урок на кажом курсе на 1 позицию вверх
conv_df.count_stud_done = conv_df.count_stud_done.shift(1)

# Создание столбца содержащего конверсию студентов на каждом модуле
conv_df['conv'] = conv_df['count_stud_start'] / conv_df['count_stud_done']

# Установка индексов 2-х уровней в датасете
conv_df.set_index(['title','module_title'], inplace = True)

# Удаление строки с первым модулем на каждом курсе
for i in list(conv_df.index.levels[0]):
    conv_df.drop(index = conv_df.loc[i].index[0], level = 1, inplace = True)
    
# Заполнение Nan-значений значением 0
conv_df.fillna(0,inplace = True)
conv_df
#4.2 bar-chart, отражающий конверсию перехода студентов из одного модуля в другой на каждом курсе.
%matplotlib inline
# Строка для отображения графиков в текущем ноутбуке

fig = plt.figure(figsize = (10,150))     # Получение доступа к Figure и Subplot. Установка размеров Figure для удобного отображения
s= 0                                     # Создание переменной для счетчика порядкового номера координатной оси
for i in list(conv_df.index.levels[0]): # Запуск цикла по каждому значению в списке курсов
    s+=1
    subplot = fig.add_subplot(15,1,s)    # Добавление координатной оси
    x_labels = [j for j in conv_df.loc[i].index]          # Создание списка подписей для оси X
    wedges = subplot.bar(conv_df.loc[i].index,            # Формирование графика по заданным значениям
                          conv_df.loc[i,'conv'],
                          color = np.random.rand(15, 3),
                          label=x_labels, alpha = 0.1)
    subplot.set_title(i)                      # Установка подписи графика соглсно текущему значению из списка курсов
    plt.box(on=None)                          # Команда для удаления рамки с графика
    plt.legend(wedges, conv_df.loc[i].index,
              loc="best",
              bbox_to_anchor=(1, 0.1, 0, 1))  # Команда для установки легенды
    plt.xticks(conv_df.loc[i].index,
               rotation = 'vertical',
               labels = x_labels)             # Установка расположения подписей на оси Х
plt.subplots_adjust(wspace=0, hspace=2)       # Установка расстояний между графиками
plt.show()   # Команда для отображения графика в ноутбуке
#4.3 Горизонтальный bar-chart, отражающий конверсию перехода студентов из одного модуля в другой на каждом курсе.
%matplotlib inline
# Строка для отображения графиков в текущем ноутбуке

fig = plt.figure(figsize = (10,300))         # Получение доступа к Figure и Subplot. Установка размеров Figure для удобного отображения
s= 0                                         # Создание переменной для счетчика порядкового номера координатной оси                       
for i in list(conv_df.index.levels[0]):     # Запуск цикла по каждому значению в списке курсов
    s+=1
    subplot = fig.add_subplot(15,1,s)        # Добавление координатной оси
    x_labels = [j for j in conv_df.loc[i].index]         # Создание списка подписей для оси X
    wedges = subplot.barh(conv_df.loc[i].index,          # Формирование графика по заданным значениям
                          conv_df.loc[i,'conv'],
                          color = np.random.rand(15, 3),
                          label=x_labels, alpha = 0.1)
    subplot.set_title(i, fontsize=20)    # Установка подписи графика соглсно текущему значению из списка курсов
    plt.box(on=None)                     # Команда для удаления рамки с графика
    plt.legend(wedges, conv_df.loc[i].index,   # Команда для установки легенды
              loc="best",
              bbox_to_anchor=(1, 0.1, 0, 1), fontsize=18)
    x = np.median(conv_df.loc[i,'conv'])      #Вычисление медианного значения конверсии на каждом курсе
    line = subplot.vlines(x,                                  
                      0,
                      1,
                      transform=subplot.get_xaxis_transform(),
                      color='red',
                      label = 'медиана') # Формирование линии медианы на графике 
    plt.xticks(fontsize=25)    # Установка размера шрифта подписей на оси Х
    plt.yticks(fontsize=25)    # Установка размера шрифта подписей на оси У
plt.show()          # Команда для отображения графика в ноутбуке
# Создаем новый датасет на основе общего
problem_stud = main_df.copy()

# Выполнение условия по датасету в котором все уроки являются домашними заданиями
problem_stud = problem_stud[problem_stud.loc[:,'is_homework'] == True]

# Фильтруем новый датасет в котором будут статусы заданий только в качестве 'fail' и 'start'
problem_stud = problem_stud[(problem_stud.loc[:,'status'] == 'fail') | (problem_stud.loc[:,'status'] == 'start')]

# Агрегация датасета по курсам и id студентов с подсчетом статусов 'fail' и 'start'
problem_stud = problem_stud.groupby(['title', 'student_id']).agg(fail_count = ('status','count'))

# Фильтруем датасет в котором количество невыполненных заданий равно больше 2
problem_stud = problem_stud[problem_stud.loc[:,'fail_count'] > 2]
problem_stud
# Сброс индексов в датасете
problem_stud.reset_index([0,1], inplace = True)

# Агрегация датасета по курсам с подсчетом студентов
problem_stud = problem_stud.groupby('title').agg(count_students = ('student_id','count'))

# Сортировка датасета по убыванию
problem_stud.sort_values('count_students', ascending=False)