import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

#1. Загрузите по ссылке набор данных о погоде в Австралии и прилегающих территориях с 1900 по 2012 год. Подробнее об этих данных можно прочитать здесь.

#2. Сохраните данные в виде числового массива NumPy. Не забудьте, что данные содержат заголовок, который необходимо пропустить. Обратите также внимание, что

#в данных есть пропуски (NA-значения).

data = np.genfromtxt(r'../input/bomregions2012.csv', delimiter=',', skip_header=True)


#3. Изобразите динамику изменений средней температуры в Южной Австралии (столбец southAVt) в виде простого линейного графика. Добавьте названия осей:

#• Ox: Year;

#• Oy: Southern region average temperature (degrees C).

#4. Измените параметры предыдущего графика следующим образом:

#• тип линии: dotted; толщина: 3; цвет: красный;

#• отобразите легенду вверху слева (подпись: Southern temperature);

#• тип маркера: circle; цвет: чёрный; размер: 40

plt.plot(np.linspace(1910, 2012, 2012-1910), data[11:, 3], linestyle = 'dotted', color = 'red', linewidth = 3, label = 'Southern temperature', marker='o', markersize = 40, 

         markerfacecolor = 'black')

plt.xlabel('Year')

plt.ylabel('Southern region average temperature (degrees C)')

plt.legend(loc = 3);

#5. Изобразите на одном графике (в одной системе координат) динамику изменения

#средней температуры на западе (westAVt), востоке (eastAVt), севере (northAVt) и

#юге (southAVt). Параметры подберите самостоятельно. В легенде укажите названия

#соответствующих переменных. Добавьте названия осей координат.

plt.plot(np.linspace(1910, 2012, 2012-1910), data[11:, 1], label = 'east')

plt.plot(np.linspace(1910, 2012, 2012-1910), data[11:, 3], label = 'south')

plt.plot(np.linspace(1910, 2012, 2012-1910), data[11:, 5], label = 'west')

plt.plot(np.linspace(1910, 2012, 2012-1910), data[11:, 6], label = 'north')

plt.xlabel('Year')

plt.ylabel('Average temperature (degrees C)')

plt.legend();
#6. Нарисуйте scatter plot, чтобы показать соотношение концентрации углекислого газа

#в атмосфере (CO2)(-2) и средней температуры (auAVt)(8) в Австралии. Добавьте сетку (см.

#plt.grid()). Добавьте названия осей координат.

fig, ax = plt.subplots(figsize=(20, 7))

plt.scatter(data[11:, 8], data[11:, -2])

plt.xlabel('auAVt')

plt.ylabel('CO2')

plt.grid();
#7. Покажите соотношение между количеством осадков (auRain)(-4) и средней температурой (auAVt)(8), 

#а также динамику годового изменения этих показателей с помощью

#столбчатой диаграммы. Используйте два разных цвета.

fig, ax = plt.subplots(2, 1, figsize = (20, 7))

ax[0].bar([1, 2], height=[data[11:, -4].mean(), data[11:, 8].mean()], color=['red', 'blue'])

auRain  = data[11:, -4]

auAVt = data[11:, 8]

first = [i for i in range(1910, 2012)]

height = [auRain[i] + auAVt[i] for i in range(len(auRain))]

color = np.array([['black', 'yellow'] for i in range(len(auRain))]).ravel()

ax[1].bar(first, height = height, color = color)
#8. Как влияло количество солнечных пятен(-1) на среднегодовую температуру в западной(5)

#и восточной(1) частях Австралии? Придумайте подходящую визуализацию.

fig, ax = plt.subplots(figsize=(20, 7))

plt.scatter(data[11:, -1], data[11:, 5])

plt.scatter(data[11:, -1], data[11:, 1])

plt.xlabel('sunspot')

plt.ylabel('AVt')

#зависимости не наблюдается - при разных кол-вах солнечных пятен разбег температуры одинаковый, уплотнений нет
#9. Как распределялось количество осадков по регионам Австралии(9-17) в период с 1990 по

#2012 год? Подходящим типом визуализации будет график-пирог (см. plt.pie()).

#Также необходимы некоторые вычисления в массиве.

labels = 'east', 'se', 'south', 'sw', 'west', 'north', 'mbd', 'au'

sizes = [sum(data[90:, i])/sum([sum(data[90:, i]) for i in range(9, 17)]) for i in range(9, 17)]

print([i for i in range(9, 17)])

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

ax1.axis('equal') 
#10. В каком из регионов количество осадков и средняя температура коррелируют

#больше всего? Придумайте подходящую визуализацию.

regions = ['east', 'se', 'south', 'sw', 'west', 'north', 'mbd', 'au']

correl = abs(np.array([np.corrcoef(data[11:, i + 1], data[11:, i + 9])[0, 1] for i in range(8)]).ravel())

print(correl)

fig, ax = plt.subplots(figsize = (20, 7))



ax.bar(regions, height = correl)

#наибольший коэфициент - для показателей 'sw', 'mbd'