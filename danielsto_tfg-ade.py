# Librerías necesarias

import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np
# Forma rápida de crear una lista para el eje x con los años de 1995 a 2019

x = list(range(1995, 2020))



# Todos los datos son de diciembre menos para el año 2001 (agosto), 2002 (septiembre)

# y 2010 (septiembre).



# Número de usarios (millones)

y_a = [16, 36, 70, 147, 248, 361, 513, 587, 719, 817, 1018,

    1093, 1319, 1574, 1802, 1971 , 2267, 2497, 2802, 3079, 3366,

    3696, 4156, 4313, 4346]



# Porcentaje sobre población mundial

y_r = [0.4, 0.9, 1.7, 3.6, 4.1, 5.8, 8.6, 9.4, 11.1, 12.7,

      15.7, 16.7, 20.0, 23.5, 26.6, 28.8, 32.7, 35.7, 39.0, 42.4,

      46.4, 49.5, 54.4, 55.6, 56.1]
int_access = {}



for i, year in enumerate(x):

    int_access[year] = (y_a[i], y_r[i])



# int_access es un diccionario cuyas claves son los años y los valores

# se corresponden con una tupla de usuarios absolutos y relativos.
y_pos = np.arange(len(x))

plt.figure(figsize=(15,7))

barlist = plt.bar(y_pos, y_a, align='center', alpha=0.7, color='red')



for i, bar in enumerate(barlist):

    # Hasta 10%

    if i < 8:

        barlist[i].set_color('#ffa600')

    # Hasta 20%

    elif i < 12:

        barlist[i].set_color('#ff6e54')

    # Hasta 30%

    elif i < 16:

        barlist[i].set_color('#dd5182')

    # Hasta 40%

    elif i < 19:

        barlist[i].set_color('#955196')

    # Hasta 50%

    elif i < 22:

        barlist[i].set_color('#444e86')

    # Más de 50%

    else:

        barlist[i].set_color('#003f5c')



# for i, v in enumerate(y_r):

#     print(i, v)

#     plt.text(i-0.38, i, str(v) + '%', color='black', fontsize=8.5)



for i, rect in enumerate(barlist):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()/2.0, height+25, str(y_r[i])+"%", ha='center', va='bottom', fontsize=8)

        

plt.xticks(y_pos, x)

plt.ylabel('Número de usuarios (en millones)')

#plt.xlabel('Año')

plt.title('Acceso a Internet en el mundo (1995-2019)\nNúmero de usuarios y porcentaje de población mundial')



# plt.show()

plt.rcParams['text.color'] = "white"

plt.rcParams['axes.labelcolor'] = "white"

plt.rcParams['xtick.color'] = "white"

plt.rcParams['ytick.color'] = "white"

plt.savefig('demo.png', transparent=True)

plt.show()

# Forma rápida de crear una lista para el eje x con los años de 2012 a 2018

x = list(range(2012, 2019))

# Número de usuarios registrados en millones

users = [2, 10, 17, 35, 58, 81, 101]

# Número de universidades proveedoras

unis = [40, 200, 400, 500, 700, 800, 900]

# Número de cursos ofrecidos

courses = [250, 1200, 2400, 4200, 6850, 9400, 11400]

fig = plt.figure(figsize=(20,20)) 

fig.suptitle('MOOCs en cifras', fontsize=20, fontname='DejaVu Serif')



plt.subplot(311)

plt.gca().set_title('Evolución de los estudiantes registrados', fontname='DejaVu Serif')

plt.xlabel('Año', fontsize=18, fontname='DejaVu Serif')

plt.ylabel('Estudiantes (en millones)', fontsize=16, fontname='DejaVu Serif')

plt.plot(x, users, alpha=0.7, color='red', marker='o')



plt.subplot(312)

plt.gca().set_title('Evolución de las universidades participantes', fontname='DejaVu Serif')

plt.xlabel('Año', fontsize=18, fontname='DejaVu Serif')

plt.ylabel('Número de universidades', fontsize=16, fontname='DejaVu Serif')

plt.plot(x, unis, alpha=0.7, color='blue', marker='o')



plt.subplot(313)

plt.gca().set_title('Evolución de los cursos ofrecidos', fontname='DejaVu Serif')

plt.xlabel('Año', fontsize=18, fontname='DejaVu Serif')

plt.ylabel('Número de cursos', fontsize=16, fontname='DejaVu Serif')

plt.plot(x, courses, alpha=0.7, color='green', marker='o')
fig = plt.figure(figsize=(9,14)) 

fig.suptitle('MOOCs en cifras (2012-2018)', fontsize=15, fontname='DejaVu Serif')



fig.subplots_adjust(hspace=0.3)

plt.subplot(311)

plt.gca().set_title('Evolución de los estudiantes registrados', fontname='DejaVu Serif')

#plt.xlabel('Año', fontsize=18, fontname='DejaVu Serif')

plt.ylabel('Estudiantes (en millones)', fontsize=13, fontname='DejaVu Serif')

users_bar = plt.bar(x, users, alpha=0.6, color='#ffa600')



for i, rect in enumerate(users_bar):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()/2.0, height+0.2, str(users[i])+"M", ha='center', va='bottom', fontsize=8)





plt.subplot(312)

plt.gca().set_title('Evolución de las universidades participantes', fontname='DejaVu Serif')

#plt.xlabel('Año', fontsize=18, fontname='DejaVu Serif')

plt.ylabel('Número de universidades', fontsize=13, fontname='DejaVu Serif')

unis_bar = plt.bar(x, unis, alpha=0.6, color='#ff6e54')





for i, rect in enumerate(unis_bar):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()/2.0, height+0.2, str(unis[i]), ha='center', va='bottom', fontsize=8)





plt.subplot(313)

plt.gca().set_title('Evolución de los cursos ofrecidos', fontname='DejaVu Serif')

#plt.xlabel('Año', fontsize=18, fontname='DejaVu Serif')

plt.ylabel('Número de cursos', fontsize=13, fontname='DejaVu Serif')

courses_bar = plt.bar(x, courses, alpha=0.6, color='#dd5182')



for i, rect in enumerate(courses_bar):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()/2.0, height+0.2, str(courses[i]), ha='center', va='bottom', fontsize=8)