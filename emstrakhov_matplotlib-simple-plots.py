import numpy as np

import matplotlib.pyplot as plt
# простейший вариант без сохранения в файл

x = np.linspace(0, 10, 1000)

plt.plot(x, np.sin(x)) # (x, f(x))

plt.xlabel('x') # подпись к оси Ох

plt.ylabel('y') # подпись к оси Оу

plt.title(r'График $y=\sin x$'); # название графика
# По фен-шую

fig, ax = plt.subplots()

x = np.linspace(0, 10, 1000)

y = np.sin(x)

ax.plot(x, y)

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_title(r'График $y=\sin x$')

fig.savefig('sin.png', dpi=300); # dpi = dots per inch (разрешение картинки) 
img = plt.imread("sin.png") # считать рисунок

plt.imshow(img) # отобразить рисунок
# To see all of them: plt.plot?
plt.plot(x, np.sin(x - 0), color='blue')        # specify color by name

plt.plot(x, np.sin(x - 1), color='g')           # short color code (rgbcmyk)

plt.plot(x, np.sin(x - 2), color='0.75')        # Grayscale between 0 and 1

plt.plot(x, np.sin(x - 3), color='#FFDD44')     # Hex code (RRGGBB from 00 to FF)

plt.plot(x, np.sin(x - 4), color=(1.0, 0.2, 0.3)) # RGB tuple, values 0 to 1

plt.plot(x, np.sin(x - 5), color='chartreuse'); # all HTML color names supported
plt.plot(x, x + 0, linestyle='solid')

plt.plot(x, x + 1, linestyle='dashed')

plt.plot(x, x + 2, linestyle='dashdot')

plt.plot(x, x + 3, linestyle='dotted');
for lw in range(1, 6):

    plt.plot(x, x + lw, linewidth=lw)
x = np.arange(1, 11)

plt.plot(x, np.sin(x), marker='o')

plt.plot(x, np.cos(x), marker='v');
x = np.arange(1, 11)

plt.plot(x, np.sin(x), marker='o', markersize=10)

plt.plot(x, np.cos(x), marker='v', markersize=30);
fig, ax = plt.subplots()



x = np.linspace(0, 10, 200)

ax.plot(x, np.sin(x))



ax.set_xlabel(r'$x$') # latex-style

ax.set_ylabel(r'$y$');
fig, ax = plt.subplots()



x = np.linspace(0, 10, 200)

ax.plot(x, np.sin(x))



ax.set_title(r'Graph of $y=\int\cos x \, dx$');
fig, ax = plt.subplots()



x = np.linspace(0, 10, 200)

ax.plot(x, np.sin(x))



ax.set_xlim(-1, 11) # (min, max)

ax.set_ylim(-1.5, 1.5); # (min, max)
plt.plot(x, np.sin(x), label='sin(x)')

plt.plot(x, np.cos(x), label='cos(x)')

plt.axis('equal')



plt.legend();
plt.plot(x, np.sin(x), label='sin(x)')

plt.plot(x, np.cos(x), label='cos(x)')

plt.axis('equal')



plt.legend(loc=3); # по умолчанию: loc=0

'''

loc=0 – оптимальное положение

loc=1 – верхний правый угол

loc=2 – верхний левый угол

loc=3 – нижний левый угол

loc=4 – нижний правый угол

'''
fig, ax = plt.subplots()

ax.plot(x, np.sin(x))

ax.set(xlim=(0, 10), ylim=(-2, 2),

       xlabel='x', ylabel='sin(x)',

       title='A Simple Plot');
fig, ax = plt.subplots(figsize=(10, 5)) # 10 x 5 дюймов



ax.plot(x, np.sin(x));
import matplotlib as mpl

mpl.rcParams.update({'font.size':18, 'font.family':'serif'}) # параметры для всех надписей

# mpl.rcParams['lines.linewidth'] = 2

# mpl.rcParams['lines.color'] = 'r'

# ...

# mpl.rc('lines', linewidth=2, color='r')



x = np.linspace(0, 5, 10)

y = x ** 2



fig, ax = plt.subplots()

ax.plot(x, y)

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_title('title');