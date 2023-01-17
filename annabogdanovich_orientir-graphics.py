# ---------------------------------------
# Подключение необходимых модулей
# Внимание: не меняйте этот блок
# ---------------------------------------

import sys
assert sys.version_info[0]==3
assert sys.version_info[1] >= 5

import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

import numpy as np
import random

np.random.seed(0)
random.seed(0)
# ----------------
# Отобразить точки (1,6), (-1,4), (10,2) и пр.
X = [1,-1,10,4,6];
y = [6,4,2,1,5];

plt.scatter (X,y);
# Построить график функции y = 2x + 3 на интервале значений (-10,10)
X = np.arange (-10, 10);
y = 2*X + 3;

plt.plot (X,y);
data = np.loadtxt("../input/ex0data1-input/ex0data1",delimiter=',',skiprows=0)  

print(data[0:10]);

X = data[:,0]; y = data[:,1]; 

# Отображение загруженных данных в виде точек

# --- Ваш код -----

# -----------------

plt.scatter (X, y);

# Предположите гипотезу и подберите вручную оптимальные параметры

# --- Ваш код -----

# h = ....
# plt.plot (X, h, color = 'red');

# -----------------
