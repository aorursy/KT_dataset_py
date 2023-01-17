import numpy as np
x = np.array([1, -2, 0.5])

y = np.array([x**2 for x in range(1, 11)])

print('x =', x, 'y =', y)
A = np.array([[1, 2], [2, -1]])

A # в режиме блокнота для вывода объекта на экран не обязательно использовать print()
a = np.zeros(10) # одномерный массив из нулей длины 10

Z = np.zeros((5, 5)) # двумерный массив (матрица) 5х5 из нулей

b = np.ones(5) # массив из единиц длины 5
I = np.eye(3) # единичная матрица 3х3

D = np.diag([1, 2, 3]) # диагональная матрица

n = np.repeat(3, 4) # массив из троек длины 4

seq = np.arange(1, 11) # то же самое, что range(1, 11)

grid = np.linspace(1, 10, 50) # массив из 50 точек разбиения отрезка [1; 10] с равномерным шагом
# Выборка из нормального распределения со средним 0, стандартным отклонением 5 и длиной 10

x = np.random.normal(0, 5, 10) 

# Выборка из равномерного распределения от 10 до 20 длиной 10

y = np.random.uniform(10, 20, 10)

# Массив случайных целых чисел от 5 до 100 длины 15

z = np.random.randint(5, 100, 15)

# Выборка из нормального распределения со средним 0, стандартным отклонением 5 и размером 5х5

x = np.random.normal(0, 5, (5, 5)) 
# Одномерный массив

x = np.random.randint(-10, 10, 20)

print(x.size) # количество элементов

print(x.ndim) # количество размерностей (измерений)

print(x.shape) # форма (размерность) массива

print(x.dtype) # тип элементов массива
# Двумерный массив

A = np.random.normal(4, 2.5, (5, 4))

print(A.size, A.ndim, A.shape, A.dtype)

print(A.shape[0]) # количество строк

print(A.shape[1]) # количество столбцов
x = np.random.randint(1, 50, 10) # вектор длины 10

y = np.ones(10) # вектор длины 10

A = np.random.uniform(0, 10, (10, 10)) # матрица 10х10

print(x)

print(x + 1) # добавляем 1 к каждому элементу

print(x * (-1)) # умножаем каждый элемент на -1

print(1 / x) # вычисляем массив обратных элементов

print(np.cos(x))

print(np.exp(-1/x))

print(x * y) # поэлементное умножение

print(x.dot(y)) # скалярное произведение

print(A - 5)

print(np.sin(A)**2 + np.cos(A)**2)
x = np.random.randint(150, 190, 20)

print(x)



print(np.sum(x)) # сумма

print(np.max(x), np.min(x)) # минимум, максимум

print(np.argmax(x), np.argmin(x)) # номер (индекс) максимального и минимального элемента

print(np.mean(x)) # среднее

print(np.median(x)) # медиана

print(np.std(x)) # стандартное отклонение



import scipy.stats as sps

print(sps.mode(x)) # мода
A = np.random.randint(0, 10, (3, 3))

B = np.random.randint(0, 10, (3, 3))

z = np.random.randint(0, 10, 3)



print(A.dot(z)) # умножение матрицы на вектор

print(np.linalg.norm(z)) # норма вектора

print(np.linalg.norm(A)) # норма матрицы, см. https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html

print(A.dot(B)) # произведение матриц в алгебраическом смысле

print(A * B) # поэлементное умножение

print(np.linalg.inv(A)) # обратная матрица

print(np.linalg.det(A)) # определитель
A = np.random.randint(-10, 10, (5, 5))

print(A)



print(np.sum(A, axis=1)) # сумма по строкам

print(np.mean(A, axis=0)) # среднее по столбцам

print(np.sum(A)) # сумма всех элементов
x = np.arange(10)

print(x)

print(x[5], x[-1]) # как для списков



A = np.eye(5)

print(A)

print(A[0, 0], A[-1, 0]) # индексы перечисляются через запятую

A[2, 3] = -1 # изменение элемента
print(x[:3], x[-5:], x[::-1], x[2:7:2]) # как для списков

print(A[1, :]) # строка матрицы с индексом 1

print(A[:, 0]) # столбец с индексом 0

print(A[:2, :2]) # верхняя левая подматрица размером 2х2

print(A[:, 1:4]) # столбцы матрицы с 1 по 4 (не включая 4)

print(A[::-1, ::-1]) # столбцы и строки "задом наперед"
# Условные индексы для вектора

x = np.random.randint(-10, 10, 10)

print(x)



print(x > 0) # поэлементное применение условия: возвращает массив True/False-значений

print(x[x > 0]) # только те элементы из x, которые больше 0

print(x[(x > 0) & (x < 5)]) # пересечение условий (and)

print(x[(x < -5) | (x >= 7)]) # объединение условий (or)

print(x[~(x > 0)]) # отрицание условия (not)
# Условные индексы для матрицы

A = np.random.normal(0, 1.5, (5, 4))

print(A)

print()



print(A[A[:, 0]>0, :]) # те строки, у которых первый элемент положителен

print()

print(A[(A[:, 0]>0) & (A[:, 1]<0.5), :]) # строки, в которых значения в первом столбце больше 0, а во втором меньше 0.5

print()

print(A[:, np.mean(A, axis=0)>0.5]) # столбцы, среднее значение в которых больше 0.5
x = np.arange(1, 10)

print(np.any(x > 5)) # есть ли хотя бы один элемент больше 5?

print(np.all(x > 5)) # верно ли, что все элементы больше 5?
x = np.array([1, 2, 3]) # массив размерности 1

print(x.ndim, x.shape)



y = x[:, np.newaxis] # вектор-столбец размерности 3х1

print(y.ndim, y.shape)



z = x[np.newaxis, :] # вектор-строка размерности 1х3

print(z.ndim, z.shape)
x = np.arange(25) # одномерный массив длины 25

A = x.reshape((5, 5)) # матрица 5х5

print(A)
x = np.array([1, 2, 3])



y = np.append(x, 4) # добавление одного элемента "по горизонтали"

print(y)



z = np.append(x, [3, 2, 1]) # добавление целого массива "по горизонтали"

print(z)
A = np.arange(24).reshape((4, 6))

B = np.random.randint(15, size=(4, 4))

C = np.random.randint(15, size=(6, 6))

print(A)

print()

print(B)

print()

print(C)

print('-'*50)



D = np.hstack([A, B]) # добавление матрицы B "справа" к матрице A

print(D)

print()

E = np.vstack([A, C]) # добавление матрицы C "снизу" к матрице A

print(E)
A = np.arange(24).reshape((4, 6))

print(A)

print()

x = np.zeros(6)

y = np.zeros(4)

print(x, y)

print('-'*50)



B1 = np.vstack([A, x[np.newaxis, :]]) # добавление строки

print(B1)

print()

B2 = np.append(A, x[np.newaxis, :], axis=0) # альтернативный способ

print(B2)

print()

print(np.all(B1 == B2)) # совпали ли матрицы?

print()



C1 = np.hstack([y[:, np.newaxis], A]) # добавление столбца

print(C1)

print()

C2 = np.append(y[:, np.newaxis], A, axis=1) # альтернативный способ

print(C2)

 
x = np.array([1, 2, 3])

y = np.delete(x, 1) # удаляем элемент по индексу, при этом сам х не меняется

print(x, y)

print()



A = np.random.normal(size=(5, 5))

print(A)

print()



B = np.delete(A, 3, axis=0) # удаление строки по индексу

print(B)

print()

C = np.delete(A, [1, 4], axis=1) # удаление столбцов по индексу

print(C)
x = np.random.randint(-10, 10, 15)

print(x)

y = np.sort(x) # сортировка одномерного массива

print(y)

print()



A = np.random.randint(-10, 10, (3, 4))

print(A)

print()

B = np.sort(A, axis=0) # сортировка по строкам

print(B)

print()

C = np.sort(A, axis=1) # сортировка по столбцам

print(C)

df = np.loadtxt('../input/cardio_train.csv', delimiter=';', skiprows=1)

print(df[:5, :])

print(df.shape)
x = np.random.normal(loc=5, scale=1.5, size=20000).reshape((1000, 20))

np.savetxt('array.csv', x, delimiter=';')