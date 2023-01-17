import pandas as pd
import numpy as np
import time
# вводим матрицы
matrix_a = [
    [1, 2, 3], 
    [4, 5, 6]
]
matrix_b = [
    [7, 10, 13, -1], 
    [8, 11, 14, -2], 
    [9, 12, 15, -3]
]

matrix_c = [
    [1, 2],
    [4, 5]
]
matrix_d = [
    [7, 10, 13, -1],
    [8, 11, 14, -2]
]

# кол-во экспериментов
exp_num = 10001
def standart_matrix_multiply(A, B):
    if len(A[0]) != len(B):
      print("Ошибка! Невозможно перемножить матрицы, т.к. они не совместимы.")
      return
    
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    
    # Создаем матрицу результатов C[rows_A x cols_B] (C[MxQ])
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
                
    return C
print('Стандартный алгоритм, матрицы A, B')
start = time.process_time()
for i in range(exp_num - 1):
    standart_matrix_multiply(matrix_a, matrix_b)
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed / exp_num} сек')
print(f"Результат умножения: \n{np.matrix(standart_matrix_multiply(matrix_a, matrix_b))}")
print('Стандартный алгоритм, матрицы C, D')
start = time.process_time()
for i in range(exp_num - 1):
    standart_matrix_multiply(matrix_c, matrix_d)
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed / exp_num} сек')
print(f"Результат умножения: \n{np.matrix(standart_matrix_multiply(matrix_c, matrix_d))}")
#  A[M * N] * B [N * Q]
def vinograd_matrix_multiply(A, B):
    if len(A[0]) != len(B):
      print("Ошибка! Невозможно перемножить матрицы, т.к. они не совместимы.")
      return
    
    rows_A = len(A)     # a  M
    cols_A = len(A[0])  # b  N
    rows_B = len(B)     # b  N
    cols_B = len(B[0])  # c  Q
    
    # Создаем матрицу результатов C[rows_A x cols_B] (C[MxQ])
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]

    # вычисление row_factors для A
    row_factor_A = [0 for col in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_A // 2):
            row_factor_A[i] = row_factor_A[i] + A[i][j * 2] * A[i][j * 2 + 1]
    
    # вычисление column_factors для B
    col_factor_B = [0 for row in range(cols_B)]
    for i in range(cols_B):
        for j in range(cols_A // 2):
            col_factor_B[i] = col_factor_B[i] + B[j * 2][i] * B[j * 2 + 1][i]
    
    # заполнение матрицы C
    for i in range(rows_A):
        for j in range(cols_B):
            C[i][j] = -row_factor_A[i] - col_factor_B[j]
            for k in range(cols_A // 2):
                C[i][j] = C[i][j] + (A[i][2 * k] + B[2 * k + 1][j]) * (A[i][2 * k + 1] + B[2 * k][j])
    
    # прибавление членов в случае нечетной общей размерности
    if cols_A % 2 == 1:
        for i in range(rows_A):
            for j in range(cols_B):
                C[i][j] = C[i][j] + A[i][cols_A - 1] * B[cols_A - 1][j]
                
                
    return C
print('Алгоритм Винограда базовый, матрицы A, B')
start = time.process_time()
for i in range(exp_num - 1):
    vinograd_matrix_multiply(matrix_a, matrix_b)
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed / exp_num} сек')
print(f"Результат умножения: \n{np.matrix(vinograd_matrix_multiply(matrix_a, matrix_b))}")
print('Алгоритм Винограда базовый, матрицы C, D')
start = time.process_time()
for i in range(exp_num - 1):
    vinograd_matrix_multiply(matrix_c, matrix_d)
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed / exp_num} сек')
print(f"Результат умножения: \n{np.matrix(vinograd_matrix_multiply(matrix_c, matrix_d))}")
# k * 2 ~ k << 1
def optimized_vinograd_matrix_multiply(A, B):
    if len(A[0]) != len(B):
        print("Ошибка! Невозможно перемножить матрицы, т.к. они не совместимы.")
        return
    
    rows_A = len(A)     # a  M
    cols_A = len(A[0])  # b  N
    rows_B = len(B)     # b  N
    cols_B = len(B[0])  # c  Q
    
    # Создаем матрицу результатов C[rows_A x cols_B] (C[MxQ])
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]

    # вычисление row_factors для A
    row_factor_A = [0 for col in range(rows_A)]
    for i in range(rows_A):
        for j in range(1, cols_A, 2):
            row_factor_A[i] -= A[i][j - 1] * A[i][j]
            
    # вычисление column_factors для B
    col_factor_B = [0 for row in range(cols_B)]
    for i in range(cols_B):
        for j in range(1, cols_A, 2):
            col_factor_B[i] -= B[j - 1][i] * B[j][i]
            
    # заполнение матрицы C
    for i in range(rows_A):
        for j in range(cols_B):
            buf = row_factor_A[i] + col_factor_B[j]
            for k in range(1, cols_A, 2):
                buf += (A[i][k - 1] + B[k][j]) * (A[i][k] + B[k - 1][j])
            C[i][j] = buf
                
    # прибавление членов в случае нечетной общей размерности
    if cols_A % 2 == 1:
        for i in range(rows_A):
            for j in range(cols_B):
                C[i][j] += A[i][cols_A - 1] * B[cols_A - 1][j]


    return C
print('Алгоритм Винограда оптимизированный, матрицы A, B')
start = time.process_time()
for i in range(exp_num - 1):
    optimized_vinograd_matrix_multiply(matrix_a, matrix_b)
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed / exp_num} сек')
print(f"Результат умножения: \n{np.matrix(optimized_vinograd_matrix_multiply(matrix_a, matrix_b))}")
print('Алгоритм Винограда оптимизированный, матрицы C, D')
start = time.process_time()
for i in range(exp_num - 1):
    optimized_vinograd_matrix_multiply(matrix_c, matrix_d)
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed / exp_num} сек')
print(f"Результат умножения: \n{np.matrix(optimized_vinograd_matrix_multiply(matrix_c, matrix_d))}")
ma = np.matrix(matrix_a)
mb = np.matrix(matrix_b)
mc = np.matrix(matrix_c)
md = np.matrix(matrix_d)

display(ma, mb, mc, md)
print('Библиотека NumPy, матрицы A, B')
start = time.process_time()
for i in range(exp_num - 1):
    ma * mb
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed / exp_num} сек')
print(f"Результат умножения: \n{ma * mb}")
print('Библиотека NumPy, матрицы C, D')
start = time.process_time()
for i in range(exp_num - 1):
    mc * md
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed / exp_num} сек')
print(f"Результат умножения: \n{mc * md}")