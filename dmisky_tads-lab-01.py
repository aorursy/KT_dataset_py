import pandas as pd
import numpy as np
import time
import nltk
word_a = 'увлечение'
word_b = 'развлечение'
exp_num = 100  # кол-во экспериментов
def levenshtein(a,b):
    '''
    Сравнение двух строк, вывод матрицы и расстояния
    '''
    size_x=len(a)+1
    size_y=len(b)+1
    matrix=np.zeros((size_x,size_y))
    for x in np.arange(size_x):
        matrix[x,0]=x
    for y in np.arange(size_y):
        matrix[0,y]=y
    for x in np.arange(1,size_x):
        for y in np.arange(1,size_y):
            if a[x-1]==b[y-1]:
                matrix[x,y]=min(
                    matrix[x-1,y]+1,
                    matrix[x-1,y-1],
                    matrix[x,y-1]+1
                )
            else:
                matrix[x,y]=min(
                    matrix[x-1,y]+1,
                    matrix[x-1,y-1]+1,
                    matrix[x,y-1]+1
                )
                
#     df=pd.DataFrame(matrix.astype(int))
#     coln=[x for x in range(1,len(b)+1)]
#     cola=[x.upper() for x in list(b)]
#     rown=[x for x in range(1,len(a)+1)]
#     rowa=[x.upper() for x in list(a)]
#     df=df.rename(columns=dict(zip(coln,cola)), index=dict(zip(rown,rowa)))
#     print(df)

    return int(matrix[size_x-1,size_y-1])
start = time.process_time()
for i in range(exp_num - 1):
    levenshtein(word_a, word_b)
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed/exp_num} сек')
print(f"Редакционное расстояние: {levenshtein(word_a, word_b)}")
levenshtein(word_a, word_b)
def levenshtein_rec(a, b):
    if not a: 
        return len(b)
    if not b: 
        return len(a)
    
    return min(levenshtein_rec(a[1:], b[1:]) + (a[0] != b[0]),
               levenshtein_rec(a[1:], b) + 1,
               levenshtein_rec (a, b[1:]) + 1
              )
start = time.process_time()
for i in range(exp_num - 1):
    levenshtein_rec(word_a, word_b)
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed/exp_num} сек')
print(f"Редакционное расстояние: {levenshtein_rec(word_a, word_b)}")
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1
 
    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion / удаление
                           d[(i,j-1)] + 1, # insertion / вставка
                           d[(i-1,j-1)] + cost, # substitution / замена
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition 
 
    return d[lenstr1-1,lenstr2-1]
start = time.process_time()
for i in range(exp_num - 1):
    damerau_levenshtein_distance(word_a, word_b)
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed/exp_num} сек')
print(f"Редакционное расстояние: {damerau_levenshtein_distance(word_a, word_b)}")
# Levenshtein distance of two strings (CHEAT)

start = time.process_time()
for i in range(exp_num - 1):
    nltk.edit_distance(word_a, word_b)
elapsed = (time.process_time() - start)
print(f'Время выполнения {exp_num} экспериментов: {elapsed} сек')
print(f'Среднее время одного эксперимента: {elapsed/exp_num} сек')
print(f"Редакционное расстояние: {nltk.edit_distance(word_a, word_b)}")
