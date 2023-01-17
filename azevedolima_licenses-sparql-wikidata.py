import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# remove resultados ocorrências nulas
def not_null_column(arr):
    aux_arr = []
    for e in arr:
        if not pd.isnull(e):
            aux_arr.append(e)
    
    return aux_arr

# enumera ocorrências de um mesmo elemento e retorna um objeto com a contagem
def count_arr(arr):
    
    l = list(dict.fromkeys(arr))
    obj = dict(zip(l, [ 0 for e in l] ))
    for e in arr:
        obj[e] += 1
    
    #for e in obj.keys():
    #    obj[e] = [obj[e]]
        
    return obj

# calcula a porcentagem do objeto à cima segundo o tamanho da amostra
def get_percentage(obj, total):
    aux_obj = {}
    for e in obj.keys():
        aux_obj[e] = float("{:.2f}".format((100 * obj[e]) / total))
        
    return aux_obj
# carrega csv
dataframe = pd.read_csv('/kaggle/input/sparql/copyright.csv')

# issue: removes duplicates informations but might lose information concerning the diversity of programming languages related to an item
aux_dataframe = dataframe.drop_duplicates(subset=['item'])

# analisa-se a coluna copyrightLabel
arr = aux_dataframe.copyrightLabel
aux_arr = not_null_column(arr)
arr_copyrights = not_null_column(aux_dataframe.copyrightLabel)
count_copyrights = count_arr(arr_copyrights)
sorted_copyrights = sorted(count_copyrights.items(), key=lambda x: x[1], reverse=True)
pd.DataFrame(dict(sorted_copyrights), index=[0]).plot(kind='bar', figsize=(50, 40))
dict(sorted_copyrights)
# 10705 resultados nulos
print([aux_dataframe.copyrightLabel.__len__(), "-" , arr_copyrights.__len__(), "=", aux_dataframe.copyrightLabel.__len__() - arr_copyrights.__len__()])
arr_copyrights.__len__()

get_percentage(dict(sorted_copyrights), arr_copyrights.__len__())

import json

arr_origins = not_null_column(aux_dataframe.originLabel)
aux_arr_origins = list(dict.fromkeys(arr_origin))
        
#dict.fromkeys(arr_origins)
count = count_arr(arr_origins)
sorted_origins = sorted(count.items(), key=lambda x: x[1], reverse=True)

pd.DataFrame(dict(sorted_origins), index=[0]).plot(kind='bar', figsize=(18, 16))
dict(sort_orders)