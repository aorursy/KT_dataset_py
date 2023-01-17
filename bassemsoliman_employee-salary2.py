import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
 
train = pd.read_csv('../input/Current_Employee_Names__Salaries__and_Position_Titles.csv', nrows= 20)
#train.head()
namelist = train[['Name']]
v = len(namelist)
s =  3
#namelist['split1'] = 0
#namelist['split2'] = 0
#namelist['split3'] = 0
name_1 = []
for n in range(v) :
    name_1.append(namelist.loc[n, 'Name'].split(',')) 

name_2 = []
for t in range(s) :
    name_2.append(namelist.loc[t, 'Name'].split(',')) 
    
print(name_1,name_2 )
def jaro(s, t):
    s_len = len(s)
    t_len = len(t)
 
    if s_len == 0 and t_len == 0:
        return 1
 
    match_distance = (max(s_len, t_len) // 2) - 1
 
    s_matches = [False] * s_len
    t_matches = [False] * t_len
    #print(s_matches , t_matches, 'done')
    #print( s_len,'------',range(s_len))
    matches = 0
    transpositions = 0
 
    for i in range(s_len):
        start = max(0, i-match_distance)
        end = min(i+match_distance+1, t_len)
        #print(start , end, match_distance)
        for j in range(start, end):
            if t_matches[j]:
                continue
            if s[i] != t[j]:
                continue
            s_matches[i] = True
            t_matches[j] = True
            matches += 1
            break
 
    if matches == 0:
        return 0
 
    k = 0
    for i in range(s_len):
        if not s_matches[i]: # 0001
            continue
        while not t_matches[k]: ### 001
            k += 1
        if s[i] != t[k]:
            print(s[i],t[k])
            transpositions += 1
        k += 1
 
    return ((matches / s_len) +
            (matches / t_len) +
            ((matches - transpositions/2) / matches)) / 3
result = []
for a in name_1 :
    for b in name_2 :
        
        h = []
        for i in a :
            ja = []
            for j in b :
                ja.append(jaro(i,j))
            h.append(max(ja))
        simi = 1/2 * sum(h)
        result.append(simi)
#print( ' ja = ' , ja ,'---', ' h = ', h  )
print('similarity =  ', result )
    

