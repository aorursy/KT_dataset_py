"""
------------------------------------------------------------------------------------
--------------------------BASSEM SOLIMAN-------------21-6-2018----------------------
------------------------------------------------------------------------------------

****this algorithm to calculate the similarity function between two names****

    Distance functions map a pair of strings s and t to a real
    number r, where a smaller value of r indicates greater similarity
    between s and t. Similarity functions are analogous,
    except that larger values indicate greater similarity 

there are several algorithms to calculate distance function (similarity function) :
    - Levenstein distance function
    - Jaro metric
    - Jaro-Winkler
    - Jaccard similarity
    - Hybrid distance functions ( which i will use in this algorithm)

*** Hybrid distance functions
    Monge and Elkan propose the following recursive matching
    scheme for comparing two long strings s and t. First, s and
    t are broken into substrings s = a1 : : : aK and t = b : : : bL.
    Then, similarity is defined as
    
            sim(s; t) = 1/k * (sum(max(sim0(Ai;Bj)))) 
            where sim0 is some secondary distance function 
            
     i will use here jaro as a secondary function
             
             sim0(Ai;Bj) = Jaro(s;t) = 1/3*(len(s')/len(s)+len(t')/len(t)+(len(s')-len(T'))/len(s))
             
*** Data set
    This dataset is a listing of all current City of Chicago employees,
    complete with full names, departments, positions, employment status 
    i will use only the name of imployee to calculate the similiarity
    between first 4 namesand the first 100 names in this data set to reduce run time  

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
 
 # -----------------------------------we will make a new dataframe (name list ) which 
 #-----------------------------------only contain list of first 100 names and another
 # ----------------------------------4 colomn for the first 4 nameto calculate the similarity between them
 

train = pd.read_csv('../input/Current_Employee_Names__Salaries__and_Position_Titles.csv', nrows= 100)
#train.head()
namelist = train[['Name']]
v = len(namelist)
s =  4
namelist['sim_name_1'] = 0
namelist['sim_name_2'] = 0
namelist['sim_name_3'] = 0
namelist['sim_name_4'] = 0

# --------------------------------- cleaning the first 100 names and save it in name_1

name_1 = []
for n in range(v) :
    z = [] 
    z.append(namelist.loc[n, 'Name'])
    zz = z[0].replace(', ' , '')
    name_1.append(zz.strip().split(' '))
    
# --------------------------------- cleaning the first 4 names and save it in name_2

name_2 = []
for t in range(s) :
    z2 = [] 
    z2.append(namelist.loc[t, 'Name'])
    zz2 = z2[0].replace(', ' , '')
    name_2.append(zz2.strip().split(' '))
    
# --------------------------------- print the two list
    
print('name_1  :  ', name_1)
print('name_2  :  ', name_2)
# --------------------------define function to calculate jaro distance function
def jaro(s, t):
    s_len = len(s)
    t_len = len(t)
 
    if s_len == 0 and t_len == 0:
        return 1
    if s_len == 1 and t_len == 1:
        if s == t :
            return 1
        else :
            return 0
            
 
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
            #print(s[i],t[k])
            transpositions += 1
        k += 1
 
    return ((matches / s_len) +
            (matches / t_len) +
            ((matches - transpositions/2) / matches)) / 3

#------------------------------------------------ loops to calculate Hybrid distance functions

result = []
for a in name_1 :
    for b in name_2 :
        
        h = []
        for i in a :
            ja = []
            for j in b :
                ja.append(jaro(i,j))
            h.append(max(ja))
        simi =   sum(h)/len(a)
        #print(len(a), h, a,b)
        result.append(simi)
#print( ' ja = ' , ja ,'---', ' h = ', h  )
#print('similarity =  ', result )


# ---------------------------------loops to allocate each similarity function in namelist dataframe
g = 0 
for p in range(v) :
    for c in ['sim_name_1','sim_name_2','sim_name_3','sim_name_4'] :
        namelist.loc[p,c] = result[g]
        g = g+1
namelist
