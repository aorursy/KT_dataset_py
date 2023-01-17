import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# gera uma população de 9999 com número de frequência para o tipo sanguíneo ---- xrange
pop_size=9999
pop=random.sample(range(0,9999),9999) 
pop2=[]
needs_organ=[]
pair_compatible=[]
pair_not_compatible=[]
cross_pair_matched=[]
donor_pool_matched=[]

#print(pop)
i=0
for i in range(0,9999):
    if pop[i] < 200:
        pop2.append('AB-')
        i+=1
    elif pop[i] < 400:
        pop2.append('AB+')
        i+=1
    elif pop[i] < 600:
        pop2.append('A-')
        i+=1
    elif pop[i] < 800:
        pop2.append('A+')
        i+=1
    elif pop[i] < 1000:
        pop2.append('B-')
        i+=1
    elif pop[i] < 3000:
        pop2.append('B+')
        i+=1
    elif pop[i] < 5000:
        pop2.append('O-')
        i+=1
    elif pop[i] < 9999:
        pop2.append('O+')
        i+=1
def split(A, n):
    return [A[i:i+n] for i in range(0, len(A), n)]

B=split(pop2,2)
#print(pop2)
#print(B)
i=0
while i<50:   
    needs_organ.append(random.choice(B))
    i+=1

print(needs_organ, 'precisam de orgão')

recbt={'AB-':['O-','AB-','B-','A-'],
       'AB+':['AB+','O-','A-','B-','O+','AB-','A+','B+'],
       'A-':['A-','O-'],
       'A+':['O-','A-','O+','A+'],
       'B-':['B-','O-'],
       'B+':['B-','B+','O+','O-'],
       'O-':['O-'],
       'O+':['O+','O-']}

def pair_compatibility_match(array):
    donor=array[i][0]
    rec=array[i][1]
    list_recbt=recbt[rec]
    if donor in list_recbt:
        pair_compatible.append(array[i])
        
    else:
        pair_not_compatible.append(array[i])
i=0
while i<50:
    pair_compatibility_match(needs_organ)
    i+=1

print (pair_compatible, 'Compatíveis')
print (pair_not_compatible, 'Incompatíveis')
def pair_wise_match(array):   
    donor1=array[j][0]
    rec1=array[j][1]
    donor2=array[i+1][0]
    rec2=array[i+1][1]
    list_recbt=recbt[rec1]
    if donor2 in list_recbt:
        list_recbt=recbt[rec2]
        if donor1 in list_recbt:
            cross_pair_matched.extend((array[j],array[i+1]))
            del array[j]
            del array[i]
i=0
j=0
while i<(len(pair_not_compatible)-1) and j<len(pair_not_compatible):
    pair_wise_match(pair_not_compatible)
    i+=1
    if i == (len(pair_not_compatible)-1):
        j+=1
        i=j

cross_pair_matched= split(cross_pair_matched,4)

print (cross_pair_matched, 'par cruzado correspondente')

pop3=random.sample(range(0,100),10)

donor_pool=[]
i=0
for i in range(0,10):
    if pop3[i] < 1:
        donor_pool.append('AB-')
        i+=1
    elif pop3[i] < 4:
        donor_pool.append('AB+')
        i+=1
    elif pop3[i] < 10:
        donor_pool.append('A-')
        i+=1
    elif pop3[i] < 44:
        donor_pool.append('A+')
        i+=1
    elif pop3[i] < 46:
        donor_pool.append('B-')
        i+=1
    elif pop3[i] < 55:
        donor_pool.append('B+')
        i+=1
    elif pop3[i] < 62:
        donor_pool.append('O-')
        i+=1
    elif pop3[i] < 100:
        donor_pool.append('O+')
        i+=1

print (donor_pool, 'Grupo de doadores')
def donor_pool_match(array1, array2):
    donor=array2[j]
    rec=array1[i][1]
    list_recbt=recbt[rec]
    if donor in list_recbt:
        donor_pool_matched.append(array1[i])
        del array2[j]
        del array1[i]
i=0
j=0
while i<(len(pair_not_compatible)) and j<len(donor_pool):
    donor_pool_match(pair_not_compatible, donor_pool)
    j+=1
    if j == (len(donor_pool)):
        i+=1

print(donor_pool_matched, 'Grupo de doadores compativéis')
print(pair_not_compatible,'Pares não compatíveis')

print('# pessoas que precisam de orgãos: ', len(needs_organ))

print('# de par de correspondencia: ', len(pair_compatible))

print('# de correspondencia cruzada: ', sum(len(x) for x in cross_pair_matched))

print('# de doador compatível: ', len(donor_pool_matched))

print('# de pessoa que não recebeu orgão: ', len(pair_not_compatible))