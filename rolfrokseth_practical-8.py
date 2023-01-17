#Modified code from

#http://blog.chapagain.com.np/hash-table-implementation-in-python-data-structures-algorithms/



#hash_table = [None] * 11

hash_table = [[] for _ in range(11)] #To avoid collisions by chaining we need a nested list





def insert(hash_table,value):

    hash_key = (3*value+5) % 11 #Function to set the key

    #hash_table[hash_key] = value

    hash_table[hash_key].append(value) 

 

insert(hash_table, 12)

insert(hash_table, 44)

insert(hash_table, 13)

insert(hash_table, 88)

insert(hash_table, 23)

insert(hash_table, 94)

insert(hash_table, 11)

insert(hash_table, 39)

insert(hash_table, 20)

insert(hash_table, 23)

insert(hash_table, 16)

insert(hash_table, 5)





print("""Using the hash function hi(i) = (3i+5) mod 11 to hash the keys given, 

handling collision by chaining, the hash table will look like this\n""")

print (hash_table)
#Handling collison using linear probing, we will look for the next available key



#hash_tableLP = [None] * 11

hash_tableLP = [[] for _ in range(11)] #To avoid collisions by chaining we need a nested list





def insertLP(hash_tableLP,value):

    hash_key = (3*value+5) % 11 #Function to set the key

    key_exists = False

    #bucket = hash_tableLP[hash_key]

    for i, kv in enumerate(hash_tableLP):

        if hash_key == i and kv != []:

            key_exists = True

            break

    if key_exists:

        start = hash_key

        pos = 0

        filled = 0

        for t, kt in enumerate(hash_tableLP):

            if start == t:

                pos = 1

            if kt == [] and pos == 1:

                hash_tableLP[t].append(value) 

                print("Value -",value,",Collision!, ","Key -", hash_key, "In use,","New key -", t)

                filled = 1

                break

        while filled != 1:

            for t, kt in enumerate(hash_tableLP):

                if kt == []:

                    hash_tableLP[t].append(value) 

                    print("Value -",value,",Collision!, ","Key -", hash_key, "In use,","New key -", t, "2nd")

                    filled = 1

                    break

                

    else:

        print("Value -",value,"Hash key -" , hash_key)

        hash_tableLP[hash_key].append(value) 

    #hash_table[hash_key] = value

    #hash_tableLP[hash_key].append(value) 

 

insertLP(hash_tableLP, 12)

insertLP(hash_tableLP, 44)

insertLP(hash_tableLP, 13)

insertLP(hash_tableLP, 88)

insertLP(hash_tableLP, 23)

insertLP(hash_tableLP, 94)

insertLP(hash_tableLP, 11)

insertLP(hash_tableLP, 39)

insertLP(hash_tableLP, 20)

insertLP(hash_tableLP, 16)

insertLP(hash_tableLP, 5)





#print("""Using the hash function hi(i) = (3i+5) mod 11 to hash the keys given, 

#handling collision by linear probing, the hash table will look like this\n""")

print(hash_tableLP)
#Handling collison using quadratic probing



#hash_tableLP = [None] * 11

hash_tableQP = [[] for _ in range(11)] #To avoid collisions by chaining we need a nested list





def insertQP(hash_tableQP,value):

    hash_key = (3*value+5) % 11 #Function to set the key

    j = 0

    collision = 0

    hash_keyj = ((3*value+5)+j^2) % 11 #Quadratic probing 

    #bucket = hash_tableLP[hash_key]

    for i, kv in enumerate(hash_tableQP):

        if hash_key == i and kv != []:

            collision = 1

    if collision == 1:

        for i in range(100):

            print("\nCollision for value - ", value)

            j += 1

            hash_keyj = ((3*value+5)+j^2) % 11 #Quadratic probing

            print("Trying key -", hash_keyj)

            if hash_tableQP[hash_keyj] == []:

                hash_tableQP[hash_keyj].append(value)

                print("Value -",value," Hash key -", hash_keyj,"Accepted" ," #Collision")

                break

    else:

        print("\nValue -",value,", Hash key -" , hash_key, "#Normal")

        hash_tableQP[hash_key].append(value)

                #hash_tableQP[hash_keyj].append(value)                 

    #hash_table[hash_key] = value

    #hash_tableLP[hash_key].append(value) 

 

insertQP(hash_tableQP, 12)

insertQP(hash_tableQP, 44)

insertQP(hash_tableQP, 13)

insertQP(hash_tableQP, 88)

insertQP(hash_tableQP, 23)

insertQP(hash_tableQP, 94)

insertQP(hash_tableQP, 11)

insertQP(hash_tableQP, 39)

insertQP(hash_tableQP, 20)

insertQP(hash_tableQP, 16)

insertQP(hash_tableQP, 5)





#print("""Using the hash function hi(i) = (3i+5) mod 11 to hash the keys given, 

#handling collision by linear probing, the hash table will look like this\n""")

print(hash_tableQP)
#h(k) = 7 − (k mod 7)?



#hash_tableLP = [None] * 11

hash_tableDH = [[] for _ in range(11)] #To avoid collisions by chaining we need a nested list





def insertDH(hash_tableDH,value):

    hash_key = (3*value+5) % 11 #Function to set the key

    j = 0

    collision = 0

    hash_keyj = (((3*value+5)+j)*(7-(value % 7))) % 11 #Double hashing

    for i, kv in enumerate(hash_tableDH):

        if hash_key == i and kv != []:

            collision = 1

    if collision == 1:

        for i in range(100):

            print("\nCollision for value - ", value)

            #print("j", j)

            hash_keyj = (((3*value+5)+j)*(7-(value % 7))) % 11 #Double hashing

            print("Trying key -", hash_keyj)

            j += 1

            if hash_tableDH[hash_keyj] == []:

                hash_tableDH[hash_keyj].append(value)

                print("Value -",value," Hash key -", hash_keyj,"Accepted" ," #Collision")

                break

    else:

        print("\nValue -",value,", Hash key -" , hash_key, "#Normal")

        hash_tableDH[hash_key].append(value)

                #hash_tableQP[hash_keyj].append(value)                 

    #hash_table[hash_key] = value

    #hash_tableLP[hash_key].append(value) 

 

insertDH(hash_tableDH, 12)

insertDH(hash_tableDH, 44)

insertDH(hash_tableDH, 13)

insertDH(hash_tableDH, 88)

insertDH(hash_tableDH, 23)

insertDH(hash_tableDH, 94)

insertDH(hash_tableDH, 11)

insertDH(hash_tableDH, 39)

insertDH(hash_tableDH, 20)

insertDH(hash_tableDH, 16)

insertDH(hash_tableDH, 5)





#print("""Using the hash function hi(i) = (3i+5) mod 11 to hash the keys given, 

#handling collision by linear probing, the hash table will look like this\n""")

print(hash_tableDH)
hash_table5 = [[] for _ in range(13)] 





def insert5(hash_table5,value):

    hash_key = value % 13 #Function to set the key

    #hash_table[hash_key] = value

    hash_table5[hash_key].append(value) 

 

insert5(hash_table5, 54)

insert5(hash_table5, 28)

insert5(hash_table5, 41)

insert5(hash_table5, 18)

insert5(hash_table5, 10)

insert5(hash_table5, 36)

insert5(hash_table5, 25)

insert5(hash_table5, 38)

insert5(hash_table5, 12)

insert5(hash_table5, 90)







print("""First table""")

print (hash_table5)





hash_table19 = [[] for _ in range(19)]



def insert19(hash_table19,hash_table5):

    #hash_key19 = (3*value) % 19 #Function to set the key

    #hash_table[hash_key] = value

    val = 0

    for i, kv in enumerate(hash_table5):

        #hash_key19 = (3*val) % 19 #Function to set the key

        #print(kv)

        if kv != []:

            #print(kv)

            for t, v in enumerate(kv):

                hash_key19 = (3*v) % 17 #Function to set the key

                hash_table19[hash_key19].append(v)



    #hash_table5[hash_key].append(value)

    

insert19(hash_table19,hash_table5)

print("\nRehashed table sized 19")

print(hash_table19)
#Parts of code from

#https://docs.python.org/2/library/bisect.html



#Count the number of 2's in a

from bisect import bisect_left

import numpy as np



a = [[2,2,2,2,2],[2,2,2,2,3],[2,2,2,3,3],[2,2,3,3,3],[2,3,3,3,3]]



x = 3

sum_of_x = 0



def index(a, x):

    'Locate the leftmost value exactly equal to x'

    i = bisect_left(a, x)

    if i != len(a) and a[i] == x:

        b = len(a) - i

        print(b,"Values equal to x\n")

        return b

    else:

        print("No value equal to x\n")



def count(a,x):

    for i in range(len(a)):

        print("In list",i+1,"there are:")

        list = a[i]

        index(list,x)    



#index(list,x)

count(a,x)





#def BinarySearch(a, x): 

#    i = bisect_left(a, x) 

#    print(a)

#    if i != len(a) and a[i] == x: 

#        return i 

#    else: 

#        return -1

#res = BinarySearch(a, x) 

#x = 0

#if res == 1: 

#    print(x, "is absent") 

#else: 

#    print("First occurrence of", x, "is present at", res)


