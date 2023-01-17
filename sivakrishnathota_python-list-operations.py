import numpy as np

import pandas as pd

import scipy as sci
li1=[1,2,3,4,5]

print("List :- ",li1,"dtype:- ",type(li1))
li2=list([6,7,8,9,0])

print("List :- ",li2,"dtype:- ",type(li2))
litemp=li1

print("List Orginal:- ",li1,"dtype:- ",type(li1))

print("List Temp:- ",litemp,"dtype:- ",type(litemp))
litemp[0]=50

print("List Orginal:- ",li1,"dtype:- ",type(li1))

print("List Temp:- ",litemp,"dtype:- ",type(litemp))
litemp1=li1.copy()

litemp2=li1[:]

litemp3=list(li1)
print("List litemp1:- ",litemp1,"dtype:- ",type(litemp1))

print("List litemp2:- ",litemp2,"dtype:- ",type(litemp2))

print("List litemp3:- ",litemp3,"dtype:- ",type(litemp3))
litemp1[0]=10

litemp2[1]=20

litemp3[2]=30

print("List modified Templist 1:- ",litemp1,"dtype:- ",type(litemp1))

print("List modified Templist 2:- ",litemp2,"dtype:- ",type(litemp2))

print("List modified Templist 3:- ",litemp3,"dtype:- ",type(litemp3))

print("-----------------------------------------------------------")

print("List Orginal List :- ",li1,"dtype:- ",type(li1))

print("-----------------------------------------------------------")
# Select item from list 

print(li1[0])

print(li1[:])

print(li1[1:3])

print(li1[1:10]) # Note :- Index 10 not exit but our program not throwing exception 
# Add items tolist

li1.append(6) # Append item to list 

print(li1)

li1.extend([7,8]) ## Append Multiple items to list

print(li1)

li1=li1+[10,20,30]

print(li1)
# Remove Items from list 

li1.remove(6) # Remove item directly from list

print(li1)

removeelement=li1.pop() # Remove index item and it will return item [Last in first out concept]

print(removeelement)

print(li1)

del li1[0] # Delete item and will not return 

print(li1)