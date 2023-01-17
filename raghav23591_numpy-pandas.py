import numpy as np

import pandas as pd
a=np.array([34,43,43,434,2,4,4])
print('mean of a:',a.mean())

print('size of a:',a.size)

print('shape of a:',a.shape )

print('dinmension of a:',a.ndim)
a.astype(float)
b=np.arange(1,21)
b
print('Shape:',b.shape)

print('Size:',b.size)

print('dimension:',b.ndim)
b1=b.reshape(5,4) #5 rows and 4 columns

b2=b.reshape(4,5) #4 rows and 5 columns



print("b1:\n",b1)

print("\nb2:\n",b2)
b1*10
b1+10
b1.dot(b2)
b1.dot(a)
print('b1:\n',b1)

print('Transpose:\n',b1.transpose())
c=np.array([[1.,2.],[3.,4.]])
c
np.linalg.inv(c)
d=np.array(np.arange(9).reshape(3,3),dtype=float)
d
#print 1.,4.,7.

#d[:,-2]

d[:,1]

#print 1,2,4,5

d[:2,1:]
#print 1,2,4,5,7,8

#d[:,1:]

d[:,-2:]
#print 3,4,6,7

d[1:3,0:2]
d[d>5]
e=[23,3,4,34,34,3,43,43,4]
e[6:]
e[-3:]
np.zeros(10)
np.ones(10)
(np.ones(25)*5).reshape(5,5)
np.ones(25).reshape(5,5)*5
np.zeros(25).reshape(5,5)+5
Name_Age={'name':['Raghav','Sharmi','Ishaan'],'Age':[28,28,1]}
type(Name_Age)
family=pd.DataFrame(Name_Age)
family

#By default, row labels(Index) are from the range(0-3) as we have 3 records
type(family)
new_data=np.arange(150)



new_col={'new_column':new_data,'sex':'M'}



new_column=pd.DataFrame(new_col)
new_column.head(5)
new_column.tail(5)
#change the column index and Column name



new_column.index=np.arange(150,300)
new_column.columns=['numbers','gender']
new_column.head(5)
test_dataA={'Id':[100,101,102,103,104],'First_Name':['Raghav','Sankar','Akhil','Deepa','Sufiya'],'Last_Name':['v','Jayaram','Vijayan','Nayak','Taranam']}
df1=pd.DataFrame(test_dataA,columns=['Id','First_Name','Last_Name'])
test_dataB={'Id':[105,106,107,108,109],'First_Name':['Renju','Saurabh','Pradeep','Joshua','Mythreyi'],'Last_Name':['Nair','Indulkar','Bhat','Singh','Matada']}
df2=pd.DataFrame(test_dataB,columns=['Id','First_Name','Last_Name'])
df1
df2
df_join=pd.concat([df1,df2])
df_join
testdataC={'Id':[100,101,102,103,104,105,106,107,108,109],'Expense_Id':[100,200,260,500,490,600,390,290,340,111]}
df3=pd.DataFrame(testdataC,columns=['Id','Expense_Id'])
df3
pd.merge(df_join,df3,on='Id')
string='India is my country'
#capitalize() helps to change the first letter of the string to capitalize



uppercase= string.capitalize()
uppercase
#islower() helps to check whether the full string has lower case or not.. Returns TRUE or FALSE 



string.islower()
#isupper() helps to check whether the full string has upper case or not.. Returns TRUE or FALSE 



string.isupper()
#len() helps to find the count of the letters available in the string.. returns 'int' datatype



len(string)
#lower() helps to lower all the letters available in the string.

#upper() helps to upper all the letters available in the string.



string.lower()
#title() helps to capitalise the first letter of all the words in a string. 



string.title()
#swapcase() helps to convert all the letters available in the sring to convert to reversecase



string.swapcase()