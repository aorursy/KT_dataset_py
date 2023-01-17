##Arithmetic Operator(+,-,,/,*,%,//)

##Addition

print(23+28)

a=23

b=28

d=26

c=a+b

print(c)

##Substraction

print(12897-3452)

e=a-d

print(e)

##Division

print(45673/23)

print(round(45673/23))

##Modular

print(45678%234)

##To the power

print(2**5)



##Comparison Operator

num1=56

num2=120

print(num1==num2)

print(num1!=num2)

print(num1>num2)

print(num1<num2)
#Bitwise Operator

a=4

b=8

print(a&b)

print(a|b)
# Logical Operator

a=True

b=False

print(a and b)
##Examples for Tuple

t1=('a','b','c','d')

print(t1)

print(t1[1])

print(t1[:3])

print(t1[2:4])

print(t1[-2])

print(type(t1))
t1[1]=x  ##As unindexed unable to change the value indexwise,so it will throw the error
t2=tuple(('a','b',1,'d'))

print(t2)

print(t2[1])

print(type(t2))

t3=('a')

t4=('a',)

print(type(t3))

print(type(t4))
##Examples of List and its operations

l1=['a','b','c','d']

print(l1)

print(type(l1))

l2=list(["a",'b','c',90])

print(l2)

print(type(l2))

print(l2[1])

l3=[1,3,6,4,5,6,7,7,"a","apple",90]

print(l3[5:10])

print(len(l3))

l3.append('123')## append will add at the end

print(l3)

l3.insert(1,'a')#insert can add at any position

print(l3)

l3.remove('a')#Will remove first occurance

print(l3)

l3.pop()#Will remove last element of the list

print(l3)

l4=l3.copy()

print(l4)

l5=l1+l4

print(l5)

l5.append(l2)#Append as a list,list within list

print(l5)

l5.extend(l2)

print(l5)#extend the list,will add at the end

l5.reverse() ##Will reverse the list

print(l5)
##Examples of Dictionary and its operation

dict1={

  "brand": "Ford",

  "model": "Mustang",

  "year": 1964

}

print(dict1)

print(type(dict1))

print(dict1.keys())

print(dict1.values())

dict1["year"]=1970

print(dict1)

dict1.update({'colr':'Red'})##will add the key & value to the dictionary

print(dict1)

dict1.update({'colr':'White'})

print(dict1)

dict1.pop('colr')

print(dict1)
s1=set(['a','b','c'])##Unordered

print(s1)

print(type(s1))

print(s1[1])## Unindexed
s2=set(['b','c','d'])

print(s2)
print(s1-s2)

print(s2-s1)

print(s1.intersection(s2))

print(s1.union(s2))
print(s1)

print(s2)

print(s1.pop())

print(s1)

s1.add('e')

print(s1)
import numpy as np

print(np.__version__)##checking numy version

a=np.array([1,2,3])

print(a)

print(type(a))

a0=np.array(12)##0-D array

a1=np.array([1,2,3,4,5])##1-D Array

a2=np.array([[1,2,3],[4,5,0]])##2-D Array

a3=np.array([[[1,2,3],[3,4,5]],[[0,2,4],[4,6,8]]])##3-D Array

print(a0)

print(a1)

print(a2)

print(a3)

print(a0.ndim)

print(a1.ndim)

print(a2.ndim)

print(a3.ndim)
##Accessing the array element

arr1 = np.array([2,4,8,6,10])##1D array



print(arr1[0])

print(arr1[1])

print(arr1[2] + arr[3])##Adding 2nd and 3rd element



arr2 = np.array([[3,5,7,9,11], [2,4,6,8,10]])## 2D array

print(arr2[0,3])## first row 3rd element
##Array Slicing

print(arr1[1:3])

print(arr1[2:])
arr3 = np.array([11, 13, 15, 17, 19, 21, 23])

print(arr3[1:6:2])#From 1 to 6 th element,every 2nd element

print(arr3[1:6:3])#From 1 to 6 th element,every 3rd element

print(arr3[::2])##every alternate number from array

print(arr3[::3])##Every 3rd number from array
## Creating Copy of the array

arr = np.array([2, 4, 6, 8, 10])

x = arr.copy() #Make a copy,but change in orginal arrary wil not reflect in copied array#Deep Copy

arr[0] = 23



print(arr)

print(x)

y=arr.view() #Make a copy,change in original array will change in copied array#Swallow copy

arr[0]=45

print(arr)

print(y)
##Array Reshaping

arr = np.array([10,15,20,25,30,35,40,45,50,60,65,70])

newar1 = arr.reshape(3,4)

newar2 = arr.reshape(2, 3, 2)



print( "Newarray1",newar1)

print( "Newarray1",newar2)
## Array Flattening(1 d array)



arr = np.array([[1,2,3],[2,4,6],[4,5,6]])



newar = arr.reshape(-1)



print(newar)
import pandas as pd

##Series

a1=pd.Series([1,2,3])

print("a1", a1)

print(type(a1))

a2=pd.Series([1,2,'a','b','c'])

print("a2",a2)

a3=pd.Series([1,2,'122','bcd','cdd'],index=['a','b','c','d','e'])

print("a3",a3)
##dataframe

df1=pd.DataFrame([['a','b','c','d'],[1,2,3,4]])

print(type(df1))

print(df1)

dict1={

  "brand": ["Ford",'honda','maruti','hyundai','skoda','bmw'],

  "model": ["Mustang",'BRV','alto','santro','skoda','bmw5'],

  "year": [1964,1865,1989,1987,2011,1999]

}

df2=pd.DataFrame(dict1,columns=['brand','model','year'])

print(df2)

print(df2.shape)

print(df2.head())

print(df2.info())

print(df2.dtypes)
print(df2.columns.tolist())

##Dataframe Slicing

print(df2[1:3])

print(df2[:2])

print(df2[::2])
print(df2.values)

print(df2.index)

print(df2.describe)

print(df2.columns)

print(df2.T)
df3=pd.DataFrame(np.random.randn(6,4),index=list('abcdef'),columns=list('ABCD'))

print(df3)

print(df3.loc[['a','b'],:])

print(df3.loc[['a','b'],['B','C']])

print(df3.loc[:,['A']])

print(df3.loc['a']>0.03)

print(df3.loc['a',"B"])
print(df3.iloc[:1])##Index

print(df3.iloc[2:4])