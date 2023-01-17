import numpy as np

#1.Create 1D, 2D, and boolean array using numpy.
a = np.array([10,20,30,40,50])

print (a)
print("\n")

b = np.array([[10,20,30],[70,80,90]])

print(b)
print("\n")

# Boolean Array

bool_ar = np.array(['x',0,1,0.5,6],dtype=bool)

print(bool_ar)
print("\n")


#2.Extract the odd numbers from a 2D array using numpy package.

arr = np.array([[10,20,30],[60,70,80]])

for i in arr:
    for j in i:
        if j%2 != 0:
            print(j, end="\t")
print("\n")
        
        
#3. How to replace items that satisfy a condition with another value in numpy array?

arr = np.array([10,69,48,24,2,10,4,5,1])

arr = np.where(arr>=19,0,arr)

print(arr)
print("\n")


#4. How to get the common items between two python numpy arrays?

a = np.array([10,20,30,60,70,80])

b = np.array([30,50,70])

c = np.intersect1d(a,b)

print(c)
print("\n")



#5.How to remove from one array those items that exist in another?

a = np.array([10,20,30,60,70,80])

b = np.array([30,50,70])

for i in b:
    for j in a:
        if i == j:
            a = a[a!=j]
print(a)
print("\n")

#6.How to get the positions where elements of two arrays match?

a = np.array([10,20,30,60,70,80])

b = np.array([30,50,70])

for i in b:
    for j in a:
        if i == j:
            index = np.argwhere(a==j)
            print(index)
