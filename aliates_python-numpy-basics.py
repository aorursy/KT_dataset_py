import numpy as np



array = np.array([[1, 2, 3, 4, 5], 

                  [6, 7, 8, 9, 10]])

print("type(array): \t{0}".format(type(array)))

print("array.dtype.name: \t{0}".format(array.dtype.name))

print("array.size: \t\t{0}".format(array.size))

print("array.shape: \t{0}".format(array.shape))

print("array.ndim: \t{0}".format(array.ndim))
array2 = array.reshape(5, 2)

print(array2)

print("array2.shape: \t{0}".format(array2.shape))

print("array2.ndim: \t{0}".format(array2.ndim))
array3 = np.zeros((3, 4))

print(array3)
array4 = np.ones((3, 5))

print(array4)
#return random values from random memory field

array5 = np.empty((2, 2))

print(array5)
#np.arange(inclusive, exclusive, increase_value)

array6 = np.arange(0, 10, 1)

print(array6)
#np.linspace(inclusive, inclusive, slice_count)

array7 = np.linspace(0, 10, 5, True, True)

print(array7)
a = np.array([[1, 2, 3],

              [4, 5, 6]])

b = np.array([[-1, -2, -3],

              [-4, -5, -6]])



print("a:\n{0}\n".format(a))

print("b:\n{0}\n".format(b))

print("a+b:\n{0}\n".format(a+b))

print("a-b: \n{0}\n".format(a-b))

print("a**2: \n{0}\n".format(a**2))

print("a<2: \n{0}\n".format(a < 2))

print("np.sin(a): \n{0}\n".format(np.sin(a)))
print("a.T: \n{0}\n".format(a.T))

print("a.shape: {0}, b.shape: {1}\n".format(a.shape, b.shape))

print("a.dot(b.T): \n{0}\n".format(a.dot(b.T)))

print("np.exp(a): \n{0}\n".format(np.exp(a)))
x = np.random.random((3, 5))



print("x:\n{0}\n".format(x))

print("x.sum():\n{0}\n".format(x.sum()))

print("x.sum(axis=0): [sum columns(5)]\n{0}\n".format(x.sum(axis=0)))

print("x.sum(axis=1): [sum rows(3)]\n{0}\n".format(x.sum(axis=1)))

print("x.max():\n{0}\n".format(x.max()))

print("x.min():\n{0}\n".format(x.min()))

print("np.sqrt(x): [√x]\n{0}\n".format(np.sqrt(x)))

print("np.square(x): [x**2]\n{0}\n".format(np.square(x)))

print("np.add(x, 2):\n{0}\n".format(np.add(x, 2)))
#Indexing and slicing

arr = np.array([1, 2, 3, 4, 5])



print("arr[0]: {0}\n".format(arr[0]))

print("arr[0:4]: [inclusive, exclusive]\n{0}\n".format(arr[0:4]))

print("arr[::-1]: [reverse array]\n{0}\n".format(arr[::-1]))

arr2 = np.array([[1, 2, 3, 4, 5],

                 [6, 7, 8, 9, 10],

                 [11, 12, 13, 14, 15]])



print("arr1[1, 1]: \t{0}\n".format(arr2[1, 1]))

print("arr1[:, :]: [all rows, all columns]\n{0}\n".format(arr2[:, :]))

print("arr1[:, 0]: [all rows, 0. columns]\n{0}\n".format(arr2[:, 0]))

print("arr1[:, 1]: [all rows, 1. columns]\n{0}\n".format(arr2[:, 1]))

print("arr1[1, :]: [1. row, all columns]\n{0}\n".format(arr2[1, :]))

print("arr1[0, :]: [0. row, all columns]\n{0}\n".format(arr2[0, :]))

print("arr1[1, 1:4]: [1. row, 1-4 columns (inclusive-exclusive)]\n{0}\n".format(arr2[1, 1:4]))

print("arr1[0:3, 0]: [0-3 rows (inc, exc), 0. columns]\n{0}\n".format(arr2[0:3, 0]))

print("arr1[-1, :]: [-1. row, all columns]\n{0}\n".format(arr2[-1, :]))

print("arr1[:, -1]: [all rows, -1. columns]\n{0}\n".format(arr2[:, -1]))
#Shape manuplation

xarr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

xarr2 = np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])



print("xarr1:\n{0}\n".format(xarr1))

print("xarr1.ravel(): [flatten]\n{0}\n".format(xarr1.ravel()))



xarr3 = np.column_stack((xarr1, xarr2))

print("xarr3: [np.column_stack(xarr1, xarr2)]\n{0}\n".format(xarr3))



xarr4 = np.hstack((xarr1, xarr2))

print("xarr4: np.hstack((xarr1, xarr2))\n{0}\n".format(xarr4))



xarr5 = np.vstack((xarr1, xarr2))

print("xarr5: np.vstack((xarr1, xarr2))\n{0}\n".format(xarr5))



#Convert

list1 = [1, 2, 3]

print("list1:\n{0}\n".format(list1))



nparr = np.array(list1)

print("nparr: np.array(list1)\n{0}\n".format(nparr))



list2 = list(nparr)

print("list2: list(nparr)\n{0}\n".format(list2))
#Copy

narr = np.random.random((3, 3))

print("narr:\n{0}\n".format(narr))



narr2 = narr.copy()

print("narr2:\n{0}\n".format(narr2))