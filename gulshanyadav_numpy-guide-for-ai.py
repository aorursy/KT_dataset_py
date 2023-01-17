import numpy as np # linear algebra
x = np.array([1,2,3,4,5,6,7,8,9])  # convert the passed list to form array

print(x.shape)

x = x.reshape(9,1)    # reshape the array into a row vector

print(x)

print(x.shape)

x = x.reshape(1,9)    # reshape the array into a column vector

print(x)

print(x.shape)

x = x.reshape(3,3)    # reshape into a 3x3 matrix

print(x)

print(x.shape)



# rows and columns in the shape can be individually accessed as

print(x.shape[0])

print(x.shape[1])
# Consider the below image of shape (3,3,3)

image = np.array([[[ 0.67826139,  0.29380381, 0.67826139],

        [ 0.90714982,  0.52835647, 0.90714982],

        [ 0.4215251 ,  0.45017551, 0.4215251]],



       [[ 0.92814219,  0.96677647, 0.92814219],

        [ 0.85304703,  0.52351845, 0.85304703],

        [ 0.19981397,  0.27417313, 0.19981397]],



       [[ 0.60659855,  0.00533165, 0.60659855],

        [ 0.10820313,  0.49978937, 0.10820313],

        [ 0.34144279,  0.94630077, 0.34144279]]])



print(image.shape)

new_shape = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)

print(new_shape)

print(new_shape.shape)
list1 = [1,2,3,4,5,6]

list2 = [10,20,30]



matrix1 = np.array(list1).reshape(6,1)      # 6 rows and 1 column

print(f"Matrix1 of size m,1 = \n{matrix1}")



matrix2 = np.array(list2[0]).reshape(1,1)

print(f"Matrix2 of size 1,1 = \n{matrix2}")



add_result = np.add(matrix1, matrix2)

print(f"Result m+1,1 = \n{add_result}")



mult_result = np.matmul(matrix1, matrix2)

print(f"Result mx1,1 = \n{mult_result}")



div_result = np.divide(matrix1, matrix2)

print(f"Result m/1,1 = \n{div_result}")



sub_result = np.subtract(matrix1, matrix2)

print(f"Result m-1,1 = \n{sub_result}")
list1 = [1,2,3,4,5,6]

list2 = [10,20,30]



matrix1 = np.array(list1).reshape(1,6)     # 1 row and 6 columns

print(f"Matrix1 of size 1,m = \n{matrix1}")



matrix2 = np.array(list2[0]).reshape(1,1)

print(f"Matrix2 of size 1,1 = \n{matrix2}")



add_result = np.add(matrix1, matrix2)

print(f"Result 1,n+1 = \n{add_result}")
list1 = [1,2,3,4,5,6]

list2 = [10,20,30]



matrix1 = np.array(list1).reshape(2,3)     # 2rows and 3 columns

print(f"Matrix1 of size m,n = \n{matrix1}")



matrix2 = np.array(list2).reshape(1,3)

print(f"Matrix2 of size 1,n = \n{matrix2}")



add_result = np.add(matrix1, matrix2)

print(f"Result m,n+n = \n{add_result}")
list1 = [1,2,3,4,5,6]

list2 = [10,20,30]



matrix1 = np.array(list1).reshape(3,2)     # 3 rows and 2 columns

print(f"Matrix1 of size m,n = \n{matrix1}")



matrix2 = np.array(list2).reshape(3,1)     # 3 rows and 1 column

print(f"Matrix2 of size 3,1 = \n{matrix2}")



add_result = np.add(matrix1, matrix2)

print(f"Result m+m,n = \n{add_result}")
list1 = [1,2,3,4,5,6]

list2 = [10,20,30,40,50,60]



matrix1 = np.array(list1).reshape(3,2)     # 3 rows and 2 columns

print(f"Matrix1 of size m,n = \n{matrix1}")



matrix2 = np.array(list2).reshape(3,2)     # 3 rows and 2 column

print(f"Matrix2 of size 3,2 = \n{matrix2}")



add_result = np.add(matrix1, matrix2)

print(f"Result m+m,n+n = \n{add_result}")
# Dot operation on 2 vectors

x = np.array([1,2,3,4])

y = np.array([5,6,7,8])

result = np.dot(x,y)

print(result)   # 1x5 + 2x6 + 3x7 + 4x8 = 70



# Dot operation on 2 matrices

x = np.array([1,2,3,4]).reshape(2,2)

print(f"x = \n{x}")

y = np.array([5,6,7,8]).reshape(2,2)

print(f"y = \n{y}")

result = np.dot(x,y)



# Output calculation:

# 1x5 + 2x7 = 19  ie. row 1 to column 1

# 3x5 + 4x7 = 43  ie. row 2 to column 1

# 1x6 + 2x8 = 22  ie. row 1 to column 2

# 3x6 + 4x8 = 50  ie. row 2 to column 2

print(f"result = \n{result}")   
print(np.exp(1))  # ie. 2.718^2 (2.718 power to 1)

print(np.exp(2))  # ie. 2.718^2

print(np.exp(3))  # ie. 2.718^3
x = np.array([1, 2, 3])

s = 1/(1+np.exp(x*-1))

print(s)
yhat = np.array([.9, 0.2, 0.1, .4, .9])

y = np.array([1, 0, 0, 1, 1])

L1_loss = np.sum(np.abs(np.subtract(y,yhat)))

print(L1_loss)
yhat = np.array([.9, 0.2, 0.1, .4, .9])

y = np.array([1, 0, 0, 1, 1])

L2_loss = np.sum(np.square(np.subtract(yhat, y)))

print(L2_loss)