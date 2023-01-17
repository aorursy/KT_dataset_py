# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# A matrix is an array of numbers arranged in rows and columns
rows = 5
cols = 8
# Lets define an array with random integer values between 0 and 99
# In numpy you can first define a series of numbers and then 
# reshape them into rows and columns
a = np.random.randint(0,100,rows*cols) # a series of rows*cols numbers. 
a = a.reshape(rows, cols)
print("A numpy matrix with %d and %d columns \n \n" % (rows,cols), a)
# A matrix with only one row or one columns is also called a vector
# A row vector has only 1 row
row_vector = np.random.randint(0,100,cols).reshape(1,cols)
print("A 1x%d Row Vector : \n" % (cols), row_vector)
# A column vector has only 1 coulmn
column_vector = np.random.randint(0,100,rows).reshape(rows,1)
print("A %dx1 Column Vector : \n" % (rows), column_vector)
# You can use a matrix to store points in space
points = np.array([[-3,-2],[3,-2],[3,2],[-3,2],[-3,-2]])
x = points[:,0] # pulls out all the "x values" 
y = points[:,1] # pulls out all the "y values"
# [:,x] picks the x column out of all the rows
print("X: ", x)
print("Y: ", y)
# Lets plot the points and draw a line between them
import matplotlib.pyplot as plt
plt.axis([-10,10,-10,10]) #Creates the graph's axises
plt.plot(x,y, marker = 'o') #Plots the points
plt.show() #Shows rect on graph
# You can also use a Matrix to store pixels.

# Lets create a 8x10 matrix of pixel brightness values between 0 and 255
my_pixels = np.linspace(0,255,8*10) #Ascending series of 80 numbers between 0 and 255
#linspace is where numbers are linearly increasing from x (e.g. 0) to y (e.g. 255)
my_pixels = my_pixels.reshape(8,10) # Reshape into 8 rows and 10 columns
plt.imshow(my_pixels, cmap = plt.cm.gist_heat) # Using a grayscale colormap(cmap)
#imshow maps each pixel as a color instead of in a graph form
# To see matrix of my_pixels, uncomment the print statement below
#print(my_pixels)

plt.show()
# A 2x2 matrix named a
a = np.array([[1,2],
             [3,4]])

# Another 2x2 matrix named b
b = np.array([[5,6], 
              [7,8]])
#Addition of matrices
a+b
#Scaling of matrices
a*3
x = np.array([1,2,3,4,5,6])
x.reshape([6,1])  # A column vector
#If last entry is .resphape or variable name, it will be printed
theta = np.array([2,4,6,8,10,12])
theta.reshape([6,1]) # Another column vector
#Transposing: converts rows to columns

# Transpose a column vector into a row vector
theta_t = np.transpose(theta)
theta_t
#"Dot" product of row and column vector
#Multiplies corresponding entries of row and column vector and adds them all
# row1*col1 + row2*col2 + .... rowN*colN

# C = A.B is possible only if :
#   Number of columns in the A equals the number of rows in B

np.dot(theta_t, x)
# Inner Product of Vectors is same as "dot" product
a = ([3,5])
b = ([12,16])


a_t = np.transpose(a) #Function for transposing
print("The dot product is : ", np.dot(a_t,b))

print("The inner product is : ", np.inner(a_t,b))
# Matrix Multiplication Example
a = np.array([[0,2],
              [4,6]])
b = np.array([[1,3],
              [5,7]])

np.dot(a,b)
# Power of Matrices a^2.For higher order powers use multiple np.dot()
b = np.dot(a,a) 
print ("Matrix a : \n", a)
print ("\nMatrix A^2 : \n", b)

c = np.dot(b,a)
print("\nMatrix A^3 : \n",c)
#Matrix Transpose - Converts rows to columns 

a = np.arange(6) # array with 6 numbers starting from 0
a = a.reshape(3,2)
print(" Matrix a: \n", a)

a_transpose = np.transpose(a)
print( "\n  Transpose of Matrix a: \n ", a_transpose )
## Optional

 # Lets show that (A.B.C)' = C'.B'.A'
a =np.random.randint(1,50,16) #"16 random integers between 1 and 50"
a = a.reshape(4,4) # arrange as 4x4 matrix

b = np.random.randint(1,50,16) #"16 random integers between 1 and 50"
b = b.reshape(4,4) # arrange as 4x4 matrix

c = np.random.randint(1,50,16) #"16 random integers between 1 and 50"
c = c.reshape(4,4) # arrange as 4x4 matrix

#Computing A.B.C
temp = np.dot(a,b)
abc = np.dot(temp,c)

#Transposing A.B.C
abc_t = np.transpose(abc)
print ("(A.B.C)' = \n", abc_t)

#Transpose of individual matrices a, b ,c 
a_t = np.transpose(a)
b_t = np.transpose(b)
c_t = np.transpose(c)

#Calculating C'.B'.A'
temp2 = np.dot(c_t, b_t)
ct_bt_at = np.dot(temp2,a_t)

print("\nC'B'A' = \n", ct_bt_at)

'''
A = a,b
    c,d
    
Determinant : Area(or volume) of a parallelogram 
                    (a+b,c+d)
        (c,d)
        
    
                (a,b)
    (0,0)

Determiant Area = ad - bc

'''
a = np.array([[4,1],[2,6]])
print(a)
np.linalg.det(a)
#Optional 
#Trace : Sum of Diagonals

np.trace(a)

# Identity matrix : I . [another matrix] = [that matrix]
# Forward (Left to right)Diagonals are 1; all other elements 0
''' 1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1'''

#Lets create a 4x4 matrix
a = np.arange(0,16,1) # 16 consecutive numbers from 0 to 16 in step of 1
a= a.reshape(4,4) # convert to 4x4 matrix
print("A matrix : \n", a)
# LEts create a 4x4 identity matrix
i4 = np.identity(4)
print("\n Identity Matrix : \n",  i4)

# Identity matrix : I . [another matrix] = [that matrix]
result = np.dot(i4,a)
print("\n Product of Identity Matrix and A : \n", result )
## Tranformantion
# Scaling a matrix
''' 10 0  0  0 
    0  20 0  0
    0  0  30 0
    0  0  0  40
    '''
    
#Generate the matrix shown above using [10, 20, 30, 40] * Identity Matrix
weights = np.identity(4)
weights = np.array([10,20,30,40])*weights
print("Weights : \n", weights)

a = np.ones([4,4])
print("\n A matrix : \n", a)

weighted_a = np.dot(weights, a)

print("\n Weighted A : = \n", weighted_a)

#Lets use a visual example of Scaling

# Remember from earlier : 
#You can use a matrix to store points in space
points = np.array([[-3,-2],[3,-2],[3,2],[-3,2],[-3,-2]])
x = points[:,0] # pulls out all the "x values"
y = points[:,1] # pulls out all the "y values"

# Lets plot the points and draw a line between them
print("Here are the original points : ")
plt.axis([-10,10,-10,10])
plt.plot(x,y, marker = 'o')
plt.show()
# Here are the original points : 
points = np.array([[-3,-2],[3,-2],[3,2],[-3,2],[-3,-2]])
#Lets scale the points to 3x
points = np.dot(points, np.identity(2)*3)
x = points[:,0] # pulls out all the "x values"
y = points[:,1] # pulls out all the "y values"

# Lets plot the points and draw a line between them
print("Here are the 3x scaled points : ")
plt.axis([-10,10,-10,10])
plt.plot(x,y, marker = 'o')
plt.show()
# Rotating a shape

# Here are the original points : 
points = np.array([[-3,-2],[3,-2],[3,2],[-3,2],[-3,-2]])

#Lets transpose it, to be able to pre-multiply with 2x2 rotation matrix
points = np.transpose(points)

#Original Points x and y
x = points[0, :] # pulls out all the "x values"
y = points[1,:] # pulls out all the "y values"

#Lets calculate the rotation matrix for angle 
angle_deg = int(input("Enter Angle : "))
angle = angle_deg*3.14/180  #Converting angle to radians
sin_a = np.sin(angle)
cos_a = np.cos(angle)
rotation_matrix = np.array([[cos_a, -sin_a],
                            [sin_a, cos_a]])

#New Rotated Points x and y
new_points = np.dot(rotation_matrix,points)

x_new = new_points[0,:] # pulls out all the "x values"
y_new = new_points[1,:] # pulls out all the "y values"

# Lets plot the points and draw a line between them
print("Rotation Angle:", angle_deg)
print("Original points in Blue, Rotated points in orange.")
plt.axis([-10,10,-10,10])
plt.plot(x,y,x_new,y_new, marker = 'o')
plt.show()
# Optional : 
# Lets try to rotate in a loop in steps of 60 degrees

# Here are the original points : 
points = np.array([[-3,-2],[3,-2],[3,2],[-3,2],[-3,-2]])

#Lets transpose it, to be able to pre-multiply with 2x2 rotation matrix
points = np.transpose(points)

#Original Points x and y
x = points[0, :] # pulls out all the "x values"
y = points[1,:] # pulls out all the "y values"

for i in range(60,361, 60):
    #Lets calculate the rotation matrix for angle 
    angle_deg = i
    angle = angle_deg*3.14/180  #Converting angle to radians
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a, cos_a]])

    #New Rotated Points x and y
    new_points = np.dot(rotation_matrix,points)

    x_new = new_points[0,:] # pulls out all the "x values"
    y_new = new_points[1,:] # pulls out all the "y values"

    # Lets plot the points and draw a line between them
    print("Rotation Angle:", angle_deg)
    #print("Original points in Blue, Rotated points in orange.")
    plt.axis([-10,10,-10,10])
    plt.plot(x,y,x_new,y_new, marker = 'o')
    plt.show()
