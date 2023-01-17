import numpy as np
mat1 = [[1,0,2],

        [2,1,3],

        [3,2,1]]

mat2 = [[6,2,2],

        [2,6,3],

        [3,2,5]]

print(np.multiply(mat1,mat2))
num = -1928

if num >= 0:

    print(num,' is positive.')

else:

    print(num,' is negative')
num = 1928

if num%2 == 0:

    print(num,' is even')

else:

    print(num,' is odd')
def factorial(n):

   if n == 1:

       return n

   else:

       return n*factorial(n-1)



n = 7

print(factorial(n))
arr = [1,9,3,5,-2,22,4,7,23,12,8]

print('Sum is = ',np.sum(arr))

print('Minimum element is = ',np.min(arr))

print('Maximum element is = ',np.max(arr))

print('Mean of the array is = ',np.mean(arr))

print('Median element is = ',np.median(arr))

print('Sorted array = ',np.sort(arr))
import cv2

from matplotlib import pyplot as plt



plt.figure(figsize=(20, 20))

plt.title("Original")

plt.imshow(cv2.imread('../input/image-for-basic-digital-image-processing-operation/crow.jpg'))
