########################################################################################

# Лабораторная работа 1 по дисциплине МРЗвИС

# Выполнена студентом группы 721702

# БГУИР Гурбович Артём Игоревич

#

# Вариант 9: Реализовать модель линейной рециркуляционной сети.

#

# 11.10.2019



########## IMPORTS #####################################################################################################################



import numpy as np

from matplotlib import pyplot as plt

from matplotlib.image import imread

from skimage.util.shape import view_as_blocks



########## IMAGE TO BLOCKS FUNCTION ####################################################################################################



def image_to_blocks(arr):

    result = []

    for i in range(h // n):

        for j in range(w // m):

            tmp = []

            for y in range(n):

                for x in range(m):

                    for k in range(3):

                        tmp.append(arr[k, i*n+y, j*m+x])

            result.append(tmp)

    return np.array(result)



########## BLOCKS TO IMAGE FUNCTION ####################################################################################################



def blocks_to_image(arr):

    result = np.array([np.arange(h*w, dtype="float64").reshape(h, w),

                       np.arange(h*w, dtype="float64").reshape(h, w),

                       np.arange(h*w, dtype="float64").reshape(h, w)])

    a = -1

    for i in range(h // n):

        for j in range(w // m):

            a += 1

            b = 0

            for y in range(n):

                for x in range(m):

                    for k in range(3):

                        result[k, i*n+y, j*m+x] = arr[a, b+k]

                    b += 3

    return result



########## SHOW IMAGE FROM ARRAY FUNCTION ###############################################################################################



def show_image_from_array(arr):

   plt.imshow(arr)

   plt.axis('off')

   plt.show()



########## RESTORE IMAGE FUNCTION #######################################################################################################



def restore_image(arr):

    arr = 1 * (arr + 1) / 2

    return arr.reshape(h, w, 3)
########## INPUT #######################################################################################################



print("Enter n: ")

n = int(input())

print("Enter m: ")

m = int(input())

print("Enter p: ")

p = int(input())
########## INIT #######################################################################################################



h, w = 256, 256

N = n * m * 3

L = int((h * w) / (n * m))

print('N =', N)

print('L =', L)

print('Z =', (N*L)/((N+L)*p+2))

image = imread("../input/lab1dataset/lena.png").reshape(3, h, w) # "../input/lab1dataset/mona.png"  "../input/lab1dataset/brick.png"

image = (2.0 * image / 1.0) - 1.0

image_blocks = image_to_blocks(image).reshape(L, 1, N)
########## TRAIN #######################################################################################################



w1 = np.random.rand(N, p) * 2 - 1

z = np.copy(w1)

w2 = z.transpose()



error_max = 3000.0

error_current = error_max + 1

alpha = 0.0007

epoch = 0



while error_current > error_max:

    error_current = 0

    epoch += 1

    for i in image_blocks:

        y = i @ w1

        x1 = y @ w2

        dx = x1 - i

        w1 -= alpha * np.matmul(np.matmul(i.transpose(), dx), w2.transpose())

        w2 -= alpha * np.matmul(y.transpose(), dx)

    for i in image_blocks:

        dx = ((i @ w1) @ w2) - i

        error = (dx * dx).sum()

        error_current += error

    print('Epoch ', epoch, '    ', 'error ', error_current)
########## TEST #######################################################################################################



res = []

for i in image_blocks:

    res.append(i.dot(w1).dot(w2))

res = np.array(res)



show_image_from_array(restore_image(blocks_to_image(res.reshape(L, N))))