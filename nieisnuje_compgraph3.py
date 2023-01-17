import cv2

import numpy as np

import os

import matplotlib.pyplot as plt



for root, dirs, files in os.walk('/kaggle/input/'):

    for file in files:

        inp_img = cv2.imread('/kaggle/input/' + file)

        gray_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)

        

        '''

        Однородный усредняющий фильтр для сглаживания 

        основан на том, что при переходе от точки 

        к точке обновляется только часть вычислений. 

        '''

        

        inp_img1 = gray_img.copy()



        for x in range(1, inp_img.shape[0]-1):

            for y in range(1, inp_img.shape[1]-1):



                total = 0.0

                total = total + inp_img1[x,y]        #centre pixel

                total = total + inp_img1[x-1,y]      #left

                total = total + inp_img1[x-1,y+1]    #bottom left

                total = total + inp_img1[x,y+1]      #bottom

                total = total + inp_img1[x+1,y+1]    #bottom right

                total = total + inp_img1[x+1,y]      #right

                total = total + inp_img1[x+1,y-1]    #top right

                total = total + inp_img1[x,y-1]      #top

                total = total + inp_img1[x-1,y-1]    #top left



                inp_img1[x,y] = int(total/9)



        print ('Оригинал vs Изображение с применением однородного усредняющего фильтра')

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = (10,10))

        ax1.imshow(inp_img)

        ax2.imshow(inp_img1)

        plt.show()

        

        

        '''

        Размытие по Гауссу — это характерный фильтр размытия 

        изображения, который использует нормальное распределение 

        (также называемое Гауссовым распределением,отсюда название) 

        для вычисления преобразования, применяемого к каждому пикселю изображения. 

        Здесь используется фильтр размытия с матрицей свертки, 

        заполненной по закону Гауссовского распределения.

        '''

        

        mask = [[1,2,1],[2,4,2],[1,2,1]]

        inp_img2 = cv2.cvtColor(cv2.resize(inp_img, (140*2,100*2)),cv2.COLOR_BGR2GRAY)

        

        mask = np.array(mask)

        new_image = []

        for i in range(1,inp_img2.shape[0]-1):

            new_image.append([])

            for j in range(1,inp_img2.shape[1]-1):

                mm = np.reshape(mask,(9,))

                tt = np.reshape(inp_img2[i-1:i+2,j-1:j+2],(9,))

                value = np.dot(mm, tt)

                new_image[-1].append(value)

        new_image = np.array(new_image)



        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = (12,4))

        

        print ('Оригинал vs Изображение с применением фильтра Гаусса')

        ax1.imshow(inp_img2)

        ax2.imshow(new_image)

        plt.show()