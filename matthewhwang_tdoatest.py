import numpy as np

import matplotlib.pyplot as plt



node_1 = [0,0]

node_2 = [100,0]

node_3 = [100,100]

sound_origin = [30,60]

len_1 = ((node_1[0]-sound_origin[0])**2+(node_1[1]-sound_origin[1])**2)**(1/2)

len_2 = ((node_2[0]-sound_origin[0])**2+(node_2[1]-sound_origin[1])**2)**(1/2)

len_3 = ((node_3[0]-sound_origin[0])**2+(node_3[1]-sound_origin[1])**2)**(1/2)

diff_1_2= abs(len_1 - len_2)

diff_2_3= abs(len_2 - len_3)



grid = np.zeros((100,100))

x = np.arange(0.0,10,0.1)

y = np.zeros_like(x)



for i in range(100):

    for j in range(100):

        len_1 = ((node_1[0]-i)**2+(node_1[1]-j)**2)**(1/2)

        len_2 = ((node_2[0]-i)**2+(node_2[1]-j)**2)**(1/2)

        diff_1_2_temp= abs(len_1 - len_2)

        if(1 > abs(diff_1_2-diff_1_2_temp)):

            grid[i][j] = 1

                   

plt.imshow(grid)

plt.show()
import numpy as np

import matplotlib.pyplot as plt



node_1 = [0,0]

node_2 = [100,0]

node_3 = [100,100]

sound_origin = [30,60]

len_1 = ((node_1[0]-sound_origin[0])**2+(node_1[1]-sound_origin[1])**2)**(1/2)

len_2 = ((node_2[0]-sound_origin[0])**2+(node_2[1]-sound_origin[1])**2)**(1/2)

len_3 = ((node_3[0]-sound_origin[0])**2+(node_3[1]-sound_origin[1])**2)**(1/2)

diff_1_2= abs(len_1 - len_2)

diff_2_3= abs(len_2 - len_3)



grid = np.zeros((100,100))

x = np.arange(0.0,10,0.1)

y = np.zeros_like(x)



for i in range(100):

    for j in range(100):

        len_1 = ((node_1[0]-i)**2+(node_1[1]-j)**2)**(1/2)

        len_2 = ((node_2[0]-i)**2+(node_2[1]-j)**2)**(1/2)

        diff_1_2_temp= abs(len_1 - len_2)

        if(1 > abs(diff_1_2-diff_1_2_temp)):

            grid[i][j] = 1

        

for i in range(100):

    for j in range(100):

        len_2 = ((node_2[0]-i)**2+(node_2[1]-j)**2)**(1/2)

        len_3 = ((node_3[0]-i)**2+(node_3[1]-j)**2)**(1/2)

        diff_2_3_temp= abs(len_2 - len_3)

        if(1 > abs(diff_2_3-diff_2_3_temp)):

            grid[i][j] = 1      



plt.imshow(grid)

plt.show()
import numpy as np

import matplotlib.pyplot as plt



node_1 = [0,0]

node_2 = [100,0]

node_3 = [100,100]

sound_origin = [30,60]

len_1 = ((node_1[0]-sound_origin[0])**2+(node_1[1]-sound_origin[1])**2)**(1/2)

len_2 = ((node_2[0]-sound_origin[0])**2+(node_2[1]-sound_origin[1])**2)**(1/2)

len_3 = ((node_3[0]-sound_origin[0])**2+(node_3[1]-sound_origin[1])**2)**(1/2)

diff_1_2= len_1 - len_2

diff_2_3= abs(len_2 - len_3)



grid = np.zeros((100,100))

x = np.arange(0.0,10,0.1)

y = np.zeros_like(x)



for i in range(100):

    for j in range(100):

        len_1 = ((node_1[0]-i)**2+(node_1[1]-j)**2)**(1/2)

        len_2 = ((node_2[0]-i)**2+(node_2[1]-j)**2)**(1/2)

        diff_1_2_temp= len_1 - len_2

        if(1 > abs(diff_1_2-diff_1_2_temp)):

            grid[i][j] = 1

                   

plt.imshow(grid)

plt.show()
import numpy as np

import matplotlib.pyplot as plt



node_1 = [0,0]

node_2 = [100,0]

node_3 = [100,100]

sound_origin = [30,60]

len_1 = ((node_1[0]-sound_origin[0])**2+(node_1[1]-sound_origin[1])**2)**(1/2)

len_2 = ((node_2[0]-sound_origin[0])**2+(node_2[1]-sound_origin[1])**2)**(1/2)

len_3 = ((node_3[0]-sound_origin[0])**2+(node_3[1]-sound_origin[1])**2)**(1/2)

diff_1_2= len_1 - len_2

diff_2_3= len_2 - len_3



grid = np.zeros((100,100))

x = np.arange(0.0,10,0.1)

y = np.zeros_like(x)



for i in range(100):

    for j in range(100):

        len_1 = ((node_1[0]-i)**2+(node_1[1]-j)**2)**(1/2)

        len_2 = ((node_2[0]-i)**2+(node_2[1]-j)**2)**(1/2)

        diff_1_2_temp= len_1 - len_2

        if(1 > abs(diff_1_2-diff_1_2_temp)):

            grid[i][j] = 1

        

for i in range(100):

    for j in range(100):

        len_2 = ((node_2[0]-i)**2+(node_2[1]-j)**2)**(1/2)

        len_3 = ((node_3[0]-i)**2+(node_3[1]-j)**2)**(1/2)

        diff_2_3_temp= len_2 - len_3

        if(1 > abs(diff_2_3-diff_2_3_temp)):

            grid[i][j] += 1 

            if(grid[i][j]>1):

                print('x:' + str(i))

                print('y:' + str(j))



plt.imshow(grid)

plt.show()