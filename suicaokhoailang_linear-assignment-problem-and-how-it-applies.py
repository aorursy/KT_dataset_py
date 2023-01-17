# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
!pip install lapjv
from lapjv import lapjv
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import matplotlib
cost_mat = np.array([ 
   [0, 0.2, 0.3,0.1, 0.15], 
   [0, 0, 0.35,0.12, 0.37], 
   [0, 0, 0, 0.1, 0.15], 
   [0, 0, 0, 0, 0.08], 
   [0, 0, 0, 0, 0], 
 ]) 

cost_mat += cost_mat.T


def draw_cost_mat(cost_mat, subplot=111):
    plt.subplot(subplot)
    size = cost_mat.shape[0]
    x_start = 3.0 
    x_end = 9.0 
    y_start = 6.0 
    y_end = 12.0 
    extent = [x_start, x_end, y_start, y_end]
    jump_x = (x_end - x_start) / (2.0 * size) 
    jump_y = (y_end - y_start) / (2.0 * size) 
    x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False) 
    y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)  

    plt.imshow(cost_mat, extent=extent)
    plt.imshow(cost_mat, extent=extent)         
    for y_index, y in enumerate(y_positions): 
         for x_index, x in enumerate(x_positions): 
            label = np.flip(cost_mat,1)[x_index, y_index] 
            text_x = x + jump_x 
            text_y = y + jump_y 
            plt.text(text_x, text_y, label, color='white', ha='center', va='center')

matplotlib.rcParams['figure.figsize'] = [15, 5]

draw_cost_mat(cost_mat,111)

plt.show() 
# set the main diagonal to infinity
cost_mat[np.arange(cost_mat.shape[0]), np.arange(cost_mat.shape[0])] = np.inf

draw_cost_mat(cost_mat, 121)

solved = cost_mat.copy()

# Solving LAP
x = lapjv(solved)[0]
y = np.arange(len(x), dtype=np.int32)
# Set the selected pairs scores to infinity
#  the x,y pair will be used for training
solved[x,y] = np.inf
solved[y,x] = np.inf

draw_cost_mat(solved, 122)

plt.show()