# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pathlib import Path



for dirpath, dirname, files in os.walk('/kaggle/input/'):

    print(dirpath)

    print(dirname)

    

data_path = Path(r'/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'



print(training_path)
# (1) Defining list of .json file names comprising training directory



training_tasks = sorted(os.listdir(training_path))



# (2) Printing contents of .json files in Python

# (3) Listing the keys of the dictionary comprising the task file

# (4) "Taking out the data" by defining labels for the dictionaries corresponding to i/o pairs. 



import json



with open(training_path / '0a938d79.json', 'r') as task:

    task_contents = json.load(task)

#(2)    print(task_contents)  

#(3)    print(list(task_contents.keys()))

#(4) Portion directly below

    demon_ex0 = task_contents['train'][0]

    demon_ex1 = task_contents['train'][1]

    demon_ex2 = task_contents['train'][2]

    demon_ex3 = task_contents['train'][3]

    test_pair = task_contents['test'][0]

    

#(5) Defining function which takes i/o matrix pair as input and outputs

# a grid with the coloring convention given in the ARC app  



from matplotlib import colors

import matplotlib.pyplot as plt 



def depiction(matrix):

    """

    A function which will depict the matrix a la the ARC html app

    """

    cmap = colors.ListedColormap(

            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']) # Ad hoc, from Starter



    norm = colors.Normalize(vmin=0, vmax=9) # Ad hoc, from Starter



    plt.matshow(matrix, cmap = cmap, norm = norm) 

    plt.grid('w')

    ax = plt.gca() 

    plt.tick_params(labelleft=False, labeltop=False) 

    ax.set_xticks(np.arange(-0.5, len(matrix[0]), 1));

    ax.set_yticks(np.arange(-0.5, len(matrix), 1));













# Defining repeated column fill and repeated row fill functions



def rep_row_fill(matrix):

    """

    Takes input grid/matrix, flushes grid accordingly 

    I.e. in accordance with how hand-solution dictates

    """

    N_colors = [] # list of number of distinct colors in each row

    color_indices = [] # indices of rows with non-black colors

    

    for i in range(len(matrix)):

        N_colors.append(len(np.unique(matrix[i])))                                            

    for i in range(len(N_colors)):

        if N_colors[i] == max(N_colors):

            color_indices.append(i)

                            

    delta = color_indices[1]-color_indices[0]

        

    # First row color flushes

    for i in range(color_indices[0], len(N_colors), 2*delta):

        for j in range(len(matrix[0])):

            matrix[i][j] = np.unique(matrix[color_indices[0]])[-1]                        

    # Second row color flushes

    for i in range(color_indices[1], len(N_colors), 2*delta):

        for j in range(len(matrix[0])):

            matrix[i][j] = np.unique(matrix[color_indices[1]])[-1]                     

    return matrix

                        

def rep_col_fill(matrix):

    """

    Takes input grid/matrix, flushes grid accordingly 

    I.e. in accordance with how hand-solution dictates

    """

    N_colors = [] # list of number of distinct colors in each column

    color_indices = [] # indices of columns with non-black colors

    col_form = np.array(matrix).T

    

    for j in range(len(col_form)):

        N_colors.append(len(np.unique(col_form[j])))                            

    for i in range(len(N_colors)):

        if N_colors[i] == max(N_colors):

            color_indices.append(i)

            

    delta = color_indices[1]-color_indices[0]

    

    # First column color flushes

    for j in range(color_indices[0], len(N_colors), 2*delta):

        for i in range(len(matrix)):

            matrix[i][j] = np.unique(col_form[color_indices[0]])[-1]

    # Second column color flushes

    for j in range(color_indices[1], len(N_colors), 2*delta):

        for i in range(len(matrix)):

            matrix[i][j] = np.unique(col_form[color_indices[1]])[-1] 

    return matrix





# Defining solution function



def output(in_matrix):

    if len(in_matrix) > len(in_matrix[0]): # Taller than wide

        rep_row_fill(in_matrix)

    elif len(in_matrix) < len(in_matrix[0]): # Wider than tall

        rep_col_fill(in_matrix)    

    return in_matrix



depiction(test_pair['input'])

depiction(test_pair['output'])

depiction(output(test_pair['input']))

    




