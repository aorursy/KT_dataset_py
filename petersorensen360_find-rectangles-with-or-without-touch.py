import numpy as np

from copy import copy, deepcopy



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

from pathlib import Path

from collections import namedtuple




data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path   = data_path / 'training'

training_tasks_list = sorted(os.listdir(training_path))



#get matrix from json structur

def get_matrix(task,i,train_or_test,input_or_output):

    return task[train_or_test][i][input_or_output]



def readTasks(files_list,path,exampletype):

    training_tasks   = {}

    for i,taskname in enumerate(files_list):

        training_tasks[i] = {}

        training_tasks[i]["filename"] = taskname

        training_tasks[i]["file"]     = str(path/taskname)



        with open(training_tasks[i]["file"], 'r') as f:

            jtask = json.load(f)

            for kind in ['train']: #,'test']:

                matrix1_list = []

                num_exam = len(jtask[kind])

                for j in range(num_exam):     

                    matrix1_list.append(np.array(get_matrix(jtask,j,'train','input')))

                training_tasks[i]["matrix1_list"] = matrix1_list 

    return training_tasks



training_tasks   = readTasks(training_tasks_list,training_path,'training')

                                    
cmap = colors.ListedColormap(

    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)

# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,

# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown



def plot_matrix(ax,matrix):

    ax.imshow(matrix, cmap=cmap, norm=norm)

    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    

    ax.set_yticks([x-0.5 for x in range(1+len(matrix))])

    ax.set_xticks([x-0.5 for x in range(1+len(matrix[0]))])     

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    
def grow_ones(matrix_in):

    # sums calculated one row at a time from y=0 and downwards

    matrix  = matrix_in.copy()

    for i in range(len(matrix)-2,-1,-1):

        matrix[i] = (matrix[i] + matrix[i+1])*matrix[i] 

    return matrix



def rowIdentifyMaxRectangle(irow,matrix,mindim):

    if mindim < 1:

        mindim = 1

    dim1 = len(matrix[0])

    maxsize = maxcol1 = maxcol2 = -1

    for j in range(dim1):

        if matrix[irow][j] < mindim:

            continue

        if j > 0 and matrix[irow][j] <= matrix[irow][j-1]:

            continue  # already included

        height = matrix[irow][j]

        if mindim == 1:

            curmaxsize = size = matrix[irow][j]

        else:

            curmaxsize = size = -1

        curmaxcol1 = curmaxcol2 = j

        for j2 in range(j+1,dim1):

            if matrix[irow][j2] < mindim:

                break

            if matrix[irow][j2] < height:

                height = matrix[irow][j2]

            if j2-j < mindim-1:

                continue

            size = height*(j2-j+1)                

            if size > curmaxsize:

                curmaxsize   = size

                curmaxcol1   = j

                curmaxcol2   = j2

        if curmaxsize > maxsize:

            maxsize   = curmaxsize

            maxcol1   = curmaxcol1

            maxcol2   = curmaxcol2

    return maxsize,maxcol1,maxcol2



def matrixIdentifyMaxRectangle(matrix,mindim):

    dim0 = len(matrix)

    maxsize = maxrow = maxcol1 = maxcol2 = -1

    for i in range(len(matrix)):

        size,col1,col2 = rowIdentifyMaxRectangle(i,matrix,mindim) 

        if size > maxsize:

            maxsize = size

            maxrow   = i

            maxcol1  = col1

            maxcol2  = col2

    return maxsize,maxrow,maxcol1,maxcol2



def matrixIdentifyMaxCountRectangleOfColor(matrix_in,color,minsize=2,mindim=1,maxcount=999):

    if minsize < 2:

        minsize = 2

    matrix01 = (matrix_in == color).astype(int)

    maxsize = maxrow = maxcol1 = maxcol2 = -1

    count = 0

    rectangles = []

    while count < maxcount:

        matrix = grow_ones(matrix01)

        maxsize,maxrow,maxcol1,maxcol2 = matrixIdentifyMaxRectangle(matrix,mindim)

        if maxsize < minsize:

            break

        count += 1

        rectangles.append((maxsize,maxrow,maxcol1,maxcol2))

        height = maxsize//(maxcol2-maxcol1+1)

        matrix01[maxrow:maxrow+height,maxcol1:maxcol2+1] = 0

    return rectangles



Rectangle = namedtuple('Rectangle', 'ymin ymax xmin xmax')



def reshapeRectanglesInfo(rectangles):

    # from (size,ymin,xmin,xmax) to (ymin,ymax,xmin,xmax)

    newlist = []

    for (size,ymin,xmin,xmax) in rectangles:

        height = size//(xmax-xmin+1)

        newlist.append(Rectangle(ymin,ymin+height-1,xmin,xmax))

    return newlist



def getRectanglesWithNoTouch(rectangles):

    # this routine uses the fact that rectangles are not overlapping

    rectangles.sort(key=lambda tup: tup[0])

    keep = [True]*len(rectangles)

    keeps = []

    for i in range(len(rectangles)):

        for j in range(i+1,len(rectangles)):

            if rectangles[j].ymin > rectangles[i].ymax+1:

                break

            if rectangles[j].ymin == rectangles[i].ymax+1:

                if not (rectangles[i].xmin > rectangles[j].xmax or rectangles[i].xmax < rectangles[j].xmin):

                    keep[i] = keep[j] = False

            elif rectangles[j].xmax == rectangles[i].xmin-1 or rectangles[j].xmin == rectangles[i].xmax+1:

                if rectangles[i].ymax >= rectangles[j].ymin:

                    keep[i] = keep[j] = False

    for i in range(len(rectangles)):

        if keep[i]:

            keeps.append(rectangles[i])

    return keeps
minsize=4     # minimum size for identified rectangle

mindim=2      # minimun sidelength for identified rectangle

maxcount=100  # find only 100 biggest rectangles 



for itask,task in training_tasks.items():

    if not itask in [13,20,48,53,76,95]:

        continue

    print("Task no",itask)

    matrix1_list = task["matrix1_list"]

    fig, axs = plt.subplots(3, 3, figsize=(3*3,3*2))

    for i in range(len(matrix1_list)):

        if i > 2:

            break

        matrix1 = matrix1_list[i]

        matrix2 = np.zeros_like(matrix1)

        matrix3 = np.zeros_like(matrix1)



        backgroundcolor = 0

        colors   = np.unique(matrix1, return_counts=False)



        for color in colors:

            if color == backgroundcolor:

                continue

            rectangles = matrixIdentifyMaxCountRectangleOfColor(matrix1,color,minsize=minsize,mindim=mindim,maxcount=maxcount)

            rectangles = reshapeRectanglesInfo(rectangles)



            for rect in rectangles:

                matrix2[rect.ymin:rect.ymax+1,rect.xmin:rect.xmax+1] = color

            

            notouch_rectangles = getRectanglesWithNoTouch(rectangles)

            for rect in notouch_rectangles:

                matrix3[rect.ymin:rect.ymax+1,rect.xmin:rect.xmax+1] = color



        for k in range(3):

            r = k    #//3

            s = i%3

            if k == 0:

                plot_matrix(axs[r,s],matrix1)

            elif k == 1:

                plot_matrix(axs[r,s],matrix2)

            elif k == 2:

                plot_matrix(axs[r,s],matrix3)

    plt.tight_layout()

    plt.show() 


