import numpy as np
import pandas as pd
import itertools
import random
import os
import json
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
training_tasks = sorted(os.listdir(training_path))
evaluation_path = data_path / 'evaluation'
evaluation_tasks = sorted(os.listdir(evaluation_path))
test_path = data_path / 'test'
test_tasks = sorted(os.listdir(test_path))
def plot_output(task, program):
    for test_part in task['test']:
        image = [np.array(test_part['input'])]
   
        for function in program:
            image=function(image)    
        plt.figure()
        
        cmap = colors.ListedColormap(
                ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
                 '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
        norm = colors.Normalize(vmin=0, vmax=9)
               
        for num in range(0,len(image)):
            plt.imshow(image[num], cmap=cmap, norm=norm)
            #plt.grid()   
            plt.show()   

cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
def plot_one(ax, i,train_or_test,input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' '+input_or_output)
    

def plot_task(task, has_testoutput=True):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """    
    num_train = len(task['train'])
    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))
    for i in range(num_train):     
        plot_one(axs[0,i],i,'train','input')
        plot_one(axs[1,i],i,'train','output')        
    plt.tight_layout()
    plt.show()        
        
    num_test = len(task['test'])
    
    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))
    
    if num_test==1: 
        plot_one(axs[0],0,'test','input')
        if has_testoutput:
            plot_one(axs[1],0,'test','output')     
    else:
        for i in range(num_test):      
            plot_one(axs[0,i],i,'test','input')
            if has_testoutput:
                plot_one(axs[1,i],i,'test','output')  
    plt.tight_layout()
    plt.show() 

    
# Display each output of the function
def show_image_list(images):
    """ Show each image contained in a list. """
    p = plt.figure().subplots(1, len(images))
    if len(images) > 1:
        for i, image in enumerate(images):
            p[i].imshow(image, cmap=cmap, norm=norm)
    elif len(images) == 1:
        p.imshow(images[0], cmap=cmap, norm=norm)


# describe a program as a human readable string.
def program_desc(program):
    """ Create a human readable description of a program. """
    desc = [x.__name__ for x in program]
    return(' >> '.join(desc))
for first_col in range(0,10):
    for second_col in range(0,10):
        if first_col != second_col:
            exec("""def c{first_color}_to_c{second_color}_unlifted(pixmap):
                         pixmap[pixmap==int({first_color})]=int({second_color})
                         return [pixmap]
            """.format(first_color=str(first_col), second_color=str(second_col))) in globals()
            
#            exec("""def unite_c{colorstr}_horizontally_in_c{colorstr2}_unlifted(pixmap):   
#               for row_index, row in enumerate(pixmap):
#                   row_nums=[]
#                   for index, value in np.ndenumerate(row):
#                       if value==int({colorstr}):
#                           row_nums.append(index)
#                   if len(row_nums)>1:
#                       for index, value in np.ndenumerate(row):
#                           if (index >= min(row_nums)) & (index <= max(row_nums)) & (value == 0):
#                               pixmap[row_index, index]=int({colorstr})
#               return [pixmap]
#           """.format(colorstr=str(color), colorstr2=str(color))) in globals()


for color in range(0,10):
    exec("""def to_c{colorstr}_unlifted(pixmap):
                pixmap[pixmap>0]=int({colorstr})
                return [pixmap]
            """.format(colorstr=str(color))) in globals()
 

    exec("""def unite_c{colorstr}_horizontally_unlifted(pixmap):   
                for row_index, row in enumerate(pixmap):
                    row_nums=[]
                    for index, value in np.ndenumerate(row):
                        if value==int({colorstr}):
                            row_nums.append(index)
                    if len(row_nums)>1:
                        for index, value in np.ndenumerate(row):
                            if (index >= min(row_nums)) & (index <= max(row_nums)) & (value == 0):
                                pixmap[row_index, index]=int({colorstr})
                return [pixmap]
            """.format(colorstr=str(color))) in globals()
    
    exec("""def unite_c{colorstr}_vertically_unlifted(pixmap):
            transp=pixmap.T
            for col_index, col in enumerate(transp):
                col_nums=[]
                for index, value in np.ndenumerate(col):
                    if value==int({colorstr}):
                        col_nums.append(index)                
                if len(col_nums)>1:
                    for index, value in np.ndenumerate(col):
                        if (index >= min(col_nums)) & (index <=max(col_nums)) & (value == 0):
                            transp[col_index, index]=int({colorstr})
            return [transp.T]
        """.format(colorstr=str(color))) in globals()
    
    exec("""def add_c{colorstr}_frame_to_single_squares_unlifted(pixmap):
                framed=np.pad(pixmap, (1,1), "constant", constant_values=(0,0))   
                for x in range(1, framed.shape[0]-2):
                    for y in range(1, framed.shape[1]-2):
                        x_=x-1
                        y_=y-1
                        if (pixmap[x_, y_]!=pixmap[x_+1, y_]) & (pixmap[x_, y_]!=pixmap[x_-1, y_]) & (pixmap[x_, y_]!=pixmap[x_, y_+1]) & (pixmap[x_, y_]!=pixmap[x_, y_-1]):
                            framed[x-1, y]=framed[x, y-1]=framed[x+1, y]=framed[x, y+1]=framed[x+1, y+1]=framed[x-1, y+1]=framed[x+1, y-1]=framed[x-1, y-1]=int({colorstr})   
                return [framed[1:-1,1:-1]]
        """.format(colorstr=str(color))) in globals()
    
    if color > 0:
        exec("""def hollow_c{colorstr}_unlifted(pixmap):
                    framed = np.pad(pixmap, 1, "constant", constant_values=0)
                    for x in range(1, framed.shape[0]-2):
                        for y in range(1, framed.shape[1]-2):
                            x_=x-1
                            y_=y-1
                            if (pixmap[x_, y_]==int({colorstr})) & (pixmap[x_+1, y_]==int({colorstr})) & (pixmap[x_-1, y_]==int({colorstr})) & (pixmap[x_, y_+1]==int({colorstr})) & (pixmap[x_, y_-1]==int({colorstr})):
                                framed[x,y]=0
                    return [framed[1:-1,1:-1]]
            """.format(colorstr=str(color))) in globals()
        

                # copied
        exec("""def fill_enclosed_area_c{colorstr}_unlifted(arr):
                        # depth first search
                        H, W = arr.shape
                        Dy = [0, -1, 0, 1]
                        Dx = [1, 0, -1, 0]
                        arr_padded = np.pad(arr, ((1,1),(1,1)), "constant", constant_values=0)
                        searched = np.zeros(arr_padded.shape, dtype=bool)
                        searched[0, 0] = True
                        q = [(0, 0)]
                        while q:
                            y, x = q.pop()
                            for dy, dx in zip(Dy, Dx):
                                y_, x_ = y+dy, x+dx
                                if not 0 <= y_ < H+2 or not 0 <= x_ < W+2:
                                    continue
                                if not searched[y_][x_] and arr_padded[y_][x_]==0:
                                    q.append((y_, x_))
                                    searched[y_, x_] = True
                        res = searched[1:-1, 1:-1]
                        res |= arr!=0  
                        return [arr+~res*int({colorstr})]
            """.format(colorstr=str(color))) in globals()
# np.array -> [np.array]
def groupByColor_unlifted(pixmap):
    """ Split an image into a collection of images with unique color """
    # Count the number of colors
    nb_colors = int(pixmap.max()) + 1
    # Create a pixmap for each color
    splited = [(pixmap == i) * i for i in range(1, nb_colors)]
    # Filter out empty images
    return [x for x in splited if np.any(x)]

# np.array -> [np.array]
def cropToContent_unlifted(pixmap):
    """ Crop an image to fit exactly the non 0 pixels """
    # Op argwhere will give us the coordinates of every non-zero point
    true_points = np.argwhere(pixmap)
    if len(true_points) == 0:
        return []
    # Take the smallest points and use them as the top left of our crop
    top_left = true_points.min(axis=0)
    # Take the largest points and use them as the bottom right of our crop
    bottom_right = true_points.max(axis=0)
    # Crop inside the defined rectangle
    pixmap = pixmap[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    return [pixmap]

# np.array -> [np.array]
def splitH_unlifted(pixmap):
    """ Split horizontally an image """
    h = pixmap.shape[0]
    if h % 2 == 1:
        h = h // 2
        return [pixmap[:h,:], pixmap[h+1:,:]]
    else:
        h = h // 2
        return [pixmap[:h,:], pixmap[h:,:]]

# np.array -> [np.array]
def negative_unlifted(pixmap):
    """ Compute the negative of an image (and conserve the color) """
    negative = np.logical_not(pixmap).astype(int)
    color = max(pixmap.max(), 1)
    return [negative * color]

# Added by fernandeslouro

# np.array -> [np.array]
def splitV_unlifted(pixmap):
    """ Split vertically an image """
    h = pixmap.shape[1] # horizontal dimension
    if h % 2 == 1: # if horizontal dimension not pair
        h = h // 2 # floor division - results after decimal points are removed - ???????????????????????????????????
        return [pixmap[:,:h], pixmap[:,h+1:]]
    else:
        h = h // 2
        return [pixmap[:,:h], pixmap[:,h:]]
    

# np.array -> [np.array]
def rotate_counterclockwise_unlifted(pixmap):
    """ Rotates array counterclockwise """
    return [np.rot90(pixmap, 1)]

# np.array -> [np.array]
def rotate_clockwise_unlifted(pixmap):
    """ Rotates array clockwise """
    return [np.rot90(pixmap, 3)]

# np.array -> [np.array]
def duplicate_unlifted(pixmap):
    """ One element of the array becomes three """
    return [np.repeat(np.repeat(pixmap, 2, axis=0), 2, axis=1)]

# np.array -> [np.array]
def triplicate_unlifted(pixmap):
    """ One element of the array becomes three """
    return [np.repeat(np.repeat(pixmap, 3, axis=0), 3, axis=1)]

# copied
def get_enclosed_area_unlifted(arr):
        # depth first search
        H, W = arr.shape
        Dy = [0, -1, 0, 1]
        Dx = [1, 0, -1, 0]
        arr_padded = np.pad(arr, ((1,1),(1,1)), "constant", constant_values=0)
        searched = np.zeros(arr_padded.shape, dtype=bool)
        searched[0, 0] = True
        q = [(0, 0)]
        while q:
            y, x = q.pop()
            for dy, dx in zip(Dy, Dx):
                y_, x_ = y+dy, x+dx
                if not 0 <= y_ < H+2 or not 0 <= x_ < W+2:
                    continue
                if not searched[y_][x_] and arr_padded[y_][x_]==0:
                    q.append((y_, x_))
                    searched[y_, x_] = True
        res = searched[1:-1, 1:-1]
        res |= arr!=0
        return [arr, ~res*2]
               
    
# np.array -> [np.array]
def extrapolate_unlifted(pixmap): #It seems that extrpolate will crash given the nature of this DSL or whatever
    """ Expand the pattern, duplicating it where different from zero (like task 0) """
    pixmap_upsampled = pixmap.repeat(pixmap.shape[0], axis=0).repeat(pixmap.shape[1], axis=1)
    pixmap_tiled = np.tile(pixmap, pixmap.shape)
    output=[pixmap_upsampled, pixmap_tiled]
    return [np.bitwise_and.reduce(np.array(output).astype(int))]

# np.array -> [np.array]
def lower_elements_unlifted(pixmap):
    pixmap=np.concatenate(([np.zeros(pixmap.shape[1])], pixmap), axis=0)
    return[pixmap[:pixmap.shape[0]-1]]

def extrapolate_unlifted(pixmap): #It seems that extrpolate will crash given the nature of this DSL or whatever
    """ Expand the pattern, duplicating it where different from zero (like task 0) """
    pixmap_upsampled = pixmap.repeat(pixmap.shape[0], axis=0).repeat(pixmap.shape[1], axis=1)
    pixmap_tiled = np.tile(pixmap, pixmap.shape)
    output=[pixmap_upsampled, pixmap_tiled]
    return [np.bitwise_and.reduce(np.array(output).astype(int))]

def duplicate_horizontally_unlifted(pixmap):
    """Adds copy to the list side by side"""
    return[np.concatenate((pixmap, pixmap), axis=1)]

def duplicate_horizontally_symmetrically_unlifted(pixmap):
    """Adds copy to the list side by side, maintaining symmetry"""
    return[np.concatenate((pixmap, np.fliplr(pixmap)), axis=1)]

def duplicate_vertically_unlifted(pixmap):
    """Adds copy to the list below"""
    return[np.concatenate((pixmap, pixmap), axis=0)]

def duplicate_vertically_symmetrically_unlifted(pixmap):
    """Adds copy to the list below, maintaining symmetry"""
    return[np.concatenate((pixmap, np.flipud(pixmap)), axis=0)]

def flip_vertically_unlifted(pixmap):
    return[np.flipud(pixmap)]

def flip_horizontally_unlifted(pixmap):
    return[np.fliplr(pixmap)]

def rotate_colors_unlifted(pixmap):
    colors=[x for x in np.unique(pixmap) if x > 0]
    if len(colors)==0:
        return[pixmap]
    switch={}
    prevcolor=colors[-1]
    for color in colors:
        switch[color]=prevcolor
        prevcolor=color
    for (x,y), value in np.ndenumerate(pixmap):
        if value > 0:
            pixmap[x,y]=switch[value]
    return [pixmap]

def hollow_all_unlifted(pixmap):
    framed = np.pad(pixmap, 1, "constant", constant_values=0)
    for color in range(1,10):
        for x in range(1, framed.shape[0]-2):
            for y in range(1, framed.shape[1]-2):
                x_=x-1
                y_=y-1
                if (pixmap[x_, y_]==color) & (pixmap[x_+1, y_]==color) & (pixmap[x_-1, y_]==color) & (pixmap[x_, y_+1]==color) & (pixmap[x_, y_-1]==color):
                    framed[x,y]=0
    return [framed[1:-1,1:-1]]

#being ignored at the moment
def paint_all_most_common_color_unlifted(pixmap):
    most_common_color=np.argmax(np.bincount(pixmap[pixmap>0].flatten().astype(int))) #SOMETIMES IT'S NOT IN INT, WHY?????????????? - THAT CAUSES ERROR HEREcolor
    pixmap[pixmap>0]=most_common_color
    return[pixmap]

# [np.array] -> [np.array]
def identity(x: [np.array]):
    return x

# [np.array] -> [np.array]
def tail(x):
    if len(x) > 1:
        return x[1:]
    else:
        return x

# [np.array] -> [np.array]
def init(x):
    if len(x) > 1:
        return x[:1]
    else:
        return x

# [np.array] -> [np.array]
def union(x):
    """ Compute the pixel union of all images in the list. """
    if len(x) < 2:
        return x
    
    # Make sure everybody have the same shape
    first_shape = tuple(x[0].shape)
    for pixmap in x[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    
    return [np.bitwise_or.reduce(np.array(x).astype(int))]
    
def intersect(x):
    """ Compute the pixel intersection of all images in the list. """
    if len(x) < 2:
        return x
    
    # Make sure everybody have the same shape
    first_shape = tuple(x[0].shape)
    for pixmap in x[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    
    return [(np.prod(np.array(x), axis=0) > 0).astype(int)]

def sortByColor(xs):
    """ Sort pictures by increasing color id. """
    xs = [x for x in xs if len(x.reshape(-1)) > 0]
    return list(sorted(xs, key=lambda x: x.max()))

def sortByWeight(xs):
    """ Sort images by how many non zero pixels are contained. """
    xs = [x for x in xs if len(x.reshape(-1)) > 0]
    return list(sorted(xs, key=lambda x: (x>0).sum()))

def reverse(x):
    """ Reverse the order of a list of images. """
    return x[::-1]

#-------
# added by fernandeslouro
def sortByNumberOfColors(xs):
    """ Sort pictures by increasing number of colors. """
    return list(sorted(xs, key=lambda x: len(np.unique(x))))


def xor(x):
    """ Compute the xor of all images in the list. """
    if len(x) < 2:
        return x
    
    # Make sure everybody have the same shape
    first_shape = tuple(x[0].shape)
    for pixmap in x[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    
    return [np.bitwise_xor.reduce(np.array(x).astype(int))]
def lift(fct):
    # Lift the function
    def lifted_function(xs):
        list_of_results = [fct(x) for x in xs]
        return list(itertools.chain(*list_of_results))
    # Give a nice name to the lifted function
    import re
    lifted_function.__name__ = re.sub('_unlifted$', '_lifted', fct.__name__)
    return lifted_function

def lift_functions_list(unlifted_functions):
    # Lift the function
    
    # Give a nice name to the lifted function
    lifted_functions=[]
    for fct in unlifted_functions:
        def lifted_function(xs):
            list_of_results = [fct(x) for x in xs]
            return list(itertools.chain(*list_of_results))
        import re
        lifted_function.__name__ = re.sub('_unlifted$', '_lifted', fct.__name__)
        lifted_functions.append(lifted_function)
    return lifted_functions
   
cropToContent = lift(cropToContent_unlifted)
groupByColor = lift(groupByColor_unlifted)
splitH = lift(splitH_unlifted)
negative = lift(negative_unlifted)

initial_functions=[tail, init, union, intersect, sortByColor, sortByWeight, reverse,\
                   cropToContent, groupByColor, splitH, negative]
#--
to_c0=lift(to_c0_unlifted)
to_c1=lift(to_c1_unlifted)
to_c2=lift(to_c2_unlifted)
to_c3=lift(to_c3_unlifted)
to_c4=lift(to_c4_unlifted)
to_c5=lift(to_c5_unlifted)
to_c6=lift(to_c6_unlifted)
to_c7=lift(to_c7_unlifted)
to_c8=lift(to_c8_unlifted)
to_c9=lift(to_c9_unlifted)

single_color_functions=[to_c1,to_c2, to_c3, to_c4, to_c5, to_c6, to_c7, to_c8, to_c9] 

#--
c1_to_c0=lift(c1_to_c0_unlifted)
c1_to_c2=lift(c1_to_c2_unlifted)
c1_to_c3=lift(c1_to_c3_unlifted)
c1_to_c4=lift(c1_to_c4_unlifted)
c1_to_c5=lift(c1_to_c5_unlifted)
c1_to_c6=lift(c1_to_c6_unlifted)
c1_to_c7=lift(c1_to_c7_unlifted)
c1_to_c8=lift(c1_to_c8_unlifted)
c1_to_c9=lift(c1_to_c9_unlifted)
c2_to_c0=lift(c2_to_c0_unlifted)
c2_to_c1=lift(c2_to_c1_unlifted)
c2_to_c3=lift(c2_to_c3_unlifted)
c2_to_c4=lift(c2_to_c4_unlifted)
c2_to_c5=lift(c2_to_c5_unlifted)
c2_to_c6=lift(c2_to_c6_unlifted)
c2_to_c7=lift(c2_to_c7_unlifted)
c2_to_c8=lift(c2_to_c8_unlifted)
c2_to_c9=lift(c2_to_c9_unlifted)
c3_to_c0=lift(c3_to_c0_unlifted)
c3_to_c1=lift(c3_to_c1_unlifted)
c3_to_c2=lift(c3_to_c2_unlifted)
c3_to_c4=lift(c3_to_c4_unlifted)
c3_to_c5=lift(c3_to_c5_unlifted)
c3_to_c6=lift(c3_to_c6_unlifted)
c3_to_c7=lift(c3_to_c7_unlifted)
c3_to_c8=lift(c3_to_c8_unlifted)
c3_to_c9=lift(c3_to_c9_unlifted)
c4_to_c0=lift(c4_to_c0_unlifted)
c4_to_c1=lift(c4_to_c1_unlifted)
c4_to_c2=lift(c4_to_c2_unlifted)
c4_to_c3=lift(c4_to_c3_unlifted)
c4_to_c5=lift(c4_to_c5_unlifted)
c4_to_c6=lift(c4_to_c6_unlifted)
c4_to_c7=lift(c4_to_c7_unlifted)
c4_to_c8=lift(c4_to_c8_unlifted)
c4_to_c9=lift(c4_to_c9_unlifted)
c5_to_c0=lift(c5_to_c0_unlifted)
c5_to_c1=lift(c5_to_c1_unlifted)
c5_to_c2=lift(c5_to_c2_unlifted)
c5_to_c3=lift(c5_to_c3_unlifted)
c5_to_c4=lift(c5_to_c4_unlifted)
c5_to_c6=lift(c5_to_c6_unlifted)
c5_to_c7=lift(c5_to_c7_unlifted)
c5_to_c8=lift(c5_to_c8_unlifted)
c5_to_c9=lift(c5_to_c9_unlifted)
c6_to_c0=lift(c6_to_c0_unlifted)
c6_to_c1=lift(c6_to_c1_unlifted)
c6_to_c2=lift(c6_to_c2_unlifted)
c6_to_c3=lift(c6_to_c3_unlifted)
c6_to_c4=lift(c6_to_c4_unlifted)
c6_to_c5=lift(c6_to_c5_unlifted)
c6_to_c7=lift(c6_to_c7_unlifted)
c6_to_c8=lift(c6_to_c8_unlifted)
c6_to_c9=lift(c6_to_c9_unlifted)
c7_to_c0=lift(c7_to_c0_unlifted)
c7_to_c1=lift(c7_to_c1_unlifted)
c7_to_c2=lift(c7_to_c2_unlifted)
c7_to_c3=lift(c7_to_c3_unlifted)
c7_to_c4=lift(c7_to_c4_unlifted)
c7_to_c5=lift(c7_to_c5_unlifted)
c7_to_c6=lift(c7_to_c6_unlifted)
c7_to_c8=lift(c7_to_c8_unlifted)
c7_to_c9=lift(c7_to_c9_unlifted)
c1_to_c0=lift(c1_to_c0_unlifted)
c8_to_c0=lift(c8_to_c0_unlifted)
c8_to_c1=lift(c8_to_c1_unlifted)
c8_to_c2=lift(c8_to_c2_unlifted)
c8_to_c3=lift(c8_to_c3_unlifted)
c8_to_c4=lift(c8_to_c4_unlifted)
c8_to_c5=lift(c8_to_c5_unlifted)
c8_to_c6=lift(c8_to_c6_unlifted)
c8_to_c7=lift(c8_to_c7_unlifted)
c8_to_c9=lift(c8_to_c9_unlifted)
c9_to_c0=lift(c9_to_c0_unlifted)
c9_to_c1=lift(c9_to_c1_unlifted)
c9_to_c2=lift(c9_to_c2_unlifted)
c9_to_c3=lift(c9_to_c3_unlifted)
c9_to_c4=lift(c9_to_c4_unlifted)
c9_to_c5=lift(c9_to_c5_unlifted)
c9_to_c6=lift(c9_to_c6_unlifted)
c9_to_c7=lift(c9_to_c7_unlifted)
c9_to_c8=lift(c9_to_c8_unlifted)

color_switch_functions = [c1_to_c0, c1_to_c2, c1_to_c3, c1_to_c4, c1_to_c5, c1_to_c6,\
                          c1_to_c7, c1_to_c8, c1_to_c9, c2_to_c0, c2_to_c1, c2_to_c3,\
                          c2_to_c4, c2_to_c5, c2_to_c6, c2_to_c7, c2_to_c8,\
                          c2_to_c9, c3_to_c0, c3_to_c1, c3_to_c2, c3_to_c4, c3_to_c5,\
                          c3_to_c6, c3_to_c7, c3_to_c8, c3_to_c9, c4_to_c0, c4_to_c1,\
                          c4_to_c2, c4_to_c3, c4_to_c5, c4_to_c6, c4_to_c7,\
                          c4_to_c8, c4_to_c9, c5_to_c0, c5_to_c1, c5_to_c2, c5_to_c3,\
                          c5_to_c4, c5_to_c6, c5_to_c7, c5_to_c8, c5_to_c9, c6_to_c0,\
                          c6_to_c1, c6_to_c2, c6_to_c3, c6_to_c4, c6_to_c5,\
                          c6_to_c7, c6_to_c8, c6_to_c9, c7_to_c0, c7_to_c1, c7_to_c2,\
                          c7_to_c3, c7_to_c4, c7_to_c5, c7_to_c6, c7_to_c8,\
                          c7_to_c9, c8_to_c0, c8_to_c1, c8_to_c2, c8_to_c3, c8_to_c4,\
                          c8_to_c5, c8_to_c6, c8_to_c7, c8_to_c9, c9_to_c0, c9_to_c1,\
                          c9_to_c2, c9_to_c3, c9_to_c4, c9_to_c5, c9_to_c6,\
                          c9_to_c7, c9_to_c8]

#--
unite_c1_horizontally=lift(unite_c1_horizontally_unlifted)
unite_c2_horizontally=lift(unite_c2_horizontally_unlifted)
unite_c3_horizontally=lift(unite_c3_horizontally_unlifted)
unite_c4_horizontally=lift(unite_c4_horizontally_unlifted)
unite_c5_horizontally=lift(unite_c5_horizontally_unlifted)
unite_c6_horizontally=lift(unite_c6_horizontally_unlifted)
unite_c7_horizontally=lift(unite_c7_horizontally_unlifted)
unite_c8_horizontally=lift(unite_c8_horizontally_unlifted)
unite_c9_horizontally=lift(unite_c9_horizontally_unlifted)

unite_c1_vertically=lift(unite_c1_vertically_unlifted)
unite_c2_vertically=lift(unite_c2_vertically_unlifted)
unite_c3_vertically=lift(unite_c3_vertically_unlifted)
unite_c4_vertically=lift(unite_c4_vertically_unlifted)
unite_c5_vertically=lift(unite_c5_vertically_unlifted)
unite_c6_vertically=lift(unite_c6_vertically_unlifted)
unite_c7_vertically=lift(unite_c7_vertically_unlifted)
unite_c8_vertically=lift(unite_c8_vertically_unlifted)
unite_c9_vertically=lift(unite_c9_vertically_unlifted)

unite_colors_functions=[unite_c1_horizontally, unite_c2_horizontally, unite_c3_horizontally,\
                        unite_c4_horizontally, unite_c5_horizontally, unite_c6_horizontally,\
                        unite_c7_horizontally, unite_c8_horizontally, unite_c9_horizontally,\
                        unite_c1_vertically, unite_c2_vertically, unite_c3_vertically,\
                        unite_c4_vertically, unite_c5_vertically, unite_c6_vertically,\
                        unite_c7_vertically, unite_c8_vertically, unite_c9_vertically]


#--
add_c1_frame_to_single_squares=lift(add_c1_frame_to_single_squares_unlifted)
add_c2_frame_to_single_squares=lift(add_c2_frame_to_single_squares_unlifted)
add_c3_frame_to_single_squares=lift(add_c3_frame_to_single_squares_unlifted)
add_c4_frame_to_single_squares=lift(add_c4_frame_to_single_squares_unlifted)
add_c5_frame_to_single_squares=lift(add_c5_frame_to_single_squares_unlifted)
add_c6_frame_to_single_squares=lift(add_c6_frame_to_single_squares_unlifted)
add_c7_frame_to_single_squares=lift(add_c7_frame_to_single_squares_unlifted)
add_c8_frame_to_single_squares=lift(add_c8_frame_to_single_squares_unlifted)
add_c9_frame_to_single_squares=lift(add_c9_frame_to_single_squares_unlifted)

add_frame_colors_functions= [add_c1_frame_to_single_squares, add_c2_frame_to_single_squares, add_c3_frame_to_single_squares, add_c4_frame_to_single_squares, add_c5_frame_to_single_squares, add_c6_frame_to_single_squares, add_c7_frame_to_single_squares, add_c8_frame_to_single_squares, add_c9_frame_to_single_squares]


#----

hollow_c1=lift(hollow_c1_unlifted)
hollow_c2=lift(hollow_c2_unlifted)
hollow_c3=lift(hollow_c3_unlifted)
hollow_c4=lift(hollow_c4_unlifted)
hollow_c5=lift(hollow_c5_unlifted)
hollow_c6=lift(hollow_c6_unlifted)
hollow_c7=lift(hollow_c7_unlifted)
hollow_c8=lift(hollow_c8_unlifted)
hollow_c9=lift(hollow_c9_unlifted)
hollow_all=lift(hollow_all_unlifted)

hollow_functions=[hollow_c1,hollow_c2,hollow_c3,hollow_c4,hollow_c5,hollow_c6,hollow_c7,hollow_c8,hollow_c9, hollow_all]

#--

fill_enclosed_area_c1=lift(fill_enclosed_area_c1_unlifted)
fill_enclosed_area_c2=lift(fill_enclosed_area_c2_unlifted)
fill_enclosed_area_c3=lift(fill_enclosed_area_c3_unlifted)
fill_enclosed_area_c4=lift(fill_enclosed_area_c4_unlifted)
fill_enclosed_area_c5=lift(fill_enclosed_area_c5_unlifted)
fill_enclosed_area_c6=lift(fill_enclosed_area_c6_unlifted)
fill_enclosed_area_c7=lift(fill_enclosed_area_c7_unlifted)
fill_enclosed_area_c8=lift(fill_enclosed_area_c8_unlifted)
fill_enclosed_area_c9=lift(fill_enclosed_area_c9_unlifted)

fill_enclosed_functions=[fill_enclosed_area_c1, fill_enclosed_area_c2, fill_enclosed_area_c3, fill_enclosed_area_c4, fill_enclosed_area_c5, fill_enclosed_area_c6, fill_enclosed_area_c7, fill_enclosed_area_c8, fill_enclosed_area_c9]


#--
splitV = lift(splitV_unlifted)
rotate_counterclockwise=lift(rotate_counterclockwise_unlifted)
rotate_clockwise=lift(rotate_clockwise_unlifted)
triplicate=lift(triplicate_unlifted)
duplicate=lift(duplicate_unlifted)
get_enclosed_area=lift(get_enclosed_area_unlifted)
extrapolate=lift(extrapolate_unlifted)
lower_elements=lift(lower_elements_unlifted)
duplicate_horizontally=lift(duplicate_horizontally_unlifted)
duplicate_horizontally_symmetrically=lift(duplicate_horizontally_symmetrically_unlifted)
duplicate_vertically=lift(duplicate_vertically_unlifted)
duplicate_vertically_symmetrically=lift(duplicate_vertically_symmetrically_unlifted)
flip_horizontally=lift(flip_horizontally_unlifted)
flip_vertically=lift(flip_vertically_unlifted)
rotate_colors=lift(rotate_colors_unlifted)
paint_all_most_common_color=lift(paint_all_most_common_color_unlifted)

added_functions=[splitV, rotate_counterclockwise, rotate_clockwise, triplicate,\
                 duplicate, get_enclosed_area, lower_elements, sortByNumberOfColors,\
                 xor, duplicate_horizontally, duplicate_horizontally_symmetrically,\
                 duplicate_vertically, duplicate_vertically_symmetrically,\
                 flip_vertically, flip_horizontally, rotate_colors]
                 #paint_all_most_common_color
#does not include extrapolate - it blows up memory, for some reason
all_functions=initial_functions+single_color_functions+color_switch_functions+unite_colors_functions+add_frame_colors_functions+\
                        hollow_functions+fill_enclosed_functions+added_functions
def evaluate(program: [], input_image: np.array):
    # Make sure the input is a np.array
    input_image = np.array(input_image)
    assert type(input_image) == np.ndarray
    
    # Apply each function on the image
    image_list = [input_image]
    for fct in program:
        # Apply the function
        image_list = fct(image_list)
        # Filter out empty images
        image_list = [img for img in image_list if img.shape[0] > 0 and img.shape[1] > 0]
        # Break if there is no data
        if image_list == []:
            return []
    return image_list        
def are_two_images_equals(a, b):
    if tuple(a.shape) == tuple(b.shape):
        if (np.abs(b-a) < 1).all():
            return True
    return False

def is_solution(program, task, verbose=True): #task is actually task['train'] - this function only analyses inputs and I am dumb
    for sample in task: # For each pair input/output
        i = np.array(sample['input'])
        o = np.array(sample['output'])

        # Evaluate the program on the input
        images = evaluate(program, i)
        if len(images) < 1:
            return False
        
        # The solution should be in the 3 first outputs
        images = images[:3]
        
        # Check if the output is in the 3 images produced
        is_program_of_for_sample = any([are_two_images_equals(x, o) for x in images])
        if not is_program_of_for_sample:
            return False
    
    return True
def width_fitness(predicted, expected_output):
    """ How close the predicted image is to have the right width. Less is better."""
    return np.abs(predicted.shape[0] - expected_output.shape[0])

def height_fitness(predicted, expected_output):
    """ How close the predicted image is to have the right height. Less is better."""
    return np.abs(predicted.shape[1] - expected_output.shape[1])

def activated_pixels_fitness(p, e):
    """ How close the predicted image to have the right pixels. Less is better."""
    shape = (max(p.shape[0], e.shape[0]), max(p.shape[1], e.shape[1]))
    diff = np.zeros(shape, dtype=int)
    diff[0:p.shape[0], 0:p.shape[1]] = (p > 0).astype(int)
    diff[0:e.shape[0], 0:e.shape[1]] -= (e > 0).astype(int)
    
    return (diff != 0).sum()

def colors_fitness(p, e):
    p_colors = np.unique(p)
    e_colors = np.unique(e)
    
    nb_inter = len(np.intersect1d(p_colors, e_colors))

    return (len(p_colors) - nb_inter) + (len(e_colors) - nb_inter)

fitness_functions = [colors_fitness, activated_pixels_fitness, height_fitness, width_fitness]

def product_less(a, b):
    """ Return True iff the two tuples a and b respect a<b for the partial order. """
    a = np.array(a)
    b = np.array(b)
    return (np.array(a) < np.array(b)).all()    
# ([[np.array] -> [np.array]], Taks) -> (int, int, ..., int)
def evaluate_fitness(program, task):
    """ Take a program and a task, and return its fitness score as a tuple. """
    score = np.zeros((len(fitness_functions)))
    
    # For each sample
    for sample in task:
        i = np.array(sample['input'])
        o = np.array(sample['output'])
        
        # For each fitness function
        for index, fitness_function in enumerate(fitness_functions):
            images = evaluate(program, i)
            if images == []: # Penalize no prediction!
                score[index] += 500
            else: # Take only the score of the first output
                score[index] = fitness_function(images[0], o)
    return tuple(score)
def build_candidates(allowed_nodes=[identity], best_candidates=[], nb_candidates=200):
    """
    Create a poll of fresh candidates using the `allowed_nodes`.
    
    The pool contain a mix of new single instructions programs
    and mutations of the best candidates.
    """
    new_candidates = []
    length_limit = 4 # Maximal length of a program
    
    def random_node():
        return random.choice(allowed_nodes)
    
    # Until we have enougth new candidates
    while(len(new_candidates) < nb_candidates):
        # Add 10 new programs
        for i in range(5):
            new_candidates += [[random_node()]]
        
        # Create new programs based on each best candidate
        for best_program in best_candidates:
            # Add one op on its right but limit the length of the program
            if len(best_program) < length_limit - 1:
                new_candidates += [[random_node()] + best_program]
            # Add one op on its left but limit the length of the program
            if len(best_program) < length_limit - 1:
                new_candidates += [best_program + [random_node()]]
            # Mutate one instruction of the existing program
            new_candidates += [list(best_program)]
            new_candidates[-1][random.randrange(0, len(best_program))] = random_node()
   
    # Truncate if we have too many candidates
    np.random.shuffle(new_candidates)
    return new_candidates[:nb_candidates]

# Test the function by building some candidates
#len(build_candidates(allowed_nodes=[identity], best_candidates=[[identity]], nb_candidates=42))
def remove_functions(all_functions, to_remove):
    reduced_functions=[]        
    for function in all_functions:
        fname=str(function.__name__)
        if '_lifted' in fname:
            fname=fname[:-7]
        if fname not in to_remove:
            reduced_functions.append(function)
    return reduced_functions
    

def color_change_heuristics(all_functions_list, task):
    """When a color is not present on the avaliable input/output grids, do not consider functions that change colors from/to that color"""
    outputs=np.zeros(0)
    for test_or_train in [task['train'], task['test']]:
        for x in test_or_train:
            if len(x)==2:
                outputs=np.append(outputs,x['output'])      
    to_remove=[]
    for color in range (0,10):
        if color not in np.unique(outputs):
            to_remove+=['to_c'+str(color)]
            other_colors= list(range(0,10))
            other_colors.remove(color)
            for other_color in other_colors:
                to_remove+=['c'+str(other_color)+'_to_c'+str(color)]
                
    inputs=np.zeros(0)
    for test_or_train in [task['train'], task['test']]:
        for x in test_or_train:
            inputs=np.append(inputs,x['input'])
    for color in range (1,10):
        if color not in np.unique(inputs):
            other_colors= list(range(1,10))
            other_colors.remove(color)
            for other_color in other_colors:
                to_remove+=['c'+str(color)+'_to_c'+str(other_color)]
                                    
    return remove_functions(all_functions, to_remove)

def initial_color_heuristic(all_functions_list, task):
    """When a color is not present either in the output or input grids, do not consider functions refer to those colors"""
    present_colors=np.zeros(0)
    for test_or_train in [task['train'], task['test']]:
        for x in test_or_train:
            present_colors=np.append(present_colors,x['input'])
            if len(x)==2:
                present_colors=np.append(present_colors,x['output'])
    colors_not_present=[]
    for colorcode in range(0,10):
        if colorcode not in np.unique(present_colors):
            colors_not_present=np.append(colors_not_present, colorcode)
    
    to_remove=[]
    for colorcode in colors_not_present:
        for function in all_functions_list:
            if 'c'+str(int(colorcode)) in function.__name__:
                fname=function.__name__
                if '_lifted' in fname:
                    fname=fname[:-7]
                to_remove+=[fname]
    
    return remove_functions(all_functions, to_remove)            

def heuristics(all_functions_list, task):
    reduced_functions=initial_color_heuristic(all_functions_list, task)
    reduced_functions=color_change_heuristics(reduced_functions, task)
    return reduced_functions
def build_model(task, candidates_nodes=all_functions, max_iterations=20, verbose=True):
    
    if verbose:
        print("Candidates nodes are:", [program_desc([n]) for n in candidates_nodes])
        print()

    best_candidates = {} # A dictionary of {score:candidate}
    for i in range(max_iterations):
        if verbose:
            print("Iteration ", i+1)
            print("-" * 10)
        
        # Create a list of candidates
        candidates = build_candidates(candidates_nodes, best_candidates.values())
        # Keep candidates with best fitness.
        # They will be stored in the `best_candidates` dictionary
        # where the key of each program is its fitness score.
        for candidate in candidates:
            score = evaluate_fitness(candidate, task)
            is_uncomparable = True # True if we cannot compare the two candidate's scores
            
            # Compare the new candidate to the existing best candidates
            best_candidates_items = list(best_candidates.items())
            for best_score, best_candidate in best_candidates_items:
                if product_less(score, best_score):
                    # Remove previous best candidate and add the new one
                    del best_candidates[best_score]
                    best_candidates[score] = candidate
                    is_uncomparable = False # The candidates are comparable
                if product_less(best_score, score) or best_score == score:
                    is_uncomparable = False # The candidates are comparable
            if is_uncomparable: # The two candidates are uncomparable
                best_candidates[score] = candidate

        # For each best candidate, we look if we have an answer
        
        
        for program in best_candidates.values():
            if is_solution(program, task):
                return program
          
            
            
        # Give some informations by selecting a random candidate
        if verbose:
            print("Best candidates lenght:", len(best_candidates))
            random_candidate_score = random.choice(list(best_candidates.keys()))
            print("Random candidate score:", random_candidate_score)
            print("Random candidate implementation:", program_desc(best_candidates[random_candidate_score]))
    
    #print(best_candidates[0])
    
    #for program in best_candidates.values(): #what to do if program isn't found? - and does it really matter? -atm I'm not even trying
    #    return program
          
    return None
task_file = str(training_path / training_tasks[1])
with open(task_file, 'r') as f:
    task = json.load(f)
#plot_task(task)
candidate_functions=heuristics(all_functions, task)
program = build_model(task['train'], candidate_functions, max_iterations=20, verbose=False)
#print(numtask)
if program is None:
    print("No program was found")
else:
    print(f'Found program: {program_desc(program)}')
task_file = str(training_path / training_tasks[119])
with open(task_file, 'r') as f:
    task = json.load(f)
plot_task(task)
program=[]
for f in all_functions:
    if 'hollow_all' in f.__name__ or 'fill_enclosed_area_c8' in f.__name__:
        program.append(f)
program_desc(program)
plot_output(task, program)