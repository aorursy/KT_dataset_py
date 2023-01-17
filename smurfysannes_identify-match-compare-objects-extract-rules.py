# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd

import os

import json

from pathlib import Path

import pprint

pp = pprint.PrettyPrinter(indent=4)

from os.path import join as path_join





import matplotlib.pyplot as plt

from matplotlib import colors, cm

import pdb

#import cv2

from skimage import measure



#from my "scratch" notebook

from scipy.stats import mode



#from object notebook

import itertools

import random

from random import *



import copy   #makes a copy so old version doesn't change. e.g. dict1 = copy.deepcopy(dict1)



import warnings

warnings.filterwarnings("ignore")  #suppress all warnings



#######THIS CODE IS FOR USE WITH ANOCONDA PYTHON EDITOR IN MY DIRECTORY###########

# training_path = 'kaggle/input/abstraction-and-reasoning-challenge/training/'

# training_tasks = os.listdir(training_path)

# Trains = []

# for i in range(400):

#     task_file = str(training_path + training_tasks[i])

#     task = json.load(open(task_file, 'r'))

#     Trains.append(task)

# train_tasks = Trains

##############################################################################



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))

eval_tasks = sorted(os.listdir(training_path))

T = training_tasks

Trains = []



for i in range(400):

    task_file = str(training_path / T[i])

    task = json.load(open(task_file, 'r'))

    Trains.append(task)

def load_data(path):

    tasks = pd.Series()

    for file_path in sorted(os.listdir(path)):

        task_file = path_join(path, file_path)

        with open(task_file, 'r') as f:

            task = json.load(f)

        tasks[file_path[:-5]] = task

    return tasks

train_tasks = load_data('../input/abstraction-and-reasoning-challenge/training/')


def plot_one(ax, i,train_or_test,input_or_output, task):

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



def plot_img(input_matrix, title ='no title given'):

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    num_train = 1

    fig, ax = plt.subplots(1, num_train, figsize=(3*num_train,3*2))

    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    

    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])

    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_title(title)



def plot_task(task):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """    

    num_train = len(task['train'])

    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))

    for i in range(num_train):     

        plot_one(axs[0,i],i,'train','input', task)

        plot_one(axs[1,i],i,'train','output', task)        

    plt.tight_layout()

    plt.show()        

    num_test = len(task['test'])

    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))

    if num_test==1: 

        plot_one(axs[0],0,'test','input', task)

        plot_one(axs[1],0,'test','output', task)     

    else:

        for i in range(num_test):      

            plot_one(axs[0,i],i,'test','input', task)

            plot_one(axs[1,i],i,'test','output', task)  

    plt.tight_layout()

    plt.show() 





##This ARC_solver is from https://www.kaggle.com/xiaoyuez/arc-identify-objects with slight modifications.

class ARC_solver:

    def __init__(self, task_num):

        self.task_num = task_num

        # standardize plotting colors

        self.cmap = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

                                         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

        self.norm = colors.Normalize(vmin = 0, vmax = 9)

        # initialize objects-related things

        self.identified_objects = []

        self.io_inx = [] # the original index of the identified objects (io)

        self.io_height = [] # height of io

        self.io_width = [] # width of io

        self.io_pixel_count = [] # count of non-background pixels

        self.io_size = [] # overall grid size

        self.io_unique_colors = [] # number of unique colors

        self.io_main_color = [] # the dominating color

        self.zzz_io_inx = []

    

    def reset(self):

        self.identified_objects = []

        self.io_inx = [] 

        self.io_height = [] 

        self.io_width = [] 

        self.io_pixel_count = [] 

        self.io_size = [] 

        self.io_unique_colors = [] 

        self.io_main_color = []

        self.zzz_io_inx = []

        

    def plot_task(self):

        # plot examples of task input-output pairs 

        task = train_tasks[self.task_num]

        cmap = self.cmap

        norm = self.norm

        fig, axs = plt.subplots(1, 5, figsize = (8,2))

        axs[0].text(0.5, 0.5, 'Task', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 15)

        axs[0].get_xaxis().set_visible(False)

        axs[0].get_yaxis().set_visible(False)

        axs[0].axis('off')

        axs[1].imshow(task['train'][0]['input'], cmap=cmap, norm=norm)

        axs[1].axis('off')

        axs[1].set_title('Train Input')

        axs[2].imshow(task['train'][0]['output'], cmap=cmap, norm=norm)

        axs[2].axis('off')

        axs[2].set_title('Train Output')

        axs[3].imshow(task['test'][0]['input'], cmap=cmap, norm=norm)

        axs[3].axis('off')

        axs[3].set_title('Test Input')

        axs[4].imshow(task['test'][0]['output'], cmap=cmap, norm=norm)

        axs[4].axis('off')

        axs[4].set_title('Test Output')

        plt.tight_layout()

        plt.show()

    

    def plot_identified_objects(self, identified_objects, title = 'objects'):

        # do not plot anything in the following situations

        if len(identified_objects) == 0:

            print('No objects were identified.')

            return

        if len(identified_objects) > 20:

            print('Way too many objects (>20). Not gonna plot them.')

            return

        fig, axs = plt.subplots(1, len(identified_objects) + 1, figsize = (8,2))

        for i in range(len(identified_objects) + 1):

            if i == 0:

                axs[0].text(0.5, 0.5, title, horizontalalignment = 'center', verticalalignment = 'center', fontsize = 15)

                axs[0].get_xaxis().set_visible(False)

                axs[0].get_yaxis().set_visible(False)

                axs[0].axis('off')

            else:

                obj = identified_objects[i-1]

                axs[i].imshow(obj, cmap = self.cmap, norm = self.norm)

                axs[i].axis('off')

                axs[i].set_title('object{}'.format(i))

        plt.tight_layout()

        plt.show()

    

    def get_background(self, image):

        # if image contains 0 

        

        if 0 in image:

          background = 0

        # else use the most frequent pixel color

        else: 

            unique_colors, counts = np.unique(image, return_counts = True)

            background = unique_colors[np.argmax(counts)]

        #TODO: its posible the if below this is breaking some things.

            if len(counts) > 1:

                if counts[np.argmax(counts)] == counts[0] and counts[np.argmax(counts)] == counts[1]:

                    background = 11

        return background

    

    def check_pairs(self, inx_pairs, this_pair, return_inx = False):

        # check if this_pair is in inx_pairs

        match = []

        for pair in inx_pairs:

          if pair[0] == this_pair[0] and pair[1] == this_pair[1]:

            match.append(True)

          else:

            match.append(False)

        if return_inx:

          return any(match), np.where(match)

        else:

          return any(match)

    

    def check_neighbors(self, all_pairs, this_pair, objectness, this_object):

        # all_pairs: an array of index pairs for all nonzero/colored pixels

        # this_pair: the index pair whose neighbors will be checked

        # objectness: an array with the shape of original image, storage for how much objectness has been identified

        # this_object: the current object we are looking at

        row_inx = this_pair[0]

        col_inx = this_pair[1]

        objectness[row_inx, col_inx] = this_object

        # find if any neighboring pixels contain color

        if self.check_pairs(all_pairs, [row_inx-1, col_inx-1]): # up-left

          objectness[row_inx-1, col_inx-1] = this_object

        if self.check_pairs(all_pairs, [row_inx-1, col_inx]): # up

          objectness[row_inx-1, col_inx] = this_object 

        if self.check_pairs(all_pairs, [row_inx-1, col_inx+1]): # up-right

          objectness[row_inx-1, col_inx+1] = this_object

        if self.check_pairs(all_pairs, [row_inx, col_inx-1]): # left

          objectness[row_inx, col_inx-1] = this_object

        if self.check_pairs(all_pairs, [row_inx, col_inx+1]): # right

          objectness[row_inx, col_inx+1] = this_object

        if self.check_pairs(all_pairs, [row_inx+1, col_inx-1]): # down-left

          objectness[row_inx+1, col_inx-1] = this_object

        if self.check_pairs(all_pairs, [row_inx+1, col_inx]): # down

          objectness[row_inx+1, col_inx] = this_object

        if self.check_pairs(all_pairs, [row_inx+1, col_inx+1]): # down-right

          objectness[row_inx+1, col_inx+1] = this_object

        return objectness



    def identify_object_by_color(self, true_image, background = 0):

        # identify obeject by the color only 

        unique_colors = np.unique(true_image)

        for i, color in enumerate(unique_colors):

          image = np.copy(true_image) # make a copy from original first

          if color == background: 

            continue

          image[image != color] = background

          inx = np.where(image == color)

          obj = image[np.min(inx[0]):np.max(inx[0])+1, np.min(inx[1]):np.max(inx[1])+1]

          # append the object attributes

          self.identified_objects.append(obj)

          self.io_inx.append(inx)

          self.io_height.append(obj.shape[0])

          self.io_width.append(obj.shape[1])

          self.io_pixel_count.append(obj[obj != background].shape[0])

          self.io_size.append(obj.size)

          nc, c = np.unique(obj, return_counts = True)

          self.io_unique_colors.append(nc)

          self.io_main_color.append(nc[np.argmax(c)])

    

    def identify_object_by_isolation(self, image, background = 0):

        # identify all objects by physical isolation on the given image

        all_pairs = np.array(np.where(image != background)).T

        objectness = np.zeros(image.shape)

        this_object = 1

        while len(all_pairs) >= 1:

          init_pair = all_pairs[0] # start with the first pair

          objectness = self.check_neighbors(all_pairs, init_pair, objectness, this_object)

          # get a list of index pairs whose neghbors haven't been checked

          unchecked_pairs = np.array(np.where(objectness == this_object)).T

          checked_pairs = np.zeros((0,2)) 

          # check all the index pairs in the expanding unchecked_pairs untill all have been checked

          while len(unchecked_pairs) != 0:

            this_pair = unchecked_pairs[0]

            objectness = self.check_neighbors(all_pairs, this_pair, objectness, this_object)

            # append the checked_pairs

            checked_pairs = np.vstack((checked_pairs, this_pair))

            # get all index pairs for the currently identified object

            current_object_pairs = np.array(np.where(objectness == this_object)).T

            # delete the checked pairs from current object pairs

            checked_inx = []

            for pair in checked_pairs:

              _, inx = self.check_pairs(current_object_pairs, pair, return_inx = True)

              checked_inx.append(inx[0][0])

            unchecked_pairs = np.delete(current_object_pairs, checked_inx, axis = 0)



          # store this object to identified_objects

          current_object_pairs = np.array(np.where(objectness == this_object)).T

          cop = current_object_pairs.T

          obj = image[np.min(cop[0]):np.max(cop[0])+1, np.min(cop[1]):np.max(cop[1])+1]

          # delete the current object pairs from all_pairs 

          cop_inx = []

          for pair in current_object_pairs:

            _, this_cop_inx = self.check_pairs(all_pairs, pair, return_inx = True)

            cop_inx.append(this_cop_inx[0][0])

          all_pairs = np.delete(all_pairs, cop_inx, axis = 0)

          # append the object attributes

          self.identified_objects.append(obj)

          

          #my code to get top left coordinates of object

          #if 'cop' doesn't work, replace cop with current_object_pairs OR objectness

          if len(inx) < 2:

              row = min(cop[0])

              column = min(cop[1])

              inx = [row, column]

          self.zzz_io_inx.append(inx)

          self.io_inx.append(inx)

          self.io_height.append(obj.shape[0])

          self.io_width.append(obj.shape[1])

          self.io_pixel_count.append(obj[obj != background].shape[0])

          self.io_size.append(obj.size)

          nc, c = np.unique(obj, return_counts = True)

          self.io_unique_colors.append(nc)

          self.io_main_color.append(nc[np.argmax(c)])

          # start identifying a new object

          this_object += 1

        return objectness, 

    

    def identify_object_by_color_isolation(self, true_image, background = 0):

        # identify objects first by color then by physical isolation

        unique_colors = np.unique(true_image)

        for i, color in enumerate(unique_colors):

          image = np.copy(true_image) # make a copy from the original first

          if color == background:

            continue

          # identify objects by isolation in this color only 

          image[image != color] = background

          self.identify_object_by_isolation(image, background = background)

    

    def identify_object(self, image, method):

        # a wrapper of different methods

        # in the future method can be a parameter to be learned

        # 1 = by_color, 2 = by_isolation, 3 = by_color_isolation

        background = self.get_background(image)

        if method == 1:

          self.identify_object_by_color(image, background)

        elif method == 2:

          self.identify_object_by_isolation(image, background)

        elif method == 3:

          self.identify_object_by_color_isolation(image, background)

          

def Vert(M):   #flip function

    n = len(M)

    k = len(M[0])

    ans = np.zeros((n,k), dtype = int)

    for i in range(n):

        for j in range(k):

            ans[i][j] = 0+M[n-1-i][j]

    return ans.tolist()



def Hor(M):   #flip function

    n = len(M)

    k = len(M[0])

    ans = np.zeros((n,k), dtype = int)

    for i in range(n):

        for j in range(k):

            ans[i][j] = 0+M[i][k-1-j]

    return ans.tolist()



def Rot1(M):

    n = len(M)

    k = len(M[0])

    ans = np.zeros((k,n), dtype = int)

    for i in range(n):

        for j in range(k):

            ans[j][i] = 0 + M[i][k-1-j]

    return ans.tolist()

            

def Rot2(M):

    n = len(M)

    k = len(M[0])

    ans = np.zeros((k,n), dtype = int)

    for i in range(n):

        for j in range(k):

            ans[j][i] = 0 + M[n-1-i][j]

    return ans.tolist()



def check_if_same_size(c, d):

    a = np.atleast_2d(c) # to allow it to work for 1d arrays

    b = np.atleast_2d(d) # to allow it to work for 1d arrays

    x, y = a.shape

    x2, y2 = b.shape

    #print('shapes are', x,y,x2,y2)

    if x ==x2 and y==y2:

        return True

    return False



def are_two_images_equals(c, d):

    a=np.array(c)

    b=np.array(d)

    if tuple(a.shape) == tuple(b.shape):

        if (np.abs(b-a) < 1).all():

            return True

    return False



def in_out_change(image_matrix_in, image_matrix_out):  #ONLY WORKS WITH ARRAYS  of size at least 2 x 2

    """ #I think it works with any array size now, even if height is only 1

    Calculate the difference between input and output image.

    (Can have different formats)    """   

    i_pre = np.atleast_2d(image_matrix_in) # to allow it to work for 1d arrays

    o_pre = np.atleast_2d(image_matrix_out) # to allow it to work for 1d arrays

    x_in, y_in = i_pre.shape

    x_out, y_out = o_pre.shape

    min_x = min(x_in, x_out)

    min_y = min(y_in, y_out)

    image_matrix_diff = np.zeros((max(x_in, x_out), max(y_in, y_out)))

    #print('image matrix diff is:', image_matrix_diff)

    image_matrix_diff[:x_in, :y_in] -= image_matrix_in

    image_matrix_diff[:x_out, :y_out] += image_matrix_out

    a = image_matrix_diff

    #print('a', a)

    b = (a != 0)             #return true for all changed cells

    c = (a != 0).astype(int) #return 1 for all changed cells

    d = c*image_matrix_in    #return original(input) colors of changed cells

    e = c*image_matrix_out   #return output colors of changed cells

    #to fix double bracket problem

    if b.shape[0] == 1:

        b = b[0]

    if c.shape[0] == 1:

        c = c[0]

    if d.shape[0] == 1:

        d = d[0]

    if e.shape[0] == 1:

        e = e[0]

    return b, c, d, e   #true(bolean), 1(int), original color cells that were changed(int), new color cells that were changed(int)



def SINGLE_PERMS(in2darray):

    """ouputs all permutations (rotates, flips)"""

    O = in2darray

    #various permutations: rotations, flips

    V = Vert(O)

    VH = Hor(Vert(O))

    H = Hor(O)

    R1 = Rot1(O) #counterclockwise

    R1V = Vert(Rot1(O))

    R2 = Rot2(O) #clockwise

    R2V = Vert(Rot2(O))

    #TWO_COMPARE_ALL_PERM function relies on the order of these

    perms = [O, V, VH, H, R1, R1V, R2, R2V] 

    perms = np.array(perms)

    return perms



def TWO_COMPARE_ALL_PERM(c, d):

    """compare 2 arrays, including all permuations(flips, rotates)

    outputs if same size, if exact same, if a permutation of, 

        #and Tranformation needed of c to get to d, if any"""

    size = False

    same = False

    perm = False # true if any permutations match

    perm_funcs = False #function needed to get the permutation that matches

    size = check_if_same_size(c, d)

    if size == True:

        same = are_two_images_equals(c, d)

        if same == True:

            perm = True

            answer = [size, same, perm, perm_funcs]

            return answer

    #if not the exact same, then check if permutations are

    cR1 = np.array(Rot1(c))

    size2 = check_if_same_size(cR1, d)

    if size2 == True:  #if the transpose the same size

        perms = SINGLE_PERMS(c)

        for i in range(len(perms)):

            perm_c_i = perms[i]  #don't need to check first one, which is original

            perm_c_i = np.array(perm_c_i)

            same_perm = are_two_images_equals(perm_c_i, d)

            if same_perm == True:

                perm = True

                if i == 1:

                    perm_funcs = Vert  #second function of SINGLE_PERMS

                if i == 2:

                    perm_funcs = Hor, Vert

                if i == 3:

                    perm_funcs = Hor

                if i == 4:

                    perm_funcs = Rot1

                if i == 5:

                    perm_funcs = Vert, Rot1

                if i == 6:

                    perm_funcs = Rot2

                if i == 7:

                    perm_funcs = Vert, Rot2

                answer = [size, same, perm, perm_funcs]

                return answer

    else: #if even the transpose isn't the same size, then no perms will match

        answer = [size, same, perm, perm_funcs]

        return answer

    answer = [size, same, perm, perm_funcs]

    return answer #format: [if same size, if exact same, if same perm, perm_funcs]



def idntfy_objs_image(image, method=3):

    """This uses the ARC_solver class above to extract objects and attributes"""

    in_objs = []

    in_objs_inx = []

    in_objs_attr = []    

    arc.input_objects = []

    arc.input_objects_inx = []

    arc.input_objects_attr = {"height": [],

                                "width": [],

                                "pixel_count": [],

                                "size": [],

                                "unique_colors": [],

                                "in_obj": [],

                                "x_coord": [],

                                "y_coord": [],

                                "shape": [],

                                "main_color": []} 

    arc.reset()

    arc.identify_object(image, method = method)

    #arc.plot_identified_objects(arc.identified_objects, title = ('Task',task_num, 'method', method))

    arc.input_objects_attr['height'].append(arc.io_height)

    arc.input_objects_attr['width'].append(arc.io_width)

    arc.input_objects_attr['pixel_count'].append(arc.io_pixel_count)

    arc.input_objects_attr['size'].append(arc.io_size)

    arc.input_objects_attr['unique_colors'].append(arc.io_unique_colors)

    arc.input_objects_attr['main_color'].append(arc.io_main_color)

    in_objs_inx = arc.io_inx

    for o in range(len(in_objs_inx)):

        x_copy = in_objs_inx[o][1]

        y_copy = in_objs_inx[o][0]

        arc.input_objects_attr['x_coord'].append(x_copy)

        arc.input_objects_attr['y_coord'].append(y_copy)

        obj = arc.identified_objects[o]

        obj_shape = (obj != 0).astype(int)

        arc.input_objects_attr['shape'].append(obj_shape)

    x_s = arc.input_objects_attr['x_coord']

    y_s = arc.input_objects_attr['y_coord']

    shape_s = arc.input_objects_attr['shape']

    arc.input_objects_attr['x_coord'] = [x_s]

    arc.input_objects_attr['y_coord'] = [y_s]

    arc.input_objects_attr['shape'] = [shape_s]

    arc.input_objects_attr['in_obj'].append(arc.identified_objects)

    in_objs.append(arc.identified_objects)

    in_objs_inx.append([arc.io_inx])

    in_objs_attr.append(arc.input_objects_attr)

    return in_objs, in_objs_inx, in_objs_attr



def get_attr_task(task_num, method_num = 3):

    """extracts the objects, and their attributes(e.g. height, location, color), 

    for all training pairs and the test pair of a task"""

    arc = ARC_solver(task_num)

    task = train_tasks[task_num] 

    aINs = []

    aOUTs = []

    aPAIRs = []

    aTEST_PAIR = []

    aINs_unchgd = []

    aOUTs_unchgd = []

    aINs_chgd = []

    aOUTs_chgd = []

    # iterate through training examples 

    num_examples = len(train_tasks[task_num]['test'])

    for i in range(num_examples):

        # identify all objects

        input_image = np.array(train_tasks[task_num]['test'][i]['input'])

        output_image = np.array(train_tasks[task_num]['test'][i]['output'])

        in_objs, in_inx_unused, A_in_objs_attr = idntfy_objs_image(input_image, method=method_num)

        out_objs, out_inx_unused, A_out_objs_attr = idntfy_objs_image(output_image, method=method_num)

        pair = [A_in_objs_attr[0], A_out_objs_attr[0]]

        aTEST_PAIR.append(pair)

    # iterate through testing examples 

    num_examples = len(train_tasks[task_num]['train'])

    for i in range(num_examples):

        # identify all objects

        input_image = np.array(train_tasks[task_num]['train'][i]['input'])

        output_image = np.array(train_tasks[task_num]['train'][i]['output'])

        in_objs, in_inx_unused, A_in_objs_attr = idntfy_objs_image(input_image, method=method_num)

        out_objs, out_inx_unused, A_out_objs_attr = idntfy_objs_image(output_image, method=method_num)

        try:

            b,c,d,e = in_out_change(input_image, output_image)

            inv_b = np.invert(b).astype(int) #get the location of pixels that did NOT change, as 0 or 1

            #d                          #input objects,  DID   change

            #e                          #output objects, DID   change

            f = inv_b*input_image      #input objects,  did NOT change

            g = inv_b*output_image     #output objects, did NOT change

            in_objs2, in_inx_unused2, A_in_objs_attr2 = idntfy_objs_image(f, method=method_num)

            out_objs2, out_inx_unused2, A_out_objs_attr2 = idntfy_objs_image(g, method=method_num)

            in_objs3, in_inx_unused3, A_in_objs_attr3 = idntfy_objs_image(d, method=method_num)

            out_objs3, out_inx_unused3, A_out_objs_attr3 = idntfy_objs_image(e, method=method_num)

            aINs_unchgd.append(A_in_objs_attr2[0])

            aOUTs_unchgd.append(A_out_objs_attr2[0])

            aINs_chgd.append(A_in_objs_attr3[0])

            aOUTs_chgd.append(A_out_objs_attr3[0])

        except:

            pass

        aINs.append(A_in_objs_attr[0])

        aOUTs.append(A_out_objs_attr[0])

        pair = [A_in_objs_attr[0], A_out_objs_attr[0]]

        aPAIRs.append(pair)

    attr_task={'aINs': aINs, 'aOUTs': aOUTs, 'aPAIRs': aPAIRs, 'aTEST_PAIR': aTEST_PAIR,

               'aINs_unchgd': aINs_unchgd, 'aOUTs_unchgd': aOUTs_unchgd, 

               'aINs_chgd': aINs_chgd, 'aOUTs_chgd': aOUTs_chgd}

    return  attr_task



def get_object_similarity(aINs, aOUTs, origin_in = [0,0], origin_out = [0,0], similar_methd = 1):

    """this function outputs a similarity matrix where aOUTs objects are rows, aINs are columns and the [i,j] value is their similarity"""

    #note for origins: [-4,-6] values should only be negative (-matrixheight/2 is likely)

    num_examples = len(aOUTs)

    all_outs_by_ins = []

    for train_pair in range(num_examples):

        aOUTS_this_pair = aOUTs[train_pair] 

        aINS_this_pair = aINs[train_pair] 

        num_input_obj = len(aINs[train_pair]['in_obj'][0])

        num_output_obj = len(aOUTS_this_pair['pixel_count'][0])

        outs_by_ins = [] #this will be the answer matrix with input objects on one axis and output objects on the other axis

        outs_by_ins = np.zeros((num_output_obj, num_input_obj), dtype = object)

        outs_by_ins_detailed = [] #this will be the answer matrix with input objects on one axis and output objects on the other axis

        outs_by_ins_detailed = np.zeros((num_output_obj, num_input_obj), dtype = object)

        for out_obj_num in range(num_output_obj):

            o_shape = aOUTS_this_pair['shape'][0][out_obj_num]

            o_obj = aOUTS_this_pair['in_obj'][0][out_obj_num]

            o_x_coord = aOUTS_this_pair['x_coord'][0][out_obj_num]

            o_y_coord = aOUTS_this_pair['y_coord'][0][out_obj_num]

            o_x_coord = o_x_coord + origin_in[0] #re-adjust for origin

            o_y_coord = o_y_coord + origin_in[1]

            o_mn_color = aOUTS_this_pair['main_color'][0][out_obj_num]

            o_pxl_cnt = aOUTS_this_pair['pixel_count'][0][out_obj_num]

            o_uniq_clrs = sorted(aOUTS_this_pair['unique_colors'][0][out_obj_num])

            for in_obj_num in range(num_input_obj):

                shape_similarity = 0

                color_similarity = 0

                location_similarity = 0

                sim_total = 0

                #print('in_obj_number', in_obj_num)

                i_shape = aINS_this_pair['shape'][0][in_obj_num]

                i_obj = aINS_this_pair['in_obj'][0][in_obj_num]

                #print('i_shape', i_shape)

                i_x_coord = aINS_this_pair['x_coord'][0][in_obj_num]

                i_y_coord = aINS_this_pair['y_coord'][0][in_obj_num]

                i_x_coord = i_x_coord + origin_out[0] #re-adjust for origin

                i_y_coord = i_y_coord + origin_out[1]

                i_mn_color = aINS_this_pair['main_color'][0][in_obj_num]

                i_pxl_cnt = aINS_this_pair['pixel_count'][0][in_obj_num]

                i_uniq_clrs = sorted(aOUTS_this_pair['unique_colors'][0][out_obj_num])

                #SHAPE: Check if same or similar

                #FIRST CHECK IF OBJ IS EXACTLY THE SAME.

                get_similarity = TWO_COMPARE_ALL_PERM(o_shape, i_shape) #returns [size, same, perm, perm_funcs]

                pxl_cnt_diff = abs((o_pxl_cnt - i_pxl_cnt)/o_pxl_cnt)

                if are_two_images_equals(o_obj, i_obj) == True:#if objs are identical in every way

                    shape_similarity = 100

                elif get_similarity[1] == True: #if obj shapes are identical

                    shape_similarity = 100

                elif get_similarity[2] == True: #if obj in a different orientation, or flipped

                    #TODO: add code for if no other object exists , 

                    #then shape similarity for this part should be much higher           

                    shape_similarity = 50

                elif pxl_cnt_diff < 0.1:

                    shape_similarity = 10

                #COLOR: Check if same or similar

                i_uniq_clrs = len(aINS_this_pair['unique_colors'][0])

                o_uniq_clrs = len(aOUTS_this_pair['unique_colors'][0]) 

                uniq_clrs_diff = abs((o_uniq_clrs - i_uniq_clrs)/o_uniq_clrs)

                if i_mn_color == o_mn_color:

                    color_similarity = 100

                elif uniq_clrs_diff == 0 and o_uniq_clrs > 2:

                    #TODO: add code for if differences in color changes were already 

                    #determined to be the same object, then higher score

                    color_similarity = 20

                #LOCATION: Check if same or similar

                x_diff = abs(o_x_coord - i_x_coord)

                y_diff = abs(o_y_coord - i_y_coord)

                if x_diff == 0 and y_diff == 0:

                    location_similarity = 100

                elif x_diff == 0 or y_diff == 0:

                    location_similarity = 90

                elif x_diff <= 1 and y_diff <= 1:

                    location_similarity = 40

                elif x_diff <= 3 and y_diff <= 3 and aOUTS_this_pair['width'][0][out_obj_num]>3 and aOUTS_this_pair['height'][0][out_obj_num]>3:

                    location_similarity = 20

                elif x_diff <= 1 or y_diff <= 1:

                    location_similarity = 20

                #TODO: add location similarity for various origins, such as center of the image

                #TODO !!! change x and y location of each object to be the center of the object, not the top left.

                #if origin_location == 'likely center', then use center origins for diff calcs

                #Calculate overall similarity score

                s_sim = shape_similarity

                c_sim = color_similarity

                l_sim = location_similarity

                if similar_methd == 1: #for example, method 2 could be to value location very highly

                    s_sim_weight = 40

                    c_sim_weight = 40

                    l_sim_weight = 19

                if similar_methd == 2: #for example, method 2 could be to value location very highly

                    s_sim_weight = 9

                    c_sim_weight = 45

                    l_sim_weight = 45

                if similar_methd == 3: #for example, method 2 could be to value location very highly

                    s_sim_weight = 45

                    c_sim_weight = 9

                    l_sim_weight = 45

                if similar_methd == 4: #for example, method 2 could be to value location very highly

                    s_sim_weight = 45

                    c_sim_weight = 45

                    l_sim_weight = 9

                if similar_methd == 5: #Random weight method

                    s = randint(1, 100)

                    c = randint(1, 100)

                    l = randint(1, 100)

                    t = s + c + l 

                    s_sim_weight = int(s/t*100)

                    c_sim_weight = int(c/t*100)

                    l_sim_weight = int(l/t*100)

                    x = s_sim_weight + c_sim_weight + l_sim_weight

                    if x==98:

                        s_sim_weight = s_sim_weight+1

                    if x==100:

                        s_sim_weight = s_sim_weight-1

                    if x==101:

                        s_sim_weight = s_sim_weight-2

                sim_total = (s_sim*s_sim_weight + c_sim*c_sim_weight + l_sim*l_sim_weight)/100

                outs_by_ins[out_obj_num][in_obj_num] = sim_total #['sim_total', sim_total, s_sim, c_sim, l_sim]

                d = np.array([sim_total, s_sim, c_sim, l_sim])

                f = np.array([out_obj_num, in_obj_num])

                e = [d,f]

                outs_by_ins_detailed[out_obj_num][in_obj_num] = e

        all_outs_by_ins.append(outs_by_ins)

        #For debug, uncomment the below line to see the detailed color, loc, etc. similarities

        #all_outs_by_ins.append(outs_by_ins_detailed)

    #the output is in the format [total similarity, shape similrty, color similrty, locatn similrty]

    all_outs_by_ins = np.array(all_outs_by_ins)

    for t in range(len(all_outs_by_ins)):

        all_outs_by_ins[t] = all_outs_by_ins[t].astype(int)

    #all_outs_by_ins = all_outs_by_ins.astype(int)

    return all_outs_by_ins



def split_dict_objs(dict_objs, obj_num_list): 

    #removes all objects not in obj_num_list, e.g. obj_num_list = [0,1,2]

    temp_dict = []

    temp_dict = dict.fromkeys(dict_objs)

    for key in sorted(dict_objs.keys()):

        temp_dict[key] = []

        for j in obj_num_list:

            temp_dict[key].append(dict_objs[key][0][j])

        temp_dict[key] = [temp_dict[key]]

    return temp_dict

#x = split_dict_objs(aOUTs[0], obj_num_list=[0,1,2])



key_list_to_reduce = (['in_obj_width','in_obj_height', 'in_obj_pixel_count', 'in_obj_size', 'in_obj_main_color','in_obj', 'input obj num','x_coord', 'y_coord'])  #,, 'train_pair', 'function', 'function_param','in_obj_unique_colors',  'funct_and_param', 'new_obj',

def remove_keyvalues_if_in_second_dict(dict1, dict2, key_list_to_reduce_ = key_list_to_reduce):

    #removes any keyvalues from first dictionary if they are present in the second dict

    failures_copy = copy.deepcopy(dict1)

    failures_copy['keys_sometimes_fail_or_succeed'] = []

    failures_copy['keyname_sometimes_fail_or_succeed'] = []

    successes_copy = copy.deepcopy(dict2)

    for i in range(len((key_list_to_reduce_))):

        key_name = key_list_to_reduce_[i]

        try:

            if failures_copy[key_name] == None or failures_copy[key_name] == []:

                continue

        except:

            continue

        failures_copy2 = copy.deepcopy(failures_copy)

        fail_key_values = failures_copy2[key_name]

        #if rule never fails then no need to try and remove failures, because they don't exist

        success_key_values = successes_copy[key_name]

        try:

            if successes_copy[key_name] == None or successes_copy[key_name] == []:

                continue

        except:

            continue

        unique_success_values = unique_objects_in_listKEY(success_key_values)

        unique_success_values = unique_objects_in_listKEY(success_key_values)

        for value_fail in fail_key_values:

        

        #TODO: #as currently set up, as objects are removed, then the indexes will become incorrect, so need to removed them a different way



            for value_succeed in unique_success_values:

                try:

                    if value_fail == value_succeed:

                        #print('removing', value_fail, 'from', key_name)

                        if 'keys_sometimes_fail_or_succeed' not in failures_copy.keys():

                            failures_copy['keys_sometimes_fail_or_succeed'] = [[key_name, value_fail]]

                            failures_copy['keyname_sometimes_fail_or_succeed'] = [key_name]

                        else:

                            failures_copy["keys_sometimes_fail_or_succeed"].append([key_name, value_fail])

                            failures_copy["keyname_sometimes_fail_or_succeed"].append(key_name)

                        failures_copy[key_name].remove(value_fail)

                except:

                    x = [value_fail, value_succeed]

                    x2 = unique_objects_in_listKEY(x)

                    if len(x2) < len(x): #if the values are identical

                        #print('removing', value_fail, 'from', key_name)

                        if 'keys_sometimes_fail_or_succeed' not in failures_copy.keys():

                            failures_copy['keys_sometimes_fail_or_succeed'] = [[key_name, value_fail]]

                            failures_copy['keyname_sometimes_fail_or_succeed'] = [key_name]

                        else:

                            failures_copy["keys_sometimes_fail_or_succeed"].append([key_name, value_fail])

                            failures_copy["keyname_sometimes_fail_or_succeed"].append(key_name)

                        try:

                            remove_array_from_list(failures_copy[key_name], value_fail)

                        except Exception as e:

                            print('Error: spot 1:', e)

                            try:

                                failures_copy[key_name].remove(value_fail)   #this fails if it is in_obj that the value is being removed from

                            except Exception as e:

                                print('Error: spot 2:', e)

    return failures_copy



def remove_array_from_list(base_array, test_array):

    for index in range(len(base_array)):

        if np.array_equal(base_array[index], test_array):

            base_array.pop(index)

            break

    return base_array



def remove_empty_keys(dict_original):

    dict1 = copy.deepcopy(dict_original)

    for key in list(dict1):

        if dict1[key] == [] or dict1[key] == None:

            del dict1[key]

    return dict1



def unique_objects_in_listKEY(key_values_or_list): #works for objects

    """equivalent of list.unique, but also works for objects"""

    s_string_not_usable = set()

    unique_objects = []

    for value_ in key_values_or_list:

        for i in range(len(key_values_or_list)):

            if str(value_) not in s_string_not_usable:

                s_string_not_usable.add(str(value_))

                unique_objects.append(value_)

    return unique_objects



def match_objs_get_diffs(aINs2, aOUTS2, asimilarity_allpairs, key_list_to_reduce_ = key_list_to_reduce, method = 1):

    """ matches output objects with closest input object, then 

    compares differences in their attributes/features"""

    exact_matches = [] #in the form of [out_obj_num, in_obj_num]

    exact_matches_not = []

    answer = [] #outputs all the different attributes for each matched obj pair in each training pair

    # object, OR match only with "objects that haven't been paired yet."

    for train_pair in range(len(aOUTS2)):

        t = [] # list of keys that are different for the input/ouput object pair.

        diffs = []

        for out_obj_num in range(len(aOUTS2[train_pair]['pixel_count'][0])):    #get closest-in-similarity input object to this output object

            input_obj_clst = asimilarity_allpairs[train_pair][out_obj_num]

            #MATCH THE OUTPUT OBJECTS TO INPUT OBJECTS: 

                 #highest similarity method:

            #get the input object number of the closest-in-similarity input object

            #TODO: add other obj matching methods: e.g. match with the second most similar obj

            in_obj_num_clse = np.argmax(input_obj_clst) 

            if max(input_obj_clst) > 98: #if exact match (equals 99), then skip?

                exact_matches.append([out_obj_num, in_obj_num_clse, train_pair])

                continue

            else:

                exact_matches_not.append([out_obj_num, in_obj_num_clse, train_pair])

            in_obj_dict = split_dict_objs(aINs2[train_pair], [in_obj_num_clse]) #make dict of only the input obj

            out_obj_dict = split_dict_objs(aOUTS2[train_pair], [out_obj_num]) #make dict of only the output obj

            #TODO !!!! modify previous dict compare function that removes all duplicates from the output and input dictionaries

            key_list_to_reduce___ = list(out_obj_dict.keys())

            diff_keys = remove_keyvalues_if_in_second_dict(in_obj_dict, out_obj_dict, key_list_to_reduce___)

            diff_keys = remove_empty_keys(diff_keys)

            if 'keys_sometimes_fail_or_succeed' in diff_keys:

                del diff_keys['keys_sometimes_fail_or_succeed']

            if 'keyname_sometimes_fail_or_succeed' in diff_keys:

                del diff_keys['keyname_sometimes_fail_or_succeed']

            diffs = copy.deepcopy(diff_keys)

            for key_nam in list(diff_keys.keys()):

                if len(aOUTS2[train_pair][key_nam]) != 0:

                    in_key_val = aINs2[train_pair][key_nam][0][in_obj_num_clse]

                    out_key_val = aOUTS2[train_pair][key_nam][0][out_obj_num]

                    diffs[key_nam] = [[in_key_val], [out_key_val]]

            t.append([diffs, 'out_obj_num', out_obj_num, 'in_obj_num_clse', in_obj_num_clse, 'train_pair', train_pair])   #t is the list of differences between the output and input object

        answer.append(t) #answer is the list of changed attrs for each object pair.

    return [answer, exact_matches, exact_matches_not]



def get_orgnzd_obj_prs_method2(answer, attrs_solved_for = [], keys_check = ['height',

                                                                            'width',

                                                                            'pixel_count',

                                                                            'size',

                                                                            'unique_colors',

                                                                            'in_obj',

                                                                            'x_coord',

                                                                            'y_coord',

                                                                            'shape',

                                                                            'main_color']): 

    #attrs_solved_for can be passed to this so that those attrs are ignored because already solved for earlier

    """separates list of changed attributes of object pairs (answer) 

    into lists of pairs with same attribute changed."""

    answer_2 = []

    for i in range(10): #searching for list of attributes that are the shortest in length first

        for key_name_1 in keys_check:

            key_0_combined = []

            split_objs = []

            for train_pair in range(len(answer)):

                for obj_pair in range(len(answer[train_pair])):

                    key_list = sorted(list(answer[train_pair][obj_pair][0].keys()))

                    if key_name_1 in key_list:

                        if len(key_list) == i:

                            if len(key_0_combined) == 0: #if original hasn't been added yet.

                                key_0_combined = [answer[train_pair][obj_pair]]

                            else:

                                key_0_combined.append(answer[train_pair][obj_pair])

            if len(key_0_combined) != 0:

                answer_2.append([[[key_name_1]], key_0_combined, split_objs])              

    #split_dict_objs(#dict_objs, obj_num_list)       

    return answer_2



def reorganize_key_values_for_rule_extraction(aaa2_orgnzd_obj_pairs):

    """This, for purposes of rule extraction, reorganizes key values in aaa2_orgnzd_obj_pairs."""

    #method 1 is by matching keys within that key list

    total_appnded = []

    total_train_pr_trackd = []

    for key_list in range(len(aaa2_orgnzd_obj_pairs)):

        key_val_appnded = []

        train_pr_trackd = []

        key_name = aaa2_orgnzd_obj_pairs[key_list][0][0][0]

        #for key_num in range(len(aaa2_orgnzd_obj_pairs[key_list][0][0])):

            #key_name = aaa2_orgnzd_obj_pairs[key_list][0][0][key_num]

        for obj_pair in range(len(aaa2_orgnzd_obj_pairs[key_list][1])):

            try:

                # key_values = aaa2_orgnzd_obj_pairs[key_list][1][obj_pair][0][key_name]

                # train_pair_num = aaa2_orgnzd_obj_pairs[key_list][1][obj_pair][6]

                key_values = aaa2_orgnzd_obj_pairs[key_list][1][obj_pair][0][key_name]

                train_pair_num = aaa2_orgnzd_obj_pairs[key_list][1][obj_pair][6]

            except:

                key_values = []

            key_val_appnded.append(key_values)

            train_pr_trackd.append(train_pair_num)

        total_appnded.append([key_name, key_val_appnded])

        total_train_pr_trackd.append([key_name, train_pr_trackd])

    #TODO: write code that can accommodate 'training-pair-specific rules' by using total_train_pr_trackd 

    return [total_appnded, total_train_pr_trackd]



def get_simple_rule(amatrix_dims):

    """extracts a simple rule for an attribute, # e.g. rule is ('add this much', 4)"""

    ##modified to work in format below for ANY attr, not just height:

    #amatrix_dims = {'in_key': [[1], [2]], 

                 #  'out_key': [[5], [6]]}   

    matrix_height_is_a_mult_of_input = False

    multiplier_height = []

    addition_height = []

    answer_height = 'unknown'

    height_param = 30

    x = sorted(list(amatrix_dims.keys())) #the input key must be listed first alphabetically

    key_attr_in = x[0]

    key_attr_out = x[1]

    num_examples = len(amatrix_dims[key_attr_in])

        #make sure num_examples is outputting correct length, not just one.

    for i in range(num_examples):

        in_height = amatrix_dims[key_attr_in][i]

        out_height = amatrix_dims[key_attr_out][i]

        try:

            if len(in_height) >= 1:

                in_height = in_height[0]

        except:

            pass

        try:

            if len(out_height) >= 1:

                out_height = out_height[0]

        except:

            pass

        mult_height = out_height / in_height

        multiplier_height.append(mult_height)

        add_height = out_height - in_height

        addition_height.append(add_height)

    mult_height_unique = np.unique(multiplier_height)

    if len(mult_height_unique) == 1:

        matrix_height_is_a_mult_of_input = True

        answer_height = 'multiply by'

        height_param = mult_height_unique[0]

    matrix_height_is_static = False

    #TODO: make this work for objects too, by using unique_obj codde that doesn't fail, made elsewhere

    #an also putting these other if statements in try blocks so it can still get 'static' right at least

    height_unique = np.unique(amatrix_dims[key_attr_out])

    if len(height_unique) == 1:

        matrix_height_is_static = True

        answer_height = 'static'

        height_param = int(height_unique[0])

    add_height_unique = np.unique(addition_height)

    if len(add_height_unique) == 1:

        answer_height = 'add this much'

        height_param = add_height_unique[0]

    return answer_height, height_param

#%%

######   TESTING ON A TASK  ###############

#task_num = 30

for task_num in [30, 7, 29]:

    arc = ARC_solver(task_num)

    task = train_tasks[task_num] 

    print('task number :', task_num)

    plot_task(task)

    attr_task = get_attr_task(task_num)

    aINs = attr_task['aINs']

    aOUTs = attr_task['aOUTs']

    aOUTs_chgd = attr_task['aOUTs_chgd']

    aINs_chgd = attr_task['aINs_chgd']

    asimilarity_allpairs = get_object_similarity(aINs, aOUTs, similar_methd = 4) 

    answer_exacts_nonexacts = match_objs_get_diffs(aINs, aOUTs, asimilarity_allpairs)

    answer = answer_exacts_nonexacts[0]

    exact_matches = answer_exacts_nonexacts[1]

    exact_matches_not = answer_exacts_nonexacts[2]

    aaa2_orgnzd_obj_pairs = get_orgnzd_obj_prs_method2(answer)  #WORKS for two attributes

    total_appnded = reorganize_key_values_for_rule_extraction(aaa2_orgnzd_obj_pairs)[0] #only works for first key in list





    print('object similarity chart for all training pairs is:')

    print(asimilarity_allpairs)



    # This caculates rule and prints it. e.g. the rule for y_coord is: ('static', 12)

    for attr in range(len(total_appnded)):

        aa = []

        bb = []

        aa = np.atleast_2d(total_appnded[attr][1])

        key_name = total_appnded[attr][0]

        if aa.shape[1] > 0:

            bb = aa.transpose()

            try:

                dict_new = {'in_key': bb[0][0], 

                            'out_key': bb[0][1]} 

                aaaaa2 = get_simple_rule(dict_new)

                # print(dict_new)

                print('the rule for', key_name, 'is:', aaaaa2)

            except:

                print('ERROR 112: the rule for', key_name, 'is:', aaaaa2)