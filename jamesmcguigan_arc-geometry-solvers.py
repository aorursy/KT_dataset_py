import numpy as np

import pandas as pd



import sys

import os

import re

import json

import cv2

import itertools

import traceback

from itertools import chain, product, combinations

from glob import glob

from pathlib import Path

from copy import copy, deepcopy



import matplotlib.pyplot as plt

from matplotlib import colors

from typing import List, Union, Callable





import toolz

import pydash as py

from pydash import py_ as _
task_files = glob('../input/abstraction-and-reasoning-challenge/**/*.json')



def load_tasks(task_files):

    if isinstance(task_files, str): task_files = glob(task_files)

        

    tasks = { re.sub(r'^.*/(.*?/.*$)','\\1',file): json.load(open(file, 'r')) for file in task_files }



    for file, task in tasks.items():

        for test_train, specs in task.items():

            for index, spec in enumerate(specs):

                for input_output, grid in spec.items():

                    tasks[file][test_train][index][input_output] = np.array(grid).astype('uint8')  # uint8 required for cv2 



    for file, task in tasks.items():

        tasks[file]['file'] = file

    return tasks



tasks = load_tasks(task_files)

task = list(tasks.values())[6]; task
# Modified from Source: https://www.kaggle.com/zaharch/visualizing-all-tasks-updated

def plot_one(task, ax, i,train_or_test,input_or_output):

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    

    try:

        input_matrix  = task[train_or_test][i][input_or_output]

        font_size     = 50 / np.sqrt(input_matrix.shape[0] * input_matrix.shape[1])

        min_font_size = 6 



        ax.imshow(input_matrix, cmap=cmap, norm=norm)

        ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    

        ax.imshow(input_matrix, cmap=cmap, norm=norm)

        # DOC: https://stackoverflow.com/questions/33828780/matplotlib-display-array-values-with-imshow

        if font_size >= min_font_size:

            for (j,i),label in np.ndenumerate(input_matrix):

                ax.text(i,j,label,ha='center',va='center', fontsize=font_size, color='black')



        ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])

        ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     

        ax.set_xticklabels([])

        ax.set_yticklabels([])

        ax.set_title(train_or_test + ' '+input_or_output)

    except: pass  # mat throw on tests, as they have not "output"

    

def plot_task(task, scale=2):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """   

    filename = None

    if isinstance(task, str):   (filename, task) = task, tasks[task]

    if isinstance(task, tuple): (filename, task) = task

    if 'file' in task: filename = task['file']

    

    num_train = len(task['train']) + len(task['test']) + 1

    if 'solution' in task: num_train += len(task['solution']) + 1

    

    fig, axs = plt.subplots(2, num_train, figsize=(scale*num_train,scale*2))

    if filename: fig.suptitle(filename)

        

    for i in range(len(task['train'])):     

        plot_one(task, axs[0,i],i,'train','input')

        plot_one(task, axs[1,i],i,'train','output')            



    axs[0,i+1].axis('off'); axs[1,i+1].axis('off')

    for j in range(len(task['test'])):      

        plot_one(task, axs[0,i+2+j],j,'test','input')

        plot_one(task, axs[1,i+2+j],j,'test','output')  

    

    if 'solution' in task:    

        axs[0,i+j+3].axis('off'); axs[1,i+j+3].axis('off')        

        for k in range(len(task['solution'])):      

            plot_one(task, axs[0,i+j+4+k],k,'solution','input')

            plot_one(task, axs[1,i+j+4+k],k,'solution','output')  



    plt.show()     

    

plot_task(list(tasks.items())[0])
solvers = {}
class Solver():

    verbose = True

    debug   = False

    def __init__(self):

        self.cache = {}

    

    @staticmethod

    def loop_specs(task, test_train='train'):

        specs = task[test_train]

        for index, spec in enumerate(specs):

            yield { 'input': spec['input'], 'output': spec['output'] }



    @staticmethod

    def is_lambda_valid(_task_, _function_, *args, **kwargs):  # _task_ = avoid namespace conflicts with kwargs=task 

        for spec in Solver.loop_specs(_task_, 'train'):

            output = _function_(spec['input'], *args, **kwargs)

            if not np.array_equal( spec['output'], output):

                return False

        return True



    @staticmethod

    def solve_lambda(_task_, _function_, *args, **kwargs):

        solutions = []

        for index, spec in enumerate(_task_['test']):

            result = _function_(spec['input'], *args, **kwargs)

            solutions.append({ 

                "input":  spec['input'],

                "output": result

            })

        return solutions



    def action(self, grid, *args):

        """This is the primary method this needs to be defined"""

        return grid

        raise NotImplementedError()



    def detect(self, task):

        """default heuristic is simply to run the solver""" 

        return self.test(task)



    def test(self, task):

        """test if the given action correctly solves the task"""

        args = self.cache.get(task['file'], ())

        return self.is_lambda_valid(task, self.action, *args, task=task)    



    def solve(self, task, force=False):

        """solve test case and persist"""

        try:

            if self.detect(task) or force:    # may generate cache

                if self.test(task) or force:  # may generate cache

                    args = self.cache.get(task['file'], ())

                    if isinstance(args, dict):

                        solutions = self.solve_lambda(task, self.action, **args, task=task)                    

                    else:

                        solutions = self.solve_lambda(task, self.action, *args, task=task)

                    task['solution'] = solutions  # TODO: fix for multiple test outputs

                    self.log_solved(task, args, solutions)

                    return solutions

        except Exception as exception:

            if self.debug: raise exception

        return None

            

    def solve_all(self, tasks, plot=False, solve_detects=False):

        count = 0

        for filename, task in tasks.items():    

            if self.detect(task):

                solution = self.solve(task, force=solve_detects)

                if solution or (solve_detects and self.test(task)):

                    count += 1

                    if plot == True:

                        plot_task((filename, task))        

        return count



    def plot(self, tasks):

        return self.solve_all(tasks, plot=True, solve_detects=False)



    def plot_detects(self, tasks, unsolved=True):

        if unsolved:

            tasks = { file: task for (file,task) in deepcopy(tasks).items() if not 'solution' in task }

        return self.solve_all(tasks, plot=True, solve_detects=True)



    

    

    ### Logging Methods ###



    def format_args(self, args):

        if isinstance(args, dict):

            args = dict(zip(args.keys(), map(self.format_args, list(args.values()))))

        elif isinstance(args, (list,set,tuple)):

            args = list(args)

            for index, arg in enumerate(args):

                if hasattr(arg, '__name__'):

                    arg = f"<{type(arg).__name__}:{arg.__name__}>"

                if isinstance(arg, (list,set,tuple,dict)):

                    arg = self.format_args(arg)

                args[index] = arg

            args = tuple(args)

        return args



    def log_solved(self, task, args: Union[list,tuple,set], solutions: List[np.ndarray]):

        if not self.verbose: return

        try:

            task_filename = getattr(task,'filename',task.get('file',''))

            if 'test' in task_filename:           label = 'test  '

            elif self.is_solved(task, solutions): label = 'solved'

            else:                                 label = 'guess '



            args  = self.format_args(args) if len(args) else None

            print(f'{label}:', task_filename, self.__class__.__name__, args)

        except Exception as exception: 

            print('Exception:', type(exception), exception)

            traceback.print_tb(exception.__traceback__)



    

    def is_solved(self, task, solutions: List):

        for solution in solutions:

            for problem in task['test']:

                if 'output' in solution and np.array_equal(problem['output'], solution['output']):

                    return True

        return False

    

    ### Helper Methods ###

    

    @staticmethod

    def grid_shape_ratio(grid1, grid2):

        try:

            return ( grid2.shape[0] / grid1.shape[0], grid2.shape[1] / grid1.shape[1] )

        except:

            return (0, 0)  # For tests

    

    @staticmethod

    def task_grids(task):

        grids = []

        for test_train in ['test','train']:

            for spec in task[test_train]:

                grids += [ spec.get('input',[]), spec.get('output',[]) ]  # tests not gaurenteed to have outputs

        return grids



    @staticmethod

    def task_grid_shapes(task):

        return [ np.array(grid).shape for grid in Solver.task_grids(task) ]

    

    @staticmethod

    def task_grid_max_dim(task):

        return max(chain(*Solver.task_grid_shapes(task)))

    

    @staticmethod

    def task_shape_ratios(task):

        ratios = set([

            Solver.grid_shape_ratio(spec.get('input',[]), spec.get('output',[]))

            for spec in Solver.loop_specs(task, 'train')

        ])

        # ratios = set([ int(ratio) if ratio.is_integer() else ratio for ratio in chain(*ratios) ])

        return ratios



    @staticmethod

    def is_task_shape_ratio_unchanged(task):

        return Solver.task_shape_ratios(task) == { (1,1) }



    @staticmethod

    def is_task_shape_ratio_consistant(task):

        return len(Solver.task_shape_ratios(task)) == 1

    

    @staticmethod

    def is_task_shape_ratio_integer_multiple(task):

        ratios = Solver.task_shape_ratios(task)

        return all([ isinstance(d, int) or d.is_integer() for d in chain(*ratios) ])

        

    

print( 'Solver().task_grid_shapes(task)',                     Solver().task_grid_shapes(task)  )

print( 'Solver().task_grid_max_dim(task)',                    Solver().task_grid_max_dim(task) )

print( 'Solver().task_shape_ratios(task)',                    Solver().task_shape_ratios(task) )

print( 'Solver().is_task_shape_ratio_consistant(task)',       Solver().is_task_shape_ratio_consistant(task) )

print( 'Solver().is_task_shape_ratio_unchanged(task)',        Solver().is_task_shape_ratio_unchanged(task)  )

print( 'Solver().is_task_shape_ratio_integer_multiple(task)', Solver().is_task_shape_ratio_integer_multiple(task) )
class DoNothingSolver(Solver):

    def action(self, grid, task):

        return grid



solvers['DoNothingSolver'] = DoNothingSolver()

solvers['DoNothingSolver'].plot(tasks)
import skimage.measure



class ZoomSolver(Solver):

    verbose = True

    

    def detect(self, task):

        ratios = self.task_shape_ratios(task)

        ratio  = list(ratios)[0] 

        detect = (

                ratios != { (1,1) }   # not no scaling

            and len(ratios) == 1      # not multiple scalings

            and ratio[0] == ratio[1]  # single consistant scaling

        )

        return detect



    def get_scale(self, task):

        return list(self.task_shape_ratios(task))[0][0]

            

    def action(self, grid, task):

        scale = self.get_scale(task)

        if scale > 1:

            resize = tuple( int(d*scale) for d in grid.shape )            

            output = cv2.resize(grid, resize, interpolation=cv2.INTER_NEAREST)

        else:

            resize = tuple( int(1/scale) for d in grid.shape )                        

            output = skimage.measure.block_reduce(grid, resize)

        if self.debug:

            print('scale', scale, 'grid.shape', grid.shape, 'output.shape', output.shape)

            print('grid', grid)

            print('output', output)            

        return output



solvers['ZoomSolver'] = ZoomSolver()

ZoomSolver().plot(tasks)
class GeometrySolver(Solver):

    optimise = True

    verbose  = True

    actions = {

        "flip":      ( np.flip,      [0,1]    ),

        "rot90":     ( np.rot90,     [1,2,3]  ),

        "roll":      ( np.roll,      product(range(-5,5),[0,1]) ),  

        "swapaxes":  ( np.swapaxes,  [(0, 1),(1, 0)] ),             

        "transpose": ( np.transpose, []       ),                      # this doesn't find anything                

    }

    

    def __init__(self):

        super().__init__()

        for key, (function, arglist) in self.actions.items():

            self.actions[key] = (function, [ (args,) if not isinstance(args, tuple) else args for args in arglist ])            

    

    def detect(self, task):

        return self.is_task_shape_ratio_unchanged(task)  # grids must remain the exact same size

        

    def test(self, task):

        if task['file'] in self.cache: return True

        

        max_roll = (self.task_grid_max_dim(task) + 1) // 2 

        for key, (function, arglist) in self.actions.items():

            if function == np.roll: arglist = product(range(-max_roll,max_roll),[0,1]) 

            for args in arglist:

                if self.is_lambda_valid(task, function, *args):

                    self.cache[task['file']] = (function, args)

                    if self.verbose: print(function, args)

                    return True

                

        # this doesn't find anything

        if self.optimise: return False        

        for ((function1, arglist1),(function2, arglist2)) in combinations( self.actions.values(), 2 ):

            if function1 == np.roll: arglist1 = product(range(-max_roll,max_roll),[0,1]) 

            if function2 == np.roll: arglist2 = product(range(-max_roll,max_roll),[0,1]) 

            for args1, args2 in product(arglist1, arglist2):

                function = lambda grid, args1, args2: function1(function2(grid, *args2), *args1)                

                if self.is_lambda_valid(task, function, *(args1,args2)):

                    self.cache[task['file']] = (function, (args1,args2))

                    if self.verbose: print(function, (args1,args2))

                    return True

        return False

    

    def action(self, grid, function=None, args=None, task=None):

        try:

            return function(grid, *args)

        except Exception as exception:

            if self.debug: print(function, args, exception)

            return grid



solvers['GeometrySolver'] = GeometrySolver()

GeometrySolver().plot(tasks)    
import skimage.measure

# from skimage.measure import block_reduce

import inspect

from fastcache import lru_cache

# from numpy_lru_cache_decorator import np_cache  # https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75



def query_true(grid,x,y):          return True

def query_not_zero(grid,x,y):      return grid[x,y]

def query_color(grid,x,y,color):   return grid[x,y] == color



# evaluation/15696249.json - max(1d.argmax())

def query_max_color(grid,x,y,exclude_zero=True):     

    return grid[x,y] == max_color(grid, exclude_zero)

# @lru_cache(1024)

def max_color(grid, exclude_zero=True):

    bincount = np.bincount(grid.flatten())

    if exclude_zero:    

        bincount[0] = np.min(bincount)  # exclude 0

    return bincount.argmax()



def query_min_color(grid,x,y, exclude_zero=True):     

    return grid[x,y] == min_color(grid, exclude_zero)

# @lru_cache(1024)

def min_color(grid,exclude_zero=True):

    bincount = np.bincount(grid.flatten())

    if exclude_zero:

        bincount[0] = np.max(bincount)  # exclude 0

    return bincount.argmin()



def query_max_color_1d(grid,x,y,exclude_zero=True):     

    return grid[x,y] == max_color_1d(grid)

# @lru_cache(16)

def max_color_1d(grid,exclude_zero=True):

    return max(

        [ max_color(row,exclude_zero) for row in grid ] +

        [ max_color(col,exclude_zero) for col in np.swapaxes(grid, 0,1) ]        

    )

def query_min_color_1d(grid,x,y):     

    return grid[x,y] == min_color_1d(grid)

# @lru_cache(16)

def min_color_1d(grid):

    return min(

        [ min_color(row) for row in grid ] +

        [ min_color(col) for col in np.swapaxes(grid, 0,1) ]        

    )





def query_count_colors(grid,x,y):     

    return grid[x,y] >= count_colors(grid)    

def query_count_colors_row(grid,x,y):     

    return x + len(grid.shape[0])*y <= count_colors(grid)    

def query_count_colors_col(grid,x,y):     

    return y + len(grid.shape[1])*x <= count_colors(grid)    

# @lru_cache(16)

def count_colors(grid):

    bincount = np.bincount(grid.flatten())

    return np.count_nonzero(bincount[1:]) # exclude 0



def query_count_squares(grid,x,y):     

    return grid[x,y] >= count_squares(grid)    

def query_count_squares_row(grid,x,y):     

    return x + len(grid.shape[0])*y <= count_squares(grid)    

def query_count_squares_col(grid,x,y):     

    return y + len(grid.shape[1])*x <= count_squares(grid)    

# @lru_cache(16)

def count_squares(grid):

    return np.count_nonzero(grid.flatten())



# BROKEN?

def rotate_loop(grid, start=0):    

    angle = start

    while True:

        yield np.rot90(grid, angle % 4)

        angle += 1 * np.sign(start)



# BROKEN?

def rotate_loop_rows(grid, start=0):    

    angle = start

    while True:

        yield np.rot90(grid, angle % grid.shape[0])

        angle += 1 * np.sign(start)



# BROKEN?

def rotate_loop_cols(grid, start=0):    

    angle = start

    while True:

        yield np.rot90(grid, angle % grid.shape[1])

        angle += 1 * np.sign(start)



def flip_loop(grid, start=0):    

    angle = start

    while True:

        if angle % 2: yield np.flip(grid)

        else:         yield grid

        angle += 1 * np.sign(start)



# BROKEN?

def flip_loop_rows(grid, start=0):    

    angle = start

    while True:

        if angle % grid.shape[0]: yield np.flip(grid)

        else:                     yield grid

        angle += 1 * np.sign(start)



# BROKEN?

def flip_loop_cols(grid, start=0):    

    angle = start

    while True:

        if angle % grid.shape[1]: yield np.flip(grid)

        else:                     yield grid

        angle += 1 * np.sign(start)



# BROKEN?

def invert(grid, color=None):

    if callable(color): color = color(grid)

    if color is None:   color = max_color(grid)

    if color:

        grid        = grid.copy()

        mask_zero   = grid[ grid == 0 ]

        mask_square = grid[ grid != 0 ]    

        grid[mask_zero]   = color

        grid[mask_square] = 0

    return grid    

        

    

        

# Source: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy         

def crop_inner(grid,tol=0): 

    mask = grid>tol; 

    return grid[np.ix_(mask.any(1),mask.any(0))]



def crop_outer(grid,tol=0):

    mask = grid>tol

    m,n  = grid.shape

    mask0,mask1 = mask.any(0),mask.any(1)

    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()

    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()

    return grid[row_start:row_end,col_start:col_end]



def make_tuple(args):

    if isinstance(args, tuple): return args

    if isinstance(args, list):  return tuple(args)

    return (args,)





class TessellationSolver(GeometrySolver):

    verbose = True

    debug   = False

    options = {

        "preprocess": { 

            "np.copy":    (np.copy, []), 

            "crop_inner": (crop_inner, range(0,9)),

            "crop_outer": (crop_outer, range(0,9)),

        },

        "transform": {  

            "none":              ( np.copy,      []        ),

            "flip":              ( np.flip,      [0,1]     ),

            "rot90":             ( np.rot90,     [1,2,3]   ),

            "roll":              ( np.roll,      product([-1,1],[0,1]) ),  

            "swapaxes":          ( np.swapaxes,  [(0, 1)]  ),                         

            "rotate_loop":       ( rotate_loop,         range(-4,4) ),

            "rotate_loop_rows":  ( rotate_loop_rows,    range(-4,4) ),  # BROKEN ?

            "rotate_loop_cols":  ( rotate_loop_cols,    range(-4,4) ),  # BROKEN ?          

            "flip_loop":         ( flip_loop,           range(0,2)  ),  # BROKEN ?

            "flip_loop_rows":    ( flip_loop_rows,      range(0,2)  ),  # BROKEN ?

            "flip_loop_cols":    ( flip_loop_cols,      range(0,2)  ),  # BROKEN ?

            "invert":            ( invert,              [max_color, min_color, max_color_1d, count_colors, count_squares, *range(1,9)]  ), # BROKEN

            # TODO: Invert

        },

        "query": {

            "query_true":              ( query_true,          [] ),

            "query_not_zero":          ( query_not_zero,      [] ),  

            "query_max_color":         ( query_max_color,     [] ),

            "query_min_color":         ( query_min_color,     [] ),

            "query_max_color_1d":      ( query_max_color_1d,  [] ),

            "query_min_color_1d":      ( query_min_color_1d,  [] ),

            "query_count_colors":      ( query_count_colors,      [] ),

            "query_count_colors_row":  ( query_count_colors_row,  [] ),

            "query_count_colors_col":  ( query_count_colors_col,  [] ),            

            "query_count_squares":     ( query_count_squares,     [] ),

            "query_count_squares_row": ( query_count_squares_row, [] ),

            "query_count_squares_col": ( query_count_squares_col, [] ),

            "query_color":             ( query_color,         range(0,10) ),  # TODO: query_max_color() / query_min_color()

        }

    }



    

    def detect(self, task):

        if self.is_task_shape_ratio_unchanged(task):            return False  # Use GeometrySolver

        if not self.is_task_shape_ratio_integer_multiple(task): return False  # Not a Tessalation problem   

        if not all([ count_colors(spec['input']) == count_colors(spec['output']) for spec in task['train'] ]): return False  # Different colors 

        if ZoomSolver().solve(task):                            return False

        #if not self.is_task_shape_ratio_consistant(task):       return False  # Some inconsistant grids are tessalations

        return True





    

    def loop_options(self):

        for (preprocess,p_args) in self.options['preprocess'].values():

            # print( (preprocess,p_args) )

            for p_arg in p_args or [()]:

                p_arg = make_tuple(p_arg)

                # print( (preprocess,p_args) )

                for (transform,t_args) in self.options['transform'].values():

                    for t_arg in t_args or [()]:

                        t_arg = make_tuple(t_arg)

                        for (query,q_args) in self.options['query'].values():

                            for q_arg in q_args or [()]:

                                q_arg = make_tuple(q_arg)

                                yield (preprocess, p_arg),(transform,t_arg),(query,q_arg)

    

    # TODO: hieraracharical nesting of solves and solutions/rules array generator

    def test(self, task):

        if task['file'] in self.cache: return True

        for (preprocess,p_arg),(transform,t_arg),(query,q_arg) in self.loop_options():         

            kwargs = {                

                "preprocess": preprocess,

                "p_arg":      p_arg,

                "transform":  transform,  # TODO: invert every other row | copy pattern from train outputs | extend lines

                "t_arg":      t_arg,

                "query":      query,  # TODO: max_colour limit counter

                "q_arg":      q_arg,

            }                        

            if self.is_lambda_valid(task, self.action, **kwargs, task=task):

                self.cache[task['file']] = kwargs

                if self.verbose: print(self.action, kwargs)

                return True

        return False

        

       

    def action(self, grid, preprocess=np.copy, p_arg=(), transform=np.copy, t_arg=(), query=query_true, q_arg=(), task=None):         

        #print('action', preprocess, transform, query)

        if inspect.isgeneratorfunction(transform):

            generator = transform(grid, *t_arg)

            transform = lambda grid, *args: next(generator)

        

        # Some combinations of functions will throw gemoetry             

        try:                                        

            grid    = preprocess(grid, *p_arg) 

            output  = self.get_output_grid(grid, task).copy()

            ratio   = ( int(output.shape[0] / grid.shape[0]), int(output.shape[1] / grid.shape[1]) )

            (gx,gy) = grid.shape

            for x,y in product(range(ratio[0]),range(ratio[1])):

                copy = np.zeros(grid.shape)

                if query(grid,x%gx,y%gy, *q_arg):            

                    copy = transform(grid, *t_arg)



                output[x*gx:(x+1)*gx, y*gy:(y+1)*gy] = copy

        except Exception as exception: 

            if self.debug: print(exception)

            pass

        return output

    

    

    def loop_ratio(self, task):

        ratio = list(self.task_shape_ratios(task))[0]

        for i in range(int(ratio[0])):

            for j in range(int(ratio[1])):            

                yield i,j



                

    def get_output_grid(self, grid, task):

        try:

            #print('get_output_grid(self, grid, task)', grid, task)

            for index, spec in enumerate(task['train']):

                if spec['input'] is grid:

                    return spec['output']

            else:

                # No output for tests

                ratio = list(self.task_shape_ratios(task))[0]

                ratio = list(map(int, ratio))

                shape = ( int(grid.shape[0]*ratio[0]), int(grid.shape[1]*ratio[1]) )

                return np.zeros(shape)

        except Exception as exception: 

            if self.debug: print(exception)

            pass

            

     

solvers['TessellationSolver'] = TessellationSolver()

solvers['TessellationSolver'].plot(tasks)
from copy import deepcopy

solvers['TessellationSolver'].plot_detects(tasks)
plot_task('test/31adaf00.json')
class BorderSolver(Solver):

    verbose = True

    debug = True

    cache = {}

    queries = [

        *range(0,10),

        max_color,     # FAIL: evaluation/fc754716.json

        max_color_1d,

        min_color,

        min_color_1d,

        count_colors,

        count_squares,

        np.count_nonzero,

    ]



    def task_has_border(self, task): 

        if not self.is_task_shape_ratio_consistant(task): return False        

        return all([ self.grid_has_border(spec['output']) for spec in task['train'] ])

        

    def grid_has_border(self, grid):

        if min(grid.shape) <= 2:                          return False  # single color problem



        grid_center = grid[1:-1,1:-1]        

        return np.count_nonzero(grid_center) == 0 and all([

            np.count_nonzero(border) == len(border)

            for border in [ grid[0,:], grid[-1,:], grid[:,0], grid[:,-1] ]

        ])

    

    def detect(self, task):

        return self.task_has_border(task)

    

    def test(self, task):

        if task['file'] in self.cache: return True

        for query in self.queries:         

            args = [ query ]

            if self.is_lambda_valid(task, self.action, *args, task=task):

                self.cache[task['file']] = args

                if self.debug: print(self.action, args)

                return True

        return False

        

    def action(self, grid, query=None, task=None):

        ratio  = list(self.task_shape_ratios(task))[0]

        output = np.zeros(( int(grid.shape[0] * ratio[0]), int(grid.shape[1] * ratio[1]) ))

        color  = query(grid) if callable(query) else query 

        output[:,:] = color

        output[1:-1,1:-1] = 0

        return output



solvers['BorderSolver'] = BorderSolver()

solvers['BorderSolver'].plot(tasks)
solvers['BorderSolver'].plot_detects(tasks)
class SingleColorSolver(Solver):

    verbose = True

    debug = True

    cache = {}

    queries = [

        *range(0,10),

        max_color,     # FAIL: evaluation/fc754716.json

        max_color_1d,

        min_color,

        min_color_1d,

        count_colors,

        count_squares,

        np.count_nonzero,

    ]



    def grid_unique_colors(self, grid):

        return np.unique(grid.flatten())

    

    def task_is_singlecolor(self, task): 

        if not self.is_task_shape_ratio_consistant(task): return False        

        return all([ len(self.grid_unique_colors(spec['output'])) == 1 for spec in task['train'] ])

                

    def detect(self, task):

        return self.task_is_singlecolor(task)

    

    def test(self, task):

        if task['file'] in self.cache: return True

        for query in self.queries:         

            args = [ query ]

            if self.is_lambda_valid(task, self.action, *args, task=task):

                self.cache[task['file']] = args

                if self.debug: print(self.action, args)

                return True

        return False

        

    def action(self, grid, query=None, task=None):

        ratio  = list(self.task_shape_ratios(task))[0]

        output = np.zeros(( int(grid.shape[0] * ratio[0]), int(grid.shape[1] * ratio[1]) ))        

        color  = query(grid) if callable(query) else query 

        output[:,:] = color

        return output



solvers['SingleColorSolver'] = SingleColorSolver()

solvers['SingleColorSolver'].plot(tasks)
solvers['SingleColorSolver'].plot_detects(tasks)
plot_task('training/1190e5a7.json')

plot_task('training/23b5c85d.json')

plot_task('training/27a28665.json')
def score_tasks(tasks: dict, label: str):

    total  = 0

    solved = 0



    for task_file, task in tasks.items():

        for index, spec in enumerate(task['test']): total += 1            



        if 'solution' in task:

            for index, spec in enumerate(task['solution']): 

                solved += 1                

        

    print(f'{label.ljust(11)}| solutions found: {str(solved).rjust(3)}/{total} | error: {round(1-solved/total,4)}')

    return solved



for label in ['training', 'evaluation', 'test']:

    score_tasks({ file: task for (file,task) in tasks.items() if label in file }, label)
test_files = glob('../input/abstraction-and-reasoning-challenge/test/*.json'); 

tests = load_tasks(test_files)

len(tests)
for name, solver in solvers.items():

    solver.cache = {}

    solver.plot(tests)
# def output_id(filename, index=0):

#     return re.sub('^.*/|\.json$', '', filename) + '_' + str(index);



# def output_grid(grid):

#     grid   = grid.astype('uint8').tolist()

#     # output = " ".join([

#     #     '|' + "".join([ str(int(cell)) for cell in row ]) + '|'

#     #     for row in grid

#     # ])

#     output = (

#         json.dumps(grid)

#             .replace(', ', '')

#             .replace('[[', '|')

#             .replace('][', '|')

#             .replace(']]', '|')

#     )

#     return output



# def output_csv(tests, submission_file='submission.csv', verbose=True):

#     output = []

#     for task_file, test in tests.items():

#         if 'solution' in test: 

#             for index, spec in enumerate(test['solution']): 

#                 output.append([ output_id(task_file, index), output_grid(spec['output']) ])

#         else:

#             for index, spec in enumerate(test['test']):             

#                 output.append([ output_id(task_file, index), output_grid(spec['input']) ])            



#     output = [['output_id','output']] + sorted(output)

#     csv = "\n".join([ ",".join(row) for row in output ])



#     with open(submission_file, 'w') as file:

#         file.write(csv)

#         if verbose: print(f'wrote: {submission_file}')



# output_csv(tests, 'submission.csv')
import pandas as pd



def output_id(filename, index=0):

    return re.sub('^.*/|\.json$', '', filename) + '_' + str(index);



# Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation

def flattener(grid):

    grid = grid.astype('uint8').tolist()    

    str_pred = str([row for row in grid])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred



def write_submission_csv(tests):

    submission_csv = pd.read_csv('../input/abstraction-and-reasoning-challenge/sample_submission.csv')

    submission_csv = submission_csv.set_index('output_id')



    total  = 0

    solved = 0

    for task_file, test in tests.items():    

        for index, spec in enumerate(test['test']): total += 1

        if 'solution' in test:

            for index, spec in enumerate(test['solution']): 

                submission_csv.loc[ output_id(task_file, index), 'output' ] = flattener(spec['output'])

                solved += 1                

        

                

    submission_csv.to_csv('submission.csv')

    print(f'wrote: submission.csv')

    print(f'solutions found: {solved}/{total} | error: {round(1-solved/total,4)}')

    return submission_csv



write_submission_csv(tests)

