import numpy as np

import pandas as pd

import os

import json

from pathlib import Path

import pprint

pp = pprint.PrettyPrinter(indent=4)

from os.path import join as path_join



import warnings

warnings.filterwarnings("ignore")  #suppress all warnings



#######THIS CODE IS FOR USE WITH ANOCONDA PYTHON EDITOR IN MY DIRECTORY###########

#training_path = 'kaggle/input/abstraction-and-reasoning-challenge/training/'

#training_tasks = os.listdir(training_path)

#Trains = []

#for i in range(400):

#    task_file = str(training_path + training_tasks[i])

#    task = json.load(open(task_file, 'r'))

#    Trains.append(task)

#train_tasks = Trains

##############################################################################



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'

training_tasks = os.listdir(training_path)

eval_tasks = os.listdir(evaluation_path)

T = training_tasks

Trains = []

for i in range(400):

    task_file = str(training_path / T[i])

    task = json.load(open(task_file, 'r'))

    Trains.append(task)

def load_data(path):

    tasks = pd.Series()

    for file_path in os.listdir(path):

        task_file = path_join(path, file_path)

        with open(task_file, 'r') as f:

            task = json.load(f)

        tasks[file_path[:-5]] = task

    return tasks

train_tasks = load_data('../input/abstraction-and-reasoning-challenge/training/')



def get_matrix_dims(task_num):

    amatrix_dims={'in_matrix_height': [], 

                  'in_matrix_width': [], 

                  'out_matrix_height': [], 

                  'out_matrix_width': [],

                  'test_in_height': [], 

                  'test_in_width': [],

                  'test_out_height': [], 

                  'test_out_width': []}

    # iterate through training examples 

    num_examples = len(train_tasks[task_num]['train'])

    ain_height = []

    ain_width = []

    aout_height = []

    aout_width = []

    for i in range(num_examples):

        input_image = np.array(train_tasks[task_num]['train'][i]['input'])

        output_image = np.array(train_tasks[task_num]['train'][i]['output'])

        in_matrix_height = input_image.shape[0]

        in_matrix_width = input_image.shape[1]

        out_matrix_height = output_image.shape[0]

        out_matrix_width = output_image.shape[1] 

        ain_height.append(in_matrix_height)

        ain_width.append(in_matrix_width)

        aout_height.append(out_matrix_height)

        aout_width.append(out_matrix_width)

    amatrix_dims['in_matrix_height'].append(ain_height)

    amatrix_dims['in_matrix_width'].append(ain_width)

    amatrix_dims['out_matrix_height'].append(aout_height)

    amatrix_dims['out_matrix_width'].append(aout_width)

    num_examples = len(train_tasks[task_num]['test'])

    ain_height = []

    ain_width = []

    aout_height = []

    aout_width = []

    for i in range(num_examples):

        input_image = np.array(train_tasks[task_num]['test'][i]['input'])

        output_image = np.array(train_tasks[task_num]['test'][i]['output'])

        in_matrix_height = input_image.shape[0]

        in_matrix_width = input_image.shape[1]

        out_matrix_height = output_image.shape[0]

        out_matrix_width = output_image.shape[1] 

        ain_height.append(in_matrix_height)

        ain_width.append(in_matrix_width)

        aout_height.append(out_matrix_height)

        aout_width.append(out_matrix_width)

    amatrix_dims['test_in_height'].append(ain_height)

    amatrix_dims['test_in_width'].append(ain_width)

    amatrix_dims['test_out_height'].append(aout_height)

    amatrix_dims['test_out_width'].append(aout_width)

    return amatrix_dims



def get_matrix_rule(amatrix_dims):

    funcs_match_not_unknown = False

    multiplier_height = []

    multiplier_width = []

    addition_height = []

    addition_width = []

    answer_height = 'unknown' # if no rule found then uses size of 30

    height_param = 30

    answer_width = 'unknown'

    width_param = 30

    num_examples = len(amatrix_dims['in_matrix_width'][0])

    for i in range(num_examples):

        in_height = amatrix_dims['in_matrix_height'][0][i]

        out_height = amatrix_dims['out_matrix_height'][0][i]

        in_width = amatrix_dims['in_matrix_width'][0][i]

        out_width = amatrix_dims['out_matrix_width'][0][i]

        mult_height = out_height / in_height

        mult_width = out_width / in_width

        multiplier_height.append(mult_height)

        multiplier_width.append(mult_width)

        add_height = out_height - in_height

        addition_height.append(add_height)

        add_width = out_width - in_width

        addition_width.append(add_width)

    mult_height_unique = np.unique(multiplier_height)

    mult_width_unique = np.unique(multiplier_width)

    if len(mult_height_unique) == 1:

        answer_height = 'multiply by'

        height_param = mult_height_unique[0]

    if len(mult_width_unique) == 1:

        answer_width = 'multiply by'

        width_param = mult_width_unique[0]

    height_unique = np.unique(amatrix_dims['out_matrix_height'][0])

    width_unique = np.unique(amatrix_dims['out_matrix_width'][0])

    if answer_height != 'unknown' and answer_width == answer_height:

        funcs_match_not_unknown = True

    if len(height_unique) == 1 and funcs_match_not_unknown == False:

        answer_height = 'static'

        height_param = int(height_unique[0])

    if len(width_unique) == 1 and funcs_match_not_unknown == False:

        answer_width = 'static'

        width_param = int(width_unique[0])

    add_height_unique = np.unique(addition_height)

    add_width_unique = np.unique(addition_width)

    if answer_height != 'unknown' and answer_width == answer_height:

        funcs_match_not_unknown = True

    if len(add_height_unique) == 1 and funcs_match_not_unknown == False:

        answer_height = 'add this much'

        height_param = add_height_unique[0]

    if len(add_width_unique) == 1 and funcs_match_not_unknown == False:

        answer_width = 'add this much'

        width_param = add_width_unique[0]

    return answer_height, height_param, answer_width, width_param



def get_test_matrix_dims(amatrix_dims, matrix_rule):

    test_in_height = amatrix_dims['test_in_height'][0][0]

    test_in_width = amatrix_dims['test_in_width'][0][0]

    if matrix_rule[0] == 'static':

        test_out_height = matrix_rule[1]

    elif matrix_rule[0] == 'multiply by':

        test_out_height = test_in_height*matrix_rule[1]

    elif matrix_rule[0] == 'add this much':

        test_out_height = test_in_height + matrix_rule[1]

    else:

        test_out_height = 30

    if matrix_rule[2] == 'static':

        test_out_width = matrix_rule[3]

    elif matrix_rule[2] == 'multiply by':

        test_out_width = test_in_width*matrix_rule[3]

    elif matrix_rule[2] == 'add this much':

        test_out_width = test_in_width + matrix_rule[3]

    else:

        test_out_width = 30

    test_out_height = int(test_out_height)

    test_out_width = int(test_out_width)

    return test_out_height, test_out_width



#%% [to test multiple tasks]



amatrix_successfully_predicted =[]

amatrix_unsuccessfully_predicted = []

##uncomment the two lines below and comment the third line to test for only certain tasks

#task_num = [0, 1, 7, 263]

#for i in task_num:     

for i in range(400):

    try:

        height_success = False

        width_success = False

        task_num=i

        task = train_tasks[task_num] 

        amatrix_dims = get_matrix_dims(task_num)

        matrix_rule = get_matrix_rule(amatrix_dims)

        test_matrix_dims = get_test_matrix_dims(amatrix_dims, matrix_rule)

        if test_matrix_dims[0] == amatrix_dims['test_out_height'][0][0]:

            height_success = True

        if test_matrix_dims[1] == amatrix_dims['test_out_width'][0][0]:

            width_success = True

        a=[i,'guess', test_matrix_dims, 'actual', amatrix_dims['test_out_height'][0][0], amatrix_dims['test_out_width'][0][0]]

        if height_success == True and width_success == True:

            amatrix_successfully_predicted.append(i)

        else:

            amatrix_unsuccessfully_predicted.append(a)

    except KeyboardInterrupt:

        print('matrix dims failed for task:', task_num)

print('predicted:', len(amatrix_successfully_predicted),'/ 400 matrix sizes.')

print('failed:   ', len(amatrix_unsuccessfully_predicted),'/ 400  matrix sizes.')

print('')

pp.pprint('failed matrixes are:')

pp.pprint(amatrix_unsuccessfully_predicted)