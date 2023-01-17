import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier, XGBRegressor

from catboost import CatBoostClassifier

import pdb

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import (ExtraTreesClassifier,

                              GradientBoostingClassifier,

                              BaggingClassifier,

                              StackingClassifier)

from sklearn.linear_model import LogisticRegression



import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", category=ConvergenceWarning)
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'
sample_sub = pd.read_csv(data_path/'sample_submission.csv')

sample_sub = sample_sub.set_index('output_id')

sample_sub.head()
def plot_result(test_input, test_prediction,

                input_shape):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    fig, axs = plt.subplots(1, 2, figsize=(15,15))

    test_input = test_input.reshape(input_shape[0],input_shape[1])

    axs[0].imshow(test_input, cmap=cmap, norm=norm)

    axs[0].axis('off')

    axs[0].set_title('Actual Target')

    test_prediction = test_prediction.reshape(input_shape[0],input_shape[1])

    axs[1].imshow(test_prediction, cmap=cmap, norm=norm)

    axs[1].axis('off')

    axs[1].set_title('Model Prediction')

    plt.tight_layout()

    plt.show()

    

def plot_test(test_prediction, task_name):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    fig, axs = plt.subplots(1, 1, figsize=(15,15))

    axs.imshow(test_prediction, cmap=cmap, norm=norm)

    axs.axis('off')

    axs.set_title(f'Test Prediction {task_name}')

    plt.tight_layout()

    plt.show()
# https://www.kaggle.com/inversion/abstraction-and-reasoning-starter-notebook

def flattener(pred):

    str_pred = str([row for row in pred])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred
def get_moore_neighbours(color, cur_row, cur_col, nrows, ncols):

    # pdb.set_trace()



    if (cur_row<=0) or (cur_col>ncols-1): top = -1

    else: top = color[cur_row-1][cur_col]

        

    if (cur_row>=nrows-1) or (cur_col>ncols-1): bottom = -1

    else: bottom = color[cur_row+1][cur_col]

        

    if (cur_col<=0) or (cur_row>nrows-1): left = -1

    else: left = color[cur_row][cur_col-1]

        

    if (cur_col>=ncols-1) or (cur_row>nrows-1): right = -1

    else: right = color[cur_row][cur_col+1]

        

    return top, bottom, left, right



def get_tl_tr(color, cur_row, cur_col, nrows, ncols):

        

    if cur_row==0:

        top_left = -1

        top_right = -1

    else:

        if cur_col==0: top_left=-1

        else: top_left = color[cur_row-1][cur_col-1]

        if cur_col==ncols-1: top_right=-1

        else: top_right = color[cur_row-1][cur_col+1]   

        

    return top_left, top_right



def get_vonN_neighbours(color, cur_row, cur_col, nrows, ncols):

        

    if cur_row==0:

        top_left = -1

        top_right = -1

    else:

        if cur_col==0: top_left=-1

        else: top_left = color[cur_row-1][cur_col-1]

        if cur_col==ncols-1: top_right=-1

        else: top_right = color[cur_row-1][cur_col+1]

        



    if cur_row==nrows-1:

        bottom_left = -1

        bottom_right = -1

    else:

        

        if cur_col==0: bottom_left=-1

        else: bottom_left = color[cur_row+1][cur_col-1]

        if cur_col==ncols-1: bottom_right=-1

        else: bottom_right = color[cur_row+1][cur_col+1]       

        

    return top_left, top_right, bottom_left, bottom_right

def make_features(input_color, nfeat):

    nrows, ncols = input_color.shape

    feat = np.zeros((nrows*ncols,nfeat))

    cur_idx = 0

    for i in range(nrows):

        for j in range(ncols):

            feat[cur_idx,0] = i

            feat[cur_idx,1] = j

            feat[cur_idx,2] = input_color[i][j]

            feat[cur_idx,3:7] = get_moore_neighbours(input_color, i, j, nrows, ncols)

            try:

                feat[cur_idx,7] = len(np.unique(input_color[i-1,:]))

                feat[cur_idx,8] = len(np.unique(input_color[:,j-1]))

            except IndexError:

                pass



            feat[cur_idx,9] = len(np.unique(input_color[i,:]))

            feat[cur_idx,10] = len(np.unique(input_color[:,j]))

            feat[cur_idx,11] = len(np.unique(input_color[i-local_neighb:i+local_neighb,

                                                        j-local_neighb:j+local_neighb]))



            feat[cur_idx,12:16] = get_moore_neighbours(input_color, i+1, j, nrows, ncols)

            feat[cur_idx,16:20] = get_moore_neighbours(input_color, i-1, j, nrows, ncols)



            feat[cur_idx,20:24] = get_moore_neighbours(input_color, i, j+1, nrows, ncols)

            feat[cur_idx,24:28] = get_moore_neighbours(input_color, i, j-1, nrows, ncols)



            feat[cur_idx,28] = len(np.unique(feat[cur_idx,3:7]))

            try:

                feat[cur_idx,29] = len(np.unique(input_color[i+1,:]))

                feat[cur_idx,30] = len(np.unique(input_color[:,j+1]))

            except IndexError:

                pass

            cur_idx += 1

        

    return feat
def features(task):

    global local_neighb, nfeat

    mode = 'train'

    cur_idx = 0

    num_train_pairs = len(task[mode])

#     total_inputs = sum([len(task[mode][i]['input'])*len(task[mode][i]['input'][0]) for i in range(num_train_pairs)])



    feat, target = [], []

    for task_num in range(num_train_pairs):

        for a in range(3):

            input_color = np.array(task[mode][task_num]['input'])

            target_color = task[mode][task_num]['output']

            if a==1:

                input_color = np.fliplr(input_color)

                target_color = np.fliplr(target_color)

            if a==2:

                input_color = np.flipud(input_color)

                target_color = np.flipud(target_color)





            nrows, ncols = len(task[mode][task_num]['input']), len(task[mode][task_num]['input'][0])



            target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])

            

            if (target_rows!=nrows) or (target_cols!=ncols):

                print('Number of input rows:',nrows,'cols:',ncols)

                print('Number of target rows:',target_rows,'cols:',target_cols)

                not_valid=1

                return None, None, 1

            

            imsize = nrows*ncols

            offset = imsize*task_num*3 #since we are using three types of aug

            feat.extend(make_features(input_color, nfeat))

            target.extend(np.array(target_color).reshape(-1,))

            cur_idx += 1

            

    return np.array(feat), np.array(target), 0
# mode = 'eval'

mode = 'test'

if mode=='eval':

    task_path = evaluation_path

elif mode=='train':

    task_path = training_path

elif mode=='test':

    task_path = test_path



all_task_ids = sorted(os.listdir(task_path))



nfeat = 31

local_neighb = 5

valid_scores = {}

model_accuracies = {'ens': []}

pred_taskids = []

for task_id in all_task_ids:



    task_file = str(task_path / task_id)

    with open(task_file, 'r') as f:

        task = json.load(f)



    feat, target, not_valid = features(task)

    if not_valid:

        print('ignoring task', task_file)

        print()

        not_valid = 0

        continue



    estimators = [

                    ('xgb', XGBClassifier(n_estimators=25, n_jobs=-1)),

                    ('extra_trees', ExtraTreesClassifier() ),

                    ('bagging', BaggingClassifier()),

                    ('LogisticRegression',LogisticRegression())

                 ]

    clf = StackingClassifier(

        estimators=estimators, final_estimator=XGBClassifier(n_estimators=10, n_jobs=-1)

    )





    clf.fit(feat, target)



#     training on input pairs is done.

#     test predictions begins here



    num_test_pairs = len(task['test'])

    cur_idx = 0

    for task_num in range(num_test_pairs):

        input_color = np.array(task['test'][task_num]['input'])

        nrows, ncols = len(task['test'][task_num]['input']), len(

            task['test'][task_num]['input'][0])



        feat = make_features(input_color, nfeat)



        print('Made predictions for ', task_id[:-5])



        preds = clf.predict(feat).reshape(nrows,ncols)

        

        if (mode=='train') or (mode=='eval'):

            ens_acc = (np.array(task['test'][task_num]['output'])==preds).sum()/(nrows*ncols)



            model_accuracies['ens'].append(ens_acc)



            pred_taskids.append(f'{task_id[:-5]}_{task_num}')



            print('ensemble accuracy',(np.array(task['test'][task_num]['output'])==preds).sum()/(nrows*ncols))

            print()



        preds = preds.astype(int).tolist()

        plot_test(preds, task_id)

        if mode=='test':

            sample_sub.loc[f'{task_id[:-5]}_{task_num}',

                        'output'] = flattener(preds)

if (mode=='train') or (mode=='eval'):

    df = pd.DataFrame(model_accuracies, index=pred_taskids)

    print(df.head(10))



    print(df.describe())

    for c in df.columns:

        print(f'for {c} no. of complete tasks is', (df.loc[:, c]==1).sum())



    df.to_csv('ens_acc.csv')
sample_sub.head()
sample_sub.to_csv('submission.csv')