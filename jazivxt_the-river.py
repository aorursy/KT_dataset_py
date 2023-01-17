import numpy as np

import pandas as pd

import glob



import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import colors

fig = plt.figure(figsize=(8., 6.))
def mask_match(task):

    result = True

    for i in range(len(task)):

        if not np.array_equal(np.clip(np.array(task[i]['input']), 0, 1), np.clip(np.array(task[i]['output']), 0, 1)):

            result = False

            break

    return result



def shape_match(task): 

    result = True

    for i in range(len(task)):

        if not np.array_equal(np.array(task[i]['input']).shape, np.array(task[i]['output']).shape):

            result = False

            break

    return result



def rotation_match(task): 

    r_ = {r:0 for r in range(1,4)}

    result = 0

    for i in range(len(task)):

        for r in range(3):

            rot = np.rot90(np.array(task[i]['input']), r+1)

            rot = np.flip(rot, axis=1)

            if np.array_equal(np.array(task[i]['output']), rot):

                r_[r + 1] += 1

    for r in r_:

        if r_[r]==len(task):

            result = r

    return result



def color_match(task): #should validate task group not individual task

    result = True

    for i in range(len(task)):

        if not np.array_equal(np.unique(np.array(task[i]['input'])), np.unique(np.array(task[i]['output']))):

            result = False

            break

    return result
path = '/kaggle/input/abstraction-and-reasoning-challenge/'

tasks = pd.DataFrame(glob.glob(path + '**/**'), columns=['path'])

tasks['tte'] = tasks['path'].map(lambda x: x.split('/')[-2])

tasks['output_id'] = tasks['path'].map(lambda x: x.split('/')[-1].split('.')[0])

tasks['file'] = tasks['path'].map(lambda x: eval(open(x).read()))

tasks['train'] = tasks['file'].map(lambda x: x['train'])

tasks['test'] = tasks['file'].map(lambda x: x['test'])

tasks.drop(columns=['file'], inplace=True)

tasks['l'] = tasks.apply(lambda r: (len(r['train']), len(r['test'])), axis=1)

tasks.tte.value_counts()
tasks['mask_match'] = tasks['train'].map(lambda x: mask_match(x))

tasks['shape_match'] = tasks['train'].map(lambda x: shape_match(x))

tasks['rotation_match'] = tasks['train'].map(lambda x: rotation_match(x))

tasks['color_match'] = tasks['train'].map(lambda x: color_match(x))



tasks.head()
#https://www.kaggle.com/nagiss/manual-coding-for-the-first-10-tasks

cmap = colors.ListedColormap(['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)



def viz(path):

    f = eval(open(path, 'r').read())

    train = f['train']

    test = f['test']

    f, ar = plt.subplots(3,len(train))

    for i in range(len(train)):

        ar[0,i].imshow(np.array(train[i]['input']), cmap=cmap, norm=norm)

        ar[1,i].imshow(np.array(train[i]['output']), cmap=cmap, norm=norm)

        if i < len(test):

            ar[2,i].imshow(np.array(test[i]['input']), cmap=cmap, norm=norm)

        else:

            ar[2,i].imshow(np.zeros(np.array(test[0]['input']).shape), cmap=cmap, norm=norm)

    plt.show()

    

df = tasks.drop_duplicates(subset=['mask_match', 'shape_match', 'rotation_match', 'color_match'])

for i in range(len(df)):

    print('mask_match:', df['mask_match'].iloc[i], 

          'shape_match:', df['shape_match'].iloc[i], 

          'rotation_match:', df['rotation_match'].iloc[i], 

          'color_match:', df['color_match'].iloc[i])

    viz(df['path'].iloc[i])
def flattener(pred):

    str_pred = '|'+ '|'.join([''.join([str(v) for v in row]) for row in pred])+'|'

    str_pred = ' '.join([str_pred for i in range(2)]) #simulating 2 predictions

    #Adding a blank prediction similar to the sample submission

    str_pred += ' |'+ '|'.join([''.join([str(0) for v in row]) for row in pred])+'|'

    return str_pred 
evaluation = tasks[tasks['tte']=='evaluation'].reset_index(drop=True)

score = 0.

denom = 0.

for i in range(len(evaluation)):

    for j in range(len(evaluation['test'][i])):

        denom += 1

        #Add your predictions here - just taking the first train ouput here for shape

        if evaluation['rotation_match'][i] > 0:

            rot = np.array(evaluation['test'][i][j]['input'])

            rot = np.flip(np.rot90(rot, evaluation['rotation_match'][i]), axis=1)

            if not np.array_equal(np.array(evaluation['test'][i][j]['output']), rot):

                score += 1

        elif evaluation['shape_match'][i] == True:

            if not np.array_equal(np.array(evaluation['test'][i][j]['output']), np.array(evaluation['test'][i][j]['input'])): score += 1

        else:

            if not np.array_equal(np.array(evaluation['test'][i][j]['output']), np.array(evaluation['train'][i][0]['output'])): score += 1



print(score/denom)
test = tasks[tasks['tte']=='test'].reset_index(drop=True)

sub = open('submission.csv','w')

sub.write('output_id,output\n')

for i in range(len(test)):

    for j in range(len(test['test'][i])):

        #Add your predictions here - just taking the first train ouput here for shape

        if test['rotation_match'][i] > 0:

            rot = np.array(test['test'][i][j]['input'])

            pred = np.flip(np.rot90(rot, test['rotation_match'][i]), axis=1)

        elif test['shape_match'][i] == True:

            pred = np.array(test['test'][i][j]['input'])

        else:

            pred = np.array(test['train'][i][0]['output'])

        sub.write(test['output_id'][i]+ '_' + str(j) + ',' + flattener(pred)+' \n')

sub.close()