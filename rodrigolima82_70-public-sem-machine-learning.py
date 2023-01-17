# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-sep-2019/X_treino.csv'

                            ,dtype = {

                                'series_id': np.int16

                               ,'measurement_number': np.int16

                               ,'orientation_X': np.float32

                               ,'orientation_X': np.float32

                                ,'orientation_Y': np.float32

                                ,'orientation_Z': np.float32

                                ,'orientation_W': np.float32

                                ,'angular_velocity_X': np.float32

                                ,'angular_velocity_Y': np.float32

                                ,'angular_velocity_Z': np.float32

                                ,'linear_acceleration_X': np.float32

                                ,'linear_acceleration_Y': np.float32

                                ,'linear_acceleration_Z': np.float32

                            })



test = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-sep-2019/X_teste.csv'

                            ,dtype = {

                                'series_id': np.int16

                               ,'measurement_number': np.int16

                               ,'orientation_X': np.float32

                               ,'orientation_X': np.float32

                                ,'orientation_Y': np.float32

                                ,'orientation_Z': np.float32

                                ,'orientation_W': np.float32

                                ,'angular_velocity_X': np.float32

                                ,'angular_velocity_Y': np.float32

                                ,'angular_velocity_Z': np.float32

                                ,'linear_acceleration_X': np.float32

                                ,'linear_acceleration_Y': np.float32

                                ,'linear_acceleration_Z': np.float32

                            })



y_train = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-sep-2019/y_treino.csv'

                            ,dtype = {

                                'series_id': np.int16

                               ,'group_id': np.int16

                            })



train.shape, test.shape, y_train.shape
full = pd.concat([train, test])

X_treino = train.iloc[:,3:].values.reshape(-1,128,10)

X_teste = test.iloc[:,3:].values.reshape(-1,128,10)

print('X_treino shape:', X_treino.shape, ', X_teste shape:', X_teste.shape)
def sq_dist(a,b):

    ''' the squared euclidean distance between two samples '''

    

    return np.sum((a-b)**2, axis=1)





def find_run_edges(data, edge):

    ''' examine links between samples. left/right run edges are those samples which do not have a link on that side. '''



    if edge == 'left':

        border1 = 0

        border2 = -1

    elif edge == 'right':

        border1 = -1

        border2 = 0

    else:

        return False

    

    edge_list = []

    linked_list = []

    

    for i in range(len(data)):

        dist_list = sq_dist(data[i, border1, :4], data[:, border2, :4]) # distances to rest of samples

        min_dist = np.min(dist_list)

        closest_i   = np.argmin(dist_list) # this is i's closest neighbor

        if closest_i == i: # this might happen and it's definitely wrong

            print('Sample', i, 'linked with itself. Next closest sample used instead.')

            closest_i = np.argsort(dist_list)[1]

        dist_list = sq_dist(data[closest_i, border2, :4], data[:, border1, :4]) # now find closest_i's closest neighbor

        rev_dist = np.min(dist_list)

        closest_rev = np.argmin(dist_list) # here it is

        if closest_rev == closest_i: # again a check

            print('Sample', i, '(back-)linked with itself. Next closest sample used instead.')

            closest_rev = np.argsort(dist_list)[1]

        if (i != closest_rev): # we found an edge

            edge_list.append(i)

        else:

            linked_list.append([i, closest_i, min_dist])

            

    return edge_list, linked_list





def find_runs(data, left_edges, right_edges):

    ''' go through the list of samples & link the closest neighbors into a single run '''

    

    data_runs = []



    for start_point in left_edges:

        i = start_point

        run_list = [i]

        while i not in right_edges:

            tmp = np.argmin(sq_dist(data[i, -1, :4], data[:, 0, :4]))

            if tmp == i: # self-linked sample

                tmp = np.argsort(sq_dist(data[i, -1, :4], data[:, 0, :4]))[1]

            i = tmp

            run_list.append(i)

        data_runs.append(np.array(run_list))

    

    return data_runs
full = full.iloc[:,3:].values.reshape(-1,128,10)
train_left_edges, train_left_linked  = find_run_edges(full, edge='left')

train_right_edges, train_right_linked = find_run_edges(full, edge='right')

print('Found', len(train_left_edges), 'left edges and', len(train_right_edges), 'right edges.')
train_runs = find_runs(full, train_left_edges, train_right_edges)
submission = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-sep-2019/sample_submission.csv')
submission['surface'] = ''

df_surface = ''



for i in range(151):

    x = train_runs[i]

    x = np.sort(x)

    if x[0]<3810:

        df_surface = y_train['surface'][x[0]]

        for j in range(len(train_runs[i])):

            if train_runs[i][j]-3810>-1:

                submission['surface'][train_runs[i][j]-3810] = df_surface
submission.head()
submission.to_csv('cv70.csv',index = False)