# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_X = pd.read_csv('../input/career-con-2019/X_train.csv')

test_X  = pd.read_csv('../input/cconlink/X_test_trial.csv' )

tot = pd.concat([train_X, test_X])

train_X = train_X.iloc[:,3:].values.reshape(-1,128,10)

test_X = test_X.iloc[:,3:].values.reshape(-1,128,10)

print('train_X shape:', train_X.shape, ', test_X shape:', test_X.shape)
df_train_y = pd.read_csv('../input/career-con-2019/y_train.csv')



# build a dict to convert surface names into numbers

surface_names = df_train_y['surface'].unique()

num_surfaces = len(surface_names)

surface_to_numeric = dict(zip(surface_names, range(num_surfaces)))

print('Convert to numbers: ', surface_to_numeric)



# y and group data as numeric values:

train_y = df_train_y['surface'].replace(surface_to_numeric).values

train_group = df_train_y['group_id'].values
fig, axes = plt.subplots(1,4)

fig.set_size_inches(20,3)



for i in range(4):

    axes[i].plot(train_X[train_group == 17][:,:,i].reshape(-1))

    axes[i].grid(True)
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
train_left_edges, train_left_linked  = find_run_edges(train_X, edge='left')

train_right_edges, train_right_linked = find_run_edges(train_X, edge='right')

print('Found', len(train_left_edges), 'left edges and', len(train_right_edges), 'right edges.')
train_runs = find_runs(train_X, train_left_edges, train_right_edges)
flat_list = [series_id for run in train_runs for series_id in run]

print(len(flat_list), len(np.unique(flat_list)))
print([ len(np.unique(train_y[run])) for run in train_runs ])
print([ (np.unique(train_y[run])) for run in train_runs ])
print([ len(np.unique(train_group[run])) for run in train_runs ])
print([ (np.unique(train_group[run])) for run in train_runs ])
fig, axes = plt.subplots(10,1, sharex=True)

fig.set_size_inches(20,15)

fig.subplots_adjust(hspace=0)



for i in range(10):

    axes[i].plot(train_X[train_runs[1]][:,:,i].reshape(-1))

    axes[i].grid(True)
train_runs[1]
df_train_y['run_id'] = 0

df_train_y['run_pos'] = 0



for run_id in range(len(train_runs)):

    for run_pos in range(len(train_runs[run_id])):

        series_id = train_runs[run_id][run_pos]

        df_train_y.at[ series_id, 'run_id'  ] = run_id

        df_train_y.at[ series_id, 'run_pos' ] = run_pos



df_train_y.to_csv('y_train_with_runs.csv', index=False)

df_train_y.tail()
df_train_y.head()
tot[487675:487700]
tot = tot.iloc[:,3:].values.reshape(-1,128,10)
train_left_edges, train_left_linked  = find_run_edges(tot, edge='left')

train_right_edges, train_right_linked = find_run_edges(tot, edge='right')

print('Found', len(train_left_edges), 'left edges and', len(train_right_edges), 'right edges.')
train_runs = find_runs(tot, train_left_edges, train_right_edges)
len(train_runs)
flat_list = [series_id for run in train_runs for series_id in run]

print(len(flat_list), len(np.unique(flat_list)))
x = train_runs[1]

y=np.sort(x)

y

#print([int(s) for s in x.split() if s.isdigit()])

#y = x.sort()

#print(y)
ss = pd.read_csv('../input/career-con-2019/sample_submission.csv')
ss['surface'] = ''
ss[:5]
l=[]

surf = ''

for i in range(151):

    x = train_runs[i]

    x = np.sort(x)

    if x[0]<3810:

        l.append((i,df_train_y['surface'][x[0]]))

        surf = df_train_y['surface'][x[0]]

        for j in range(len(train_runs[i])):

            if train_runs[i][j]-3810>-1:

                ss['surface'][train_runs[i][j]-3810] = surf
l[:5]
ss[:5]
ss.to_csv('x.csv',index = False)
sub96 =  pd.read_csv('../input/096acc/woodconc.csv')
sub96.to_csv('sub96.csv',index = False)