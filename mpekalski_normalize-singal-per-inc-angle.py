# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_json("../input/train.json").fillna(-1.0).replace('na', -1.0)

test = pd.read_json("../input/test.json").fillna(-1.0).replace('na', -1.0)
# For each image get 10 pixel stripe from the left

# append all stripes for images grouped by inc_angle

# rounded to some point.



stripes={'band_1':{}, 'band_2':{}}

n_round = 1

for x in round(train.inc_angle, n_round):

    stripes[str(int(x*100))]={'band_1':[],'band_2':[]}
y = 10 

for i in range(1604):   

    for band in ['band_1','band_2']: 

        x = np.reshape(train.loc[i][band],(75,75))

        round_inc = str(int(round(train.loc[i].inc_angle,n_round)*100))

        slice_1 = x[:,0:y]

        slice_2 = x[:,75-y:75]

        slice_3 = x[0:y,:]

        slice_4 = x[75-y:75,:]

        slices = [slice_1,slice_2,slice_3,slice_4]

        std_1 = np.std(np.hstack(slice_1))

        std_2 = np.std(np.hstack(slice_2))

        std_3 = np.std(np.hstack(slice_3))

        std_4 = np.std(np.hstack(slice_4))

    

        min_std = np.argmin([std_1,std_2,std_3,std_4])



        stripes[round_inc][band].append(np.hstack(slices[min_std]))
for k in stripes.keys():

    if stripes[k]:

        for band in ['band_1','band_2']:

            stripes[k][band] = np.concatenate(stripes[k][band])
stats = {}

for z in stripes.keys():

    if z not in ['band_1','band_2']:

        if z not in stats:

            stats[z] = {'band_1':{},'band_2':{}}

            for band in ['band_1','band_2']:

                stats[z][band] = {'mean': np.mean(stripes[z][band]), 'std': np.std(stripes[z][band])}        

    
def normalize(ds, setname):

    ds = train

    first=True

    with open('normalized_'+setname+'.json', 'w') as f:

        f.write('[')

        for i in range(ds.shape[0]):

            if not first:

                f.write(',{')

            else:

                f.write('{')

            f.write('"id":"{}","inc_angle":{},"is_iceberg":{}'.format(train.loc[i]['id'], train.loc[i]['inc_angle'],train.loc[i]['is_iceberg']))

            for band in ['band_1', 'band_2']:

                x = np.reshape(train.loc[i][band],(75,75))

                round_inc = str(int(round(train.loc[i].inc_angle,n_round)*100))

                z = (x.copy()-stats[round_inc][band]['mean'])/stats[round_inc][band]['std']

                f.write(',"{}":{}'.format(band, str(list(np.hstack(z)))))

            f.write('}')

            first=False

        f.write(']')
normalize(train, 'train')
normalize(test, 'test')
print(check_output(["ls", "."]).decode("utf8"))