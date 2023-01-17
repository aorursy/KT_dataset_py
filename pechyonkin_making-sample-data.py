import os, sys

import numpy as np

from glob import glob
# for setting aside sample

p_sample = 0.1

def calc_sample_size(n_files, proportion=p_sample):

    return int(np.round(n_files * proportion))
# data for this competition should be in folder /data/fishes/

cwd = os.getcwd()

DATA_DIR = cwd + '/data/fishes/' # for full training

DATA_DIR_SAMPLE = cwd + '/data/fishes/sample/' # for local training (will be created if doesn't yet exist)
dirs = os.listdir(DATA_DIR)

dirs = [d for d in dirs if (os.path.isdir(DATA_DIR + d) & (d != 'sample'))]

dirs
if not os.path.exists(DATA_DIR+'/test_stg1/unknown'):

    os.makedirs(DATA_DIR+'/test_stg1/unknown')
if not os.path.exists(DATA_DIR_SAMPLE):

    os.makedirs(DATA_DIR_SAMPLE)
for d in dirs:

    in_dirs = os.listdir(DATA_DIR+d)

    in_dirs = [dd for dd in in_dirs if (os.path.isdir(DATA_DIR + d +'/'+ dd) & (dd[0] != '.'))]

    for nd in in_dirs:

        full_dir = DATA_DIR + d + '/' + nd

        os.chdir(full_dir)

        g = glob('*.jpg')

        

        # shuffle filenames

        shuf = np.random.permutation(g)

        sample_size = calc_sample_size(len(shuf), proportion=p_sample)

        

        sample_dir = DATA_DIR + 'sample/' + d + '/' + nd

        # create same dir in sample/d

        if not os.path.exists(sample_dir):

            os.makedirs(sample_dir)

            

        # copy sample files to /sample/d/nd

        for i in range(sample_size): 

            new_filename = sample_dir+'/'+shuf[i]

            if not os.path.exists(new_filename):

                os.rename(shuf[i], new_filename)