from tqdm import tqdm_notebook

import bson

import numpy as np

import io

import matplotlib.pyplot as plt

from PIL import Image

% matplotlib inline
num_images = 12371293

num_points = 1000
checkpoints = np.linspace(0, num_images, num_points, dtype=np.int32)

file_pointers = [0]
bar = tqdm_notebook(total=num_images)

i = 0

current_checkpoint = 0

with open('../input/train.bson', 'rb') as fbson:

    data = bson.decode_file_iter(fbson)



    for c, d in enumerate(data):

        category = d['category_id']

        _id = d['_id']

        for e, pic in enumerate(d['imgs']):

            i += 1

            bar.update()

        

        if i > checkpoints[current_checkpoint + 1] and i < checkpoints[current_checkpoint + 2]:

            file_pointers.append(fbson.tell())

            current_checkpoint += 1
file_pointers
bar = tqdm_notebook(total=len(file_pointers))

for i in range(len(file_pointers) - 1):

    with open('train_example.bson', 'rb') as fbson:

        fbson.seek(file_pointers[i])

        bytes_chunk = fbson.read(file_pointers[i + 1] - file_pointers[i])

        # Do something with bytes_chunk, for example: write to file, upload to Amazon S3, etc.