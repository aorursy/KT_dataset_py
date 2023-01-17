import bson 

from bson import BSON

import os

from tqdm import tqdm_notebook as tqdm

from subprocess import check_output

print(check_output(["ls", ".."]).decode("utf8"))
if os.path.isfile("train_small.bson"):

    os.remove("train_small.bson")



if os.path.isfile("test_small.bson"):

    os.remove("test_small.bson")

    

f = open('train_small.bson', 'ab+')



MAX_TRAIN_FILE_SIZE = 40 * 1024 * 1024

MAX_TEST_FILE_SIZE = 5 * 1024 * 1024



data = bson.decode_file_iter(open('../input/train.bson', 'rb'))

cat_list = list()

pbar = tqdm(total=MAX_TRAIN_FILE_SIZE)

sz_last = 0

for d in data:

    category_id = d['category_id']

    if not category_id in cat_list:

        f.write(BSON.encode(d))

        cat_list.append(category_id)        

        fsz = f.tell()

        sz_diff = fsz - sz_last

        pbar.update(sz_diff)

        sz_last = fsz

        if fsz > MAX_TRAIN_FILE_SIZE:

            break

pbar.close()

f.close()

print("Saved {:d} categories in train.".format(len(cat_list)))



## Saving test sample as well.

f = open('test_small.bson', 'ab+')

data = bson.decode_file_iter(open('../input/test.bson', 'rb'))

pbar = tqdm(total=MAX_TEST_FILE_SIZE)

sz_last = 0

num_samples = 0

for d in data:

    f.write(BSON.encode(d))

    num_samples += 1

    fsz = f.tell()

    sz_diff = fsz - sz_last

    pbar.update(sz_diff)

    sz_last = fsz

    if fsz > MAX_TEST_FILE_SIZE:

        break

pbar.close()

f.close()

print("Saved {:d} samples in test.".format(num_samples))