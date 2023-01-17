import os

import time

import h5py

import numpy as np

import pandas as pd
!wc -l ../input/train.csv
n_lines_train = 629145481
chunk_size = 150000

n_segments = (n_lines_train-1)//chunk_size

n_segments
leftover = n_lines_train - n_segments*chunk_size

leftover
leftover/n_lines_train
input_dir = "../input"

output_dir = ""
#create the hdf5 file

h5_file = h5py.File(os.path.join(output_dir, "train.h5"), "w")
#create datasets within the top level group in that file

sound_dset = h5_file.create_dataset("sound", shape=(n_segments, chunk_size), dtype=np.int16)

ttf_dset = h5_file.create_dataset("ttf", shape=(n_segments,), dtype=np.float32)
#iterate over all 629 million lines and save them out in chunks of 150,000

#this takes a while ...

chunk_size = 150000

lines_to_read = chunk_size*n_segments



printed_warning = False

with open(os.path.join(input_dir, "train.csv")) as f:

    x_stack, y_stack = [], []

    last_ttf = np.inf

    hdr = f.readline()

    for line_idx in range(lines_to_read):

        cx, cy = f.readline().split(",")

        cx = int(cx)

        if np.abs(cx) > 32767:

            if not printed_warning:

                printed_warning = True

                print("line {} is too big to be an int16".format(line_idx+1))

        cy = float(cy)

        if cy < last_ttf:

            last_ttf = cy

            y_stack.append(cy)

        x_stack.append(cx)

        if line_idx % chunk_size == chunk_size-1:

            sound_dset[line_idx//chunk_size] = np.array(x_stack).astype(np.int16)

            ttf_dset[line_idx//chunk_size] = np.mean(y_stack)

            x_stack, y_stack = [], []

            last_ttf = np.inf
h5_file.close()
start_time = time.time()

hf = h5py.File(os.path.join(output_dir, "train.h5"))

segs = np.array(hf["sound"][:100])

seg_ttf = np.array(hf["ttf"][:100])

hf.close()

end_time = time.time()

print("{} seconds".format(end_time-start_time))
start_time = time.time()

df = pd.read_csv(

    os.path.join(input_dir, "train.csv"), 

    nrows=chunk_size*100,#limit to the first 100 segments 

    dtype={"acoustic_data":np.int16, "time_to_failure":np.float32}

)

end_time = time.time()

print("{} seconds".format(end_time-start_time))
test_files = os.listdir(os.path.join(input_dir, "test"))



with h5py.File(os.path.join(output_dir, "test.h5"), "w") as h5_file:

    sound_dset = h5_file.create_dataset("sound", (len(test_files), chunk_size))

    seg_ids = []



    for fidx, fname in enumerate(test_files):

        cdata = pd.read_csv(os.path.join(input_dir, "test", fname), dtype=np.int16)["acoustic_data"].values

        sound_dset[fidx] = cdata

        seg_ids.append(fname.split(".")[0])



    h5_file["seg_id"] = np.array(seg_ids).astype(np.string_)
#loading back in the test data segment strings

hf = h5py.File(os.path.join(output_dir, "test.h5"))
#without a .astype(str) the resulting array is of type bytes

seg_ids = np.array(hf["seg_id"])

seg_ids
#adding a astype call gets us nice unicode strings

#that play well with python 3

seg_ids = np.array(hf["seg_id"]).astype(str)

seg_ids
hf.close()