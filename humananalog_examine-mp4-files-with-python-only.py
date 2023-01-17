import os, sys

import struct

import numpy as np
def find_boxes(f, start_offset=0, end_offset=float("inf")):

    """Returns a dictionary of all the data boxes and their absolute starting

    and ending offsets inside the mp4 file.



    Specify a start_offset and end_offset to read sub-boxes.

    """

    s = struct.Struct("> I 4s") 

    boxes = {}

    offset = start_offset

    f.seek(offset, 0)

    while offset < end_offset:

        data = f.read(8)               # read box header

        if data == b"": break          # EOF

        length, text = s.unpack(data)

        f.seek(length - 8, 1)          # skip to next box

        boxes[text] = (offset, offset + length)

        offset += length

    return boxes
def scan_mvhd(f, offset):

    f.seek(offset, 0)

    f.seek(8, 1)            # skip box header



    data = f.read(1)        # read version number

    version = int.from_bytes(data, "big")

    word_size = 8 if version == 1 else 4



    f.seek(3, 1)            # skip flags

    f.seek(word_size*2, 1)  # skip dates



    timescale = int.from_bytes(f.read(4), "big")

    if timescale == 0: timescale = 600



    duration = int.from_bytes(f.read(word_size), "big")



    print("Duration (sec):", duration / timescale)
def examine_mp4(filename):

    print("Examining:", filename)

    

    with open(filename, "rb") as f:

        boxes = find_boxes(f)

        print(boxes)



        # Sanity check that this really is a movie file.

        assert(boxes[b"ftyp"][0] == 0)



        moov_boxes = find_boxes(f, boxes[b"moov"][0] + 8, boxes[b"moov"][1])

        print(moov_boxes)



        trak_boxes = find_boxes(f, moov_boxes[b"trak"][0] + 8, moov_boxes[b"trak"][1])

        print(trak_boxes)



        udta_boxes = find_boxes(f, moov_boxes[b"udta"][0] + 8, moov_boxes[b"udta"][1])

        print(udta_boxes)



        scan_mvhd(f, moov_boxes[b"mvhd"][0])
test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"

test_files = [x for x in os.listdir(test_dir) if x[-4:] == ".mp4"]



train_dir = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/"

train_files = [x for x in os.listdir(train_dir) if x[-4:] == ".mp4"]
examine_mp4(os.path.join(train_dir, np.random.choice(train_files)))
examine_mp4(os.path.join(test_dir, np.random.choice(test_files)))