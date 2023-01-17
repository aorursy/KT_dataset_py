


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import csv

sub_files = ['../input/lstm_mobilenet.csv',
             '../input/lstm_mobilenet.csv',
             '../input/lstm_mobilenet.csv']

## Weights of the individual subs ##
sub_weight = [1.75, 1.45, 1.2]

place_weights = {}
for i in range(3):
    place_weights[i] = 10-i*2

Hlabel = 'key_id'
Htarget = 'word'

lg = len(sub_files)
sub = [None]*lg
for i, file in enumerate( sub_files ):
    ## input files ##
    print("Reading {}: w={} - {}". format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file,"r"))
    sub[i] = sorted(reader, key=lambda d: float(d[Hlabel]))

## output file ##
out = open("submition_diego2.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel,Htarget])
p=0
for row in sub[0]:
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt,0) + (place_weights[ind]*sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:3]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
    p+=1
out.close()