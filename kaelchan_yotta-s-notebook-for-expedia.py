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
print("hello")
from math import log
from collections import defaultdict

restricted = []
for i in range(23):
    if i not in [0,11,12]:
        restricted.append(i)

file = open("../input/train.csv", "r")
cluster = defaultdict(int)
feature = [defaultdict(lambda:defaultdict(int)) for i in range(23)]
count = 0
for line in file:
    if count % 2000000 == 0:
        print(count)
    raw = line.strip().split(",")
    c = int(raw[-1])
    cluster[c] += 1
    for i in restricted:
        if raw[i] != "":
            feature[i][raw[i]][c] += 1
    count += 1
def gainratio(D, infoCluster):
    info_a = 0
    split_info = 0
    d = 0
    for i, Di in D.items():
        di = 0
        for j, dij in Di.items():
            info_a -= dij*log(dij)
            di += dij
        info_a += di*log(di)
        split_info -= di*log(di)
        d += di
    split_info += d*log(d)
    return (infoCluster - info_a) / split_info

d, infoCluster = 0, 0
for j, cj in cluster.items():
    infoCluster -= cj*log(cj)
    d += cj
infoCluster += d*log(d)
file = open("../input/train.csv", "r")
attr = file.readline().strip().split(",")
file.close()
ratio = []
for i in restricted:
    ratio.append( (gainratio(feature[i], infoCluster), attr[i]) )
ratio.sort(reverse=True)
ratio





