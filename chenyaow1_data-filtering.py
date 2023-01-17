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
vertex_set = set();
sink_dict = {};

with open("../input/iclp18/train.txt") as trainfile:
    for i, line in enumerate(trainfile):
        line_list = [int(k) for k in line[:-1].split("\t")];
        vertex_set.add(line_list[0]);
        for s in line_list[1:]:
            if s in sink_dict:
                sink_dict[s] += 1;
            else:
                sink_dict[s] = 1;
        if i % 1000 == 0:
            print(i);

print(len(sink_dict));
print(len(vertex_set));
new_sink_dict = {};
threshold = 10;
for k in sink_dict:
    if sink_dict[k] >= threshold:
        new_sink_dict[k] = sink_dict[k];
        
new_sink_set = set(new_sink_dict);
print(len(new_sink_set))
test_vertex_and_sink_set = set();

with open("../input/testpublic-txt/test-public.txt") as testfile:
    for i, line in enumerate(testfile):
        if i == 0:
            continue;
        line_list = [int(k) for k in line[:-1].split("\t")];
        for s in line_list:
            test_vertex_and_sink_set.add(s);
print(len(test_vertex_and_sink_set));

total_set = test_vertex_and_sink_set.union(new_sink_set).union(vertex_set);
print(len(total_set));
total_dict = {};
total_list = []
for i, p in enumerate(total_set):
    total_dict[p] = i
    total_list.append(p)
print(total_dict[4394015])
print(total_dict[2397416])
print(total_dict[1272125])
print(total_dict[2984819])
print(total_list[83517])
print(total_list[124658])
print(total_list[93341])
print(total_list[348183])
max_neighbors = 1000

import numpy as np
total_array = np.array(total_list);

pairs = [];

with open("../input/iclp18/train.txt") as trainfile:
    for i, line in enumerate(trainfile):
        line_list = [int(k) for k in line[:-1].split("\t")];
        v = line_list[0];
        ranking = [-sink_dict[k] for k in line_list[1:]];
        sorting = np.argsort(ranking);
        filtered_linelist = np.array(line_list[1:])[sorting];
        for s in filtered_linelist[1:max_neighbors]:
            if s in total_set:
                pairs.append([total_dict[v], total_dict[s]]);
        if i % 1000 == 0:
            print(i);

test_pairs = [];

with open("../input/testpublic-txt/test-public.txt") as testfile:
    for i, line in enumerate(testfile):
        if i == 0:
            continue;
        line_list = [int(k) for k in line[:-1].split("\t")];
        test_pairs.append([line_list[0], total_dict[line_list[1]], total_dict[line_list[2]]]);

np.savez_compressed("filtered_data", correspondence = total_array, pairs = pairs, test_pairs = test_pairs);

print(os.path.getsize('filtered_data.npz') / 1000000)

with np.load('filtered_data.npz') as fd:
    print(fd["pairs"].shape);
    print(fd["pairs"][0:9]);
    print(fd["test_pairs"].shape);
    print(fd["test_pairs"][0:9]);
    print(fd["correspondence"][136288]);
    print(fd["correspondence"][208308]);