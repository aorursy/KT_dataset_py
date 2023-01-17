import numpy as np 

import pandas as pd 

import datetime

import os

from collections import defaultdict

import operator

f = open('../input/train_ver2.csv','r')

first_line = f.readline().strip()

first_line = first_line.replace("\"","")

map_names = first_line.split(',')[24:]

                                
custormer = dict()

best_valid = defaultdict(lambda: defaultdict(int))

overallbest_valid = defaultdict(int)



valid_part = []
total = 0

while 1:

    line = f.readline()[:-1]

    total += 1

    

    if line=='':

        break

    tmp1 = line.split("\"")

    arr = tmp1[0][:-1].split(',') + [tmp1[1]] + tmp1[2][1:].split(',')

    arr = [a.strip() for a in arr]#去掉空格
f.readline()