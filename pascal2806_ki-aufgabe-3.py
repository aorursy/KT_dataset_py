# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import csv
import os
import math
import random
print(os.listdir("../input/"))


def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

trainingsset = read_data("../input/kiwhs-comp-1-complete/train.arff")
#trainingsset = read_data("../input/neueTrainingsdaten/train-skewed.arff")

#trainset = random

def teile(trainingsdaten):
    pos = []
    neg = []
    for i in trainingsdaten:
        if(i[2] == 1):
            pos.append([i[0],i[1]])       
        else:
            neg.append([i[0],i[1]])
    return pos, neg

pos, neg = teile(trainingsset)

def finde_schwerpunkt(gruppe):
    x = 0.0
    y = 0.0
    for i in gruppe:
        x = x + i[0]
        y = y + i[1]
    x = x / len(gruppe)
    y = y / len(gruppe)
    return x, y

x_pos, y_pos = finde_schwerpunkt(pos)
x_neg, y_neg = finde_schwerpunkt(neg)

def berechne_abstand(x,y,x1,y2):
    return math.sqrt(((x-x1)**2) + ((y - y2)**2))

def ordne_hinzu(x,y):
    return 1 if ((berechne_abstand(x,y,x_pos,y_pos)) < (berechne_abstand(x,y,x_neg,y_neg))) else -1


test = pandas.read_csv('../input/kiwhs-comp-1-complete/test.csv',index_col = 'Id', header = 0)

with open('submission.csv','w') as submission:
    writer = csv.writer(submission)
    writer.writerow(['Id (String)','Category (String)'])
    for t, x, y in test.itertuples():
        writer.writerow([t,ordne_hinzu(x,y)])


