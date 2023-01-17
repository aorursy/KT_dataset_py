# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import xgboost as xgb
# Define training and testing sets

train_df = open('../input/train.csv', "r")
test_df  = open('../input/test.csv', "r")
# Construct dest_clusters dictionary

from collections import defaultdict
import math

def scale(X,Y):
    X = math.floor(float(X) * 100)
    Y = math.floor(float(Y) * 100)
    return X,Y;

cluster = defaultdict(lambda: defaultdict(int))
train_df.readline()

while True:
	line = train_df.readline().strip()
	if line == '': break
	row = line.split(',')
	X,Y = scale(row[1],row[2])
	place_id = row[5]
    
    # for every (X,Y) - after scaling, increment it's palce_id by 1
	cluster[(X,Y)][place_id]+= 1

train_df.close()
# For every (X,Y), get the top frequent places

freq_places = dict()

for coord in cluster:
        d = cluster[coord]
        freq_places[coord] = sorted(d, key=d.get, reverse=True)[:3]
# Create Submission

submission = open("facebook.csv",'w')
submission.write("row_id,place_id\n")

test_df.readline()

while True:
    line = test_df.readline().strip()
    if line == '': break
    row = line.split(',')
    row_id = row[0]
    X = math.floor(float(row[1]) * 100)
    Y = math.floor(float(row[2]) * 100)
    
    submission.write(str(row_id)+",")
    if (X,Y) in freq_places:
        places = freq_places[(X,Y)]
        submission.write(" ".join(places))
        
    submission.write("\n")

submission.flush()
submission.close()