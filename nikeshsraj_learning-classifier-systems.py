# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import csv 
import matplotlib.pyplot as plt
import pickle
import random
from keras.layers import Concatenate, concatenate
from keras.layers.normalization import BatchNormalization
import h5py
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
!pip install xcs
import xcs
from xcs import XCSAlgorithm
from xcs.scenarios import MUXProblem, ScenarioObserver
import logging
logging.root.setLevel(logging.INFO)
import random
from xcs.scenarios import Scenario
from xcs.bitstrings import BitString
df = pd.read_csv("../input/advertising-cmaes/advertising.csv")
df.head(5)
df.drop(['City', 'Country','Timestamp','Ad Topic Line'], axis = 1, inplace=True) 
print(df["Daily Internet Usage"].min())
df["Daily Time Spent on Site"] = pd.cut(df["Daily Time Spent on Site"],4,labels=[0,1,2,3])
df["Age"] = pd.cut(df["Age"],3,labels=[0,1,2])
df["Area Income"] = pd.cut(df["Area Income"],4,labels=[0,1,2,3])
df["Daily Internet Usage"] = pd.cut(df["Daily Internet Usage"],3,labels=[0,1,2])
print(df.head(10))
df1 = pd.get_dummies(df["Daily Time Spent on Site"], prefix = 'Daily Time Spent on Site')
df2 = pd.get_dummies(df["Age"], prefix = 'Age')
df3 = pd.get_dummies(df["Area Income"], prefix = 'Area Income')
df4 = pd.get_dummies(df["Daily Internet Usage"], prefix = 'Daily Internet Usage')
DataFrame = pd.concat([df["Clicked on Ad"],df["Male"],df1, df2,df3,df4], axis=1)
print(DataFrame.head(10))
pid_dataset = DataFrame.values
arr = []
for i in range(len(pid_dataset)):
    arr.append(''.join([str(x) for x in pid_dataset[i]]))
class Advertising(Scenario):
    def __init__(self, training_cycles=10000, input_size=500, inputs=[]):
        self.input_size = input_size
        self.possible_actions = (1, 0)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles
        self.needle_index = random.randrange(input_size)
        self.needle_value = None
        self.inputs = inputs
    
    @property
    def is_dynamic(self):
        return False
        
    def get_possible_actions(self):
        return self.possible_actions
    
    def reset(self):
        self.remaining_cycles = self.initial_training_cycles
        self.needle_index = random.randrange(self.input_size)
        
    def more(self):
        return self.remaining_cycles > 0
    
    def sense(self):
        random.seed()
        i = random.randint(0,len(self.inputs) - 1)
        c_feat = BitString(self.inputs[i][:-1])
        self.needle_value = self.inputs[i][-1]
        return c_feat
        
    def execute(self, action):
        self.remaining_cycles -= 1
        return int(action) == int(self.needle_value)
inp = len(arr[0])-1
problem = Advertising(training_cycles=10000, input_size=inp, inputs = arr)

algorithm = xcs.XCSAlgorithm()

# Default parameter settings in test()
algorithm.exploration_probability = .1
algorithm.discount_factor = 0
algorithm.do_ga_subsumption = True
algorithm.do_action_set_subsumption = True

# Modified parameter settings
algorithm.ga_threshold = 1
algorithm.crossover_probability = .5
algorithm.wildcard_probability = .5
algorithm.deletion_threshold = 2 
algorithm.subsumption_threshold = 10         # theta_sub
algorithm.mutation_probability = .003
model = algorithm.new_model(ScenarioObserver(problem))
history = model.run(ScenarioObserver(problem), learn=True)
print(model)
for rule in model:
    if rule.fitness > .5 and rule.experience >= 10:
        print(rule.condition, '=>', rule.action, ' [%.5f]' % rule.fitness) 