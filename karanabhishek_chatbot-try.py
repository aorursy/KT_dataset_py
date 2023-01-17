# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time
 
 
 
########## PART 1 - DATA PREPROCESSING ##########
 
 
 
# Importing the dataset
lines = open('./movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('./movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
 
# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
