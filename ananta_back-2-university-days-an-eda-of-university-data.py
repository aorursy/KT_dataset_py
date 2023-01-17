""" Loading Libraries """

import os

import pandas as pd

import matplotlib as plt

import seaborn as sn
""" Explore the Input directory (Working with operating system with os library) """

# os.chdir() method in Python used to change the current working directory to specified path. It takes only a single argument as new directory path.

print(os.listdir("../input/"))



# os.getcwd(): Function os.getcwd(), returns the Current Working Directory(CWD) of the file used to execute the code, can vary from system to system.

#print(os.getcwd())
