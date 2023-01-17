# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Dictionaries are key value pairs

# Data in dictionaries is not accessed by index as is done in Lists

# Data in dictionaries is access via key



fruit = {"orange": "a sweet citrus fruit",

        "apple": "good for making cider",

        "lemon": "a sour, yellow citrus fruit",

        "grape": "a small, sweet fruit growing in bunches",

        "lime": "a sour, green citrus fruit"}

print(fruit)

print()

print(fruit["lemon"])

print("\n"*5)



#add a record in dictionaries like this

fruit["pear"] = "an odd shaped apple" # By identifying the key value pair

print(fruit)



#you can override the entry like this

fruit["pear"] = "great with tequila"

print(fruit)



#you can delete entry like this

del fruit["lemon"]

print(fruit)



#del fruit will delete entire dictionary

#in order to delete contents of dictionary use this command

fruit.clear()

print(fruit)
