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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
%matplotlib notebook
import seaborn as sns
%matplotlib inline 
import networkx as nx

char_data = pd.read_csv("../input/wowah_data.csv")
print (char_data.describe())
print (char_data.char.nunique()) #no of unique players
print (len(char_data)) #no of rows in the dataset
# note that this is not equal. i initially thought it was. So silly of me!
max_level = []
x={}
char_id = char_data.char.unique()
for i in char_data.index:
    x[char_data["char"][i]] = max(char_data["level"][i], x.get(char_data["char"][i],0))
for i in x.keys():
    max_level.append([i,x[i]])
print (max_level[:10])
