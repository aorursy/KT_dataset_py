# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
filenames = check_output(["ls", "../input/WEB"]).decode("utf8").strip().split('\n')

# Any results you write to the current directory are saved as output.
filenames

filenames2 = check_output(["ls", "../input/WEB/Matthew"]).decode("utf8").strip().split('\n')
filenames2
textpath = "../input/WEB/transcript.txt"
lines =[]
with open(textpath,'r') as f:
    lines.extend(f.readline() for i in range(5))
lines
df = pd.read_csv(textpath, sep = '\t', header = None)
df.head()
