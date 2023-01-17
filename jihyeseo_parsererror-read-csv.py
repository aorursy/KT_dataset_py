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
import chardet
from subprocess import check_output
filename = check_output(["ls", "../input"]).decode("utf8").strip()

filename
# Any results you write to the current directory are saved as output.
lines =[]
with open('../input/1000sents.csv','r') as f:
    lines.extend(f.readline() for i in range(5))

lines
for size in range(1,11):
    length = 10 ** size
    with open("../input/1000sents.csv"  , 'rb') as rawdata:
        result = chardet.detect(rawdata.read(length))

    # check what the character encoding might be
    print(size, length, result)
#df = pd.read_csv('../input/1000sents.csv', encoding = 'UTF-8-SIG')

#df.head()

# ParserError: Error tokenizing data. C error: Expected 12 fields in line 676, saw 13