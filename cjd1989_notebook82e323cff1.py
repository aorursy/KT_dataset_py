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

import os
#print(os.path.abspath(os.curdir))

print(os.listdir("/kaggle/input"))

#os.chdir("input")
path = os.getcwd()

print(path)
data_sub = pd.read_csv("/kaggle/input/sample_submission.csv")

data_test = pd.read_csv("/kaggle/input/test_ver2.csv", low_memory=False)
data_sub
data_test
chunks = pd.read_csv("/kaggle/input/train_ver2.csv", low_memory=False, chunksize=1000000)
num=1

for chunk in chunks:

    print(num)

    print(chunk)

    num+=1

  
print(len(data_test))
data_train = pd.concat(chunk for chunk in chunks)