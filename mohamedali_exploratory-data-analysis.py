import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import glob 
from IPython.display import display 
from IPython.display import display, HTML

files = glob.glob("../input/*")
for file_ in files:
    print(file_)
    #display(pd.read_csv(file_,nrows=100).head(2))
    #HTML(pd.read_csv(file_,nrows=5).to_html())
    print(pd.read_csv(file_,nrows=2).head(2))
##
import random 
def gimme_a_sample(filename, sample_size):
    #n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
    n = pd.read_csv(filename, usecols=[0]).shape[0]
    skip = sorted(random.sample(range(1,n+1),n-sample_size)) #the 0-indexed header will not be included in the skip list
    return pd.read_csv(filename, skiprows=skip)
train = gimme_a_sample("../input/train.csv",10000)
train.head()
train.corr()