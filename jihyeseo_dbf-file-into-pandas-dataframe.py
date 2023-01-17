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

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd



from simpledbf import Dbf5



dbf = Dbf5('../input/afrbeep020.dbf')



df = dbf.to_dataframe()
import pysal as ps 
# this block of code copied from https://gist.github.com/ryan-hill/f90b1c68f60d12baea81 



import pysal as ps

import pandas as pd

'''

Arguments

---------

dbfile  : DBF file - Input to be imported

upper   : Condition - If true, make column heads upper case

'''

def dbf2DF(dbfile, upper=True): #Reads in DBF files and returns Pandas DF

    db = ps.open(dbfile) #Pysal to open DBF

    d = {col: db.by_col(col) for col in db.header} #Convert dbf to dictionary

    #pandasDF = pd.DataFrame(db[:]) #Convert to Pandas DF

    pandasDF = pd.DataFrame(d) #Convert to Pandas DF

    if upper == True: #Make columns uppercase if wanted 

        pandasDF.columns = map(str.upper, db.header) 

    db.close() 

    return pandasDF
df = dbf2DF('../input/afrbeep020.dbf')
df.head()
df.tail()
df.describe()
df['STATE'].value_counts()
df['AFRBEEP020'].value_counts()
df['FIPS'].value_counts() 
df['STATE_FIPS'].value_counts() 