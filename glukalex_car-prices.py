# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# linear algebra

import numpy as np 

# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 

# Input data files are available in the "../input/" directory.

# For example, 

# running this (by clicking run or pressing Shift+Enter) 

# will list the files 

# in the input directory

from subprocess import check_output

#

print( check_output( [ "ls", "../input" ] ).decode( "utf8" ) )

# Any results you write to the current directory are saved as output.
import glob

#

#>>> glob.glob('./[0-9].*')

#['./1.gif', './2.txt']

#>>> glob.glob('*.gif')

#['1.gif', 'card.gif']

#>>> glob.glob('?.gif')

#['1.gif']

glob.glob('**', recursive=True )
data_Url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/"

file1_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

data_Frame = pd.read_csv( 

    #data_Url 

    file1_URL,

    header = None

)

data_Frame.head( 7 )