# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Uncomment the following line of code and make sure the 'Internet' option of the kernel is turned 'ON'

#!pip install --upgrade bamboolib>=1.2.1
# Importing the necessary libraries 

import bamboolib as bam

bam.enable()

import pandas as pd

#Importing the training dataset

df = pd.read_csv('/kaggle/input/titanic/train.csv')

df
