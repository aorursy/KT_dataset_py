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
# Import libraries necessary for this project

import numpy as np

import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames



# Import supplementary visualizations code visuals.py

#import visuals as vs



# Pretty display for notebooks

%matplotlib inline



# Load the dataset

in_file = '../input/train.csv'

full_data = pd.read_csv(in_file)



# Print the first few entries of the RMS Titanic data

display(full_data.head())
# Store the 'Survived' feature in a new variable and remove it from the dataset

outcomes = full_data['Survived']

data = full_data.drop('Survived', axis = 1)



# Show the new dataset with 'Survived' removed

display(data.head())

#print(data.loc[0])

#print(outcomes[0])