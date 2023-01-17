# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
#import a csv file 

data=pd.read_csv('../input/data.csv')

#when we want to look the data we write this code statemant

data.info()

# it is a large data but it is not important because this data too good for learn 



# I be curious columns of the data

data.columns
# Ok we learnt the columns ,now we will learn about  Messi 

data[data['Name']=='L. Messi']

# his values are €118.5M (he is too expensive :) 

data[data['Name']=='L. Messi'].Value

# I want to learn best of the player (for potential )

data.sort_values('Potential',ascending=[False])[0:10]

#I want to sort value of player but value is string it is not integer or float