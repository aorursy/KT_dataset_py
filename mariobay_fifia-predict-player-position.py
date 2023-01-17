# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from sklearn.model_selection import train_test_split



fullData = pd.read_csv('../input/FullData.csv')



train, test = train_test_split(fullData, test_size = 0.2)



#The best way to do this in pandas is to use drop:

#df = df.drop('column_name', 1)

#where 1 is the axis number (0 for rows and 1 for columns.)



train = train.drop("National_Position", 1)






