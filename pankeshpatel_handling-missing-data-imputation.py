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
df = pd.read_csv("../input/Datapreprocessing.csv")

df
from sklearn.preprocessing import Imputer



imputer = Imputer(strategy="median")



# Since the median can only be computed on numerical attributes, we need to create a copu of the data without the text 

# attributes 



df_numerical = df[["Age", "Salary"]]

df_numerical
# Now you can fit the imputer instance to the training data using the fit() method

imputer.fit(df_numerical)
# The imputer has simply computed the median of each attributes and stored the result in its statistics_ instance variable.



imputer.statistics_
# Now you can use the trained imputer to transform the training set by replacing the missing values

X = imputer.transform(df_numerical)



# The result is a plain Numpy array containing the transformed features. 

X
# If you want to put the Numpy array back into Pandas DataFrame

pd.DataFrame(X,columns=["Age", "Salary"])