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
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
input_data = pd.read_csv("../input/US_graduate_schools_admission_parameters_dataset.csv") #,index_col=0 
input_data.head()
X= input_data[['GRE Score','toefl score']]
y= input-data['chance of admit']
x=sm.add_constant(x)
multiple