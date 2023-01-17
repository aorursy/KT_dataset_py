# Helpful packages to load in 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")
print(train_df.head(5))
print('-'*97)
# Any results you write to the current directory are saved as output.
# Data Exploration using Pandas
print(train_df.dtypes)
print(train_df.Pclass. unique())
print(train_df.info())
print('-'*100)
print(train_df.describe(include='all'))
# Data imputation using Fillna
train_df = train_df.Age.fillna(train_df.Age.median(),inplace=True)
