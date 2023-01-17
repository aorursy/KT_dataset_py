#Importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Loading our dataset
data=pd.read_csv("../input/spacedatacsv/spacedataset.csv")
data.head()

#Dropping unnecessary columns
data=data.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
data.head()
#Info about data
data.info()
#A small description about data
data.describe()