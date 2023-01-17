import numpy as np 
import pandas as pd 

data = pd.read_csv('../input/daily-inmates-in-custody.csv')
data.head()
data.isnull().sum()
import matplotlib.pyplot as plt
%matplotlib inline
data['CUSTODY_LEVEL'].value_counts().plot(kind='bar');
data['RACE'].value_counts().plot(kind='bar');