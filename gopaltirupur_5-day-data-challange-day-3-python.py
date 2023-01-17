import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mt
import seaborn as sns

%matplotlib inline
%pylab inline
cereals = pd.read_csv("../input/cereal.csv")
cereals.head()
column_names = cereals.columns
print(column_names)
print(len(cereals.columns))
for columnName in column_names:
    print(columnName,' : ',cereals[columnName].dtype)
cereals.describe()        
#1. There is no missing values
cereals.plot(kind='box',subplots=True,figsize=(12,5),use_index=True)
cereals['fat'].plot(kind='hist')
cereals['fat'].max()
sugar_cold = cereals[cereals['type']=='C']['sugars']
sugar_hot = cereals[cereals['type']=='H']['sugars']
from scipy import stats
from scipy.stats import ttest_ind,norm,skew


ttest_ind(sugar_cold,sugar_hot,axis=0,equal_var=False)
#since the pvalue of cold and hot sugar is 0.0187 which is greater than 0.01, hence they they
#different sugars 
sodium_cold = cereals[cereals['type']=='C']['sodium']
sodium_hot = cereals[cereals['type']=='H']['sodium']

ttest_ind(sodium_cold,sodium_hot,axis=0,equal_var=False)
#since the pvalue of cold and hot sugar is 0.0.02411 which is greater than 0.01, hence they they
#different sugars 