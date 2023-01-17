import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("train shape - ",train.shape)
print("test shape - ", test.shape)
pd.options.display.max_rows = 100
pd.options.display.max_columns = 10
qualitative = [x for x in train.columns if train.dtypes[x] == 'object']
print("Number of quanlitative features :",len(qualitative))
print("Qualitative features -",*qualitative)
quantitative = [x for x in train.columns if train.dtypes[x] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
print('Number of quantitative features : ', len(quantitative))
print('Quantitative features -',*quantitative)
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace = True)
missing.plot.bar()
#print('Features with missing values -',missing.shape[0])
#print('Features with less than 5 missing values -',missing[missing < 5].shape[0])
#print('Features with 5 -50 missing values -',missing[ ( missing >5 & missing < 50)].shape[0])
#print('Features with more than 50% missing values -',missing[missing > (train.shape[0]/2)].shape[0])












train['Fence'].dtypes