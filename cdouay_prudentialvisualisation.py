import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt 
%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# See nan proportion per columns

print('proportion of nan values in train set : ')
print(train.isnull().sum(axis = 0).sort_values(ascending = False).head(14)/len(train))
print('\n')
print('proportion of nan values in test set : ')
print(test.isnull().sum(axis = 0).sort_values(ascending = False).head(14)/len(train))
# Influence of BMI on risk level 
data = []
for i in range(1,9): 
    data.append(train.BMI[train.Response == i])
plt.figure(figsize = (15,8))
plt.subplot(2,2,1)
plt.boxplot(data)
plt.title('Response VS BMI')

data = []
for i in range(1,9): 
    data.append(train.Ins_Age[train.Response == i])
plt.subplot(2,2,2)
plt.boxplot(data)
plt.title('Response VS Ins_Age')

data = []
for i in range(1,9): 
    data.append(train.Wt[train.Response == i])
plt.subplot(2,2,3)
plt.boxplot(data)
plt.title('Response VS weight')

data = []
for i in range(1,8): 
    data.append(train.Ht[train.Response == i])
plt.subplot(2,2,4)
plt.boxplot(data)
plt.title('Response VS height')


# Study of employement
#Continuous variables  1-4-6
plt.figure(figsize = (15,8))

data  = []
for i in range(1,9):
    x = train.Employment_Info_1[train.Response == i]
    data.append(np.sqrt(x[~np.isnan(x)]))

plt.subplot(2,2,1)
plt.boxplot(data)
plt.title('Response VS Employement info 1')

data  = []
for i in range(1,9):
    x = train.Employment_Info_4[train.Response == i]
    data.append(x[~np.isnan(x)])

plt.subplot(2,2,2)
plt.boxplot(data)
plt.title('Response VS Employement info 4')

data  = []
for i in range(1,9):
    x = train.Employment_Info_6[train.Response == i]
    data.append(x[~np.isnan(x)])

plt.subplot(2,2,3)
plt.boxplot(data)
plt.title('Response VS Employement info 6')
#plot histogram of log of employement info 1
plt.figure(figsize = (15,9))
for i in range(1,9): 
    plt.subplot(3,3,i)
    x = train.Employment_Info_1[train.Response==i]
    x = x[~np.isnan(x)]
    x = [i for i in x]
    plt.hist(x, bins = 50)
    plt.title('Response '+ str(i))
#plot histogram of log of employement info 4
plt.figure(figsize = (15,9))
for i in range(1,9): 
    plt.subplot(3,3,i)
    x = train.Employment_Info_4[train.Response==i]
    x = x[~np.isnan(x)]
    x = ([i for i in x if i!=0])
    plt.hist(np.log(x), bins = 50)
    plt.title('Response '+ str(i))
data = []
plt.figure(figsize = (15,9))
for i in range(1,9): 
    x = train.Employment_Info_4[train.Response==i]
    x = x[~np.isnan(x)]
    x = ([i for i in x if i!=0])
    x = np.log(x)
    data.append(x)
plt.boxplot(data)
print('')
# Boxplot for variables : Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5

plt.figure(figsize = (15,8))

data  = []
for i in range(1,9):
    x = train.Family_Hist_2[train.Response == i]
    data.append(x[~np.isnan(x)])
plt.subplot(2,2,1)
plt.boxplot(data)
plt.title('Response VS Family_Hist_2')

data  = []
for i in range(1,9):
    x = train.Family_Hist_3[train.Response == i]
    data.append(x[~np.isnan(x)])

plt.subplot(2,2,2)
plt.boxplot(data)
plt.title('Response VS Family_Hist_3')

data  = []
for i in range(1,9):
    x = train.Family_Hist_4[train.Response == i]
    data.append(x[~np.isnan(x)])

plt.subplot(2,2,3)
plt.boxplot(data)
plt.title('Response VS Family_Hist_4')
          
data  = []
for i in range(1,9):
    x = train.Family_Hist_5[train.Response == i]
    data.append(x[~np.isnan(x)])

plt.subplot(2,2,4)
plt.boxplot(data)
plt.title('Response VS Family_Hist_5')

#plot histogram of log of Family_Hist_2
plt.figure(figsize = (15,9))
for i in range(1,9): 
    plt.subplot(3,3,i)
    x = train.Family_Hist_2[train.Response==i]
    x.hist(bins = 50)
    x = x[~np.isnan(x)]
    plt.title('Response '+ str(i))
# Discrete variables Medical_History_1, Medical_History_10, 
# Medical_History_15, Medical_History_24, Medical_History_32

plt.figure(figsize = (15,8))

data  = []
for i in range(1,9):
    x = train.Medical_History_1[train.Response == i]
    data.append(np.log1p(x[~np.isnan(x)]))
#Apply a log1p on Medical history 1
plt.subplot(3,2,1)
plt.boxplot(data)
plt.title('Response VS Medical_History_1')

data  = []
for i in range(1,9):
    x = train.Medical_History_10[train.Response == i]
    data.append(x[~np.isnan(x)])

plt.subplot(3,2,2)
plt.boxplot(data)
plt.title('Response VS Medical_History_10')

data  = []
for i in range(1,9):
    x = train.Medical_History_15[train.Response == i]
    data.append(x[~np.isnan(x)])

plt.subplot(3,2,3)
plt.boxplot(data)
plt.title('Response VS Medical_History_15')
          
data  = []
for i in range(1,9):
    x = train.Medical_History_24[train.Response == i]
    data.append(x[~np.isnan(x)])

plt.subplot(3,2,4)
plt.boxplot(data)
plt.title('Response VS Medical_History_24')

data  = []
for i in range(1,9):
    x = train.Medical_History_32[train.Response == i]
    data.append(x[~np.isnan(x)])

plt.subplot(3,2,5)
plt.boxplot(data)
plt.title('Response VS Medical_History_32')

# Feature selection of catégorical variables 
# Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7
print('Values for product info 1 :{}'.format(train.Product_Info_1.unique().shape))
print('Values for product info 2 :{}'.format(train.Product_Info_2.unique().shape))
print('Values for product info 3 :{}'.format(train.Product_Info_3.unique().shape))
print('Values for product info 5 :{}'.format(train.Product_Info_5.unique().shape))
print('Values for product info 6 :{}'.format(train.Product_Info_6.unique().shape))
print('Values for product info 7 :{}'.format(train.Product_Info_7.unique().shape))
print(train.pivot_table(columns = 'Response', index = 'Product_Info_1', values = 'Id', aggfunc = len))
print(train.pivot_table(columns = 'Response', index = 'Product_Info_5', values = 'Id', aggfunc = len))
print(train.pivot_table(columns = 'Response', index = 'Product_Info_6', values = 'Id', aggfunc = len))
print(train.pivot_table(columns = 'Response', index = 'Product_Info_7', values = 'Id', aggfunc = len, fill_value = 0))
