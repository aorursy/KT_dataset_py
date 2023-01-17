import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import os
os.chdir('../input')
train = pd.read_csv('home.csv') 
train
train.columns
train.describe() 
#missing values

missing = train.isnull().sum()

missing = missing[missing>0]

missing.sort_values(inplace=True)

missing.plot.bar() 
sns.countplot(train['SalePrice'], color= 'gold')
sns.scatterplot(train['Condition1'], train['Condition2'], hue= train['SalePrice'], palette= 'winter')
X1= train[['Condition1', 'Condition2', 'SalePrice']] 
#to find categorical data

cat = train.select_dtypes(exclude=['int64','float64'])

list_of_cat = list(cat.columns) 

dict_of_cat = {i:list_of_cat[i] for i in range(len(list_of_cat))}  
cat.columns
f, axes = plt.subplots(1, 2,figsize=(12, 8))

g = sns.swarmplot(x= train['Street'] ,y= train['SalePrice'], ax= axes[0]) # y is Saleprice

g = g.set_ylabel("Sale Price for Diffrent Streets")



labels=['Pave','Grvl']

slices=[cat.loc[cat.Street=="Pave"].shape[0],cat.loc[cat.Street=="Grvl"].shape[0]]

plt.pie(slices,labels=labels,startangle=90,explode=(0.5,0.7),autopct='%.2f%%',colors=['purple', 'red'])

plt.show()
f, axes = plt.subplots(1, 2,figsize=(12, 8))

h = sns.swarmplot(x= train['Alley'] ,y= train['SalePrice'], ax= axes[0]) # y is Saleprice

h = h.set_ylabel("Sale Price for Diffrent Alleys")



labels=['Pave','Grvl']

slices=[cat[cat.Alley=="Pave"].shape[0],cat[cat.Alley=="Grvl"].shape[0]]

plt.pie(slices,labels=labels,startangle=90,autopct='%1.2f%%',colors=['purple', 'red'])

plt.show() 
#analysing categorical data
qwe = list(i for i in range(5))

f,axes=plt.subplots(1, 5,figsize=(20,8))

f.subplots_adjust(hspace=0.5)



for j in qwe:

    g = sns.swarmplot(x=cat[dict_of_cat[j]],y= train['SalePrice'],ax=axes[j], palette= 'twilight')

    g.set_title(label=dict_of_cat[j].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[j]),fontsize=25)

    g.set_ylabel('SalePrice',fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 3,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(5,8)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight_r') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout()
f,axes=plt.subplots(1, 2,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(9,11)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight_r') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 3,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(11,14)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight_r') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 2,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(14,16)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight_r') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 3,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(17,20)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight_r') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 3,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(20,23)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight_r') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 3,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(23,26)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight_r') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 3,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(26,29)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 3,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(29,32)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 3,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(32,35)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 3,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(35,38)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 2,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(39,41)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
f,axes=plt.subplots(1, 2,figsize=(15,8))

f.subplots_adjust(hspace=0.5)

for j,i in zip(qwe,range(41,43)):

    g = sns.swarmplot(x= cat[dict_of_cat[i]], y= train['SalePrice'] ,ax=axes[j], palette= 'twilight') 

    g.set_title(label= dict_of_cat[i].upper(),fontsize=20)

    g.set_xlabel(str(dict_of_cat[i]),fontsize=25)

    g.set_ylabel('SalePrice', fontsize=25)

    plt.tight_layout() 
figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

figure.set_size_inches(20,10)

sns.regplot(train['TotalBsmtSF'], train['SalePrice'], ax=ax1)

sns.regplot(train['2ndFlrSF'], train['SalePrice'], ax=ax2)

sns.regplot(train['TotalBsmtSF'] + train['2ndFlrSF'], train['SalePrice'], ax=ax3) 
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

figure.set_size_inches(14,10)

sns.barplot(train['BsmtFullBath'], train['SalePrice'], ax= ax1)

sns.barplot(train['FullBath'], train['SalePrice'], ax= ax2)

sns.barplot(train['BsmtHalfBath'], train['SalePrice'], ax= ax3)

sns.barplot(train['BsmtFullBath'] + train['FullBath'] + train['BsmtHalfBath'] + train['HalfBath'], train['SalePrice'], ax= ax4)
figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

figure.set_size_inches(18,8)

sns.regplot(train['YearBuilt'], train['SalePrice'], ax=ax1)

sns.regplot(train['YearRemodAdd'], train['SalePrice'], ax=ax2)

sns.regplot((train['YearBuilt']+train['YearRemodAdd'])/2, train['SalePrice'], ax=ax3) 
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)

figure.set_size_inches(20,10)

sns.regplot(train['OpenPorchSF'], train['SalePrice'], ax=ax1)

sns.regplot(train['3SsnPorch'], train['SalePrice'], ax=ax2)

sns.regplot(train['EnclosedPorch'], train['SalePrice'], ax=ax3)

sns.regplot(train['ScreenPorch'], train['SalePrice'], ax=ax4)

sns.regplot(train['WoodDeckSF'], train['SalePrice'], ax=ax5)

sns.regplot((train['OpenPorchSF']+train['3SsnPorch']+train['EnclosedPorch']+train['ScreenPorch']+train['WoodDeckSF']), train['SalePrice'], ax=ax6)
fig = plt.figure(figsize=(11,11))

print("Skew of SalePrice:", train.SalePrice.skew())

plt.hist(train.SalePrice, density =1, color='red')

plt.show() 
fig = plt.figure(figsize=(11,11))

print("Skew of Log-Transformed SalePrice:", np.log1p(train.SalePrice).skew())

plt.hist(np.log1p(train.SalePrice), color='green')

plt.show()