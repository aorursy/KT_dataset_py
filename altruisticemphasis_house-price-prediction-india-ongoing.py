

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

import statsmodels.formula.api as smf



from sklearn.preprocessing import LabelEncoder





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv('/kaggle/input/house-price-prediction-challenge/train.csv')

test_data = pd.read_csv('/kaggle/input/house-price-prediction-challenge/test.csv')

submission = pd.read_csv('/kaggle/input/house-price-prediction-challenge/sample_submission.csv')

train_data.info()
test_data.info()
train_data.dtypes
train_data.isnull().sum()

train_data.head()
train_data = train_data.rename(columns={"TARGET(PRICE_IN_LACS)": 'TARGET'})
# Outlier Analysis

fig, axs = plt.subplots(2,3, figsize = (10,5))

plt1 = sns.boxplot(train_data['BHK_NO.'], ax = axs[0,0])

plt2 = sns.boxplot(train_data['SQUARE_FT'], ax = axs[0,1])

plt3 = sns.boxplot(train_data['LONGITUDE'], ax = axs[0,2])

plt1 = sns.boxplot(train_data['LATITUDE'], ax = axs[1,0])

plt2 = sns.boxplot(train_data['TARGET'], ax = axs[1,1])





plt.tight_layout()
objList = train_data.select_dtypes(include = "object").columns

print (objList)
train_data.drop(['ADDRESS'], inplace=True, axis=1, errors='ignore')



#label_encoder object knows how to understand word labels. 

le = LabelEncoder() 

train_data['POSTED_BY']= le.fit_transform(train_data['POSTED_BY']) 

train_data['BHK_OR_RK']= le.fit_transform(train_data['BHK_OR_RK']) 
train_data.head()
sns.pairplot(train_data)

plt.show()
# Calculate correlation between each pair of variable

corr_matrix=train_data.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(230, 20, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot= True)


