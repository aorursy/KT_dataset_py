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
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
train.columns
sample_submission.columns
train.describe()
train.info()
print(train['SaleType'].unique())
print(train['SaleCondition'].unique())
print(train['GarageCond'].unique())
print(train['Functional'].unique())
print(train['Heating'].unique())
'''nn_train = train.dropna(axis='columns')
nn_train.info()'''
max_size = train.shape[0]
max_size = max_size - int((train.shape[0]*20)/100)
# For takeing in the min row
nn_train = train.dropna(axis='columns', thresh=max_size)
nn_train = nn_train.select_dtypes(exclude=[np.object])
nn_train.fillna(nn_train.median(), inplace=True)
'''nn_train['LotFrontage'] = nn_train['LotFrontage'].astype(np.int64)
nn_train['MasVnrArea'] = nn_train['MasVnrArea'].astype(np.int64)
nn_train['GarageYrBlt'] = nn_train['GarageYrBlt'].astype(np.int64)'''
nn_train.info()
#nn_train.fillna(nn_train.mean(), inplace=True)
nn_train.info()
'''
nn_coloums = nn_train.columns
for i,x in enumerate(nn_coloums):
    print(x)
    print(nn_train[x].mode())
    nn_train[x]=nn_train[x].fillna(nn_train[x].median(), inplace=True)
'''
'''nn_train.'''
nn_train.info()
import seaborn as sns
import matplotlib.pyplot as plt

corr = nn_train.corr(method='pearson', min_periods=1)
#plt.figure(figsize=(60,32))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
corr_wrt_saleprice = corr['SalePrice']
coloum_name = corr_wrt_saleprice.keys()
final_coloums = list()
for i,x in enumerate(corr_wrt_saleprice):
    if( (x >= 0.0) or (x<=-0.0)):
        final_coloums.append(coloum_name[i])
del final_coloums[-1]
del final_coloums[0]
print(final_coloums)
x = final_coloums
y = ['SalePrice']
'''from sklearn.model_selection import train_test_split

train, test = train_test_split(train, test_size=0.2)'''
from sklearn.linear_model import LinearRegression as lr

model = lr()
model.fit(nn_train[x], nn_train[y])
print(model.score(nn_train[x], nn_train[y]))
'''pridected_salesPrice = model.predict(test[x])
model.score(test[x], test[y])'''
print(model.coef_)
print(model.intercept_)
from sklearn.ensemble import RandomForestClassifier as rf
model = rf()
model.fit(nn_train[x], nn_train[y])
print(model.score(nn_train[x], nn_train[y]))
'''from sklearn.linear_model import LinearRegression as lr

model = lr()
model.fit(train[x], train[y])
print(model.score(train[x], train[y]))
pridected_salesPrice = model.predict(test[x])
model.score(test[x], test[y])
print(model.coef_)
print(model.intercept_)'''
'''print(pridected_salesPrice)
df = pd.DataFrame({
    "Original": test[y], 
    "Pridicted": pridected_salesPrice
})'''
test = pd.read_csv("../input/test.csv")
nn_train = test.dropna(axis='columns')
nn_train = nn_train.select_dtypes(exclude=[np.object])
test = test.interpolate()
x = final_coloums
y_pred = model.predict(test[x])
test['SalePrice'] = y_pred
submission_df = pd.DataFrame({
    "Id": test['Id'],
    "SalePrice": test['SalePrice']
})
submission_df.to_csv('output.csv', index=False)
#print(submission_df)
submission_df