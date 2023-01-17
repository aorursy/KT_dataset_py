# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns; 

sns.set(style="ticks", color_codes=True)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#If you need more computing power, I recommend Google Colab: https://colab.research.google.com

#from google.colab import drive

#drive.mount('/content/drive')
# Load the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

features = pd.read_csv('../input/features.csv')

stores = pd.read_csv('../input/stores.csv')



# Combine training sets and test sets, of course, we can't do that in the real world

full = pd.concat([train,test],ignore_index=True )



# The training set is combined with other feature sets

features_d = features.drop(['IsHoliday'],axis=1)

full = full.merge(features_d,how='left').merge(stores,how='left')
# The bool type is cast to int

full['IsHoliday'] = full['IsHoliday'].values + 0



# Date converted to time type

full['Date'] = pd.to_datetime(full['Date'])

full['Year'] = pd.to_datetime(full['Date']).dt.year

full['Month'] = pd.to_datetime(full['Date']).dt.month

full['WDay'] = pd.to_datetime(full['Date']).dt.weekofyear

full['Day'] = pd.to_datetime(full['Date']).dt.day



# The categorization feature uses One-Hot

full = pd.get_dummies(full,columns=['Type'])



# There are missing values in CPI, Unemployment, Temperature and MarkDown

full['CPI'] = full['CPI'].fillna(full['CPI'].mean())

full['Temperature'] = full['Temperature'].fillna(full['Temperature'].mean())

full['Unemployment'] = full['Unemployment'].fillna(full['Unemployment'].mean())



MarkDown_features=['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']

for b in range(len(MarkDown_features)):

    full[MarkDown_features[b]] = full[MarkDown_features[b]].fillna(0)


# Looking at the bigger picture

# Draw a multivariable graph

x_vars = full.columns.drop(['Weekly_Sales'])

sns.pairplot(full ,x_vars=x_vars ,y_vars=['Weekly_Sales'] , plot_kws={'alpha': 0.1})
# Draw correlation heat map to find correlation

plt.subplots(figsize=(18,9))

corrDf = full.corr()

sns.heatmap(corrDf,annot=True)



corrDf['Weekly_Sales'].sort_values(ascending=False)
# Sequential analysis

# Highly correlated features：MarkDown1 - MarkDown4

contFeaturelist = []

contFeaturelist.append('MarkDown4')

contFeaturelist.append('MarkDown1')

contFeaturelist.append('Weekly_Sales')



correlationMatrix = full[contFeaturelist].corr().abs()

plt.subplots()

sns.heatmap(correlationMatrix, annot=True)



#Mask unimportant features

sns.heatmap(correlationMatrix, mask=correlationMatrix < 1, cbar = False)

plt.show()
#Check for suspicious features

fig, axes = plt.subplots(2, 2)

fig.set_size_inches(12,10)



sns.distplot(full['Temperature'],ax=axes[0,0])

sns.distplot(full['Fuel_Price'],ax=axes[0,1])

sns.distplot(full['CPI'],ax=axes[1,0])

sns.distplot(full['Unemployment'],ax=axes[1,1])



axes[0,0].set(xlabel='Temperature',title='Distribution of temp',)

axes[0,1].set(xlabel='Fuel_Price',title='Distribution of atemp')

axes[1,0].set(xlabel='CPI',title='Distribution of humidity')

axes[1,1].set(xlabel='Unemployment',title='Distribution of windspeed')
# Remove irrelevant features. The kernel:

full= full.drop(['MarkDown4','Unemployment','CPI'],axis=1)
# Use train_test_split

from sklearn.model_selection import train_test_split



# Simple event handling

fullDf = full.drop(columns='Weekly_Sales')



sourceRow = len(train)

source_X = fullDf.loc[0:sourceRow-1,:]

source_y = full.loc[0:sourceRow-1,'Weekly_Sales']

pred_X = fullDf.loc[sourceRow:,:]



train_X,test_X,train_y,test_y = train_test_split(source_X,

                                                 source_y,

                                                 test_size=.3,

                                                 random_state=0)



# Use Date as index (the required model cannot use date-type numerical calculation, so this step is required)

train_X = train_X.set_index('Date')

test_X = test_X.set_index('Date')

pred_X = pred_X.set_index('Date')
# Use RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(

               n_estimators=360,

               min_samples_leaf=1,

               oob_score=True,

               n_jobs=-1)

rfr.fit(train_X,train_y)

pred_y=rfr.predict(pred_X)
# Export data set

predDf = pd.DataFrame( 

    { 'Id': test['Store'].astype(str)+'_'+test['Dept'].astype(str)+'_'+test['Date'].astype(str),

    'Weekly_Sales':pred_y

    })



predDf.to_csv('pred.csv',index=False)
# Use GridSearchCV

from sklearn.model_selection import GridSearchCV



rfr = RandomForestRegressor(

               n_estimators=10, 

               min_samples_leaf=1,

               n_jobs=-1)



parameters = {  

    'n_estimators':[20,50,100,200,350,360],

    'min_samples_leaf':[1,5,10,30,50,100]

}



gs = GridSearchCV(rfr, param_grid =parameters, cv=3 )

gs.fit(train_X,train_y.values)



gs.best_params_, gs.best_score_