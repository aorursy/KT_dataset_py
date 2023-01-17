#Perfect Libraries

import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_path = '/kaggle/input/home-data-for-ml-course/train.csv'

test_path = '/kaggle/input/home-data-for-ml-course/test.csv'

sub_path = '/kaggle/input/home-data-for-ml-course/sample_submission.csv'
#Data extraction to Pandas (Panel Data Structure)

train = pd.read_csv(train_path)

test = pd.read_csv(test_path)

sub = pd.read_csv(sub_path)
#Observing Dataframes

display(train, test)
#Correlation matrix of top 6 features with sale price in a descending manner

train.corr()['SalePrice'].sort_values(ascending = False)[:6]
#Those nice visalizations using seaborn and matplotlib

fig=plt.figure(figsize=(15,7))

sns.set_style('whitegrid')

sns.set_color_codes = 'g'

ax1=fig.add_subplot(221)

ax2=fig.add_subplot(222)

ax3=fig.add_subplot(223)

ax4=fig.add_subplot(224)



sns.scatterplot(train['OverallQual'], train['SalePrice'], ax=ax1)

sns.scatterplot(train['GrLivArea'], train['SalePrice'], hue=train['ExterQual'],ax=ax2)

sns.scatterplot(train['GarageCars'], train['SalePrice'], ax=ax3)

sns.scatterplot(train['GarageArea'], train['SalePrice'], ax=ax4)



plt.show()
#Taking out the outliers SalePrice > 300000 or GrLivArea < 2000

train = train.loc[(train['SalePrice']>300000) | (train['GrLivArea']<2000)]
#Train data corelation with SalePrice in a descending manner

train.corr()['SalePrice'].sort_values(ascending = False)
#Joining train and test data to have more data for training

alldata = pd.concat([train,test],sort = False)
#Dropping Id and SalePrice columns (dropped Id to avoid data leakage)

alldata = alldata.drop(['Id','SalePrice'],axis = 1)
#Viewing entire data after combining it

alldata
#Adding dummy (categorical features) and 

#filling out NA values with -1 so it doesn't affect the performance of model

alldata_2 = pd.get_dummies(alldata)

alldata_2 = alldata_2.fillna(-1)
#Standard Scaler dor transforming data

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

alldata_3=ss.fit_transform(alldata_2)
#Creating new train test datasets

train_2=alldata_3[:len(train)]

test_2=alldata_3[len(train):]
#This is quiet intuitive (trial run)

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.model_selection import cross_val_score



ridgecv=RidgeCV(alphas=[1])

np.sqrt(-cross_val_score(ridgecv , train_2, train['SalePrice'], n_jobs=-1, cv=10, scoring='neg_mean_squared_error').mean())
#With multiple alphas and rmse obtained from it

rmse=[]

alphas=[1,5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 700, 750, 900, 1000] 

for i in alphas :

    ridgecv=RidgeCV(alphas=[i])

    result=np.sqrt(-cross_val_score(ridgecv , train_2, train['SalePrice'], n_jobs=-1, cv=10, scoring='neg_mean_squared_error').mean())

    rmse.append(result)

    print(result)
#Check the optimal value for alpha visually

plt.figure(figsize=(8,3))

sns.lineplot(x=alphas, y=rmse)

plt.suptitle('RMSE by alpha of ridge regression')

plt.xlabel('alpha')

plt.ylabel('RMSE')

plt.show()
#Checking Coe Vs Features effect using two different Alphas as 1 and 100

ridgecv=RidgeCV(alphas=[1])

ridgecv.fit(train_2,np.log(train['SalePrice']))



ridgecv_2=RidgeCV(alphas=[100])

ridgecv_2.fit(train_2,np.log(train['SalePrice']))





fig=plt.figure(figsize=(25,70))



ax1=fig.add_subplot(121)

ax2=fig.add_subplot(122)



#x=alldata_2.columns.tolist(),

chart_1=sns.barplot(y=alldata_2.columns.tolist(), x=ridgecv.coef_.tolist(), ax=ax1)

chart_2=sns.barplot(y=alldata_2.columns.tolist(), x=ridgecv_2.coef_.tolist(), ax=ax2)



ax1.set_xlim(-0.02, 0.05)

ax2.set_xlim(-0.02, 0.05)



ax1.set_title('coefficient with alpha 1')

ax2.set_title('coefficient with alpha 100')





plt.show()
#Distrbution of train data with np.log and unprocessed data

fig=plt.figure(figsize=(15,4))



ax1=fig.add_subplot(121)

ax2=fig.add_subplot(122)



sns.distplot(train['SalePrice'],ax=ax1)

sns.distplot(np.log(train['SalePrice']),ax=ax2)



bdict = {'facecolor' : 'r', 'alpha' : 0.5, 'boxstyle' : 'rarrow', 'linewidth' : 2}





fig.suptitle('Target distribution from unlogged to logged', bbox=bdict, color='black')

ax1.set_title('Un-logged')

ax2.set_title('Logged')



plt.show()
#Using alphas from 1 to 1000 using np.linspace(1,1000,1000) to get the best model

ridgecv=RidgeCV(alphas=np.linspace(1,1000,1000))

ridgecv.fit(train_2,np.log(train['SalePrice']))

result=ridgecv.predict(test_2)

sub['SalePrice']=np.exp(result)

sub.to_csv('result.csv',index=False)