# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_train.columns
df_train.info()
df_train.describe()
df_train.isnull().sum().sort_values(ascending =False).head(20)
df_train.isnull().sum().sort_values(ascending =False).head(20)/len(df_train)
df_train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=1,inplace=True)

df_test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage',],axis=1,inplace=True)
import random

missing_values = df_train.columns[df_train.isna().any()].to_list()

for each in missing_values:

    if (df_train[each].dtypes =='float64'):

        minimum= int(df_train[each].quantile(0.25))

        maximum= int(df_train[each].quantile(0.75))

        A=df_train[df_train[each].isnull()].index.tolist()

        for i in A:

            df_train.loc[i,each]=random.randint(minimum,maximum)

        df_train[each]=pd.to_numeric(df_train[each])

   



    elif(df_train[each].dtypes == 'object'):

        if ('True' in str(df_train[each].str.contains('No').unique().tolist())):

            df_train[each].fillna('No',inplace=True)

        elif('True' in str(df_train[each].str.contains('None').unique().tolist())):

            df_train[each].fillna('None',inplace=True)

        elif('True' in str(df_train[each].str.contains('Unf').unique().tolist())):

            df_train[each].fillna('Unf',inplace=True)

        else:

            A=df_train[df_train[each].isnull()].index.tolist()

            unique = df_train[each].unique().tolist()

            unique=pd.Series(unique).dropna().tolist()

            for i in A:

                df_train.loc[i,each]=random.choice(unique)





missing_values = df_test.columns[df_test.isna().any()].to_list()

for each in missing_values:

    if (df_test[each].dtypes =='float64'):

        minimum= int(df_test[each].quantile(0.25))

        maximum= int(df_test[each].quantile(0.75))

        A=df_test[df_test[each].isnull()].index.tolist()

        for i in A:

            df_test.loc[i,each]=random.randint(minimum,maximum)

        df_test[each]=pd.to_numeric(df_test[each])

   



    elif(df_test[each].dtypes == 'object'):

        if ('True' in str(df_test[each].str.contains('No').unique().tolist())):

            df_test[each].fillna('No',inplace=True)

        elif('True' in str(df_test[each].str.contains('None').unique().tolist())):

            df_test[each].fillna('None',inplace=True)

        elif('True' in str(df_test[each].str.contains('Unf').unique().tolist())):

            df_test[each].fillna('Unf',inplace=True)

        else:

            A=df_test[df_test[each].isnull()].index.tolist()

            unique = df_test[each].unique().tolist()

            unique=pd.Series(unique).dropna().tolist()

            for i in A:

                df_test.loc[i,each]=random.choice(unique)

df_train.drop(['Id'],axis=1,inplace=True)

df_test.drop(['Id'],axis=1,inplace=True)
plt.figure(figsize=(25,25))

sns.heatmap(df_train.corr(),annot=True,cmap='coolwarm')
plt.figure(figsize=(10,5))

df_train.corr()['SalePrice'].sort_values().drop('SalePrice').plot(kind='bar')
fig = plt.figure(figsize=(15,10));   

ax1 = fig.add_subplot(3,4,1);  

ax2 = fig.add_subplot(3,4,2);

ax3 = fig.add_subplot(3,4,3);  

ax4 = fig.add_subplot(3,4,4);

ax5 = fig.add_subplot(3,4,5);  

ax6 = fig.add_subplot(3,4,6);

ax7 = fig.add_subplot(3,4,7);  

ax8 = fig.add_subplot(3,4,8);

ax9 = fig.add_subplot(3,4,9);  

ax10 = fig.add_subplot(3,4,10);

ax11 = fig.add_subplot(3,4,11);  

ax12 = fig.add_subplot(3,4,12);



sns.boxplot("OverallQual", "SalePrice", data=df_train,ax=ax1)

sns.scatterplot("GrLivArea", "SalePrice", data=df_train, ax=ax2)

sns.boxplot("GarageCars", "SalePrice", data=df_train,ax=ax3)

sns.scatterplot("GarageArea", "SalePrice", data=df_train, ax=ax4)

sns.scatterplot("TotalBsmtSF", "SalePrice", data=df_train,ax=ax5)

sns.scatterplot("1stFlrSF", "SalePrice", data=df_train, ax=ax6)

sns.boxplot("FullBath", "SalePrice", data=df_train,ax=ax7)

sns.boxplot("TotRmsAbvGrd", "SalePrice", data=df_train, ax=ax8)

sns.scatterplot("YearBuilt", "SalePrice", data=df_train,ax=ax9)

sns.scatterplot("YearRemodAdd", "SalePrice", data=df_train, ax=ax10)

sns.boxplot("MasVnrType", "SalePrice", data=df_train,ax=ax11)

sns.boxplot("Fireplaces", "SalePrice", data=df_train, ax=ax12)

plt.tight_layout()
fig = plt.figure(figsize=(15,10));   

ax1 = fig.add_subplot(2,1,1);  

ax2 = fig.add_subplot(2,1,2);

sns.distplot(df_train['YearBuilt'],bins=50,color='black',ax=ax1)

sns.distplot(df_train['YearRemodAdd'],bins=50,color='black',ax=ax2)
plt.figure(figsize=(12,8))

sns.violinplot(x='TotRmsAbvGrd',y='GrLivArea',data=df_train)
plt.figure(figsize=(12,8))

sns.boxplot(x='FullBath',y='2ndFlrSF',data=df_train,hue='HalfBath',palette="BuGn_r")
plt.figure(figsize=(10,5))

df_train.corr()['OverallQual'].sort_values().drop(['OverallQual','SalePrice']).plot(kind='bar')
catogorical_features_ = np.array([df_train.columns[df_train.dtypes == 'object'].to_list()])

catogorical_features_
df_train['Utilities'].value_counts()
# All the records in utilities are mostly AllPub 

df_train.drop('Utilities',axis=1,inplace=True)

df_test.drop('Utilities',axis=1,inplace=True)
fig = plt.figure(figsize=(20,15));   

ax1 = fig.add_subplot(4,4,1);  

ax2 = fig.add_subplot(4,4,2);

ax3 = fig.add_subplot(4,4,3);  

ax4 = fig.add_subplot(4,4,4);

ax5 = fig.add_subplot(4,4,5);  

ax6 = fig.add_subplot(4,4,6);

ax7 = fig.add_subplot(4,4,7);  

ax8 = fig.add_subplot(4,4,8);

ax9 = fig.add_subplot(4,4,9);  

ax10 = fig.add_subplot(4,4,10);

ax11 = fig.add_subplot(4,4,11);  

ax12 = fig.add_subplot(4,4,12);

ax13 = fig.add_subplot(4,4,13);  

ax14 = fig.add_subplot(4,4,14);

ax15 = fig.add_subplot(4,4,15);  

ax16 = fig.add_subplot(4,4,16);



sns.boxplot(x="LotShape",y= "SalePrice", data=df_train,ax=ax1)

sns.boxplot("SaleCondition", "SalePrice", data=df_train, ax=ax2)

sns.boxplot("LandSlope", "SalePrice", data=df_train,ax=ax3)

sns.boxplot("Condition1", "SalePrice", data=df_train, ax=ax4)

sns.boxplot("BldgType", "SalePrice", data=df_train,ax=ax5)

sns.boxplot("HouseStyle", "SalePrice", data=df_train, ax=ax6)

sns.boxplot("RoofStyle", "SalePrice", data=df_train,ax=ax7)

sns.boxplot("Exterior1st", "SalePrice", data=df_train, ax=ax8)

sns.boxplot("Exterior2nd", "SalePrice", data=df_train,ax=ax9)

sns.boxplot("ExterQual", "SalePrice", data=df_train, ax=ax10)

sns.boxplot("ExterCond", "SalePrice", data=df_train,ax=ax11)

sns.boxplot("Foundation", "SalePrice", data=df_train, ax=ax12)

sns.boxplot("HeatingQC", "SalePrice", data=df_train,ax=ax13)

sns.boxplot("CentralAir", "SalePrice", data=df_train, ax=ax14)

sns.boxplot("KitchenQual", "SalePrice", data=df_train,ax=ax15)

sns.boxplot("SaleType", "SalePrice", data=df_train, ax=ax16)

plt.tight_layout()
df_train['Foundation'].value_counts()
fig = plt.figure(figsize=(20,10));   

ax1 = fig.add_subplot(1,2,1);  

ax2 = fig.add_subplot(1,2,2);

sns.boxplot("Exterior1st", "SalePrice", data=df_train, ax=ax1)

sns.boxplot("Exterior2nd", "SalePrice", data=df_train,ax=ax2)

plt.tight_layout()
fig = plt.figure(figsize=(15,10));   

ax1 = fig.add_subplot(2,1,1);  

ax2 = fig.add_subplot(2,1,2);

sns.distplot(df_train['1stFlrSF'],bins=30,color='black',ax=ax1)

sns.distplot(df_train['2ndFlrSF'],bins=10,color='black',ax=ax2)
catogorical_features_ = np.delete(catogorical_features_,np.where(catogorical_features_=='Utilities'))
test_match=[]

for i,feature in enumerate(catogorical_features_): 

    test_match.append( (feature,(df_train[feature].nunique()  -  df_test[feature].nunique())))

    if (df_train[feature].nunique()  -  df_test[feature].nunique()) != 0:

        df_train.drop(feature,axis=1,inplace=True)

        df_test.drop(feature,axis=1,inplace=True)
print(test_match)
catogorical_features_ = np.array([df_train.columns[df_train.dtypes == 'object'].to_list()])

dummies = []

concat_dummies=[]

for i,feature in enumerate(catogorical_features_[0]):

    dummies.append(pd.get_dummies(df_train[feature],drop_first=True))

    df_train = pd.concat([df_train,dummies[i]],axis=1) 
df_train.drop(['MSZoning', 'Street', 'LotShape', 'LandContour','LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',

               'BldgType', 'RoofStyle','MasVnrType', 'ExterQual','Foundation',  'HeatingQC', 'CentralAir',

         'KitchenQual', 'Functional', 'GarageFinish','PavedDrive', 'SaleType', 'SaleCondition','ExterCond',

               'GarageCond',

               'GarageType','GarageYrBlt','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual'],

              axis=1,inplace=True)
catogorical_features_ = np.array([df_test.columns[df_test.dtypes == 'object'].to_list()])

dummies = []

concat_dummies=[]

for i,feature in enumerate(catogorical_features_[0]):

    dummies.append(pd.get_dummies(df_test[feature],drop_first=True))

    df_test = pd.concat([df_test,dummies[i]],axis=1) 
df_test.drop(['MSZoning', 'Street', 'LotShape', 'LandContour','LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',

               'BldgType', 'RoofStyle','MasVnrType', 'ExterQual','Foundation',  'HeatingQC', 'CentralAir',

         'KitchenQual', 'Functional', 'GarageFinish','PavedDrive', 'SaleType', 'SaleCondition','ExterCond',

             'GarageCond','GarageType','GarageYrBlt','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual'],axis=1,inplace=True)
X_train = np.array(df_train.drop('SalePrice',axis=1))

y_train = np.array(df_train['SalePrice'])

X_test = np.array(df_test)
print('Shape of X_train {} \nShape of y_test {}\nShape of X_test {}'.format(X_train.shape,y_train.shape,X_test.shape))
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

X_train = mms.fit_transform(X_train)

X_test = mms.transform(X_test)
y_train = mms.fit_transform(y_train.reshape(-1,1))
from keras.models import Sequential

from keras.layers import Dense,Dropout
regressor = Sequential()

regressor.add(Dense(units=512,activation='relu',kernel_initializer='uniform'))

regressor.add(Dense(units=256,activation='relu',kernel_initializer='uniform'))

regressor.add(Dense(units=256,activation='relu',kernel_initializer='uniform'))

regressor.add(Dense(units=128,activation='relu',kernel_initializer='uniform'))

regressor.add(Dense(units=1,activation='relu',kernel_initializer='uniform'))
regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
regressor.fit(X_train,y_train,epochs=100,batch_size=50)
losses = regressor.history.history

losses = np.array(pd.DataFrame(losses))

plt.plot(losses)

plt.xlabel('Epochs')

plt.ylabel('loss')
'''

def build_classifier(optimizer,units1,units2,units3,units4):

    regressor = Sequential()

    regressor.add(Dense(units=units1,activation='relu',kernel_initializer='uniform'))

    regressor.add(Dense(units=units2,activation='relu',kernel_initializer='uniform'))

    regressor.add(Dense(units=units3,activation='relu',kernel_initializer='uniform'))

    regressor.add(Dense(units=units4,activation='relu',kernel_initializer='uniform'))

    regressor.add(Dense(units=1,activation='relu',kernel_initializer='uniform'))

    regressor.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mean_squared_error'])

    return regressor



from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV



regressor = KerasRegressor(build_fn=build_classifier)

parameters = {'batch_size':[10,15,25,32],

             'epochs':[100,300,500],

             'optimizer':['adam','rmsprop'],

             'units1':[512,256],

             'units2':[256,128],

             'units3':[256,128],

             'units4':[256,128,64]}



grid_search = GridSearchCV(estimator = regressor,

                           param_grid = parameters,

                           scoring = 'neg_mean_squared_error',

                           cv = 3)

grid_search = grid_search.fit(X_train, y_train)

'''

'''print(grid_search.best_score_)

print('\n')

print(grid_search.best_params_)'''
Best = {'batch_size': 15, 'epochs': 500, 'optimizer': 'adam', 'units1': 512, 'units2': 128, 'units3': 128, 'units4': 64}
#from keras.models import load_model

#regressor.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

#del model  # deletes the existing model

# returns a compiled model

# identical to the previous one

#regressor1 = load_model('my_model.h5')
regressor = Sequential()

regressor.add(Dense(units=512,activation='relu',kernel_initializer='uniform'))

regressor.add(Dense(units=256,activation='relu',kernel_initializer='uniform'))

regressor.add(Dense(units=256,activation='relu',kernel_initializer='uniform'))

regressor.add(Dense(units=128,activation='relu',kernel_initializer='uniform'))

regressor.add(Dense(units=1,activation='relu',kernel_initializer='uniform'))

regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
regressor.fit(X_train,y_train,epochs=500,batch_size=15)
losses = regressor.history.history

losses = np.array(pd.DataFrame(losses))

plt.plot(losses)

plt.xlabel('Epochs')

plt.ylabel('loss')
regressor.summary()
y_pred = regressor.predict(X_test) 

y_pred_original = mms.inverse_transform(y_pred.reshape(-1,1))

y_pred_original = y_pred_original.tolist()

y_pred_original = [pred for i in y_pred_original for pred in i]
test_set =pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

submission = pd.DataFrame({'Id': test_set['Id'],'SalePrice': y_pred_original})
#submission.to_csv('submission.csv', index=False)
from sklearn.ensemble import RandomForestRegressor

regressor1 = RandomForestRegressor(n_estimators= 100)

regressor1.fit(X_train,y_train)
y_pred1 = regressor1.predict(X_test) 

y_pred_original1 = mms.inverse_transform(y_pred1.reshape(-1,1))

y_pred_original1 = y_pred_original1.tolist()

y_pred_original1 = [pred for i in y_pred_original1 for pred in i]
y_pred_final=[]

for i in range(0,1459):

    y_pred_final.append((y_pred_original[i]*0.5)+(y_pred_original1[i]*0.5))
submission = pd.DataFrame({'Id': test_set['Id'],'SalePrice': y_pred_final})
submission.to_csv('submission.csv', index=False)