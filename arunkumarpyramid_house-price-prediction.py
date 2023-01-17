# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the dataset and put it in pandas dataframe



import pandas as pd

training_set=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_set=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Display the first 5 row in training_set data using .head() method

# .head() method used to print upto how many rows need to print from dataframe

training_set.head(5)
# Disply the first 5 row in test_set using .head() method

test_set.head(5)
# Save the ID column in another varaible and drop it from both training and test set



# Save ID column

train_id=training_set['Id']

test_id=test_set['Id']



# Drop Id column from train and test set

# axis=1 droping column cell

training_set.drop('Id',axis=1,inplace = True)

test_set.drop('Id',axis=1,inplace = True)



# Display the shape of training and test set

print("Training set shape : ",training_set.shape)

print("Test set shape : ",test_set.shape)
# Describe the Statistical summary of Saleprice column

training_set['SalePrice'].describe()
# Well with min and max value it is clear that our distribution falls in b/n 34900 to 755000 price.



# Now we are going to visualise the sale price column with distribution 

# Histogram descriptive visualisation for SalePrice



import seaborn as sb

sb.distplot(training_set['SalePrice'])
# As per above distribution it is clear that our SalePrice Column is Normal or gaussian Distribution..
# To Know what kind of skewness of our column distribution is..

print("Skewness :%f"%training_set['SalePrice'].skew())
# We Need to find the correlation between input and output attributes.



# Scatter plot for GrLivArea/SalePrice grlivarea----living area for house.



plotdata=pd.concat([training_set['GrLivArea'],training_set['SalePrice']],axis=1)

plotdata.plot.scatter(x='GrLivArea',y='SalePrice')
# Well with above visualisation both grlivarea and saleprice both are having good relationship like best frnds



# scatter plot for totalbsmtsf/saleprice

plotdata=pd.concat([training_set['TotalBsmtSF'],training_set['SalePrice']],axis=1)

plotdata.plot.scatter(x='TotalBsmtSF',y='SalePrice')
# Corr() method use to find all correlation between all varaibles in our training dataset



correlation_matrix=training_set.corr()

sb.heatmap(correlation_matrix,square=True)
# SalePrice Correlation matrix

import numpy as np

import matplotlib.pyplot as plt

k=10 #No.of varaibles from heatmap with best correlation with other variables

# Top Correlation taking from above heatmap and index is weual to saleprice index.

cols=correlation_matrix.nlargest(k,'SalePrice')['SalePrice'].index



# Storing the Correlation coefficient value for all k=15 attribute

corr_coeff=np.corrcoef(training_set[cols].values.T)

sb.set(font_scale=1.25)

zoomin_heatmap=sb.heatmap(corr_coeff,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 10},yticklabels=cols.values,xticklabels=cols.values)

plt.show()
# Checking outlier there in GrLivArea column with the SalePrice

plt.scatter(training_set['GrLivArea'],training_set['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.show()
# Removing the outliers in GrLiveArea column.



# Deleting outliers 

training_set=training_set.drop(training_set[(training_set['GrLivArea']>4000) & (training_set['GrLivArea']<300000)].index)



#Check visualisation Again that outlier values removed or not

plt.scatter(training_set['GrLivArea'],training_set['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.show()

from scipy.stats import norm, skew #for some statistics



# fit=norm is normal distribution in saleprice

sb.distplot(training_set['SalePrice'],fit=norm)



# get the fitted  parameters used by the function

(mean,std)=norm.fit(training_set['SalePrice'])

print("mean :%f  Std: %f "%(mean,std))



# Now plot the distribution

plt.legend(['Norm Dist mean: %f and Std: %f'%(mean,std)])

plt.ylabel('Frequency')

plt.xlabel('SalePrice')



fig=plt.figure()



from scipy import stats

#probplot is the probability distribution.

result=stats.probplot(training_set['SalePrice'],plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column



import numpy as np

training_set['SalePrice']=np.log1p(training_set['SalePrice'])



# Checking the Distribution

sb.distplot(training_set['SalePrice'],fit=norm)



# Mean and std for salePrice column 

(mean,std)=norm.fit(training_set['SalePrice'])



print(" SalePrice Mean: %f and Std: %f"%(mean,std))



# Now plot the distribution legend

plt.legend("Norm Dist  mean:%.3f and std:%.3f"%(mean,std))

plt.xlabel('SalePrice')

plt.ylabel('Frequency')

plt.title('Sales Distribution')



# probability distribution

fig=plt.figure()

stats.probplot(training_set['SalePrice'],plot=plt)

plt.show()
y_train=training_set['SalePrice'].values

print(y_train[:5,])



n_train=training_set.shape[0]

n_test=test_set.shape[0]

# Reassign the training_set to x_train and test_set to x_test

x_train_df=training_set

x_test_df=test_set



print("Before droping the output attribute column and row shape")

print("x_train shape:",x_train_df.shape)

print("x_test shape",x_test_df.shape)



var=['SalePrice']

x_train_df=x_train_df.drop(var,axis=1)



print("After droping the output attribute column and row shape")

print("x_train shape:",x_train_df.shape)

print("x_test shape",x_test_df.shape)
# Counting the missing values in each column by using the isna() method for both x_train and x_test

print("---------------------x_train--------------------------")

print(x_train_df.isna().sum())

print("--------------------x_test----------------------------")

print(x_test_df.isna().sum())
# We are concat the x_train and x_test to replace all null values at once



all_data=pd.concat((x_train_df,x_test_df)).reset_index(drop=True)

# find the missing values instead of number getting in ratio for each column

all_data_null=(all_data.isna().sum()/len(all_data))*100



# Now we are droping all values ratio is equal zero in all_data_null variable

all_data_null=all_data_null.drop(all_data_null[all_data_null==0].index).sort_values(ascending=False)



missing_data=pd.DataFrame({'Missing Ratio ': all_data_null})

missing_data.head(20)
# Now plotting the missing  ratio in barplot

sb.barplot(x=all_data_null.index,y=all_data_null)

plt.xlabel("features")

plt.xticks(rotation='90')

plt.ylabel('percentage of missing values')

plt.title("percentage of missing data in features")
# PoolQC : data description says NA means "No Pool". 

# That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.

all_data['PoolQC']=all_data['PoolQC'].fillna('None')

print(all_data['PoolQC'].head(10))
# MiscFeature : data description says NA means "no misc feature"

all_data['MiscFeature']=all_data['MiscFeature'].fillna('None')
# Alley : data description says NA means "no alley access"

all_data['Alley']=all_data['Alley'].fillna('None')
# Fence : data description says NA means "no fence"

all_data['Fence']=all_data['Fence'].fillna('None')
# FireplaceQu : data description says NA means "no fireplace"

all_data['FireplaceQu']=all_data['FireplaceQu'].fillna('None')
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
# GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')
# GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : 

# missing values are likely zero for having no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : 

#  For all these categorical basement-related features, NaN means that there is no basement.

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
# MasVnrArea and MasVnrType :

# NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.



all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
# MSZoning (The general zoning classification) : 

# 'RL' is by far the most common value. So we can fill in missing values with 'RL'

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)

# Functional : data description says NA means typical

all_data["Functional"] = all_data["Functional"].fillna("Typ")

# Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

# Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# SaleType : Fill in again with most frequent which is "WD"

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# MSSubClass : Na most likely means No building class. We can replace missing values with None

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

# Is there any remaining missing value ?



# find the missing values instead of number getting in ratio for each column

all_data_null=(all_data.isna().sum()/len(all_data))*100



# Now we are droping all values ratio is equal zero in all_data_null variable

all_data_null=all_data_null.drop(all_data_null[all_data_null==0].index).sort_values(ascending=False)



missing_data=pd.DataFrame({'Missing Ratio ': all_data_null})

missing_data.head(20)
# Transforming some numerical variables that are really categorical

#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
# Label Encoding some categorical variables that may contain information in their ordering set

from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))
# Getting dummy categorical features

all_data = pd.get_dummies(all_data)

print(all_data.shape)

# Getting the new train and test sets.

x_train=all_data[:n_train].values

x_test=all_data[n_train:].values

print(x_train[:5,:])

print(x_test[:5,:])
# Spot Checking and Comparing Algorithms Without standard Scaler

models=[]

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

models.append(('linear_reg',LinearRegression()))

models.append(('knn',KNeighborsRegressor()))

models.append(('SVR',SVR()))

models.append(("decision_tree",DecisionTreeRegressor()))



# Evaluating Each model

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

names=[]

predictions=[]

error='neg_mean_squared_error'

for name,model in models:

    fold=KFold(n_splits=10,random_state=0)

    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)

    predictions.append(result)

    names.append(name)

    msg="%s : %f (%f)"%(name,result.mean(),result.std())

    print(msg)

    



# Visualizing the Model accuracy

fig=plt.figure()

fig.suptitle("Comparing Algorithms")

axis=fig.add_subplot(111)

plt.boxplot(predictions)

axis.set_xticklabels(names)

plt.xticks(rotation='90')

plt.show()
# Create Pipeline with Standardization Scale and models

# Standardize the dataset

from sklearn.pipeline import Pipeline

from sklearn. preprocessing import StandardScaler,MinMaxScaler

pipelines=[]

pipelines.append(('scaler_lg',Pipeline([('scaler',MinMaxScaler()),('lg',LinearRegression())])))

pipelines.append(('scale_KNN',Pipeline([('scaler',StandardScaler()),('KNN',KNeighborsRegressor())])))

pipelines.append(('scale_SVR',Pipeline([('scaler',StandardScaler()),('SVR',SVR())])))

pipelines.append(('scale_decision',Pipeline([('scaler',StandardScaler()),('decision',DecisionTreeRegressor())])))



# Evaluate Pipelines

predictions=[]

names=[]

for name, model in pipelines:

    fold=KFold(n_splits=10,random_state=5)

    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)

    predictions.append(result)

    names.append(name)

    msg='%s : %f (%f)'%(name,result.mean(),result.std())

    print(msg)

    

#Visualize the compared algorithms

fig=plt.figure()

fig.suptitle("Algorithms Comparisions")

plt.boxplot(predictions)

plt.show()
# SVR Tuning

import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

kernel=['linear','poly','rbf','sigmoid']

c=[0.2,0.4,0.6,0.8,1.0]

param_grid=dict(C=c,kernel=kernel)

model=SVR()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# DecisionTreeRegressor Tuning



import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

criterion=['mse','friedman_mse','mae']

param_grid=dict(criterion=criterion)

model=DecisionTreeRegressor()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Ensemble and Boosting algorithm to improve performance



#Ensemble

# Boosting methods

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

# Bagging methods

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor

ensembles=[]

ensembles.append(('scaledAB',Pipeline([('scale',StandardScaler()),('AB',AdaBoostRegressor())])))

ensembles.append(('scaledGBR',Pipeline([('scale',StandardScaler()),('GBR',GradientBoostingRegressor())])))

ensembles.append(('scaledRF',Pipeline([('scale',StandardScaler()),('rf',RandomForestRegressor(n_estimators=10))])))

ensembles.append(('scaledETR',Pipeline([('scale',StandardScaler()),('ETR',ExtraTreesRegressor(n_estimators=10))])))

ensembles.append(('scaledRFR',Pipeline([('scale',StandardScaler()),('RFR',RandomForestRegressor(n_estimators=10))])))

# Evaluate each Ensemble Techinique

results=[]

names=[]

for name,model in ensembles:

    fold=KFold(n_splits=10,random_state=5)

    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)

    results.append(result)

    names.append(name)

    msg="%s : %f (%f)"%(name,result.mean(),result.std())

    print(msg)

    

# Visualizing the compared Ensemble Algorithms

fig=plt.figure()

fig.suptitle('Ensemble Compared Algorithms')

plt.boxplot(results)

plt.show()
# ExtraTreesRegressor Tuning



import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

n_estimators=[5,10,15,20,25,30,40,50,100,200]

param_grid=dict(n_estimators=n_estimators)

model=ExtraTreesRegressor()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# GradientBoostingRegressor Tuning



import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

learning_rate=[0.1,0.2,0.3,0.4,0.5]

n_estimators=[5,10,15,20,25,30,40,50,100,200]

param_grid=dict(n_estimators=n_estimators,learning_rate=learning_rate)

model=GradientBoostingRegressor()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Finalize Model

# we will finalize the gradient boosting regression algorithm and evaluate the model for house price predictions.



from sklearn.metrics import mean_squared_error

scaler=StandardScaler().fit(x_train)

scaler_x=scaler.transform(x_train)

model=GradientBoostingRegressor(random_state=5,n_estimators=200,learning_rate=0.2)

model.fit(scaler_x,y_train)



#Transform the validation test set data

scaledx_test=scaler.transform(x_test)

y_pred=model.predict(scaledx_test)
# output

sub = pd.DataFrame()

sub['Id'] = test_id

sub['SalePrice'] = y_pred

sub.to_csv('submission.csv',index=False)

print("Successfully updated")