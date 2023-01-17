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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_rows', 100)
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.shape
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
print(df.isnull().sum())
# Dropping features with more than 50 % missing values.

df.drop(['Id','Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

# Filling missing values with mean of column,

df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())

df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())

df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])

df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])

df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])

df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])

df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])

df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])

df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])

df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])

df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])

df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])

df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])

df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])

df.dropna(inplace=True)
df.shape
# Handling categorical features
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',

         'Condition2','BldgType','Condition1','HouseStyle','SaleType',

        'SaleCondition','ExterCond',

         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',

         'CentralAir',

         'Electrical','KitchenQual','Functional',

         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
len(columns)
test_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



# Dropping features with more than 50 % missing values.

test_df.drop(['Id','Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

# Filling missing values with mean of column,

test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())

test_df['GarageYrBlt']=test_df['GarageYrBlt'].fillna(test_df['GarageYrBlt'].mean())



listWithNULL = ["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath","KitchenQual","Functional",

                "GarageCars","GarageArea","SaleType","BsmtCond","BsmtQual","FireplaceQu","GarageType","GarageFinish","GarageQual"

               ,"GarageCond","MasVnrType","MasVnrArea","BsmtExposure","BsmtFinType2","BsmtFinType1","Utilities","MSZoning",

                "Exterior1st","Exterior2nd"]

for col in listWithNULL:

    test_df[col]=test_df[col].fillna(test_df[col].mode()[0])

    

test_df.to_csv('/kaggle/working/formulatedtest.csv',index=False)
def category_onehot_multcols(multcolumns):

    df_final=final_df

    i=0

    for fields in multcolumns:

        

        print(fields)

        df1=pd.get_dummies(final_df[fields],drop_first=True)

        

        final_df.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:

            

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([final_df,df_final],axis=1)

        

    return df_final
main_df=df.copy()
df.shape
test_df.shape
final_df=pd.concat([df,test_df],axis=0)
final_df.shape
final_df=category_onehot_multcols(columns)
final_df.shape
final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
final_df
df_Train=final_df.iloc[:1459,:]

df_Test=final_df.iloc[1459:,:]
df_Train.dropna(inplace=True)
df_Test.drop(['SalePrice'],axis=1,inplace=True)
df_Train.shape
X_train=df_Train.drop(['SalePrice'],axis=1)

y_train=df_Train['SalePrice']
import xgboost

from sklearn.model_selection import RandomizedSearchCV

regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]
## Hyper Parameter Optimization





n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
# Set up the random search with 4-fold cross validation

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.1, max_delta_step=0, max_depth=2,

             min_child_weight=1, missing=None, monotone_constraints=None,

             n_estimators=900, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None)
regressor.fit(X_train,y_train)
import pickle

filename = 'finalized_model.pkl'

pickle.dump(regressor, open(filename, 'wb'))
df_Test.shape
pred=pd.DataFrame(regressor.predict(df_Test))
sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],pred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('/kaggle/working/sub.csv',index=False)
X_train.shape
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LeakyReLU,PReLU,ELU

from keras.layers import Dropout



from keras import backend as K

def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    

    

# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu',input_dim = 176))



# Adding the second hidden layer

classifier.add(Dense(output_dim = 25, init = 'he_uniform',activation='relu'))



# Adding the third hidden layer

classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu'))

# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'he_uniform'))



# Compiling the ANN

classifier.compile(loss=root_mean_squared_error, optimizer='Adamax')



# Fitting the ANN to the Training set

model_history=classifier.fit(X_train.values, y_train.values,validation_split=0.20, batch_size = 10, nb_epoch = 1000)
pred=pd.DataFrame(classifier.predict(df_Test))

sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],pred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('/kaggle/working/sub.csv',index=False)