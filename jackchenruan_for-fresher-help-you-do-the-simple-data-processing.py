import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb



class FillUp():

    def __init__(self):

        pass

    

    # Fill the missing value by mode.

    # Input the feature matrix and the name of column with missing value

    def mode(self,df,name):       

        df[name] = df[name].fillna(df[name].mode()[0])

        return df

    

    # Fill the missing value by median.

    def median(self,df,name):

        df[name] = df[name].fillna(df[name].median())

        return df

    

    # Fill the missing value by mean.

    def mean(self,df,name):

        df[name] = df[name].fillna(df[name].mean())

        return df

    

    # Fill the missing value by RandomForestRegressor.

    def RFR(self,df,name):

        

        # Get the name of columns without missing value

        cols = list(df.dropna(axis=1).columns)

        

        # Transform 'int64' into 'float64'

        for col in cols:

            if df[col].dtype == 'int64':

                df[col] = df[col].astype('float64')

                

        # We can't use 'Object' value in RandomForestRegressor

        for col in cols:

            if df[col].dtype == 'O':

                cols.remove(col)

                

        # Insert the name of column which will be filled.

        cols.insert(0,name)

        

        # Define a new feature matrix

        df_ = df[cols]

        

        # To separate the known data and unknown data.

        known = df_[df_[name].notnull()].values

        unknown = df_[df_[name].isnull()].values

        

        # Get training data

        y = known[:,0]

        x = known[:,1:]

        

        # Modeling and training

        rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)

        rfr.fit(x, y)

        

        # Make a prediction

        predicted = rfr.predict(unknown[:,1:])

        

        # Fill the predicted data back to the original feature matrix

        df.loc[(df[name].isnull()),name] = predicted

        return df

    

    # Fill the missing value by XgBoostingRegressor.

    # The follows are the same as above but the model

    def XGBR(self,df,name):

        

        cols = list(df.dropna(axis=1).columns)

        

        for col in cols:

            if df[col].dtype == 'int64':

                df[col] = df[col].astype('float64')

        for col in cols:

            if df[col].dtype == 'O':

                cols.remove(col)

                

        cols.insert(0,name)

        df_ = df[cols]

        

        known = df_[df_[name].notnull()].values

        unknown = df_[df_[name].isnull()].values

        y = known[:,0]

        x = known[:,1:]

        

        xgbr = xgb.XGBRegressor()

        xgbr.fit(x, y)

        

        predicted = xgbr.predict(unknown[:,1:])

        df.loc[(df[name].isnull()),name] = predicted

        return df
import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
# Save ID

train_Id = train.loc[:,'PassengerId']

test_Id = test.loc[:,'PassengerId']



# Delete useless information

train.drop('PassengerId',axis=1,inplace=True)

test.drop('PassengerId',axis=1,inplace=True)

train.drop('Name',axis=1,inplace=True)

test.drop('Name',axis=1,inplace=True)

train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
#concat the train data and test data

ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.Survived

df = pd.concat((train,test)).reset_index(drop=True)

df.drop(['Survived'],axis=1,inplace=True)
df.head()
df.info()
# Percentage of missing values

df.isnull().mean()
Fup = FillUp()
# Fill 'Embarked' with mode

Fup.mode(df,'Embarked')
df.info()
# Fill 'Fare' with mean

Fup.mean(df,'Fare')
df.info()
# Fill 'Age' with RandomForestRegessor

Fup.RFR(df,'Age')
df.info()
# Percentage of missing values

df.isnull().mean()