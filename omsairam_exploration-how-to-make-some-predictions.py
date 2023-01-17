# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#load in the train and test csvs

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')





#print out their dimensions

print(train.shape)

print(test.shape)
print(train.head())
print(test.head())
numbers = train.select_dtypes(include=[np.number])



print(train.select_dtypes(include=[np.number]).dtypes)
import seaborn as sns

corr = numbers.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
corrWithSales = corr['SalePrice']

print(corrWithSales.head(5))
corrWithSales.sort_values(ascending = False)[1:11]
train_relevant_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF','1stFlrSF',

                'FullBath', 'TotRmsAbvGrd','YearBuilt','YearRemodAdd', 'SalePrice']



test_relevant_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF','1stFlrSF',

                'FullBath', 'TotRmsAbvGrd','YearBuilt','YearRemodAdd']



trainR = train[train_relevant_cols]

testR = test[test_relevant_cols]



trainR.head(4)

testR.head(4)
trainX = trainR.drop(['SalePrice'], axis=1)

testX = trainR['SalePrice']





print(trainX.shape)

print(testX.shape)
from sklearn import linear_model

lm = linear_model.LinearRegression()



model = lm.fit(trainX, testX)
#https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for

#user Boem from stackoverflow created the following function to clean datasets



def clean_dataset(df):

    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)

    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep].astype(np.float64)
testClean = clean_dataset(testR)

predictions = model.predict(testClean)

print(predictions)
