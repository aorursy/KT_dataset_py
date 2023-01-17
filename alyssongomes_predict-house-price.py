# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler #lib for preprocesss



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
def select_cat_atributes(datas):

    dfc = datas.select_dtypes([object])

            

    for c in dfc.columns:

        dfc[c].fillna(dfc[c].mode().values[0], inplace=True)

    

    #discretizar atributos categóricos

    discretization = pd.DataFrame()

    for c in dfc.columns:

        discretization[c] = dfc[c]

        discretization[c] = discretization[c].astype('category')

        discretization[c] = discretization[c].cat.codes

    

    return discretization
def pre_process(datas):

    X = pd.DataFrame()

    

    #Substituindo por Modas para valores numéricos

    cols = ["OverallQual","GarageCars","YearBuilt","YearRemodAdd","GarageYrBlt","Fireplaces","HalfBath","BedroomAbvGr","MoSold"]

    for column in cols:

        X[column] = datas[column]

        X[column].fillna(X[column].mode().values[0],inplace=True)

    

    #Substituindo por Médias para valores numéricos

    cols = ["GrLivArea","TotalBsmtSF","1stFlrSF","FullBath","MasVnrArea","BsmtFinSF1","LotFrontage","WoodDeckSF","2ndFlrSF","OpenPorchSF","LotArea","BsmtFullBath","BsmtUnfSF","ScreenPorch","PoolArea","3SsnPorch"]

    for column in cols:

        X[column] = datas[column]

        X[column].fillna(X[column].mean(),inplace=True)

    

    #Normalizar valores

    values = X.values

    column = X.columns

    scaler = StandardScaler().fit(values)

    normalized = scaler.transform(values)

    X = pd.DataFrame(normalized,columns=column)

    

    #Discretização para valores categóricos

    discretization = select_cat_atributes(datas)

    cols = ['Foundation','CentralAir','PavedDrive','RoofStyle','SaleCondition','Neighborhood','HouseStyle','RoofMatl','ExterCond','Functional','Exterior2nd','Exterior1st','Condition1','LandSlope','Street']

    for column in cols:

        X[column] = discretization[column]

        #Substituindo os nulos (NaN) pela Moda dos valores

        X[column].fillna(X[column].mode().values[0], inplace=True)

    

    return X
# Training

Y = train.SalePrice

X = pre_process(train)



#Model Predictor

model_linear = LinearRegression()

model_linear.fit(X,Y)

print ('Score: {}'.format(model_linear.score(X, Y)))
X_test = pre_process(test)



#Predict

prediction = model_linear.predict(X_test)



#Save prediction

submission = pd.DataFrame()

submission['Id'] = test.Id

submission['SalePrice'] = pd.Series(prediction)

submission.to_csv("kaggle.csv", index=False)

submission.head()