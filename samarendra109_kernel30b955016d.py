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
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['Province_State'] = '_'+df['Province_State']

df['Province_State'].fillna("",inplace=True)

df['Location'] = df['Country_Region']+df['Province_State']
df.ConfirmedCases = df.ConfirmedCases.cummax()

df.Fatalities = df.Fatalities.cummax()
tmStamp = ((pd.to_datetime(df['Date']).astype(np.int64)//100000000000) - 15796512)//864

df['nDay'] = tmStamp
df.head()
from sklearn.linear_model import LinearRegression



class NewLinearRegressor():

    

    def __init__(self):

        self.regressor = LinearRegression()

        

    def processX(self,X):

        X = X.copy()

        minDay = self.minDay

        X['nDay'] = X['nDay']-minDay

        tmax = self.maxDay

        X['nDay'] = X['nDay']/tmax

        

        X['nDay2'] = X.nDay**2

        #X['nDay4'] = X.nDay**4

        #X['nDay6'] = X.nDay**6

        #X['nDay8'] = X.nDay**8

        

        return X[['nDay','nDay2']]#,'nDay4','nDay6','nDay8']]

        

    def fit(self,X,y):

        

        tempDf = pd.DataFrame({"nDay":X,"Cases":y})

        tempDf = tempDf[tempDf.Cases!=0]

        y = np.log(tempDf.Cases)

        X = tempDf[['nDay']]

        

        self.minDay = X.nDay.min()-1

        #self.maxDay = X.nDay.max()

        self.maxDay = 120

        

        #Processing X

        X = self.processX(X)

        #Fitting

        self.regressor.fit(X,y)

        return self

        

    def predict(self,X):

        

        #Processing X

        X = self.processX(X)

        #Predict

        y_log = self.regressor.predict(X)

        return np.exp(y_log)
regressor = {}

count = 0



for loc,gdf in df.groupby('Location'):

    

    #ConfirmCases

    x,y = gdf['nDay'],gdf['ConfirmedCases']

    c_reg = NewLinearRegressor()

    c_reg = c_reg.fit(x,y)

    

    #Fatalities

    x,y = gdf['nDay'],gdf['Fatalities']

    f_reg = NewLinearRegressor()

    f_reg = f_reg.fit(x,y)

    

    regressor[loc] = {'ConfirmCases':c_reg,'Fatalities':f_reg}

    count+=1

    print(count,end=" ")
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
df_test['Province_State'] = '_'+df_test['Province_State']

df_test['Province_State'].fillna("",inplace=True)

df_test['Location'] = df_test['Country_Region']+df_test['Province_State']
tmStamp = ((pd.to_datetime(df_test['Date']).astype(np.int64)//100000000000) - 15796512)//867

df_test['nDay'] = tmStamp
df_test.head()
ids = []

c_res = []

f_res = []



for loc,gdf in df_test.groupby('Location'):

    

    c_reg,f_reg = regressor[loc].values()

    

    g_ids = list(gdf['ForecastId'])

    

    cy,fy = [],[]

        

    '''ConfirmedCases'''

    x = gdf[['nDay']]

    cy = list(c_reg.predict(x))



    '''Fatalities'''

    x = gdf[['nDay']]

    fy = list(f_reg.predict(x))

        

    ids += g_ids

    c_res += cy

    f_res += fy
sub = pd.DataFrame({'ForecastId':ids,

                    'ConfirmedCases':c_res,

                    'Fatalities':f_res})

sub.to_csv('submission.csv',index=False)