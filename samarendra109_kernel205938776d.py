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
df.head()
df['Province_State'] = '_'+df['Province_State']

df['Province_State'].fillna("",inplace=True)

df['Location'] = df['Country_Region']+df['Province_State']
df.head()
df = df[['Id','Location', 'Date','ConfirmedCases', 'Fatalities']].copy()
tmStamp = ((pd.to_datetime(df['Date']).astype(np.int64)//100000000000) - 15796512)//864

df['Date'] = tmStamp
tmStamp.unique().__len__()
df.head()
groups = df.groupby(['Location'])

group_dict = {}

for g,gdf in groups:

    gdf = gdf.drop(['Location'],axis=1)

    group_dict[g] = gdf.copy()
from scipy.optimize import curve_fit



'''

def logistic_curve(t,a,b,c):

    return c/(1+a*np.exp(-b*t))

'''

def logistic_curve(t,L,k,x1):

    return L/(1+np.exp(k*(x1-t)))



class NewLogisticRegressor:

    

    def fit(self,x,y):

        #bounds = (0,[100000,3,1000000000])

        bounds =  (0,[1000000000,3,79])

        self.count = 5

        for c1 in range(self.count):

            p0 = np.random.exponential(size=3)

            try:

                (a,b,c),cov = curve_fit(logistic_curve,x,y,bounds=bounds,p0=p0)

                break

            except Exception as e:

                if c1==(self.count-1):

                    return None

            

        self.variables = (a,b,c)

        return self

        

    def predict(self,x):

        a,b,c = self.variables

        return np.vectorize(logistic_curve)(gdf['Date'],a,b,c)
regressor = {}

count = 0

count_cfrm,count_fatal = 0,0
for loc in group_dict.keys():



    gdf = group_dict[loc]

    

    #ConfirmCases

    x,y = gdf['Date'],gdf['ConfirmedCases']

    c_reg = NewLogisticRegressor()

    c_reg = c_reg.fit(x,y)

    

    if c_reg==None:

        count_cfrm+=1

    

    #Fatalities

    x,y = gdf['Date'],gdf['Fatalities']

    f_reg = NewLogisticRegressor()

    f_reg = f_reg.fit(x,y)

    

    if f_reg == None:

        count_fatal +=1

    

    regressor[loc] = {'ConfirmCases':c_reg,'Fatalities':f_reg}

    count+=1

    print(count,end=" ")
print(count_cfrm,count_fatal)
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
df_test['Province_State'] = '_'+df_test['Province_State']

df_test['Province_State'].fillna("",inplace=True)

df_test['Location'] = df_test['Country_Region']+df_test['Province_State']
tmStamp = ((pd.to_datetime(df_test['Date']).astype(np.int64)//100000000000) - 15796512)//867

df_test['Date'] = tmStamp
groupsTest = df_test.groupby(['Location'])

group_dict_test = {}

for g,gdf in groupsTest:

    #gdf = gdf.drop(['Location'],axis=1)

    group_dict_test[g] = gdf.copy()
ids = []

c_res = []

f_res = []



for loc in group_dict_test.keys():

    

    c_reg,f_reg = regressor[loc].values()

    

    g_ids = list(group_dict_test[loc]['ForecastId'])

    

    cy,fy = [],[]

    if c_reg!=None and f_reg!=None:

        

        '''ConfirmedCases'''

        x = gdf['Date']

        #cy = list(np.round(c_reg.predict(x)))

        cy = list(c_reg.predict(x))

        

        '''Fatalities'''

        x = gdf['Date']

        #fy = list(np.round(f_reg.predict(x)))

        fy = list(f_reg.predict(x))

        

    ids += g_ids

    c_res += cy

    f_res += fy
sub = pd.DataFrame({'ForecastId':ids,

                    'ConfirmedCases':c_res,

                    'Fatalities':f_res,

                   })
sub.head()
sub.to_csv('submission.csv',index=False)