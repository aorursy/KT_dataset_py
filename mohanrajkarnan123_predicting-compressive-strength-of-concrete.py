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
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.metrics import r2_score

file = pd.read_csv("../input/concrete-compressive-strength-data-set/compresive_strength_concrete.csv")

file.head(5)
# Changing Columns

file.columns = [col[:col.find("(")].strip() for col in file.columns]
file
file.shape
# Method for finding adjusted R2 and R2



def calc_adjstr2(x,y,predict):

    r2 = r2_score(y,predict)

    n= len(y)

    p= x.shape[1]

    adjstr2 = 1-(1-r2) * (n-1) / (n-p-1)

    return("r2 = ",r2 ,"adjstr2 : ",adjstr2)
linearModel = LinearRegression()

x= file.loc[:,['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate',

       'Fine Aggregate', 'Age']]

y= file.loc[:,'Concrete compressive strength']

RFE_model = RFE(linearModel,1)

RFE_model.fit(x,y)



Ranking = pd.DataFrame(RFE_model.ranking_).T

Ranking.columns = x.columns

Ranking
linearModel = LinearRegression()

x= file.loc[:,['Superplasticizer','Water','Age','Cement','Blast Furnace Slag','Fly Ash']]

y= file.loc[:,'Concrete compressive strength']

linearModel.fit(x,y)

predictedStr = linearModel.predict(x) #Test data

print(calc_adjstr2(x,y,predictedStr))
Residual = pd.DataFrame()

Residual['OrginalStrength'] = y

Residual['PredictedStrength'] =predictedStr

Residual['Residual'] = y -predictedStr

Residual = Residual.sort_values(by='OrginalStrength') # Sorting

%matplotlib inline

Residual.plot.scatter(x='OrginalStrength',y='Residual')
from sklearn import metrics

import numpy as np

print('Mean Absolute Error:', metrics.mean_absolute_error(y, predictedStr))

print('Mean Squared Error:', metrics.mean_squared_error(y, predictedStr))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, predictedStr)))