# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn import linear_model
import statsmodels.api as sm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
givenTrain = pd.read_csv("../input/train.csv",parse_dates = ['Date'])
givenTest = pd.read_csv("../input/test.csv",parse_dates = ['Date'])
givenStore = pd.read_csv("../input/store.csv")
givenTrain[givenTrain.StateHoliday=="0"] = 0  #To maintain consistency in data type in this column
#Considering only open stores and stores with continous promotions
openStores = givenTrain[givenTrain['Open'] == 1]
givenStore = givenStore[givenStore['Promo2'] == 1]
combinedData = pd.merge(openStores,givenStore,how="inner")
interimData = combinedData.drop(combinedData[['CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear']],axis= 1)
print("Ensuring if any anomalies present with Promo2 and Promo2SinceWeek :",((interimData['Promo2'] == 0) & (pd.notnull(interimData['Promo2SinceWeek']))).sum())
print("Ensuring if any anomalies present with Promo2 and Promo2SinceYear :",((interimData['Promo2'] == 0) & (pd.notnull(interimData['Promo2SinceYear']))).sum())
print("Ensuring if any anomalies present with Promo2 and PromoInterval :",((interimData['Promo2'] == 0) & (pd.notnull(interimData['PromoInterval']))).sum())
continuousPromo['Month'] = continuousPromo['Date'].dt.month
dataAfterDropping = continuousPromo.drop(continuousPromo[['Date','Promo2SinceWeek','Promo2SinceYear','PromoInterval']],axis =1)

#To process discrete values to dummies it is converted to string
dataAfterDropping['Month'] = dataAfterDropping['Month'].astype(str)
dataAfterDropping['DayOfWeek'] = dataAfterDropping['DayOfWeek'].astype(str)

cleanDataSet = pd.get_dummies(dataAfterDropping)
cleanDataSet.drop(cleanDataSet[['Open','Promo2']],axis=1,inplace=True) #it has same value for all observations.
cleanDataSet.head(5)
X1 = cleanDataSet.drop(cleanDataSet[['Sales','Store']],axis =1)
y = cleanDataSet['Sales']
X2 = sm.add_constant(X1)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
corr = cleanDataSet.corr()
sns.heatmap(corr)
newX = cleanDataSet.drop(cleanDataSet[['Sales','StateHoliday_b','Assortment_a','Month_3','StoreType_a','Month_11','Month_4']],axis =1)
y1 = cleanDataSet['Sales']
X2 = sm.add_constant(newX)
est = sm.OLS(y1, X2)
est2 = est.fit()
print(est2.summary())
lr = linear_model.Lasso(alpha=0.2)
lr.fit(newX,y)
print("Lasso Regression Co-efficients are:","\n")
[print(a, b) for a,b in zip(newX.columns,lr.coef_)][0]

print("\n")

rf = Ridge()
rf.fit(newX,y)
print("Ridge Regression Co-oefficients are:","\n")
[print(a, b) for a,b in zip(newX.columns,rf.coef_)][0]
