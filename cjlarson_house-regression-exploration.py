import numpy as np 

import pandas as pd 

import statsmodels.api as sm

from matplotlib import pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





df = pd.read_csv("../input/train.csv")

df



def stories(HouseStyle):

    hs = HouseStyle

    hs = hs.str.replace('Story','')

    hs = hs.str.replace('Fin','')

    hs = hs.str.replace('Unf','')

    hs = hs.str.replace('SLvl','1')

    hs = hs.str.replace('SFoyer','1')

    return hs.apply(pd.to_numeric,errors = 'ignore')



print(len(df.columns))

print(len(df.mean()))

df['BuiltAge'] = 2015 - df['YearBuilt']

df['RemodAge'] = 2015 - df['YearRemodAdd']

df['Stories'] = stories(df['HouseStyle'])

df.mean()



target = df[['SalePrice']]

start_features = df[['LotArea','OverallQual','OverallCond','TotRmsAbvGrd','BuiltAge']]

model = sm.OLS(target,start_features).fit()

predictions = model.predict(start_features)



model.summary()


target = df[['SalePrice']]

start_features = df[['LotArea','OverallQual','OverallCond','TotRmsAbvGrd','BuiltAge','Stories']]

model_withstories = sm.OLS(target,start_features).fit()

predictions_withstories = model_withstories.predict(start_features)



model_withstories.summary()
start_features = sm.add_constant(start_features)

model_withc = sm.OLS(target,start_features).fit()

predictions_withc = model_withc.predict(start_features)



model_withc.summary()
## The line / model

plt.scatter(target, predictions)

plt.xlabel('True Values')

plt.ylabel('Predictions')
## The line / model

plt.scatter(target, predictions_withstories)

plt.xlabel('True Values')

plt.ylabel('Predictions')
## The line / model

plt.scatter(target, predictions_withc)

plt.xlabel('True Values')

plt.ylabel('Predictions')