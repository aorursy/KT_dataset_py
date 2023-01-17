#Model Random  Forest

import pandas as pd

from sklearn.ensemble import RandomForestClassifier



print('Preparing training and test data...') 

dftrain = pd.read_csv('../input/train.csv')

dftrain = dftrain.head(100000)

dftest = pd.read_csv('../input/test.csv', index_col="Id")

dftest = dftest.head(100000)



#independent variable

train_WITH = dftrain["Category"]



#dependent variables

train_WITHOUT = dftrain.drop("Category", axis=1)



#remove insignificant variables for training

train_WITHOUT = train_WITHOUT.drop("Descript", axis=1) 

train_WITHOUT = train_WITHOUT.drop("Resolution", axis=1)



print('Making training a categorical structure...') 

hours = pd.get_dummies(train_WITHOUT.Dates.map(lambda x: pd.to_datetime(x).hour), prefix="hour")

months = pd.get_dummies(train_WITHOUT.Dates.map(lambda x: pd.to_datetime(x).month), prefix="month")

years = pd.get_dummies(train_WITHOUT.Dates.map(lambda x: pd.to_datetime(x).year), prefix="year")

district = pd.get_dummies(train_WITHOUT["PdDistrict"])

day_of_week = pd.get_dummies(train_WITHOUT["DayOfWeek"])



print('Train cleaning...')

train_WITHOUT = pd.concat([train_WITHOUT, hours, months, years, district, day_of_week], axis=1)

train_WITHOUT = train_WITHOUT.drop(['PdDistrict', 'Dates', 'DayOfWeek', 'Address'], axis = 1)



print('Making test a categorical structure...') 

hours = pd.get_dummies(dftest.Dates.map(lambda x: pd.to_datetime(x).hour), prefix="hour")

months = pd.get_dummies(dftest.Dates.map(lambda x: pd.to_datetime(x).month), prefix="month")

years = pd.get_dummies(dftest.Dates.map(lambda x: pd.to_datetime(x).year), prefix="year")

district = pd.get_dummies(dftest["PdDistrict"])

day_of_week = pd.get_dummies(dftest["DayOfWeek"])



print('Train cleaning...')

dftest = pd.concat([dftest, hours, months, years, district, day_of_week], axis=1)

dftest = dftest.drop(['PdDistrict', 'Dates', 'Address', 'DayOfWeek'], axis = 1)



#Start predicting

print("Classifier...")   

rfcl = RandomForestClassifier(n_estimators=100)

rfcl.fit(train_WITHOUT, train_WITH)

print('Predictor...')

result = pd.DataFrame(rfcl.predict_proba(dftest), index=dftest.index, columns=rfcl.classes_)



print("Result...")

print(result)