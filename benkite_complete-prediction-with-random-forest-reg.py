# Import necessary libraries and data



import pandas, os, numpy

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt



dat = pandas.read_csv("../input/train.csv")

testdat = pandas.read_csv("../input/test.csv")

macro = pandas.read_csv("../input/macro.csv")

## Merge the macro data with the train and test data

dat = pandas.merge(dat, macro, "left", on = "timestamp")

testdat = pandas.merge(testdat, macro, "left", on = "timestamp")
## Add new features

dat["top_floor"] = numpy.where(dat["floor"] == dat["max_floor"], 1, 0)

testdat["top_floor"] = numpy.where(testdat["floor"] == testdat["max_floor"], 1, 0)

dat["pdens"] = dat["full_all"]/dat["area_m"]

testdat["pdens"] = testdat["full_all"]/testdat["area_m"]

dat["workpop"] = dat["work_all"]/dat["full_all"]

testdat["workpop"] = testdat["work_all"]/testdat["full_all"]



dat[["top_floor", "pdens", "workpop"]][0:5]
## Fix extreme values that could influence results 

## This probably could be done in a better way

## These decisions were made quickly.

dat["build_year"].loc[dat["build_year"] > 2015] = None

dat["build_year"].loc[dat["build_year"] < 1900] = None

dat["kitch_sq"].loc[dat["kitch_sq"] > 15] = None

dat["life_sq"].loc[dat["full_sq"] < dat["life_sq"]] = None

dat["full_sq"].loc[dat["full_sq"] > 200] = None

dat["life_sq"].loc[dat["full_sq"] < dat["life_sq"]] = None



testdat["build_year"].loc[testdat["build_year"] > 2017] = None

testdat["build_year"].loc[testdat["build_year"] < 1900] = None

testdat["kitch_sq"].loc[testdat["kitch_sq"] > 15] = None

testdat["life_sq"].loc[testdat["full_sq"] < testdat["life_sq"]] = None

testdat["full_sq"].loc[testdat["full_sq"] > 200] = None

testdat["life_sq"].loc[testdat["full_sq"] < testdat["life_sq"]] = None
## Add a variable for the age of the home

dts = []

for ds in dat["timestamp"]:

    dts.append(int(ds.split("-")[0]))

dat["cur_year"] = dts

tts = []

for ts in testdat["timestamp"]:

    tts.append(int(ts.split("-")[0]))

testdat["cur_year"] = tts

dat["age"] = dat["cur_year"] - dat["build_year"]

testdat["age"] = testdat["cur_year"] - testdat["build_year"]

## Drop these variables

del dat["timestamp"]

del testdat["timestamp"]

del dat["cur_year"]

del testdat["cur_year"]

del dat["build_year"]

del testdat["build_year"]
## Dummy code the string variables

dat = pandas.get_dummies(dat, drop_first = True)

testdat = pandas.get_dummies(testdat, drop_first = True)

## Remove variables with all missings and variables that are not present in both data sets.

dat.dropna(axis=1, how='all', inplace=True)

testdat.dropna(axis=1, how='all', inplace=True)

datnames = dat.columns

testnames = testdat.columns

dnames = []

tnames = []

for d in datnames:

    dnames.append(d)

for t in testnames:

    tnames.append(t)

usevars = list(set(dnames).intersection(tnames))

dv = "price_doc"

testdat = testdat[usevars]

trainusevars = usevars

trainusevars.append(dv)

dat = dat[trainusevars]
## Impute missing values with variable mean replacement

imputer = Imputer(axis = 0, strategy = "mean")

trainX = imputer.fit_transform(dat)

trainX = pandas.DataFrame(trainX)

trainX.columns = dat.columns
## Fit a random forest regressor

## My high score came with 400 estimators

model = RandomForestRegressor(n_estimators = 5)



dv = "price_doc"

preds = trainX.columns[:len(trainX.columns) - 2]

model.fit(trainX[preds], trainX[dv])
testnames = testdat.columns

testX = imputer.fit_transform(testdat)

testX = pandas.DataFrame(testX)

testX.columns = testnames

#testX = testdat



testdat["price_doc"] = model.predict(testX[preds])

## These are my predictions

testdat["price_doc"][0:20]
## Finish by preparing the submission set.

output = testdat[["id", "price_doc"]]

output