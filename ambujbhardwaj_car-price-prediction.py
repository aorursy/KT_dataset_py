import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import legend_handler

import seaborn as sns; sns.set()

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor





lb = LabelBinarizer()

# Read data from csv file into a pandas DataFrame

carsdata = pd.read_csv("../input/CarPrice_Assignment.csv")
carsdata.describe()
carsdata.info()
# check for null/missing values

carsdata.isnull().sum()
carsdata.select_dtypes(include = "object").head()
## From DATA DICTIONARY we know symboling is a categorical feature

newcarsdata = pd.DataFrame(carsdata["symboling"], columns=["symboling"])
# creating derived column from CarName for car company / manufacturer

carsdata["carCompany"] = carsdata["CarName"].apply(lambda x: str.split(x," ")[0]).astype("category")
print(set(carsdata["carCompany"]))

print(len(set(carsdata["carCompany"])))
carsdata = carsdata.replace({"Nissan":"nissan", "maxda":"mazda", "toyouta": "toyota","porcshce":"porsche", "vokswagen": "volkswagen", "vw":"volkswagen"})
print(set(carsdata["carCompany"]))

print(len(set(carsdata["carCompany"])))
newcarsdata.head()
## lets create dummy variables from carcompany and set them into newcarsdata dataframe

carmakers = pd.get_dummies(carsdata["carCompany"])

newcarsdata = pd.concat([newcarsdata,carmakers], axis=1)

newcarsdata.head()


# encoding car companies with numerical digits using label encoder from sklearn library

#newcarsdata["carCompany"] = LabelEncoder().fit_transform(carsdata["carCompany"])



# encoding fueltype with numerical digits

# 1 for gas, 0 for diesel

newcarsdata["fueltype"] = LabelEncoder().fit_transform(carsdata["fueltype"])



# encoding aspiration column with binary numeric data

# 0 for std and 1 for turbo

newcarsdata["aspiration"] = LabelEncoder().fit_transform(carsdata["aspiration"])
#Using LabelBinarizer function from sklearn to transform categorical values into separate columns

lb_results = lb.fit_transform(carsdata["carbody"])

carbody = pd.DataFrame(lb_results, columns=lb.classes_)

newcarsdata = pd.concat([newcarsdata,carbody],axis=1)

#Converting doornumbers from string to numeric category using simple replace function

newcarsdata["doornumber"] = carsdata["doornumber"].replace({"two":2, "four":4})
# transforming drivewheels into separate columns

lb_drivewheel = lb.fit_transform(carsdata["drivewheel"])

drivewheel = pd.DataFrame(lb_drivewheel, columns=lb.classes_)

newcarsdata = pd.concat([newcarsdata, drivewheel], axis=1)
carsdata["enginelocation"].value_counts()
# transforming engine location into numreic categories

newcarsdata["enginelocation"] = carsdata["enginelocation"].replace({"front":1,"rear":0})
# transforming enginetype into dummy features using pandas get_dummies function

enginetype = pd.get_dummies(carsdata["enginetype"])

newcarsdata = pd.concat([newcarsdata,enginetype],axis=1)
# transforming cylinder numbers into numeric categories

lb_cylinder = lb.fit_transform(carsdata["cylindernumber"])

cylindernumber = pd.DataFrame(lb_cylinder, columns=lb.classes_)

cylindernumber.columns = ['eight_cyl', 'five_cyl', 'four_cyl', 'six_cyl', 'three_cyl', 'twelve_cyl', 'two_cyl']

newcarsdata = pd.concat([newcarsdata,cylindernumber], axis=1)
#transforming fuel system to numeric category 



lb_fuelsystem = lb.fit_transform(carsdata["fuelsystem"])

fuelsystem = pd.DataFrame(lb_fuelsystem, columns=lb.classes_)

newcarsdata = pd.concat([newcarsdata, fuelsystem], axis = 1)
# new dataframe with categorical features only

newcarsdata.head()
# checking for continuous type features

carsdata.select_dtypes(exclude="object").loc[:,"wheelbase":"price"].head()
## adding continuous features to newcarsdata Dataframe

newcarsdata = pd.concat([newcarsdata,carsdata.select_dtypes(exclude="object").loc[:,"wheelbase":"price"]],axis=1)
newcarsdata.head()
newcarsdata.shape
newcarsdata.columns


symboling = newcarsdata["symboling"].value_counts()



plt.figure(figsize = [16,6])

plt.subplots_adjust(wspace=0.5)

plt.suptitle("symboling", size = 16)



plt.subplot(121)

plt.bar(x = symboling.index, height = symboling.values)



for i in symboling.index:

    plt.annotate(s = symboling.values[i], xy = (symboling.index[i], symboling.values[i]/3), color ="white", ha = "center")





plt.subplot(122)



sns.boxplot(newcarsdata["symboling"], newcarsdata["price"])



plt.show()
plt.figure(figsize= [16,6])

plt.suptitle("Car Companies", size = 16)

plt.subplots_adjust(wspace= 0.5)



cc =  carsdata["carCompany"].value_counts()



plt.subplot(121)



sns.barplot(x = cc.keys(), y = cc.values)

plt.xticks(rotation = 90)

#plt.yticks(labels = ["counts"])

plt.subplot(122)



sns.boxplot(carsdata["carCompany"], newcarsdata["price"])

plt.xticks(rotation = 90)





plt.show()
plt.figure(figsize = [16,6])



plt.suptitle("Fuel Type", size = 16)

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

#plt.title("Fuel Type", size = 16)

p1 = sns.countplot(x= newcarsdata["fueltype"])

#plt.legend(title = "Fuel type", labels = ["Diesel", "Gas"])

p1.set_xticklabels(labels = ["Diesel", "Gas"])

plt.xlabel("Fuel")



plt.subplot(122)

p2 = sns.boxplot(newcarsdata["fueltype"], newcarsdata["price"])

p2.set_xticklabels(labels = ["Diesel", "Gas"])

plt.xlabel("Fuel")



plt.show()
# 0 for std and 1 for turbo



plt.figure(figsize = [16,6])

plt.suptitle("Aspiration", size = 16)

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)



p1 = sns.countplot(newcarsdata["aspiration"])

p1.set_xticklabels(["Standard", "Turbo"])



plt.subplot(122)



p2 = sns.boxplot(newcarsdata["aspiration"], newcarsdata["price"])

p2.set_xticklabels(["Standard", "Turbo"])





plt.show()
plt.figure(figsize = [16,6])

plt.suptitle("Carbody Type", size = 16)

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

sns.countplot(carsdata["carbody"])



plt.subplot(122)

sns.boxplot(carsdata["carbody"], newcarsdata["price"])



plt.show()
plt.figure(figsize = [16,6])

plt.suptitle("Number of Doors", size = 16)

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

sns.countplot(newcarsdata["doornumber"])



plt.subplot(122)

sns.boxplot(newcarsdata["doornumber"], newcarsdata["price"])



plt.show()
plt.figure(figsize = [16,6])

plt.suptitle("Drive Wheel Type", size = 16)

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

sns.countplot(carsdata["drivewheel"])



plt.subplot(122)

sns.boxplot(carsdata["drivewheel"], newcarsdata["price"])



plt.show()
plt.figure(figsize = [16,6])

plt.suptitle("Engine Location", size = 16)

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

sns.countplot(carsdata["enginelocation"])



plt.subplot(122)

sns.boxplot(carsdata["enginelocation"], newcarsdata["price"])



plt.show()
plt.figure(figsize = [16,6])

plt.suptitle("Engine Type", size = 16)

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

sns.countplot(carsdata["enginetype"])



plt.subplot(122)

sns.boxplot(carsdata["enginetype"], newcarsdata["price"])



plt.show()
plt.figure(figsize = [16,6])

plt.suptitle("Number of Cylinders", size = 16)

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

sns.countplot(carsdata["cylindernumber"])



plt.subplot(122)

sns.boxplot(carsdata["cylindernumber"], newcarsdata["price"])



plt.show()
plt.figure(figsize = [16,6])

plt.suptitle("Fuel System", size = 16)

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

sns.countplot(carsdata["fuelsystem"])



plt.subplot(122)

sns.boxplot(carsdata["fuelsystem"], newcarsdata["price"])



plt.show()
plt.figure(figsize=[16,6])

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

sns.scatterplot(newcarsdata["wheelbase"], newcarsdata["price"])

sns.regplot(newcarsdata["wheelbase"],newcarsdata["price"])



plt.subplot(122)

sns.scatterplot(newcarsdata["enginesize"], newcarsdata["price"])

sns.regplot(newcarsdata["enginesize"], newcarsdata["price"])



plt.show()
plt.figure(figsize=[16,4])

plt.subplots_adjust(wspace=0.5)



plt.subplot(141)

sns.scatterplot(newcarsdata["carlength"], newcarsdata["price"])

sns.regplot(newcarsdata["carlength"],newcarsdata["price"])



plt.subplot(142)

sns.scatterplot(newcarsdata["carwidth"], newcarsdata["price"])

sns.regplot(newcarsdata["carwidth"],newcarsdata["price"])



plt.subplot(143)

sns.scatterplot(newcarsdata["carheight"], newcarsdata["price"])

sns.regplot(newcarsdata["carheight"],newcarsdata["price"])



plt.subplot(144)

sns.scatterplot(newcarsdata["curbweight"], newcarsdata["price"])

sns.regplot(newcarsdata["curbweight"], newcarsdata["price"])



plt.show()
plt.figure(figsize=[16,6])

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

sns.scatterplot(newcarsdata["boreratio"], newcarsdata["price"])

sns.regplot(newcarsdata["boreratio"],newcarsdata["price"])



plt.subplot(122)

sns.scatterplot(newcarsdata["compressionratio"], newcarsdata["price"])

sns.regplot(newcarsdata["compressionratio"], newcarsdata["price"])



plt.show()
plt.figure(figsize=[16,6])

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

sns.scatterplot(newcarsdata["stroke"], newcarsdata["price"])

sns.regplot(newcarsdata["stroke"],newcarsdata["price"])



plt.subplot(122)

sns.scatterplot(newcarsdata["horsepower"], newcarsdata["price"])

sns.regplot(newcarsdata["horsepower"], newcarsdata["price"])



plt.show()
plt.figure(figsize=[16,6])

plt.subplots_adjust(wspace=0.5)



plt.subplot(121)

sns.scatterplot(newcarsdata["citympg"], newcarsdata["price"])

sns.regplot(newcarsdata["citympg"],newcarsdata["price"])



plt.subplot(122)

sns.scatterplot(newcarsdata["highwaympg"], newcarsdata["price"])

sns.regplot(newcarsdata["highwaympg"], newcarsdata["price"])



plt.show()
sns.scatterplot(newcarsdata["peakrpm"], newcarsdata["price"])

sns.regplot(newcarsdata["peakrpm"], newcarsdata["price"])



plt.show()
# Cretaing new dataframes as a set of predictor features and target features

predictors = newcarsdata.iloc[:,:-1]

target  = newcarsdata.iloc[:,-1]
#Splitting the data into train and test sets

# xtrain - tarining predictors

# ytrain - training target

# xtest - testing predictors

# ytest - testing target

## train_size = %age data to assigned for training datasets

## random_state = seed for random number generator

xtrain, xtest, ytrain, ytest = train_test_split(predictors,target , train_size = 0.7, random_state = 42)
xtrain.shape
xtest.shape
ytrain.shape
ytest.shape
features_for_scaling = ['wheelbase', 'carlength', 'carwidth',

       'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke',

       'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']
# Feature description before scaling

xtrain[features_for_scaling].describe()
xtest[features_for_scaling].describe()
xtrain[features_for_scaling] = MinMaxScaler().fit_transform(xtrain[features_for_scaling].values)

xtest[features_for_scaling] = MinMaxScaler().fit_transform(xtest[features_for_scaling].values)
#Feature description after scaling

xtrain[features_for_scaling].describe()
#Feature description after scaling

xtest[features_for_scaling].describe()
xtrain.columns
xtrain_lm = sm.add_constant(xtrain)
smlr1 = sm.OLS(ytrain,xtrain_lm[list(carbody.columns)+list(carmakers.columns)+['fueltype', 'aspiration']]).fit()
smlr1.summary()
smlr2 = sm.OLS(ytrain,xtrain_lm[list(carbody.columns)+list(carmakers.columns)+['fueltype', 'aspiration'] + list(drivewheel.columns)]).fit()
smlr2.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = features_for_scaling

vif['VIF'] = [variance_inflation_factor(xtrain_lm[features_for_scaling].values, i) for i in range(xtrain_lm[features_for_scaling].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
xtrain_lm["mpg"] = (xtrain_lm["citympg"]+ xtrain_lm["highwaympg"])/2
xtrain_lm.drop(["citympg","highwaympg"], axis=1, inplace=True)
features_vif =['carlength', 'curbweight', 'wheelbase', 'carwidth',

'horsepower', 'enginesize', 'stroke', 'boreratio', 

'carheight', 'peakrpm', 'compressionratio',

'mpg']
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = features_vif

vif['VIF'] = [variance_inflation_factor(xtrain_lm[features_vif].values, i) for i in range(xtrain_lm[features_vif].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
xtrain_lm.drop(['stroke', 'boreratio','curbweight', 'horsepower'], axis = 1, inplace= True)
features_vif = features_vif =['carlength', 'wheelbase', 'carwidth',

                              'enginesize', 'carheight', 'peakrpm', 'compressionratio','mpg']
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = features_vif

vif['VIF'] = [variance_inflation_factor(xtrain_lm[features_vif].values, i) for i in range(xtrain_lm[features_vif].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
xtrain_lm.drop(["wheelbase"],axis=1, inplace=True)
features_vif = features_vif =['carlength', 'carwidth',

                              'enginesize', 'carheight', 'peakrpm', 'compressionratio','mpg']
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = features_vif

vif['VIF'] = [variance_inflation_factor(xtrain_lm[features_vif].values, i) for i in range(xtrain_lm[features_vif].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
xtrain_lm.drop(["carheight"],axis=1, inplace=True)
features_vif = features_vif =['carlength', 'carwidth',

                              'enginesize', 'peakrpm', 'compressionratio','mpg']
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = features_vif

vif['VIF'] = [variance_inflation_factor(xtrain_lm[features_vif].values, i) for i in range(xtrain_lm[features_vif].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
xtrain_lm.drop(["carwidth"],axis=1, inplace=True)
features_vif = features_vif =['carlength', 'enginesize', 'peakrpm', 'compressionratio','mpg']
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = features_vif

vif['VIF'] = [variance_inflation_factor(xtrain_lm[features_vif].values, i) for i in range(xtrain_lm[features_vif].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
xtrain_lm.drop(["peakrpm"],axis=1, inplace=True)
features_vif = features_vif =['carlength', 'enginesize', 'compressionratio','mpg']
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = features_vif

vif['VIF'] = [variance_inflation_factor(xtrain_lm[features_vif].values, i) for i in range(xtrain_lm[features_vif].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
xtrain_lm.columns
smlr3 = sm.OLS(ytrain,xtrain_lm).fit()

smlr3.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = xtrain_lm.columns

vif['VIF'] = [variance_inflation_factor(xtrain_lm.values, i) for i in range(xtrain_lm.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
xtrain_lm.drop(["compressionratio"], axis=1, inplace=True)
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = xtrain_lm.columns

vif['VIF'] = [variance_inflation_factor(xtrain_lm.values, i) for i in range(xtrain_lm.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
xtrain_lm.drop(["enginesize"], axis=1, inplace=True)
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = xtrain_lm.columns

vif['VIF'] = [variance_inflation_factor(xtrain_lm.values, i) for i in range(xtrain_lm.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
smlr4 = sm.OLS(ytrain,xtrain_lm).fit()

smlr4.summary()
xtrain_lm.columns
features_vif = ['const', 'alfa-romero', 'audi', 'bmw', 'buick',

       'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury',

       'mitsubishi', 'nissan', 'peugeot', 'plymouth', 'porsche', 'renault',

       'saab', 'subaru', 'toyota', 'volkswagen', 'volvo', 'fueltype',

       'aspiration', 'convertible', 'hardtop', 'hatchback', 'sedan', 'wagon',

       'doornumber', '4wd', 'fwd', 'rwd', 'enginelocation', 'dohc', 'dohcv',

       'l', 'ohc', 'ohcf', 'ohcv', 'rotor', 'eight_cyl', 'five_cyl',

       'four_cyl', 'six_cyl', 'three_cyl', 'twelve_cyl', 'two_cyl', '1bbl',

       '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'carlength',

       'mpg']
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = features_vif

vif['VIF'] = [variance_inflation_factor(xtrain_lm[features_vif].values, i) for i in range(xtrain_lm[features_vif].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
smlr5 = sm.OLS(ytrain,xtrain_lm[features_vif]).fit()

smlr5.summary()
xtrain_lm.drop(["enginelocation", "hatchback", "doornumber"],axis=1, inplace=True)
smlr6 = sm.OLS(ytrain,xtrain_lm).fit()

smlr6.summary()
xtrain_lm.drop(list(cylindernumber.columns)+list(fuelsystem.columns),axis=1, inplace=True)
smlr6 = sm.OLS(ytrain,xtrain_lm).fit()

smlr6.summary()
xtrain_lm.drop(["symboling", "dohc"],axis=1, inplace=True)
smlr7 = sm.OLS(ytrain,xtrain_lm).fit()

smlr7.summary()
xtrain_lm.drop(["fueltype", "mercury"],axis=1, inplace=True)
smlr8 = sm.OLS(ytrain,xtrain_lm).fit()

smlr8.summary()
xtest["mpg"] = (xtest["citympg"]+xtest["highwaympg"])/2
xtest_lm = sm.add_constant(xtest)
xtest_lm = xtest_lm[xtrain_lm.columns]
predictions_df = pd.DataFrame()

predictions_df["pred"] = smlr8.predict(xtest_lm)

predictions_df["actual"] = ytest.values
predictions_df
r2_score(y_pred=predictions_df["pred"], y_true=predictions_df["actual"])