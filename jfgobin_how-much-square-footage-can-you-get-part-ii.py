import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from scipy.stats import norm

from scipy.stats import linregress

%matplotlib inline



train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



# First, some cleaning and conversions

# Conversion of categorical to type "category" and replacement of

# missing values in categorical with "missing"

for i_name in ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',

               'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',

               'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

               'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',

               'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

               'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

               'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',

               'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',

               'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',

               'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']:

        train_data[i_name].fillna("missing", inplace=True)

        train_data[i_name] = train_data[i_name].astype('category')

        test_data[i_name] = test_data[i_name].astype('category')



# 

train_data['SFPerBuck'] = (train_data['GrLivArea'])/(train_data['SalePrice'])

log_stat = np.log(train_data['SFPerBuck'])

train_data['logSFPerBuck'] = log_stat

train_data["QualBinned"] = pd.cut(train_data['OverallQual'],[0,5.5,6.5,10], 

                                  labels=["Low","Medium","High"])

train_data["CondBinned"] = pd.cut(train_data['OverallCond'],[0,4.5,5.5,10], 

                                  labels=["Low","Medium","High"])

train_data['C1C2'] = (train_data['Condition1'].astype("str") + \

                     ','+train_data['Condition2'].astype("str")).astype("category")

# Transformed variables: 



for varToTrsf in ['BsmtFinSF1', 'TotalBsmtSF', 'GarageArea', 

          'EnclosedPorch', '2ndFlrSF', 'LowQualFinSF']:

    varName = varToTrsf+'Transf' 

    varPresent = train_data[varToTrsf] > 0

    train_data[varName] = varPresent.astype('category')

    

# Variables for which some elements are merged

# BsmtCond (after converting the Po to TA)

train_data.loc[train_data['BsmtCond']=='Po',['BsmtCond']] = 'TA'

# BsmtFinType2 (after converting between missing and non-missing)

def finTypePresent(a):

    if a == "missing":

        return 0

    return 1

bsmtFinType2 = train_data['BsmtFinType2'].apply(finTypePresent)

train_data.drop(['BsmtFinType2'], inplace=True, axis=1)

train_data['BsmtFinType2'] = bsmtFinType2.astype("category")

# Heating (after converting Floor to GasW)

train_data.loc[train_data['Heating']=='Floor',['Heating']] = 'GasW'

# Electrical (after converting Missing and Mix to SBrkr)

def redefElectrical(a):

    if a in ('FuseA', 'FuseF', 'FuseP'):

        return 'Fuse'

    return 'SBrkr'

electrical = train_data['Electrical'].apply(redefElectrical)

train_data.drop(['Electrical'], inplace=True, axis=1)

train_data['Electrical'] = electrical.astype('category')

# HalfBath (after creating two categories for 0/1 and 2+)

def redefHalfBath(a):

    if a in (0,1):

        return 0

    return 1

halfBath = train_data['HalfBath'].apply(redefHalfBath)

train_data.drop(['HalfBath'], inplace=True, axis=1)

train_data['HalfBath'] = halfBath.astype('category')

# BedroomAbvGr (after grouping 0/1, 2/3, and 4+)

def redefBRAbGr(a):

    if a in (0,1):

        return "few"

    if a in (2,3):

        return "normal"

    return "many"

bedroomAbvGr = train_data['BedroomAbvGr'].apply(redefBRAbGr)

train_data.drop(['BedroomAbvGr'], inplace=True, axis=1)

train_data['BedroomAbvGr'] = bedroomAbvGr.astype("category")

# KitchenAbvGr (after grouping 0/1 and 2+)

kitchenAbvGr = train_data['KitchenAbvGr'].apply(redefHalfBath)

train_data.drop(['KitchenAbvGr'], inplace=True, axis=1)

train_data['KitchenAbvGr'] = kitchenAbvGr.astype('category')

# Functional (after grouping the maj1/maj2/sev and min1/min2 together)

def redefFunctional(a):

    if a in ("Maj1", "Maj2", "Sev"):

        return "Major"

    if a in ("Min1", "Min2", "Mod"):

        return "Minor"

    return "Typical"

functional = train_data['Functional'].apply(redefFunctional)

train_data.drop(['Functional'], inplace=True, axis=1)

train_data['Functional'] = functional.astype("category")

# GarageQual (after grouping Ex with Gd)

train_data.loc[train_data['GarageQual']=='Ex',['GarageQual']] = 'Gd'

# Condition1

def redefCondition(a):

    if a in ("RRNe", "RRAe", "RRAn", "RRNn"):

        return "RR"

    if a in ("PosN", "PosA"):

        return "Pos"

    return a

condition1 = train_data['Condition1'].apply(redefCondition)

train_data.drop(['Condition1'], inplace=True, axis=1)

train_data['Condition1'] = condition1.astype("category")
mean_t = train_data['logSFPerBuck'].mean()

var_t = train_data['logSFPerBuck'].var()

print("Mean    : {: 5.3f}\nVariance: {: 5.3f}\n".format(mean_t,

                                                        var_t))

# Let's make an hypothetical plot of SalePrice vs grLivArea based on these values.

x0 = train_data['GrLivArea'].min()

x1 = train_data['GrLivArea'].max()



x = np.linspace(x0, x1, num=100)

y0 = x/np.exp(mean_t)

y1 = x/np.exp(mean_t+3*np.sqrt(var_t))

ym1 = x/np.exp(mean_t-3*np.sqrt(var_t))



plt.figure(figsize=(16,8))

mplt = plt.plot(x,y0, x,y1, x,ym1)

plt.scatter(train_data['GrLivArea'].values,

             train_data['SalePrice'].values)

plt.grid()



grLivAreaAvg = train_data['GrLivArea'].mean()

minSalePrice = grLivAreaAvg/np.exp(mean_t+3*np.sqrt(var_t))

maxSalePrice = grLivAreaAvg/np.exp(mean_t-3*np.sqrt(var_t))



print("Average Living Area (sf2): {:> 10.2f}\nPrice (lower bound)      : {:> 10.2f}\nPrice (upper bound)      : {:> 10.2f}".format(grLivAreaAvg,minSalePrice,maxSalePrice))
# Let's make a split on QualBinned

tot_var = 0

plt.figure(figsize=(16,8))

for (qb_cat, mark) in zip(train_data['QualBinned'].cat.categories, ("o","x","+")):

    td_slice = train_data.loc[train_data['QualBinned']==qb_cat,('GrLivArea',

                                                                'SalePrice',

                                                                'logSFPerBuck')]

    xp = td_slice['GrLivArea']

    yp = td_slice['SalePrice']

    a = td_slice['logSFPerBuck'].mean()

    y0 = x/np.exp(a)

    plt.scatter(xp.values, yp.values, marker=mark, label=qb_cat)

    plt.plot(x,y0,linestyle=':', label=qb_cat)

    tot_var += td_slice['logSFPerBuck'].var()

    print("Variance for {:>6s}: {: 5.3f}".format(qb_cat,td_slice['logSFPerBuck'].var()))



plt.grid()

plt.legend()



print("Sum of variance: {: 5.3f}".format(tot_var))
# Let's make a split on CondBinned

tot_var = 0

plt.figure(figsize=(16,8))

for (qb_cat, mark) in zip(train_data['CondBinned'].cat.categories, ("o","x","+")):

    td_slice = train_data.loc[train_data['CondBinned']==qb_cat,('GrLivArea',

                                                                'SalePrice',

                                                                'logSFPerBuck')]

    xp = td_slice['GrLivArea']

    yp = td_slice['SalePrice']

    a = td_slice['logSFPerBuck'].mean()

    y0 = x/np.exp(a)

    plt.scatter(xp.values, yp.values, marker=mark, label=qb_cat)

    plt.plot(x,y0,linestyle=':', label=qb_cat)

    tot_var += td_slice['logSFPerBuck'].var()

    print("Variance for {:>6s}: {: 5.3f}".format(qb_cat,td_slice['logSFPerBuck'].var()))



plt.grid()

plt.legend()



print("Sum of variance: {: 5.3f}".format(tot_var))
td_slice = train_data.loc[train_data['CondBinned']=='High',('GrLivArea',

                                                            'SalePrice',

                                                            'logSFPerBuck')]

mean_t = td_slice['logSFPerBuck'].mean()

var_t = td_slice['logSFPerBuck'].var()

y0 = x/np.exp(mean_t)

y1 = x/np.exp(mean_t+3*np.sqrt(var_t))

ym1 = x/np.exp(mean_t-3*np.sqrt(var_t))



plt.figure(figsize=(16,8))

mplt = plt.plot(x,y0, x,y1, x,ym1)

plt.scatter(td_slice['GrLivArea'].values,

            td_slice['SalePrice'].values)

plt.grid()



#grLivAreaAvg = td_slice['GrLivArea'].mean()

minSalePrice = grLivAreaAvg/np.exp(mean_t+3*np.sqrt(var_t))

maxSalePrice = grLivAreaAvg/np.exp(mean_t-3*np.sqrt(var_t))



print("Average Living Area (sf2): {:> 10.2f}\nPrice (lower bound)      : {:> 10.2f}\nPrice (upper bound)      : {:> 10.2f}".format(grLivAreaAvg,minSalePrice,maxSalePrice))
td_slice = train_data.query('CondBinned == "Low" and QualBinned =="Low"')[['GrLivArea',

                                                                             'SalePrice',

                                                                             'logSFPerBuck']]

mean_t = td_slice['logSFPerBuck'].mean()

var_t = td_slice['logSFPerBuck'].var()

y0 = x/np.exp(mean_t)

y1 = x/np.exp(mean_t+3*np.sqrt(var_t))

ym1 = x/np.exp(mean_t-3*np.sqrt(var_t))



plt.figure(figsize=(16,8))

mplt = plt.plot(x,y0, x,y1, x,ym1)

plt.scatter(td_slice['GrLivArea'].values,

            td_slice['SalePrice'].values)

plt.grid()



#grLivAreaAvg = td_slice['GrLivArea'].mean()

minSalePrice = grLivAreaAvg/np.exp(mean_t+3*np.sqrt(var_t))

maxSalePrice = grLivAreaAvg/np.exp(mean_t-3*np.sqrt(var_t))



print("Average Living Area (sf2): {:> 10.2f}\nPrice (lower bound)      : {:> 10.2f}\nPrice (upper bound)      : {:> 10.2f}".format(grLivAreaAvg,minSalePrice,maxSalePrice))

mean_t = train_data['logSFPerBuck'].mean()

var_t = train_data['logSFPerBuck'].var()

x = train_data['GrLivArea'].values

y0 = x/np.exp(mean_t)



error_pred = np.abs((y0 - train_data['SalePrice'].values)/y0)



sns.distplot(error_pred)



print("Mean relative error : {: 5.3f}".format(error_pred.mean()))

print("Max relative error  : {: 5.3f}".format(error_pred.max()))

error_pred = np.array([])

for cb_cat in train_data['CondBinned'].cat.categories:

    for qb_cat in train_data['QualBinned'].cat.categories:

        td_slice = train_data.query('CondBinned == "' +\

                                cb_cat + '" and QualBinned == "'+qb_cat+'"')[['GrLivArea',

                                                                       'SalePrice',

                                                                       'logSFPerBuck']]

    xp = td_slice['GrLivArea'].values

    yp = td_slice['SalePrice'].values

    a = td_slice['logSFPerBuck'].mean()

    y0 = xp/np.exp(a)

    error_pred = np.concatenate((error_pred,np.abs((y0-yp)/y0)))

    

sns.distplot(error_pred)



print("Mean relative error : {: 5.3f}".format(error_pred.mean()))

print("Max relative error  : {: 5.3f}".format(error_pred.max()))
var_list = ['Overall']

var_mean = [0.211]

var_max = [1.375]

var_warn = [False]

for cat_var in ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',

               'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',

               'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

               'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',

               'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

               'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

               'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',

               'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',

               'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',

               'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition',

               "QualBinned", "CondBinned", "BsmtFinSF1Transf", "TotalBsmtSFTransf", 

                "GarageAreaTransf",]:

    error_pred = np.array([])

    warning = False

    for cat_val in train_data[cat_var].cat.categories:

        td_slice = train_data.loc[train_data[cat_var]==cat_val,('GrLivArea',

                                                            'SalePrice',

                                                            'logSFPerBuck')]

        if td_slice.shape[0] < 10:

            warning = True

        xp = td_slice['GrLivArea'].values

        yp = td_slice['SalePrice'].values

        a = td_slice['logSFPerBuck'].mean()

        y0 = xp/np.exp(a)

        error_pred = np.concatenate((error_pred,np.abs((y0-yp)/y0)))

    var_list.append(cat_var)

    var_warn.append(warning)

    var_mean.append(error_pred.mean())

    var_max.append(error_pred.max())



onevar = pd.DataFrame({"Variable": var_list, "Warning": var_warn,

                       "Mean_Error": var_mean, "Max_Error": var_max})

onevar.sort_values(["Mean_Error"], inplace=True, ascending=False)

f, ax = plt.subplots(figsize=(12, 9))

sns.barplot(x="Variable", y="Mean_Error", data=onevar, hue="Warning")

plt.xticks(rotation=90);



f, ax = plt.subplots(figsize=(12, 9))

sns.barplot(x="Variable", y="Max_Error", data=onevar, hue="Warning")

plt.xticks(rotation=90);
only_nowarn = onevar.loc[onevar['Warning']==False]

ordered_var = only_nowarn.sort_values(["Mean_Error"], ascending=False)['Variable']



f, ax = plt.subplots(figsize=(12, 9))

sns.barplot(x="Variable", y="Mean_Error", 

            data=only_nowarn, order=ordered_var)

plt.xticks(rotation=90);



f, ax = plt.subplots(figsize=(12, 9))

sns.barplot(x="Variable", y="Max_Error", 

            data=only_nowarn, order=ordered_var)

plt.xticks(rotation=90);
error_pred = np.array([])

warning = False

for cb_cat in train_data['BsmtQual'].cat.categories:

    for qb_cat in train_data['BsmtFinType1'].cat.categories:

        td_slice = train_data.query('BsmtQual == "' +\

                                cb_cat + '" and BsmtFinType1 == "'+qb_cat+'"')[['GrLivArea',

                                                                       'SalePrice',

                                                                       'logSFPerBuck']]

    if td_slice.shape[0] < 10:

        warning = True

    xp = td_slice['GrLivArea'].values

    yp = td_slice['SalePrice'].values

    a = td_slice['logSFPerBuck'].mean()

    y0 = xp/np.exp(a)

    error_pred = np.concatenate((error_pred,np.abs((y0-yp)/y0)))

    

f, ax = plt.subplots(figsize=(12, 9))

sns.distplot(error_pred)



print("Mean relative error : {: 5.3f}".format(error_pred.mean()))

print("Max relative error  : {: 5.3f}".format(error_pred.max()))

if warning:

    print("There were warnings.")
error_pred = np.array([])

warning = False

for cb_cat in train_data['BsmtQual'].cat.categories:

    for qb_cat in train_data['ExterQual'].cat.categories:

        td_slice = train_data.query('BsmtQual == "' +\

                                cb_cat + '" and ExterQual == "'+qb_cat+'"')[['GrLivArea',

                                                                       'SalePrice',

                                                                       'logSFPerBuck']]

    if td_slice.shape[0] < 10:

        warning = True

    xp = td_slice['GrLivArea'].values

    yp = td_slice['SalePrice'].values

    a = td_slice['logSFPerBuck'].mean()

    y0 = xp/np.exp(a)

    error_pred = np.concatenate((error_pred,np.abs((y0-yp)/y0)))

    

f, ax = plt.subplots(figsize=(12, 9))

sns.distplot(error_pred)



print("Mean relative error : {: 5.3f}".format(error_pred.mean()))

print("Max relative error  : {: 5.3f}".format(error_pred.max()))

if warning:

    print("There were warnings.")
f, ax = plt.subplots(figsize=(12, 9))

sns.boxplot(x="BsmtQual", y="SalePrice", data=train_data, hue="ExterQual")

plt.grid()

slope, intercept, r_value, p_value, std_err = stats.linregress(train_data['YearBuilt'], 

                                                               train_data['logSFPerBuck'])

xpred = train_data['YearBuilt']*slope+intercept

ypred = train_data['GrLivArea'].values/np.exp(xpred)

yreal = train_data['SalePrice'].values

error_pred = np.abs((ypred-yreal)/ypred)



f, ax = plt.subplots(figsize=(12, 9))

sns.distplot(error_pred)



print("Mean relative error : {: 5.3f}".format(error_pred.mean()))

print("Max relative error  : {: 5.3f}".format(error_pred.max()))

print("Slope               : {: 6.5f}".format(slope))

print("Intercept           : {: 6.5f}".format(intercept))

error_pred = np.array([])

val_pred = np.array([])

val_real = np.array([])

warning = False

for cb_cat in train_data['Condition1'].cat.categories:

    td_slice = train_data.loc[train_data['Condition1']==cb_cat,['GrLivArea',

                                                              'SalePrice',

                                                              'logSFPerBuck',

                                                              'YearBuilt', 

                                                               'MiscVal']]

    if td_slice.shape[0] < 10:

        print(cb_cat)

        print(td_slice.shape[0])

        warning = True

        

    slope, intercept, r_value, p_value, std_err = stats.linregress(td_slice['YearBuilt'], 

                                                                   td_slice['logSFPerBuck'])

    xpred = td_slice['YearBuilt']*slope+intercept

    ypred = td_slice['GrLivArea'].values/np.exp(xpred) + td_slice['MiscVal']

    yreal = td_slice['SalePrice'].values

    error_pred = np.concatenate((error_pred,np.abs((ypred-yreal)/ypred)))

    val_pred = np.concatenate((val_pred, ypred))

    val_real = np.concatenate((val_real, yreal))

    

f, ax = plt.subplots(figsize=(12, 9))

sns.distplot(error_pred)



f, ax = plt.subplots(figsize=(12, 9))

plt.scatter(val_pred, val_real)

plt.plot([val_pred.min(), val_pred.max()], [val_pred.min(), val_pred.max()])



print("Mean relative error : {: 5.3f}".format(error_pred.mean()))

print("Max relative error  : {: 5.3f}".format(error_pred.max()))

if warning:

    print("There were warnings.")