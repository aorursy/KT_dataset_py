# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-

"""

Created on Thu Mar 16 06:21:08 2017



@author: d

This is my first kernel submission, so bear with me. It's also a first draft, which I'll hopefully clean up as I go.

"""



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('/home/d/data_sets/kaggle_houseprices/train.csv')



#Remove id

df.drop(['Id'], axis=1, inplace=True)



#There are too many columns for the default pandas display. Let's change that. If you ever want to change it back, use   pd.reset_option('display.max_columns')

pd.set_option('display.max_columns', len(df.columns))

#Take a quick overview

df.columns

df.head()

df.describe()

df.corr()



#Take a looko at NaN's. We have quite a few to deal with.

pd.set_option('display.max_rows', len(df))

df.isnull().sum().apply(lambda x:x*100) # Why multiply by 100? It's just a hack to make it easier to see columns with single-digit NaN values. I'm open to hearing a cleaner way to do this.

pd.reset_option('display.max_rows')



corr = df.corr()



#Fix missing data

      # Count NaN's: df['FireplaceQu'].notnull().sum()        or use isnull()

      # Look at the NaN's: df['FireplaceQu'].loc[df['FireplaceQu'].notnull()]

      # If you want to see all the data: pd.set_option('display.max_rows', len(df))

      # And here's how to undo the above when you don't want to see everything anymore: pd.reset_option('display.max_rows')

      # View the contents of a column; it's useful for seeing what's going on with NaN's.  df['Fence'].value_counts() 



#Fill continuous variables with only a few missing entries

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt']) #Gonna just assume garage and house are built same year.

df['LotFrontage'] = df['LotFrontage'].fillna(df['LotArea']*(df['LotFrontage'].mean() / df['LotArea'].mean())) # Gonna assume lot frontage is a function of area. More area = more frontage.



#Fill categoricals where NaN just means 0 or equivalent. Will turn these into categorical columns later

df['Fence'].fillna(0, inplace=True)  #this is just categorical of quality, not continuous of sq ft. So fillna()

df['MiscFeature'].fillna(0, inplace=True) # These are almost entirely sheds. So keep them, and fillna. df['MiscFeature'].loc[df['MiscFeature'].notnull()]

df['FireplaceQu'].fillna(0, inplace=True) # There are NaN's when there are 0 fireplaces. So we'll convert those into a 0 for later.

df['Alley'].fillna(0, inplace=True)

df['MasVnrArea'].fillna(0, inplace=True) # Only 8 missing. not sure why.['MasVnrType'].value_counts()

df['MasVnrType'].fillna('None', inplace=True) #  It looks like these NaNs don't have MasVnr.

df['BsmtQual'].fillna('NA', inplace=True)

df['BsmtCond'].fillna('NA', inplace=True)

df['BsmtExposure'].fillna('NA', inplace=True) #There's on that's unfinished, but I'm not going to worry about it

df['BsmtFinType1'].fillna('NA', inplace=True)

df['BsmtFinType2'].fillna('NA', inplace=True)

df['Electrical'].fillna('SBrkr', inplace=True) #Looking at the data, it looks like a normal newer house. Assuming electrical is as well.

df['GarageType'].fillna('NA', inplace=True) #Change the "No garage" NaN entries to NA

df['GarageFinish'].fillna('NA', inplace=True)

df['GarageQual'].fillna('NA', inplace=True)

df['GarageCond'].fillna('NA', inplace=True)



#Drop the variables with too many missing to be able to use it effectively. 

df.drop(['PoolQC'], axis=1, inplace=True) # There are seven of these. Not much you can do with that.

#df['PoolQC'] = df.loc[df['PoolArea'] == 0].apply(lambda x: 0, axis=1) # A snippet of code I didn't use but want to keep around



#Feature Engineering

df['TotalSqft'] = df['GrLivArea'] + df['TotalBsmtSF'] #create a "total sq ft" column

df['TotalLivSqft'] = df['TotalSqft'] + df['BsmtFinSF1'] + df['BsmtFinSF2']







''' Need to figure out how to make these box plots appear in separate windows#################

for i in types['float64']:

    df.boxplot(i)



for i in types['int64']:

    df.boxplot(i)



for i in types['float64']:

    df[i].hist(bins=40)

for i in types['int64']:

    df[i].hist(bins=40)

'''



'''

#Collinearity matrix. This is a big one and takes a long time to run, so only run it when you really want to see it.

sns.set(style="white")

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

   # Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

   # Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

   # Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

'''

            

######The collinearities matching to SalesPrice may be a good place to start.

import datetime # Man this takes forever. What's taking so long? I'll drop times in occasionally to see what model is taking how long. 

                # Eventually I'm going to refactor this so the model runs are done either using separate python instances or just separate processes.



#Use this next line to make generating some analyses automatically easier

types = df.columns.to_series().groupby(df.dtypes.astype('string')).groups



#Create categoricals

mapping = {}

for i in types['object']:

    df[i] = df[i].astype('category')

    mapping[i] = dict( enumerate(df[i].cat.categories)) #This lets me come back later and see what variables map to what numbers

    df[i] = pd.Categorical.from_array(df[i]).codes

    

#A first look at what features matter

X_train = df.drop('SalePrice', axis=1)

y_train = df['SalePrice']



import cPickle



# Recursive Feature Elimination. 



# create a base classifier used to evaluate a subset of attributes

'''

#I try not to do this more than once. It takes half an hour to run on my laptop with an i7 6700HQ proc. Dumps a 571mb pickle file. Do this concurrent to doing other things.

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2',C=1000000)

# create the RFE model and select 3 attributes

rfe = RFE(model, 3)

rfe = rfe.fit(X_train, y_train)

# summarize the selection of the attributes

print("RFE: %s" % type(model))

print(rfe.support_)

print(rfe.ranking_)

print("\n")

with open ('RFE_logReg.pk', 'wb') as RFE_log:

    cPickle.dump(rfe, RFE_log)

'''

#If you've already run and pickled the model, then load it.

with open ('RFE_logReg.pk', 'rb') as RFE_log: 

    rfe_loaded = cPickle.load(RFE_log)

print("RFE: %s" % rfe_loaded.estimator_)

print(rfe_loaded.support_)

print(rfe_loaded.ranking_)

print("\n")





# Feature Importance

#from sklearn import metrics



'''

# fit an Extra Trees model to the data. 

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(X_train, y_train)

# display the relative importance of each attribute

print("Feature Importance: %s" % type(model))

print(model.feature_importances_)

print("\n")



with open ('exTreeClass.pk', 'wb') as xTrCl:

    cPickle.dump(model, xTrCl)

'''





#If you've already run and pickled the model, then load it.

with open ('exTreeClass.pk', 'rb') as xTrCl:

    model_loaded = cPickle.load(xTrCl)    

print("Feature Importance: %s" % type(model_loaded))

print(model_loaded.feature_importances_)

print("\n")



#features_to_keep = idx[model_loaded.feature_importances_ > np.mean(model_loaded.feature_importances_)]

#pd.DataFrame(model_loaded.transform(X_train))



#Select K Best

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

test = SelectKBest(score_func = chi2, k=4)

chifit = test.fit(X_train, y_train)

np.set_printoptions(precision=3)

print("Fit Scores")

print(chifit.scores_)

features = chifit.transform(X_train)

print("\n")

print("Features")

print(chifit.get_support())

print(features[0:5,:])



#PCA... I need to learn how PCA works.

from sklearn.decomposition import PCA

pca = PCA(n_components=3)

pcafit = pca.fit(X_train)

print("Explained Variance: %s") % pcafit.explained_variance_ratio_

print(pcafit.components_)





feature_selector = pd.DataFrame()

feature_selector = feature_selector.append(pd.DataFrame(np.arange(0, X_train.shape[1]), columns=['column_number']), ignore_index = True)

feature_selector['column_name'] = pd.DataFrame(X_train.columns)#.to_series

feature_selector['chi2'] = pd.DataFrame(chifit.get_support())

feature_selector['rfe_logreg'] = pd.DataFrame(rfe_loaded.ranking_)

feature_selector['extratreesclassifier'] = pd.DataFrame(model_loaded.feature_importances_ > .025)

#feature_selector['pca'] = # I need to learn how PCA works.



feature_selector[(feature_selector == True).any(axis=1)]





idx = np.arange(0, X_train.shape[1])



X_train = pd.get_dummies(X_train) #Used for model development







#!#!#!#! I have a first working cut of classifiers above. Now to use some logic to pick the top X variables from the last two columns, and run some models.











###

#High-level look at data. Takes a long while though, and the chart is impossibly smaLL font. so I leave it commented unless I really want to see it. 

#sns.pairplot(df)



###TO-DOs:

###There is a time series component to this data, and that needs to be factored in. Espcially given the recession of '08.###

###Given the above, YrSold becomes a categorical I think

###We also want to create dummy variables reflecting what stage of the recession we were in. Need to hit some econ data on this###

###Might use time-series charts to show how house prices shifted for similar houses to define the variables.###

###Might also develop categoricals for the type of house (luxury, normal, etc. or something) and create interactions between those typs and the recession phase###

###Original Construction Date - curious to see how trends follow here. Should look to see what phases houses were built in, and where? Might make categoricals for each build phase / project (by neighborhood and when built)###

###Per above, could also categorize the types of homes in each neighborhood. Not sure if it's worth the effort though.###





#5. Visualizations

######################################################################################All the further I've gotten




