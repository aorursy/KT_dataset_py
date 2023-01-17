# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # some plotting!

import seaborn as sns # so pretty!

from scipy import stats # I might use this

from sklearn.ensemble import RandomForestClassifier # checking if this is available

# from sklearn import cross_validation

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# import the training data set and make sure it's in correctly...

train = pd.read_csv('../input/train.csv')

train_original = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.info()
# define a function to convert an object (categorical) feature into an int feature

# 0 = most common category, highest int = least common.

def getObjectFeature(df, col, datalength=1460):

    if df[col].dtype!='object': # if it's not categorical..

        print('feature',col,'is not an object feature.')

        return df

    elif len([i for i in df[col].T.notnull() if i == True])!=datalength: # if there's missing data..

        print('feature',col,'is missing data.')

        return df

    else:

        df1 = df

        counts = df1[col].value_counts() # get the counts for each label for the feature

        df1[col] = [counts.index.tolist().index(i) for i in df1[col]] # do the conversion

        return df1 # make the new (integer) column from the conversion

# and test the function...

fcntest = getObjectFeature(train,'LotShape')

fcntest.head(10)
#histogram and normal probability plot

from scipy.stats import norm

sns.distplot(train['SalePrice'],fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice'] = np.log(train['SalePrice'])

sns.distplot(train['SalePrice'],fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
from sklearn.tree import DecisionTreeRegressor as dtr

# define the training data X...

X = train[['MoSold','YrSold','LotArea','BedroomAbvGr']]

Y = train[['SalePrice']]

# and the data for the competition submission...

X_test = test[['MoSold','YrSold','LotArea','BedroomAbvGr']]

print(X.head())

print(Y.head())
# let's set up some cross-validation analysis to evaluate our model and later models...

from sklearn.model_selection import cross_val_score

# try fitting a decision tree regression model...

DTR_1 = dtr(max_depth=None) # declare the regression model form. Let the depth be default.

# DTR_1.fit(X,Y) # fit the training data

scores_dtr = cross_val_score(DTR_1, X, Y, cv=10,scoring='explained_variance') # 10-fold cross validation

print('scores for k=10 fold validation:',scores_dtr)

print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_dtr.mean(), scores_dtr.std() * 2))
from sklearn.ensemble import RandomForestRegressor as rfr

estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

mean_rfrs = []

std_rfrs_upper = []

std_rfrs_lower = []

yt = [i for i in Y['SalePrice']] # quick pre-processing of the target

np.random.seed(11111)

for i in estimators:

    model = rfr(n_estimators=i,max_depth=None)

    scores_rfr = cross_val_score(model,X,yt,cv=10,scoring='explained_variance')

    print('estimators:',i)

#     print('explained variance scores for k=10 fold validation:',scores_rfr)

    print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))

    print('')

    mean_rfrs.append(scores_rfr.mean())

    std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting

    std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting
# and plot...

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

ax.plot(estimators,mean_rfrs,marker='o',

       linewidth=4,markersize=12)

ax.fill_between(estimators,std_rfrs_lower,std_rfrs_upper,

                facecolor='green',alpha=0.3,interpolate=True)

ax.set_ylim([-.3,1])

ax.set_xlim([0,80])

plt.title('Expected Variance of Random Forest Regressor')

plt.ylabel('Expected Variance')

plt.xlabel('Trees in Forest')

plt.grid()

plt.show()
# list all the features we want. This is still arbitrary...

included_features = ['MoSold','YrSold','LotArea','BedroomAbvGr', # original data

                    'FullBath','HalfBath','TotRmsAbvGrd', # bathrooms and total rooms

                    'YearBuilt','YearRemodAdd', # age of the house

                    'LotShape','Utilities'] # some categoricals 

# define the training data X...

X = train[included_features]

Y = train[['SalePrice']]

# and the data for the competition submission...

X_test = test[included_features]

# transform categorical data if included in X...

for col in list(X):

    if X[col].dtype=='object':

        X = getObjectFeature(X, col)

X.head()
# define the number of estimators to consider

estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

mean_rfrs = []

std_rfrs_upper = []

std_rfrs_lower = []

yt = [i for i in Y['SalePrice']]

np.random.seed(11111)

# for each number of estimators, fit the model and find the results for 8-fold cross validation

for i in estimators:

    model = rfr(n_estimators=i,max_depth=None)

    scores_rfr = cross_val_score(model,X,yt,cv=10,scoring='explained_variance')

    print('estimators:',i)

#     print('explained variance scores for k=10 fold validation:',scores_rfr)

    print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))

    print("")

    mean_rfrs.append(scores_rfr.mean())

    std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting

    std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting
# and plot...

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

ax.plot(estimators,mean_rfrs,marker='o',

       linewidth=4,markersize=12)

ax.fill_between(estimators,std_rfrs_lower,std_rfrs_upper,

                facecolor='green',alpha=0.3,interpolate=True)

ax.set_ylim([-.2,1])

ax.set_xlim([0,80])

plt.title('Expected Variance of Random Forest Regressor')

plt.ylabel('Expected Variance')

plt.xlabel('Trees in Forest')

plt.grid()

plt.show()
import sklearn.feature_selection as fs # feature selection library in scikit-learn

train = pd.read_csv('../input/train.csv') # get the training data again just in case

train['SalePrice'] = np.log(train['SalePrice'])

# first, let's include every feature that has data for all 1460 houses in the data set...

included_features = [col for col in list(train)

                    if len([i for i in train[col].T.notnull() if i == True])==1460

                    and col!='SalePrice' and col!='id']

# define the training data X...

X = train[included_features] # the feature data

Y = train[['SalePrice']] # the target

yt = [i for i in Y['SalePrice']] # the target list 

# and the data for the competition submission...

X_test = test[included_features]

# transform categorical data if included in X...

for col in list(X):

    if X[col].dtype=='object':

        X = getObjectFeature(X, col)

X.head()

# Y.head()
mir_result = fs.mutual_info_regression(X, yt) # mutual information regression feature ordering

feature_scores = []

for i in np.arange(len(included_features)):

    feature_scores.append([included_features[i],mir_result[i]])

sorted_scores = sorted(np.array(feature_scores), key=lambda s: float(s[1]), reverse=True) 

print(np.array(sorted_scores))
# and plot...

fig = plt.figure(figsize=(13,6))

ax = fig.add_subplot(111)

ind = np.arange(len(included_features))

plt.bar(ind,[float(i) for i in np.array(sorted_scores)[:,1]])

ax.axes.set_xticks(ind)

plt.title('Feature Importances (Mutual Information Regression)')

plt.ylabel('Importance')

# plt.xlabel('Trees in Forest')

# plt.grid()

plt.show()
# define a function to do the necessary model building....

def getModel(sorted_scores,train,numFeatures):

    included_features = np.array(sorted_scores)[:,0][:numFeatures] # ordered list of important features

    # define the training data X...

    X = train[included_features]

    Y = train[['SalePrice']]

    # transform categorical data if included in X...

    for col in list(X):

        if X[col].dtype=='object':

            X = getObjectFeature(X, col)

    # define the number of estimators to consider

    estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    mean_rfrs = []

    std_rfrs_upper = []

    std_rfrs_lower = []

    yt = [i for i in Y['SalePrice']]

    np.random.seed(11111)

    # for each number of estimators, fit the model and find the results for 8-fold cross validation

    for i in estimators:

        model = rfr(n_estimators=i,max_depth=None)

        scores_rfr = cross_val_score(model,X,yt,cv=10,scoring='explained_variance')

        mean_rfrs.append(scores_rfr.mean())

        std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting

        std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting

    return mean_rfrs,std_rfrs_upper,std_rfrs_lower



# define a function to plot the model expected variance results...

def plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,numFeatures):

    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(111)

    ax.plot(estimators,mean_rfrs,marker='o',

           linewidth=4,markersize=12)

    ax.fill_between(estimators,std_rfrs_lower,std_rfrs_upper,

                    facecolor='green',alpha=0.3,interpolate=True)

    ax.set_ylim([-.2,1])

    ax.set_xlim([0,80])

    plt.title('Expected Variance of Random Forest Regressor: Top %d Features'%numFeatures)

    plt.ylabel('Expected Variance')

    plt.xlabel('Trees in Forest')

    plt.grid()

    plt.show()

    return
# top 15...

mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,15)

plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,15)
# top 20...

mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,20)

plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,20)
# top 30...

mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,30)

plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,30)
# top 40...

mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,40)

plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,40)
# top 50...

mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,50)

plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,50)
# build the model with the desired parameters...

numFeatures = 40 # the number of features to inlcude

trees = 60 # trees in the forest

included_features = np.array(sorted_scores)[:,0][:numFeatures]

# define the training data X...

X = train[included_features]

Y = train[['SalePrice']]

# transform categorical data if included in X...

for col in list(X):

    if X[col].dtype=='object':

        X = getObjectFeature(X, col)

yt = [i for i in Y['SalePrice']]

np.random.seed(11111)

model = rfr(n_estimators=trees,max_depth=None)

scores_rfr = cross_val_score(model,X,yt,cv=10,scoring='explained_variance')

print('explained variance scores for k=10 fold validation:',scores_rfr)

print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))

# fit the model

model.fit(X,yt)
# let's read the test data to be sure...

test = pd.read_csv('../input/test.csv')
# re-define a function to convert an object (categorical) feature into an int feature

# 0 = most common category, highest int = least common.

def getObjectFeature(df, col, datalength=1460):

    if df[col].dtype!='object': # if it's not categorical..

        print('feature',col,'is not an object feature.')

        return df

    else:

        df1 = df

        counts = df1[col].value_counts() # get the counts for each label for the feature

#         print(col,'labels, common to rare:',counts.index.tolist()) # get an ordered list of the labels

        df1[col] = [counts.index.tolist().index(i) 

                    if i in counts.index.tolist() 

                    else 0 

                    for i in df1[col] ] # do the conversion

        return df1 # make the new (integer) column from the conversion
# apply the model to the test data and get the output...

X_test = test[included_features]

for col in list(X_test):

    if X_test[col].dtype=='object':

        X_test = getObjectFeature(X_test, col, datalength=1459)

# print(X_test.head(20))

y_output = model.predict(X_test.fillna(0)) # get the results and fill nan's with 0

print(y_output)
# transform the data to be sure

y_output = np.exp(y_output)

print(y_output)
# define the data frame for the results

saleprice = pd.DataFrame(y_output, columns=['SalePrice'])

# print(saleprice.head())

# saleprice.tail()

results = pd.concat([test['Id'],saleprice['SalePrice']],axis=1)

results.head()
# and write to output

results.to_csv('housepricing_submission.csv', index = False)