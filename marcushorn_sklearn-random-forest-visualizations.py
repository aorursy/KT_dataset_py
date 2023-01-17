# Data processing libraries

import numpy as np

import pandas as pd



# Plotting libraries for data exploration

import matplotlib.pyplot as plt

import seaborn as sns



# Misc libraries

import math



# List all files in working directory:

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Allows us to see every column in the [dataset].head() calls, keep this commented

# out unless you want to see the name of every variable (takes a while to run)

#pd.set_option('display.max_columns', None)



train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

train_trans = train_trans.set_index('TransactionID').sort_index()

print("Training transaction set has %d rows and %d columns" % train_trans.shape)

train_trans.head()
train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

train_id = train_id.set_index('TransactionID').sort_index()

print("Training identification set has %d rows and %d columns" % train_id.shape)

train_id.head()
test_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

test_trans = test_trans.set_index('TransactionID').sort_index()

print("Test transaction set has %d rows and %d columns" % test_trans.shape)

test_trans.head()
test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

test_id = test_id.set_index('TransactionID').sort_index()

print("Test identification set has %d rows and %d columns" % test_id.shape)

test_id.head()
### Preparing submission file

submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

print("Submission file has %d rows and %d columns" % submission.shape)

submission.head()
def checkMatches(transaction, identification):

    isIdentified = []

    trans_ids = transaction.index.array

    id_ids = identification.index.array

    # Since we guarantee order of the transaction ids, we don't need to check the

    # entirety of id_ids for each id and can just check the current index instead

    length = len(id_ids)

    i = 0

    for id in trans_ids:

        if i >= length or id != id_ids[i]:

            isIdentified.append(0) # False - no match found

        else:

            i += 1

            isIdentified.append(1) # True - match found

    transaction["isIdentified"] = isIdentified

    return transaction
checkMatches(train_trans,train_id)

train_trans.head()
checkMatches(test_trans,test_id)

test_trans.head()
train_combined = train_trans.join(train_id)

print("Combined training set has %d rows and %d columns" % train_combined.shape)

train_combined.head()
test_combined = test_trans.join(test_id)

print("Combined test set has %d rows and %d columns" % test_combined.shape)

test_combined.head()
numGraphs = 3 # If multiple variables are plotted, this is how many graphs to plot per row

subWidth = 12 # Width to allocate for each row of subplots

subHeight = 6 # Height to allocate for each subplot

font = 8 # Font size for subplots



def plotNum(setType,dataset,fields):

    # Plots KDE plots of numerical data, to show distribution of frequencies

    # setType is a string, either "train" or "test"

    sns.set(style='darkgrid')

    n = len(fields) # Number of variables to plot

    if n == 1:

        field = fields[0]

        data = dataset[field].dropna()

        plt.xticks(rotation=90)

        sns.kdeplot(data).set_title("%s set %s" % (setType, field))

        #print("Minimum %s value: %d" % (field, data.min()))

        #print("Maximum %s value: %d" % (field, data.max()))

        #print("Average %s value: %d" % (field, (data.sum()/len(data))))

        #print("Median %s value: %d" % (field, data.median()))

    else:

        size = (subWidth, subHeight * math.ceil(n/numGraphs)) # Allot 4 in of height per row

        if n > numGraphs: 

            fig, axes = plt.subplots(math.ceil(n/numGraphs), numGraphs, figsize=size)

        else:

            fig, axes = plt.subplots(1, n, figsize=size)

        for i in range(n):

            field = fields[i]

            data = dataset[field].dropna()

            if n > numGraphs:

                sns.kdeplot(data,ax=axes[i//numGraphs,i % numGraphs]).set_title("%s set %s" % (setType, field))

            else:

                sns.kdeplot(data,ax=axes[i]).set_title("%s set %s" % (setType, field))

            #print("Minimum %s value: %d" % (field, data.min()))

            #print("Maximum %s value: %d" % (field, data.max()))

            #print("Average %s value: %d" % (field, (data.sum()/len(data))))

            #print("Median %s value: %d" % (field, data.median()))

        for ax in axes.flatten():

            for tick in ax.get_xticklabels():

                tick.set_rotation(90)

            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.show()

    return None



def plotCat(setType,dataset,fields):

    maxCat = 10 # Number of categories to reduce plots to showing

    sns.set(style='darkgrid')

    n = len(fields) # Number of variables to plot

    if n == 1:

        field = fields[0]

        data = dataset[field].dropna()

        plt.xticks(rotation=90)

        sns.countplot(data, order = data.value_counts().iloc[:maxCat].index).set_title("%s set %s" % (setType, field))

    else:

        size = (subWidth, subHeight * math.ceil(n/numGraphs))

        if n > numGraphs:

            fig, axes = plt.subplots(math.ceil(n/numGraphs), numGraphs, figsize=size)

        else:

            fig, axes = plt.subplots(1, n, figsize=size)

        for i in range(n):

            field = fields[i]

            data = dataset[field].dropna()

            if n > numGraphs:

                sns.countplot(data, order = data.value_counts().iloc[:maxCat].index,

                    ax=axes[i//numGraphs,i % numGraphs]).set_title("%s set %s" % (setType, field))

            else:

                sns.countplot(data, order = data.value_counts().iloc[:maxCat].index,

                    ax=axes[i]).set_title("%s set %s" % (setType, field))

        for ax in axes.flatten():

            for tick in ax.get_xticklabels():

                tick.set_rotation(90)

    plt.show()

    return None
nulls = train_combined.isnull().sum()

#nulls = nulls.sort_values()

nulls.plot(kind='barh', figsize=(10,60), fontsize=10, 

           title = "Number of Training Set Null Values")
nulls = test_combined.isnull().sum()

print(nulls)

#nulls = nulls.sort_values()

nulls.plot(kind='barh', figsize=(10,60), fontsize=10, 

           title = "Number of Test Set Null Values")
plotNum('train',train_combined,['TransactionAmt'])
plotNum('test',test_combined,['TransactionAmt'])
plotCat('train',train_combined,['ProductCD'])
plotCat('test',test_combined,['ProductCD'])
catFields = []

numFields = []

for i in [4,6]:

    catFields.append('card%d' % i)

for i in [1,2,3,5]:

    numFields.append('card%d' % i)
plotCat('train',train_combined,catFields)
plotCat('test',test_combined,catFields)
plotNum('train',train_combined,numFields)
plotNum('test',test_combined,numFields)
catFields = ['addr1','addr2']
plotNum('train',train_combined,catFields)
plotNum('test',test_combined,catFields)
numFields = ['dist1','dist2']
plotNum('train',train_combined,numFields)
plotNum('test',test_combined,numFields)
# Extracting email services from a list of addresses:

def emailServices(emails):

    services = []

    top = 10 # How many of the most frequent providers to display before lumping into "other"

    for email in emails:

        if type(email) != str:

            service = float('nan')

        else:

            splitIndex = email.find('@')

            service = email[splitIndex + 1:]

        services.append(service)

    return services
# Adding each set of servicers to the datasets:

train_combined['P_emailprovider'] = emailServices(train_combined['P_emaildomain'])

train_combined['R_emailprovider'] = emailServices(train_combined['R_emaildomain'])

test_combined['P_emailprovider'] = emailServices(test_combined['P_emaildomain'])

test_combined['R_emailprovider'] = emailServices(test_combined['R_emaildomain'])
catFields = ['P_emailprovider','R_emailprovider']
plotCat('train',train_combined,catFields)
plotCat('test',test_combined,catFields)
numFields = []

for i in range(1,15):

    numFields.append('C%d' % i)
plotNum('train',train_combined,numFields)
plotNum('test',test_combined,numFields)
numFields = []

for i in range(1,16):

    numFields.append('D%d' % i)
plotNum('train',train_combined,numFields)
plotNum('test',test_combined,numFields)
catFields = []

for i in range(1,10):

    catFields.append('M%d' % i)
plotCat('train',train_combined,catFields)
plotCat('test',test_combined,catFields)
catFields = ['DeviceType','DeviceInfo']
plotCat('train',train_combined,catFields)
plotCat('test',test_combined,catFields)
ids = []

numFields = []

catFields = []

for i in range(1,10):

    ids.append('id_0%d' % i)

for i in range(10,39):

    ids.append('id_%d' % i)

for id in ids:

    idType = train_combined[id].dtype

    if idType == 'float64':

        numFields.append(id)

    else:

        catFields.append(id)

    print(id, train_combined[id].dtype)
plotNum('train',train_combined,numFields)
plotNum('test',test_combined,numFields)
plotCat('train',train_combined,catFields)
plotCat('test',test_combined,catFields)
def resolutionParams(res):

    # Retrieves width, height, and area from a list of strings

    widths = []

    heights = []

    areas = []

    for r in res:

        if type(r) != str:

            # Catch missing values

            width = float('nan')

            height = float('nan')

            area = float('nan')

        else:

            w, h = r.split('x')

            width = int(w)

            height = int(h)

            area = width * height

        widths.append(width)

        heights.append(height)

        areas.append(area)

    return widths, heights, areas
widths, heights, areas = resolutionParams(train_combined["id_33"])

train_combined["width"] = widths

train_combined["height"] = heights

train_combined["area"] = areas



widths, heights, areas = resolutionParams(test_combined["id_33"])

test_combined["width"] = widths

test_combined["height"] = heights

test_combined["area"] = areas
numFields = ["width", "height", "area"]
plotNum('train',train_combined,numFields)
plotNum('test',test_combined,numFields)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
catColumns = []

for col in train_combined.columns:

    if train_combined[col].dtype == 'object':

        catColumns.append(col)



cats = {}

for col in catColumns:

    train_combined[col].fillna('missing',inplace=True)

    test_combined[col].fillna('missing',inplace=True)

    

    train_combined[col] = train_combined[col].astype('category')

    train_combined[col].cat.add_categories('unknown',inplace=True)

    cats[col] = train_combined[col].cat.categories
for k, v in cats.items():

    test_combined[k][~test_combined[k].isin(v)] = 'unknown'
from pandas.api.types import CategoricalDtype



for k, v in cats.items():

    new_dtype = CategoricalDtype(categories=v, ordered=True)

    test_combined[k] = test_combined[k].astype(new_dtype)
for col in catColumns:

    train_combined[col] = train_combined[col].cat.codes

    test_combined[col] = test_combined[col].cat.codes
train_combined.fillna(-999,inplace=True)

test_combined.fillna(-999,inplace=True)
y = train_combined.pop("isFraud") # Separates results from parameters

x_train, x_test, y_train, y_test = train_test_split(train_combined, y, train_size = .2)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
model = RandomForestRegressor(

    n_estimators=400, max_features=0.3,

    min_samples_leaf=20, n_jobs=-1, verbose=1)

model.fit(x_train, y_train)
pred_test = model.predict(x_test)
roc_auc_score(y_test, pred_test)
model = RandomForestRegressor(

    n_estimators=400, max_features=0.3,

    min_samples_leaf=20, n_jobs=-1, verbose=1)

model.fit(train_combined,y)
pred = model.predict(test_combined)
submission["isFraud"] = pred

submission.to_csv('submission.csv', index=False)