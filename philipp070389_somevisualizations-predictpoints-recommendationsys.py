# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

from math import ceil

import numpy as np

from collections import Counter

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

import nltk

import time

import gc

from sklearn.model_selection import train_test_split

import re



df_wines = pd.read_csv("../input/winemag-data-130k-v2.csv").drop(['Unnamed: 0'], axis=1).drop_duplicates()

print(df_wines.columns.values + '\n')

df_wines.head()
print('min points: ' + str(df_wines['points'].min()))

print('max points: ' + str(df_wines['points'].max()))

print('mean points: ' + str(df_wines['points'].mean()))

print('variance points: ' + str(df_wines['points'].var()))
print(df_wines.groupby(['taster_name'])['points'].mean().sort_values(ascending = False))
print(df_wines.groupby(['country'])['points'].mean().sort_values(ascending = False))
df_england = df_wines[df_wines['country'] == 'England']

df_england['title'].head()
del df_england

print(df_wines['country'].value_counts())
df_multirating = df_wines[df_wines.duplicated(['title'], keep=False)].drop_duplicates()

df_multirating['title'].value_counts().head()
print(df_multirating.groupby(['title'])['taster_name'].value_counts())
print(df_wines[df_wines['title'] == 'Willm 2011 Clos Gaensbroennel Kirchberg de Barr Grand Cru Gewurztraminer (Alsace)'][['taster_name', 'points']])
print(df_wines['taster_name'].value_counts())

print('# different tasters: ' + str(len(df_wines['taster_name'].value_counts())))
plt.figure()



df_wines['points'].value_counts().sort_index().plot(figsize=(40,20), fontsize = 20)



plt.xlabel('Points', fontsize = 30)

plt.ylabel('# Ratings with Points', fontsize = 30)

plt.show()
def plotvalues (columnsubplotby, df = df_wines, columncountvalues = 'points', subplotcolumnnumber = 4, figuresizebynumberentries = True, figsize = (40,40)):

    columnnumber = subplotcolumnnumber

    x_min = df[columncountvalues].min() - 5

    x_max = df[columncountvalues].max() + 5

    if figuresizebynumberentries:

        fig, axes = plt.subplots(nrows=ceil(len(df[columnsubplotby].value_counts()) / columnnumber), 

                             ncols=columnnumber, 

                             figsize=(round(len(df[columnsubplotby].value_counts()) / columnnumber) * 10,subplotcolumnnumber * 10))

    else:

        fig, axes = plt.subplots(nrows=ceil(len(df[columnsubplotby].value_counts()) / columnnumber), 

                             ncols=columnnumber, 

                             figsize=figsize)

    i,j = 0,0

    for entry in df[columnsubplotby].value_counts().index:

        axi = axes[i,j]

        axi.text(0.7, 0.85,'# wines: ' + str(len(df[df[columnsubplotby] == entry])) + 

                            '\n mean ' + columncountvalues +': ' + str(round(df[df[columnsubplotby] == entry][columncountvalues].mean(), 2)), 

                 fontsize=12, transform=axi.transAxes)

        df[df[columnsubplotby] == entry][columncountvalues].value_counts().sort_index().plot(ax = axi)

        axi.set_title(entry) 

        axi.set_xlim(x_min, x_max)

        j += 1

        if j == columnnumber:

            j = 0

            i += 1



    plt.show()
plotvalues('taster_name')
df_reg = df_wines[df_wines['region_1'].isin(df_wines['region_1'].value_counts()[df_wines['region_1'].value_counts() > 1000].nlargest(100).index)]

df_reg['region_1'].value_counts().nlargest(100)
plotvalues ('region_1', df = df_reg, columncountvalues = 'points', subplotcolumnnumber = 4, figuresizebynumberentries = False, figsize = (75,75))
print("Extract years from df." + str('title') + " and count the appearances of these years." '\n')

df_wines['year'] =  df_wines['title'].apply(lambda x: [int(y) for y in re.findall('\d+', x) if int(y) < 2019 and int(y) > 1800])

starttime = time.time()

results = Counter()

df_wines['year'].apply(results.update)

df_result = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index':'name', 0:'count'})



df_wines['year'] = df_wines['year'].apply(lambda x: x[0] if len(x) > 0 else None )
plt.figure()

df_wines.groupby(['year'])['points'].mean().plot(figsize=(40,20), fontsize = 30)

plt.xlabel('Year', fontsize = 30)

plt.ylabel('Mean Points', fontsize = 30)

plt.show()
plt.figure()

df_wines.groupby(['year'])['price'].mean().plot(figsize=(40,20), fontsize = 30)

plt.xlabel('Year', fontsize = 30)

plt.ylabel('Mean Price', fontsize = 30)

plt.show()
plt.figure()

df_wines.groupby(['price'])['points'].mean().plot(figsize=(40,20), fontsize = 30, logx = True)

plt.xlabel('Price', fontsize = 30)

plt.ylabel('Mean Points', fontsize = 30)

plt.show()
df_wines = pd.read_csv('../input/winemag-data-130k-v2.csv').drop(['Unnamed: 0'], axis=1).drop_duplicates()

df_wines['price'] = df_wines['price'].fillna(0)

df_train, df_test = train_test_split(df_wines, test_size=0.2)

del df_wines
wordsToDrop = ['fruit', 'fruits', 'much', '%', 'feel', 'glass', 'drink', 'drinking', 'wines', 'next', 'offers', 'preserved',

              'footprint','touch', 'picture', 'otherwise', 'years', 'generations', 'california', 'prove', 'appeal', 'time', 

              'sit', 'other', 'franc', 'bit', 'value', 'available', 'month', 're-staved', 'alongside', 'district', 'earth', 

              'site', 'small', 'fettuccine', 'new', 'many', 'con', 'opens', 'same', "'08", "can't-miss", 'aÿ', 'big', 'game',

              'pencil', 'meet', 'such', 'front', 'center', 'lot', 'makeup', 'delivers', 'close', 'lots', 'take', 'more', 'two-thirds',

              'one-third', 'now-2018', 'most', 'smaller', 'amounts', 'le', 'add', 'tiny', 'it´s', 'next', '4-6', 'prune—no', 'now–2020',

              'road', 'half', 'newest', '“', '”', '’', 'points', 'price']
def mem_usage(pandas_obj):

    """

    Function which gives you the memory usage of a pandas_obj

    """

    if isinstance(pandas_obj,pd.DataFrame):

        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else: # we assume if not a df it's a series

        usage_b = pandas_obj.memory_usage(deep=True)

    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes

    return "{:03.2f} MB".format(usage_mb)



def AddCol_DropCol(df, columnname, threshold_cor = 0.01, columnCorrWith = 'points'):

    """

    This function first add a column for each unique value of df.columnname to the dataframe df and 

    removes the columns which have a correlation less than threshold_cor with df.columnCorrWith

    """

    to_drop = []

    for name in df[columnname].unique():

        df[name] = df[columnname].apply(lambda x: 1 if x == name else 0)

        abscorr = abs(df[name].corr(df[columnCorrWith]))

        if abscorr <= threshold_cor:

            to_drop.append(name)

        else:

            df[name] = df[name].astype('int8')     

    df = df.drop(to_drop, axis=1)

    return df



def get_redundant_pairs(df):

    """

    Get diagonal and lower triangular pairs of correlation matrix

    """

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    """

    Returns top n correlation of columns of dataframe df and its correlation matrix

    """

    df = df.select_dtypes(exclude=['object'])

    stacked_au_corr = df.corr()

    au_corr = stacked_au_corr.abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop, errors = 'ignore').sort_values(ascending=False)

    return au_corr[0:n], stacked_au_corr



def PreProc(df, dropLessThan = 500, wordsToDrop= wordsToDrop, columnnameY = 'points', columnnameDescr = 'description',

            columnnameLookYears = 'title', columnnamePrice = 'price', BoolTrain = True, Booldropobj=True,

            corrLessDrop = 0.01, corrMoreDrop = 0.8, 

            dictFeatToAdd = {'province': 0.1, 'taster_name': 0.1}, columnsDfTrain = df_train.columns):

    

    """

    Does the whole preprocessing of a dataframe.

    1.) Select nouns and adjectives from df.columnnameDescr and add new columns to df for words which appear > dropLessThan

    2.) Select years from df.columnnameLookYears and add new columns to df for words which appear > dropLessThan

    3.) Fills nans in df.columnnamePrice with 0s

    4.) Convert columns in smaller formats where possible, e.g. int64 -> int8

    5.) Drops 'object' columns

    6.) Drops one of two columns when correlation is more than corrMoreDrop

    7.) Drops columns when correlation of that column and df.columnnameY is less than corrLessDrop

    8.) Rearrange columns, s.t. df.columnnameY is the first column

    8.1.) Drop / Add columns such that the columns of the df fit to the columns of columnsDfTrain

    """

    

    print("Start Preprocessing" + '\n')

    print("Select nouns and adjectives from df." + columnnameDescr + '\n')

    

    df[columnnameDescr] = df[columnnameDescr].str.lower()

    starttime = time.time()

    df['des'] = df[columnnameDescr].apply(lambda x: [word for (word,pos) in nltk.pos_tag(nltk.word_tokenize(x)) if pos[0] in ['N','J']])

    endtime = time.time()

    print("required time: " + str(endtime - starttime) + '\n')

    gc.collect()

    

    print("Count the appearance of these words."  + "\n")

    starttime = time.time()

    results = Counter()

    df['des'].apply(results.update)

    df_result = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index':'name', 0:'count'})

    df_result = df_result[df_result['count'] >= dropLessThan]

    endtime = time.time()

    print("required time: " + str(endtime - starttime) + '\n')

    gc.collect()



    print("Create columns for words which appear at least " + str(dropLessThan) + "times" + " and which are not in wordsToDrop\n")

    df_result = df_result[~df_result['name'].isin(wordsToDrop)]

    

    starttime = time.time()

    for item in df_result['name'].values:

        df[item] = df[columnnameDescr].apply(lambda x: 1 if item in x else 0)

    endtime = time.time()

    print("required time: " + str(endtime - starttime) + '\n')

    gc.collect()



    print("Extract years from df." + str(columnnameLookYears) + " and count the appearances of these years." '\n')

    df['year'] =  df[columnnameLookYears].apply(lambda x: [int(y) for y in re.findall('\d+', x) if int(y) < 2019 and int(y) > 1800])

    starttime = time.time()

    results = Counter()

    df['year'].apply(results.update)

    df_result = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index':'name', 0:'count'})

    

    df_result = df_result[df_result['count'] >= dropLessThan]

    endtime = time.time()

    print("required time: " + str(endtime - starttime) + '\n')

    gc.collect()

    

    print("Create columns for years which appear at least " + str(dropLessThan) + "times"  + "\n")

    starttime = time.time()

    for item in df_result['name'].values:

        df[item] = df['year'].apply(lambda x: 1 if item in x else 0)

    endtime = time.time()

    print("required time: " + str(endtime - starttime) + '\n')

    gc.collect()



    

    del results

    del df_result

    print("Memory usage of the df is: " + "\n")

    print(df.info(memory_usage='deep'))

    print("Start to converte columns into smaller data formats" + "\n")

    df[columnnamePrice] = df[columnnamePrice].fillna(0).astype('int64')

    

    gl_int = df.select_dtypes(include=['int'])

    converted_int = gl_int.apply(pd.to_numeric,downcast='signed')

    

    print("Memory usage of all columns of format intx before: " + str(mem_usage(gl_int)) + "\n")

    print("Memory usage of all columns of format intx after:" + str(mem_usage(converted_int)) +"\n")

    

    print("Start to replace old columns with new columns" + "\n")

    starttime = time.time()

    

    df[list(converted_int.columns)] = converted_int

    endtime = time.time()

    print("required time: " + str(endtime - starttime) + '\n')

    gc.collect()

    

    print("Actual memory usage of df" + "\n")

    print(df.info(memory_usage='deep'))

    

    if BoolTrain:

        print("Start to drop columns when correlation with df." + str(columnnameY) + " is smaller than " + str(corrLessDrop) + "\n")

        corr_points = df.corrwith(df[columnnameY])

        to_drop = corr_points.index[corr_points.abs() < corrLessDrop].tolist()

        df = df.drop(to_drop, axis=1, errors = 'ignore')

        print("Add columns to df according to dictFeatToAdd")

        for key in dictFeatToAdd:

            df = AddCol_DropCol(df, key, threshold_cor = dictFeatToAdd[key], columnCorrWith = columnnameY)

            

    else:

        print("Add columns to df according to dictFeatToAdd" + '\n')

        print("No columns will be dropped, since BoolTrain = False" + '\n')

        for key in dictFeatToAdd:

            df = AddCol_DropCol(df, key, threshold_cor = 0, columnCorrWith = columnnameY) 

    

    gc.collect()

    if Booldropobj:

        print("Drop columns with dtype 'object'" + "\n")

        df = df.select_dtypes(exclude=['object'])

        print("Memory usage of the df is: " + "\n")

        print(df.info(memory_usage='deep'))

    

    if BoolTrain:

        print("Compute correlation matrix of df" + "\n")

        starttime = time.time()

        top_corr, corr_matr = get_top_abs_correlations(df, 100)

        endtime = time.time()

        print("required time: " + str(endtime - starttime) + '\n')

        print("Top Absolute Correlations")

        print(top_corr)

        print("Drop one of two columns whose correlation is larger than " + str(corrMoreDrop) + "\n")

        c1 = corr_matr.abs()

        upper = c1.where(np.triu(np.ones(c1.shape), k=1).astype(np.bool))

        to_drop = [column for column in upper.columns if any(upper[column] > corrMoreDrop)]

        df = df.drop(to_drop, axis=1)

        print("Memory usage of the df is: " + "\n")

        print(df.info(memory_usage='deep'))

        print("Last Step: Rearrange columns")

        a = list(df.drop(columnnameY, axis = 1).columns)

        a = [columnnameY] + a

        df = df[a]

        del a

        return df

    

    else:

        print("Last Step: Rearrange columns")

        a = list(set(df.columns).intersection(set(columnsDfTrain)))

        df = df[a]

        

        a = list(set(columnsDfTrain).difference(set(df.columns)))

        for name in a:

            df[name] = 0

        

        a = list(columnsDfTrain.drop(columnnameY))

        a = [columnnameY] + a

        df = df[a]

        del a

        return df
df_train = PreProc(df_train, columnsDfTrain = df_train.columns)

gc.collect()
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesClassifier



from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import median_absolute_error



def BoundResult(x):

    if x > 100:

        y = 100

    elif x < 0:

        y = 0

    else:

        y = x

    return y



def Feat_Imp(df, max_depth_ExTreeReg = 42, n_estimators_ExTreeReg = 500, max_features_ExTreeReg = "sqrt", n_jobs_ExTreeReg = -1, featureImpDropSmallerThan = 0.0004):

    print("Use ExtraTreesRegressor to select important features" + "\n")

    array_all = df.values

    X_all = array_all[:,1:]

    y_all = array_all[:,0]

    

    X_trainETR, X_testETR, y_trainETR, y_testETR = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    

    ETR = ExtraTreesRegressor(max_depth= max_depth_ExTreeReg, n_estimators=n_estimators_ExTreeReg, max_features=max_features_ExTreeReg, n_jobs = n_jobs_ExTreeReg)

    print("Start fitting of ExtraTreesRegressor"+ "\n")

    starttime = time.time()

    ETR.fit(X_trainETR, y_trainETR)

    endtime = time.time()

    print("required time: " + str(endtime - starttime) + '\n')

    

    pred = list(map(BoundResult,ETR.predict(X_testETR)))

    

    

    print('Mean Squared Error: ' + str(mean_squared_error(pred, y_testETR)) + '\n' +

          'Mean Absolute Error: ' + str(mean_absolute_error(pred, y_testETR)) + '\n' +

          'Median Absolute Error: ' + str(median_absolute_error(pred, y_testETR))

         )

    

    del X_trainETR, X_testETR, y_trainETR, y_testETR

    

    print("Drop features whose importance is smaller than " + str(featureImpDropSmallerThan))

    df_featureImp = pd.DataFrame(index=df.columns[1:])

    df_featureImp['Importance'] = ETR.feature_importances_

    df_featureImp.sort_values(by='Importance')

    

    to_drop = df_featureImp.index[df_featureImp['Importance'] < featureImpDropSmallerThan].tolist()

    df = df.drop(to_drop, axis=1, errors = 'ignore')

    df.info(memory_usage='deep')

    del df_featureImp, ETR

    return df
df_train = Feat_Imp(df_train)

gc.collect()
from sklearn import linear_model



def TrainLRM(df):

    array_train = df.values

    X_train = array_train[:,1:]

    y_train = array_train[:,0]

    

    starttime = time.time()

    lrm = linear_model.LinearRegression().fit(X_train, y_train)

    endtime = time.time()

    print(endtime - starttime)

    

    return lrm
lrm = TrainLRM(df_train)
df_test = PreProc(df_test, BoolTrain = False, columnsDfTrain = df_train.columns)
def CheckLRM(df, lrm):

    array_test = df.values

    X_test = array_test[:,1:]

    y_test = array_test[:,0]

    starttime = time.time()

    pred = list(map(BoundResult,lrm.predict(X_test)))

    endtime = time.time()

    print(endtime - starttime)

    

    print('Mean Squared Error: ' + str(mean_squared_error(pred, y_test)) + '\n' +

          'Mean Absolute Error: ' + str(mean_absolute_error(pred, y_test)) + '\n' +

          'Median Absolute Error: ' + str(median_absolute_error(pred, y_test))

     )

    

    return pred, y_test
pred, y_test = CheckLRM(df_test, lrm)
test = np.around(abs(y_test - pred), decimals=1)

df_errors = pd.DataFrame(test, columns = ['AbsErrors'])

xticks = np.arange(0, max(test) + 2, step=2)

df_errors['AbsErrors'].value_counts().sort_index().plot(xticks = xticks, 

                                                        logy = False, 

                                                        kind = 'bar', 

                                                        figsize=(40,20), 

                                                        color = 'r',

                                                       fontsize = 20)
df_errors['AbsErrors'].value_counts().sort_index().plot(xticks = xticks, 

                                                        logy = True, 

                                                        kind = 'bar', 

                                                        figsize=(40,20), 

                                                        color = 'r',

                                                       fontsize = 20)
test = np.around(abs(y_test - pred), decimals=0)

df_errors = pd.DataFrame(test, columns = ['AbsErrors'])

xticks = np.arange(0, max(test) + 2, step=2)

df_errors['AbsErrors'].value_counts().sort_index().plot(xticks = xticks, 

                                                        logy = False, 

                                                        kind = 'bar', 

                                                        figsize=(40,20), 

                                                        color = 'r',

                                                       fontsize = 20)
df_wines = pd.read_csv("../input/winemag-data-130k-v2.csv").drop(['Unnamed: 0'], axis=1).drop_duplicates()

df_wines['price'] = df_wines['price'].fillna(0)

df_train, df_test = train_test_split(df_wines[df_wines['taster_name'] == 'Anne Krebiehl\xa0MW'], test_size=0.2)

print(df_wines['taster_name'].unique())

del df_wines
df_train = PreProc(df_train, columnsDfTrain = df_train.columns)

gc.collect()

df_train = Feat_Imp(df_train)

gc.collect()

lrm = TrainLRM(df_train)



df_test = PreProc(df_test, BoolTrain = False, columnsDfTrain = df_train.columns)

pred, y_test = CheckLRM(df_test, lrm)
df_wines = pd.read_csv("../input/winemag-data-130k-v2.csv").drop(['Unnamed: 0'], axis=1).drop_duplicates().set_index('title', drop = False)

df_wines['price'] = df_wines['price'].fillna(0)

gc.collect()
df_wines = PreProc(df_wines, dropLessThan = len(df_wines) * 0.005, columnsDfTrain = df_train.columns)

gc.collect()
df_ownRatings = pd.DataFrame(index = df_wines.index)

df_ownRatings['points'] = np.nan

df_ownRatings = df_ownRatings.sample(23)

df_ownRatings = df_ownRatings[~df_ownRatings.index.duplicated(keep='first')]

df_ownRatings['points'] = df_ownRatings['points'].apply(lambda x: round(np.random.normal(np.mean(df_wines['points']), np.std(df_wines['points']))))

print(df_ownRatings.head)
a = list(df_wines.drop('points', axis = 1).columns)

a = ['points'] + a

df_wines = df_wines[a]

del a



df_help = df_wines.loc[list(df_ownRatings.index), :]

for name, row in df_help.iterrows():

    df_help.loc[name, 'points'] = df_ownRatings.loc[name, 'points'].copy()

    

array_all = df_help.loc[list(df_ownRatings.index), :].values

X_all = array_all[:,1:]

y_all = array_all[:,0]



X_trainETR, X_testETR, y_trainETR, y_testETR = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
df_help = Feat_Imp(df_help.loc[list(df_ownRatings.index), :])

lrm = TrainLRM(df_help)

X_wines = df_wines.loc[:, list(df_help.columns)].values[:,1:]
PredRatings = X_wines.dot(lrm.coef_)

ind = np.argpartition(PredRatings, -5)[-5:]

print(list(df_wines.iloc[ind, :].index))