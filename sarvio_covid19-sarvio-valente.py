# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import librares

import matplotlib.pyplot as plt

import seaborn as sns

#import dataset and pre-visualize it.

data = pd.read_excel('/kaggle/input/covid19/dataset.xlsx', sheet_name='All')

data.head()
# percentage of negative and postive

count1 = data.loc[data[data.columns[2]]== 'negative'][data.columns[2]].shape[0]

count2 = data.loc[data[data.columns[2]]== 'positive'][data.columns[2]].shape[0]

print('negative values is :', round(count1*100/(count1+count2), 1), '% in dataset')

print('positive values is :', round(count2*100/(count1+count2), 1), '% in dataset')
#copy dataset for df dataframe

df = data.copy()



#define a dictionary to storage original columns names

col_names ={}



#define first 3 column manually

columns =['patient', 'f1', 'result' ]



#save first 3 column original names in dictionary

col_names = {'patient': data.columns[0], 'f1': data.columns[1], 'result': data.columns[2]}



#define new columns names as f2, f3,...and save original name in dictionary

for i in range(3, 111):

    new_name = 'f' + str(i-1)

    columns.append(new_name)

    col_names.update( {new_name : data.columns[i]})



#change columns names

df.columns = columns



#remove patient column from df dataframe

df = df.drop('patient', axis = 1)



#result

df.head()
#change value of result columns as negative to 0, and positive to 1

df['result'] = [0 if a == 'negative' else 1 for a in df['result'].values]

df.head()
#modify some NAN value to -99 and change string value to numbers



#to make some change possible I changed null value to -99

df = df.replace(np.nan, -99)



df = df.replace('not_detected', 0)

df = df.replace('detected', 1)

df = df.replace('negative', 0)

df = df.replace('absent', 0)

df = df.replace('normal', 0)

df = df.replace('Ausentes', 0)

df = df.replace('clear', 0)

df = df.replace('positive', 1)

df = df.replace('present', 1)

df = df.replace('not_done', -99) #it's look like null value



df.iloc[:,87] = df.iloc[:,87].replace('light_yellow', 0)

df.iloc[:,87] = df.iloc[:,87].replace('yellow', 1)

df.iloc[:,87] = df.iloc[:,87].replace('light_yellow', 2)

df.iloc[:,87] = df.iloc[:,87].replace('citrus_yellow', 3)

df.iloc[:,87] = df.iloc[:,87].replace('orange', 4)



df.iloc[:, 81] = df.iloc[:, 81].replace('<1000', '1000')

df.iloc[:, 81] = df.iloc[:, 81].replace('', '1000')

df.iloc[:, 81] = df.iloc[:, 81].astype('int32')



df.iloc[:, 71] = df.iloc[:, 71].replace('cloudy', 0)

df.iloc[:, 71] = df.iloc[:, 71].replace('lightly_cloudy', 1)

df.iloc[:, 71] = df.iloc[:, 71].replace('lightly', 2)

df.iloc[:, 71] = df.iloc[:, 71].replace('altered_coloring', 3)



df.iloc[:, 82] = df.iloc[:, 82].replace('Urato Amorfo --+', 0)

df.iloc[:, 82] = df.iloc[:, 82].replace('Urato Amorfo +++', 1)

df.iloc[:, 82] = df.iloc[:, 82].replace('Oxalato de Cálcio -++', 2)

df.iloc[:, 82] = df.iloc[:, 82].replace('Oxalato de Cálcio +++', 3)



df.iloc[:, 72] = df.iloc[:, 72].replace('Não Realizado', 0)

df.iloc[:, 72] = df.iloc[:, 72].astype('float')



#change back -99 to null value

df = df.replace(-99, np.nan)
#The result

df.head()
#remove f2, f3, f4, leave just result and lab

df = df.drop(['f2', 'f3', 'f4'], axis = 1)

df.head()
#change NULL to -99

df = df.replace(np.nan, -99)



#create a list for storage feature filtered.

features_filtered = list()

features_filtered.append('result')



#get columns of the dataset

cols = list(df.columns)



#remove feature result of the cols list.

cols.remove('result')



#starting to filtering feature by feature

for name in cols:

    #get count1 - number of valid values of actural feature for positive samples 

    result1 = df.loc[(df[name] != -99) & (df['result'] == 1)]

    count1 = len(result1.loc[:,[name, 'result']])

    

    #get count2 - count2 = number of valid values of actural feature for negative samples

    result2 = df.loc[(df[name] != -99) & (df['result'] == 0)]

    count2 = len(result2.loc[:,[name, 'result']])

    

    if (count1 + count2) > 0:                       #first filter

        if (count1*100/(count1 + count2) >= 9):    #second filter

            if (count2 >= 500):                     #third filter

                features_filtered.append(name)      #storage name of feature

                



#after correlation analyzis I decided to remove  f9 because it is correlatated with f5

features_filtered.remove('f9')



#change -99 to NULL values

df = df.replace(-99, np.nan)

print(features_filtered)
#change -99 values to null value to remove it

df = df.replace(-99,np.nan)



#create new Dataframe with selected features

df_filtered = df[features_filtered]



#remove null values - whole line

df_filtered = df_filtered.dropna()



#results

df_filtered.head(), df_filtered.shape
df_filtered.head()
# percentage of negative and postive

count1 = df_filtered.loc[df_filtered['result']== 0]['result'].shape[0]

count2 = df_filtered.loc[df_filtered['result']== 1]['result'].shape[0]

print('negative values is :', round(count1*100/(count1+count2), 1), '% in dataset')

print('positive values is :', round(count2*100/(count1+count2), 1), '% in dataset')
#import library

from sklearn.model_selection import train_test_split



#copy dataframe to X and y

X = df_filtered.copy()

X = X.drop('result', axis = 1)

y = df_filtered.loc[:,'result']



#split dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state = 0)



#reset index

X_train.reset_index(drop = True, inplace = True)

X_test.reset_index(drop = True, inplace = True)

y_train.reset_index(drop = True, inplace = True)

y_test.reset_index(drop = True, inplace = True)



#train data set with results

train_dataset = X_train.copy()

train_dataset['result'] = y_train
import matplotlib.pyplot as plt

import numpy as np

train_dataset.hist(bins=10, figsize= (20,15))

plt.show()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



#normalize basend on train data (X_train)

X_t = scaler.fit_transform(X_train)

X_t = pd.DataFrame(X_t)

X_t.columns = X_train.columns

X_train = X_t.copy()



#normalize test data basend on train data (X_train)

X_t = scaler.transform(X_test)

X_t = pd.DataFrame(X_t)

X_t.columns = X_test.columns

X_test = X_t.copy()



#show histogram for all features

X_train.hist(bins=10, figsize= (20,15))

plt.show()
#create a new dataframe with just train data include result one it. It for correlation.

train_dataset = X_train.copy()

train_dataset['result'] = y_train



#correlation graphic

corr = train_dataset.corr('pearson', 2)

f, ax = plt.subplots(figsize=(24, 18))

sns.heatmap(corr, vmax=.8, square=True, cmap='RdYlBu')

corr.head(50)
import seaborn as sns



n=len(train_dataset.columns)

fig,ax = plt.subplots(n,1, figsize=(6,n*2), sharex=True)

for i in range(n):

    plt.sca(ax[i])

    col = train_dataset.columns[i]

    sns.violinplot(x='result', y=col, data=train_dataset, split=False)

#Features combinations (Thanks Marcio filho for this code)

from itertools import combinations

combos_ = []

for f1,f2 in combinations(X_train.columns, 2):

    f1_ = X_train[f1].corr(y_train)

    f2_ = X_train[f2].corr(y_train)

    f1_f2 = ((X_train[f2] - X_train[f1])).corr(y_train)

    best_single = max(abs(f1_), abs(f2_))

    combo_score = abs(f1_f2)

    

    res = dict()

    res['f1'] = f1

    res['f2'] = f2

    res['f1_'] = f1_

    res['f2_'] = f2_

    res['f1_f2_'] = f1_f2 

    res['f1_f2'] = combo_score - best_single

    combos_.append(res)

df_combos = pd.DataFrame(combos_).sort_values("f1_f2", ascending=False)

df_combos = df_combos[df_combos['f1_f2'] > 0.01]

print(df_combos.shape)

df_combos.head(5)
#create new feature in train data

X_train['f15_f17'] = X_train['f17'] - X_train['f15']

X_train['f14_f16'] = X_train['f16'] - X_train['f14']



#create new feature in test data

X_test['f15_f17'] = X_test['f17'] - X_test['f15']

X_test['f14_f16'] = X_test['f16'] - X_test['f14']



#updata dictionary of original names features

new_name = 'f15_f17'

col_names.update( {new_name : col_names['f15'] + ' and ' + col_names['f17']})



new_name = 'f14_f16'

col_names.update( {new_name : col_names['f14'] + ' and ' + col_names['f16']})
from lightgbm import LGBMClassifier

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from skopt import gp_minimize



def my_kappa (model, X, y):

    y_pred = model.predict(X)

    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(y, y_pred)

    return kappa



def train_model(params):

    

    model = LGBMClassifier(learning_rate= params[0], num_leaves= params[1], min_child_samples=params[2],

                           subsample=params[3], colsample_bytree=params[4], n_estimators=params[5], random_state=0,

                           subsample_freq=1)



    scores = cross_val_score(estimator = model, X = X_train, y = y_train, scoring= my_kappa, cv = 5, verbose = 0)



    mean_score = np.mean(scores)



    return -mean_score



params =     [(1e-3, 1e-1, 'log-uniform'), # learning rate

              (2, 128),                    # num_leaves

              (1, 100),                    # min_child_samples

              (0.05, 1.0),                 # subsample

              (0.1, 1.0),                  # colsample bytree

              (100, 1000)]                 # number of tree



#search for better params

opt = gp_minimize(train_model, params, random_state=0, verbose=1, n_calls=50, n_random_starts=10)



print ("best kappa", opt.fun)

print("best params", opt.x)



model = LGBMClassifier(learning_rate= opt.x[0], num_leaves= opt.x[1], min_child_samples=opt.x[2],

                           subsample=opt.x[3], colsample_bytree=opt.x[4], n_estimators=opt.x[5], random_state=0,

                           subsample_freq=1)



scores = cross_val_score(estimator = model, X = X_train, y = y_train, scoring= my_kappa, cv = 5, verbose = 1)



model.fit(X_train, y_train)



mean_score = np.mean(scores)



print("k-folds kappa ", scores)

print("mean kappa CV", mean_score)
#confusion matrix with Train data

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report





y_pred = model.predict(X_test)



print('summary report')

print(classification_report(y_test, y_pred))

print('confusion matrix')

mc_test = confusion_matrix(y_test, y_pred)

print(mc_test)

print('kappa coeficient')

kappa = cohen_kappa_score(y_test, y_pred)

print(kappa)

print('ROC_AUC score')

roc_auc = roc_auc_score(y_test, y_pred)

print(roc_auc)

#confusion matrix with Train data

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report





y_proba = model.predict_proba(X_test)

y_pred = np.where(y_proba[:,0]> 0.999999, 0,1)



print('summary report')

print(classification_report(y_test, y_pred))

print('confusion matrix')

mc_test = confusion_matrix(y_test, y_pred)

print(mc_test)

print('kappa coeficient')

kappa = cohen_kappa_score(y_test, y_pred)

print(kappa)

print('ROC_AUC score')

roc_auc = roc_auc_score(y_test, y_pred)

print(roc_auc)
cols = list(X_test.columns)

original_names = list()

for name in cols:

    original_names.append(col_names[name])

X_test.columns = original_names

X_test.head()
import shap

shap.initjs()

# use Kernel SHAP to explain test set predictions

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)