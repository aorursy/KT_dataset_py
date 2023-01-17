# Importing the required packages:

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# importing train and test data sets and 

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
testRecordCount = test.shape[0]

trainRecordCount = train.shape[0]
# Dimentions of Size of the train data

print('This dataset contain',train.shape[0],'rows and',train.shape[1],'columns')
#Lets describe the data

train.describe()
columdatatypes = pd.DataFrame({'Feature': train.columns , 'Data Type': train.dtypes.values})
## Fixing -1 with NaN values

train_withNull = train.replace(-1, np.NaN)

test_withNull = test.replace(-1, np.NaN)
# Listing columns which contain null values

NullColumns = train_withNull.isnull().any()[train_withNull.isnull().any()].index.tolist()

NullColumns
# Heat map of null value columns in the data 

#In the data, NULL values have been coded as -1

plt.figure(figsize=(10,3))

sns.heatmap(train_withNull[NullColumns].isnull().astype(int), cmap='viridis')
# percentage of values that are null in each column

print((train_withNull[NullColumns].isnull().sum()/train_withNull[NullColumns].isnull().count())*100)

print((test_withNull[NullColumns].isnull().sum()/test_withNull[NullColumns].isnull().count())*100)
#We can feed these values with the median values of these columns

train_median_values = train_withNull.median(axis=0)

test_median_values = test_withNull.median(axis=0)

train_NoNull = train_withNull.fillna(train_median_values, inplace=False)

test_NoNull = test_withNull.fillna(test_median_values, inplace=False)
# HEat map after replacing all NULL values with the corresponding column medians

plt.figure(figsize=(10,4))

sns.heatmap(train_NoNull.isnull(), cmap='viridis')
#Segregating binary, categorical and continuous columns 

CatColumns = [c for c in train_NoNull.columns if c.endswith("cat")]

BinColumns = [c for c in train_NoNull.columns if c.endswith("bin")]

ContColumns = [c for c in train_NoNull.columns if (c not in CatColumns and c not in BinColumns) ]

print('# of categorical columns =',len(CatColumns))

print('# of Binary columns =',len(BinColumns))

print('# of Continuous columns =',len(ContColumns))

#Analysing Binary featuresns:

plt.figure(figsize=(9,5))

for i,c in enumerate(BinColumns):

    ax = plt.subplot(3,7,i+1)

    sns.countplot(train_NoNull[c],orient ='v')
#Analysing output variable 'target:

plt.figure(figsize=(9,5))

sns.countplot(train_NoNull['target'],orient ='v',)
# % of true values

((train_NoNull['target']==1).sum()/(train_NoNull['target']==1).count())*100
#Within continuous variables, there are many different groups denoted by tags 'ind','reg', 'car' and calc. LEle

#analyse those groups separately

indContColumns = [c for c in ContColumns if c.find('ind')!=-1]

regContColumns = [c for c in ContColumns if c.find('reg')!=-1]

carContColumns = [c for c in ContColumns if c.find('car')!=-1]

calcContColumns = [c for c in ContColumns if c.find('calc')!=-1]
print('# of independent continuous columns =',len(indContColumns))

print('# of reg continuous columns=',len(regContColumns))

print('# of car continuous columns',len(carContColumns))

print('# of calculated continuous columns',len(calcContColumns))
# Check for correlation between various continuous columns

plt.figure(figsize=(10,5))

sns.heatmap(train_NoNull[ContColumns].corr(), annot  = False,cmap= plt.cm.inferno)
#Plotting count of individual categories in each category attribute

plt.figure(figsize=(15,10))

for i,c in enumerate(CatColumns):

    ax = plt.subplot(4,4,i+1)

    sns.countplot(train_NoNull[c],orient ='v')
# Let's deep dive into ps_car_11_cat attribute as it has a large number of categories

plt.figure(figsize=(20,5))

ax = plt.subplot()

sns.countplot(train_NoNull['ps_car_11_cat'],orient ='v')
# Let's look at the top 20 categories in 'ps_car_11_cat' attribute

train_NoNull['ps_car_11_cat'].value_counts().head(20).plot(kind='bar')
# Lets convert categorical attributes to their corresponding dummy variables by one hot encoding

train_NoNull_wDummies = pd.get_dummies(train_NoNull,columns = CatColumns,prefix=None, drop_first=True)

test_NoNull_wDummies = pd.get_dummies(test_NoNull,columns = CatColumns,prefix=None, drop_first=True)

train_NoNull_wDummies.head()
# Getting rid of target column to create input train data

X = train_NoNull_wDummies.drop(['target','id'],axis=1)

y = train_NoNull_wDummies['target']
# Divding input train data into train and test dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size =0.3,random_state=10)
# Computing gini coefficient ( Coursey Kaggle)

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation @jit

def eval_gini(y_true, y_prob):

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini
# Importing Xgboost library and generating xgb's internal object Dmatrix

import xgboost as xgb

dX_train = xgb.DMatrix(X_train, label = y_train)

dX_test = xgb.DMatrix(X_test, label = y_test)
# Parameters to be used in xgboost model

param = {}

param['objective'] = 'binary:logistic'

param['eta'] = 0.1

param['silent'] = True

param['max_depth'] = 20

param['subsample'] = 0.8

param['colsample_bytree'] = 0.8

param['eval_metric'] = 'auc'

evallist  = [(dX_test,'eval'), (dX_train,'train')]
# training xgb model on training dataset

model=xgb.train(param, dX_train, 100, evallist, early_stopping_rounds=20, maximize=True, verbose_eval=9)
# Predicting probabilities from the learned xgb model

y_prob = model.predict(dX_test)

eval_gini(y_test, y_prob)
#Plotting ROC AUC curve for the predictions

import matplotlib.pyplot as plt

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_prob)

roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic (ROC Curve)')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
test_prob = model.predict(xgb.DMatrix(test_NoNull_wDummies.drop('id',axis=1)))
#Creating Submission file

sub = pd.DataFrame()

sub['id'] =test['id']

sub['target'] = test_prob

sub.to_csv('xgboost.csv', index=False,float_format='%.2f')