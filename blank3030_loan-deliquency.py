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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling
train = pd.read_csv('../input/hackathon/train.csv')

test = pd.read_csv('../input/new-test/test.csv')
train.describe()
X_train = train.iloc[:, :-1]

y_train = train.iloc[:, -1]
X_train = X_train.drop(columns = ['loan_id','financial_institution', 'loan_purpose'])
test = test.drop(columns = ['financial_institution', 'loan_purpose'])
X_train['source'].unique()
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
def plotCorrelationMatrix(df, graphWidth):

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for give dataframe', fontsize=15)

    plt.show()
def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
X_train['first_payment_date'].unique()
test['first_payment_date'].unique()
test['first_payment_date'] = test['first_payment_date'].map({'Apr-12': '04/2012', 'Mar-12':'03/2012', 'May-12': '05/2012', 'Feb-12':'02/2012'})
X_train['origination_date'].unique()
test['origination_date'].unique()
test['origination_date'] = test['origination_date'].map({'01/02/12': '2012-02-01', '01/01/12': '2012-01-01', '01/03/12': '2012-03-01'})
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X_train['source'] = le.fit_transform(X_train['source'])

test['source'] = le.transform(test['source'])
le1 = LabelEncoder()

X_train['first_payment_date'] = le1.fit_transform(X_train['first_payment_date'])

test['first_payment_date'] = le1.transform(test['first_payment_date'])
le2 = LabelEncoder()

X_train['origination_date'] = le2.fit_transform(X_train['origination_date'])

test['origination_date'] = le2.transform(test['origination_date'])
X_train['to_pay'] = X_train['unpaid_principal_bal'] + (X_train['unpaid_principal_bal']*X_train['loan_term']*X_train['interest_rate'])/100

test['to_pay'] = test['unpaid_principal_bal'] + (test['unpaid_principal_bal']*test['loan_term']*X_train['interest_rate'])/100
X_train['loan%'] = X_train['loan_to_value']/(1/X_train['debt_to_income_ratio'])/X_train['insurance_percent']

test['loan%'] = test['loan_to_value']/(1/test['debt_to_income_ratio'])/test['insurance_percent']
X_train['interest'] = (X_train['unpaid_principal_bal']*X_train['loan_term']*X_train['interest_rate'])/36000

test['interest'] = (test['unpaid_principal_bal']*test['loan_term']*X_train['interest_rate'])/36000
X_train['comp_interest'] = X_train['unpaid_principal_bal']*((1 + (X_train['interest_rate']/12))**(X_train['loan_term']/360))

test['comp_interest'] = test['unpaid_principal_bal']*((1 + test['interest_rate']/12)**(test['loan_term']/360))
from xgboost import XGBClassifier
remove = ['loan%']
from sklearn.preprocessing import StandardScaler
X_train['m13'] = y_train
X_train['m13'].value_counts()
target = 'm13'

IDcol = ['loan_id']

from sklearn import model_selection, metrics
clf = XGBClassifier()
predictors = [x for x in X_train.columns if x not in [target]+IDcol + remove]
X_train[predictors] = X_train[predictors].astype(np.float64)
predictors
clf.fit(X_train[predictors], X_train[target])
train_pred = clf.predict(X_train[predictors])
n = X_train.shape[0]

p = X_train.shape[1] - 1
from sklearn.metrics import r2_score

r2 = r2_score(X_train['m13'], train_pred)

adj_r2 = 1 - ((1-r2)*((n-1)/(n-p-1)))

print(adj_r2)
test[target] = clf.predict(test[predictors])
IDcol.append(target)
submission = pd.DataFrame({x: test[x] for x in IDcol})
submission['m13'].sum()
submission.to_csv('alg0.csv', index = False)
featimp = pd.Series(clf.feature_importances_,index= predictors).sort_values(ascending=False)

featimp.plot(kind='bar', title='Feature Importances')

plt.ylabel('Feature Importance Score')
Count_paid_del = len(X_train[X_train['m13'] == 0])

Count_unpaid_del = len(X_train[X_train['m13'] == 1])

percentage_of_paid_del = Count_paid_del/(Count_paid_del+Count_unpaid_del)

print('percentage of paid delequency is',percentage_of_paid_del*100)
def undersample(df, times):

    unpaid_indices = np.array(df[df.m13 == 1].index)

    paid_indices = np.array(df[df.m13 == 0].index) 

    paid_indices_undersample = np.array(np.random.choice(paid_indices, (times*Count_unpaid_del), replace = False))

    undersample_data = np.concatenate([unpaid_indices, paid_indices_undersample])

    undersample_data = df.iloc[undersample_data, :]

    return(undersample_data)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, classification_report
predictors = [x for x in X_train.columns if x not in [target]+IDcol + remove]
from sklearn.model_selection import cross_validate 

from sklearn.model_selection import learning_curve,GridSearchCV
import xgboost as xgb
def modelfit(alg, train, test, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(train[predictors].values, label=train[target].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='error', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(train[predictors], train[target],eval_metric='error')

        

    #Predict training set:

    train_predictions = alg.predict(train[predictors])

    train_predprob = alg.predict_proba(train[predictors])[:,1]

        

    #Print model report:

    print ("\nModel Report")

    print ("Accuracy : %.4g" % metrics.accuracy_score(train[target].values, train_predictions))

    print ("AUC Score (Train): %f" % metrics.roc_auc_score(train[target], train_predprob))

    featimp = pd.Series(alg.feature_importances_,index= predictors).sort_values(ascending=False)

    featimp.plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')
i = 25

train_undersample = undersample(X_train, i)

test_undersample = undersample(test, i)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from vecstack import stacking
predictors
col = X_train.columns

for col in predictors:

    print(max(X_train[col]))
sc = StandardScaler()

X_train[predictors] = sc.fit_transform(X_train[predictors])

test[predictors] = sc.transform(test[predictors])
import keras

from keras.models import Sequential

from keras.layers import Dense



classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 28))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train[predictors], X_train[target], batch_size = 100, epochs = 100)
test.shape
test[target] = classifier.predict(test[predictors])

test[target] = (test[target] > 0.19)

IDcol.append(target)

subm = pd.DataFrame({x: test[x] for x in IDcol})

print(subm['m13'].sum())

subm.to_csv('XGB35S.csv', index = False)