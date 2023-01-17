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
from sklearn.neural_network import MLPClassifier as NNetC

from sklearn.neural_network import MLPRegressor as NNetR

from sklearn.linear_model import LogisticRegression as LogReg

from sklearn.ensemble import RandomForestClassifier as RanFor

from sklearn.ensemble import ExtraTreesClassifier as ExtTre

from sklearn.ensemble import VotingClassifier as Voting

from sklearn.tree import DecisionTreeClassifier as DecTree



from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score



from sklearn.feature_selection import VarianceThreshold



# from statsmodels.stats.outliers_influence import variance_inflation_factor

# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# from scipy import stats



# Feature selection + K-fold cross validation?
df_data = pd.read_csv("/kaggle/input/credit-card-approval-prediction/application_record.csv")

df_result = pd.read_csv("/kaggle/input/credit-card-approval-prediction/credit_record.csv")
# Delete rows with no loans in that month

indexNames = df_result[df_result['STATUS'] == 'X' ].index # Get row indices with STATUS == 'X'

df_result.drop(indexNames, inplace = True) # Delete these row indices from dataFrame



# Change 'STATUS' column to numeric

df_result.loc[df_result['STATUS'] == 'C', 'STATUS'] = '-1'

df_result['STATUS'] = pd.to_numeric(df_result['STATUS'])

df_result['STATUS'] = df_result['STATUS'] + 1



# Group by ID with the 'MONTH' and 'STATUS' being averaged

df_grouped = df_result.groupby(['ID']).mean() # Group by ID

df_grouped.index.name = 'ID'

df_grouped.reset_index(inplace = True)



# Merge df_data and df_result by ID

df = pd.merge(df_data, df_grouped, on = 'ID')



# Drop useless ID column

df.drop(['ID'], axis = 1, inplace = True)



# Convert categorical columns to dummy columns

df = pd.get_dummies(df)



# Shift 'STATUS' column to the back

status = df['STATUS']

df.drop(labels = ['STATUS'], axis = 1,inplace = True)

df.insert(55, 'STATUS', status)



# Write cleaned data to csv file

df.to_csv(path_or_buf = "/kaggle/working/combined_record.csv")

df
def getScaledTrainTestSets(df):

    ncol = df.shape[1]

    

    # Get train-test splits

    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:(ncol - 2)], df['STATUS'], test_size = 0.20, random_state = 0)



    # Transform to mean 0 and unit variance

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test) # scale with the same parameters



    return x_train, x_test, y_train, y_test

  



def doPCA(x_train, x_test):

    pca = PCA(n_components = 0.99)

    pca.fit(x_train)



    originalCol = x_train.shape[1]



    x_train = pca.transform(x_train)

    x_test = pca.transform(x_test)



    newCol = x_train.shape[1]



    print("PCA Transformation: Number of predictors reduced from " + str(originalCol) + " to " + str(newCol) + "\n\n")

    

    return x_train, x_test



def runModelC(df, model):

    df_this = df

    df_this['STATUS'] = round(df_this['STATUS'])

        

    x_train, x_test, y_train, y_test = getScaledTrainTestSets(df_this)

    x_train, x_test = doPCA(x_train, x_test)

    

    # Feature selection: actually not needed here, since we did PCA

    FS = VarianceThreshold(threshold = 0.10)

    x_train = FS.fit_transform(x_train, y_train)

    x_test = FS.transform(x_test)

    

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    

    accuracy = model.score(x_test, y_test)

    f1 = f1_score(y_test, y_pred, average = 'weighted')

    

    print("Accuracy = " + str(accuracy))

    print("F1 Score = " + str(f1))

    

    return accuracy, f1



def runModelR(df, model):

    df_this = df

        

    x_train, x_test, y_train, y_test = getScaledTrainTestSets(df_this)

    x_train, x_test = doPCA(x_train, x_test)



    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    

    y_test = np.around(y_test)

    y_pred = np.around(y_pred)

    

    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average = 'weighted')

    

    print("Accuracy = " + str(accuracy))

    print("F1 Score = " + str(f1))

    

    return accuracy, f1

    
# MLPRegressor: since the 'STATUS' is actually mathematically related (we took the average)

NNR = NNetR()

score = runModelR(df, NNR)
# MLPClassifier

NNC = NNetC(learning_rate_init = 0.01, max_iter = 1000)

score = runModelC(df, NNC)
# LogisticRegressionClassifier

LR = LogReg(solver= 'lbfgs', max_iter = 1000, multi_class = 'multinomial', random_state = 0)

score = runModelC(df, LR)
# RandomForestClassifier

RF = RanFor(n_estimators = 10, bootstrap = False, random_state = 0)

score = runModelC(df, RF)
# DecisionTreeClassifier

DT = DecTree()

score = runModelC(df, DT)
# ExtraTreesClassifier

ET = ExtTre()

score = runModelC(df, ET)
# VotingClassifier

VT = Voting(estimators=[('et', ET), ('rf', RF), ('nnc', NNC)], voting = 'soft') # Similar results to 'hard'

score = runModelC(df, VT)