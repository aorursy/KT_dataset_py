#Email : tariqha@gmail.com 
#This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np

# import metric function
from sklearn.metrics import log_loss

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from warnings import filterwarnings; filterwarnings('ignore')

# for time measurement
import time


RANDOM_SEED = 44
N_FOLDS = 5

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
PATH = '/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/'
train = pd.read_csv(PATH + 'train.csv').drop(columns = 'uid')
test = pd.read_csv(PATH + 'test.csv').drop(columns = 'uid')
submission = pd.read_csv(PATH + 'submit.csv')
print(f'Train shape: {train.shape} Test.shape {test.shape}')
# checking if the data has any NaN values, this might harm the model
print(f'TRAIN \n\n{train.isna().sum()} \n----------------------\
      \nTEST \n\n{test.isna().sum()}')
# find columns with non numerical data
stringColumns = list(train.columns[train.dtypes == 'object'])
print(f'''Non numerical columns: ["{'","'.join(stringColumns)}"]''')

# find columns with numerical data (all the columns not in the stringColumns list)
numericalColumns = list(train.columns[train.dtypes != 'object'])
print(f'''Numerical columns: ["{'","'.join(numericalColumns)}"]''')
# percentage of 0s and 1s in the train set
train['target'].value_counts() / train['target'].shape[0] * 100
#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = "YlGn",
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':11 }
    );
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(train[numericalColumns])
plt.figure(figsize = (15, 15))
sns.pairplot(train[numericalColumns]);
for column in stringColumns:
    plt.figure(figsize = (15, 5))
    plt.xticks(rotation = '80')
    sns.violinplot(x=column, y='target', data=train)
    plt.show()
from sklearn.metrics import log_loss
print(f'''All 1 baseline {log_loss(train['target'].values, np.ones_like(train['target'].values))}''')
print(f'''All 0 baseline {log_loss(train['target'].values, np.zeros_like(train['target'].values))}''')
print(f'''All 0.5 {log_loss(train['target'].values, 0.5 * np.ones_like(train['target'].values))}''')
print(f'''Random baseline {log_loss(train['target'].values, np.random.random(train['target'].values.shape))}''')
# get numerical and categorical features (taken from EDA)
stringColumns = list(train.columns[train.dtypes == 'object'])
numericalColumns = list(train.columns[train.dtypes != 'object'])
# connect train and test datasets for label encoding
test['target'] = np.nan
allData = pd.concat([train, test]).reset_index(drop = True)
# helper function
def doFeatureCombination(col1, col2, f):
    return f(col1.values, col2.values)

plus = lambda c1, c2: c1+c2
minus = lambda c1, c2: c1-c2
def preprocessDataTree(data):
    
    '''
    Function to preprocess data for Decision Tree model
    '''
    
    le = LabelEncoder()
    for column in ['age', 'education-num', 'hours-per-week']:
        data[column+'_binned'] = le.fit_transform(pd.cut(data[column], 10))
    
    # categorical features label encoding
    droppedColumns = []
    for column in stringColumns + ['age_binned', 'education-num_binned', 'hours-per-week_binned']:
        if column != 'target':
            NewFeat = data[
                data['target'].notna()
            ].groupby([column])['target'].agg([
                'mean', 
                'std',
            ]).rename(
                columns={'mean': 'target_mean_' + column, 
                         'std': 'target_std_' + column,
                         'median': 'target_median_' + column,
                        }
            ).fillna(0.0).reset_index()
            data = pd.merge(data, NewFeat, how='left', on = column).drop(columns = column)
            droppedColumns.append(column)
            
    #numerical features linear combinations
    for i1 in range(len(numericalColumns) - 2):
        for i2 in range(i1 + 1, len(numericalColumns) - 1):
            col1, col2 = numericalColumns[i1], numericalColumns[i2]
            # sum of 2 columns
            data[col1 + ' + ' + col2] = doFeatureCombination(data[col1], data[col2], plus)
    
    return data.iloc[:train.shape[0], :], data.iloc[train.shape[0]:, :].reset_index(drop = True).drop(columns = 'target')

def preprocessDataKNN(data):
    '''
    Function to preprocess data for the KNN model
    '''
    
    
    le = LabelEncoder()
    for column in ['age', 'education-num', 'hours-per-week']:
        data[column+'_binned'] = le.fit_transform(pd.cut(data[column], 10))
        
    # target mean encoding
    droppedColumns = []
    for column in stringColumns + ['age_binned', 'education-num_binned', 'hours-per-week_binned']:
        if column != 'target':
            NewFeat = data[
                data['target'].notna()
            ].groupby([column])['target'].agg([
                'mean', 
                'std',
            ]).rename(
                columns={'mean': 'target_mean_' + column, 
                         'std': 'target_std_' + column,
                         'median': 'target_median_' + column,
                        }
            ).fillna(0.0).reset_index()
            data = pd.merge(data, NewFeat, how='left', on = column).drop(columns = column)
            droppedColumns.append(column)
    
    # standard scaling
    sc = StandardScaler()
    data.loc[:, data.columns != 'target'] = sc.fit_transform(data.loc[:, data.columns != 'target'])
    
    return data.iloc[:train.shape[0], :], data.iloc[train.shape[0]:, :].reset_index(drop = True).drop(columns = 'target')
trainDT, testDT = preprocessDataTree(allData)
trainKNN, testKNN = preprocessDataKNN(allData)
# perform a simple KFold crossvalidation with out of fold prediction

kf = KFold(n_splits = N_FOLDS, shuffle = True, random_state=RANDOM_SEED)

X_train_DT, X_test_DT = trainDT.drop(columns = 'target'), testDT
X_train_KNN, X_test_KNN = trainKNN.drop(columns = 'target'), testKNN

y_train = train['target']


treeParams = {
    'criterion': 'gini',
    'splitter':'best',
    'max_depth': 7,
    'min_samples_split': 80,
    'min_samples_leaf': 80,
    'random_state': RANDOM_SEED,
    'max_leaf_nodes': 60,
    'class_weight': {0:1, 1:1}
}



knnParams = {
    'n_neighbors': 200,
    'weights':'distance',
    'algorithm':'auto',
    'leaf_size':30,
    'p':2,
    'metric':'minkowski',
    'n_jobs':-1
}

tree = DecisionTreeClassifier(**treeParams)
knn = KNeighborsClassifier(**knnParams)

kFoldScores = {}

trainPredictionsDT = np.zeros(trainDT.shape[0])
trainPredictionsKNN = np.zeros(trainKNN.shape[0])
testPredictionsDT = np.zeros(testDT.shape[0])
testPredictionsKNN = np.zeros(testKNN.shape[0])
print('Modelling Stage 1...')
for index, (tr_index, val_index) in enumerate(kf.split(X_train_DT)):
    t1 = time.time()
    # fit and predict with decision tree
    X_te = X_test_DT
    X_tr, X_val = X_train_DT.iloc[tr_index, :], X_train_DT.iloc[val_index, :]
    y_tr, y_val = y_train.iloc[tr_index], y_train.iloc[val_index]
    

    # Keep 30 features
    selector = SelectKBest(f_classif, k=35)

    X_tr = selector.fit_transform(X_tr, y_tr)
    X_val, X_te = selector.transform(X_val), selector.transform(X_te)
    
    # fit the DT model
    tree.fit(X_tr, y_tr)
    y_pred_DT = tree.predict_proba(X_val)[:, 1]
    testPrediction_DT = tree.predict_proba(X_te)[:, 1]
    
    
    # fit and predict with KNN
    X_te = X_test_KNN
    X_tr, X_val = X_train_KNN.iloc[tr_index, :], X_train_KNN.iloc[val_index, :]
    y_tr, y_val = y_train.iloc[tr_index], y_train.iloc[val_index]
    
    knn.fit(X_tr, y_tr)
    y_pred_KNN = knn.predict_proba(X_val)[:, 1]
    testPrediction_KNN = knn.predict_proba(X_te)[:, 1]
    
    trainPredictionsDT[val_index] = y_pred_DT
    trainPredictionsKNN[val_index] = y_pred_KNN
    
    testPredictionsDT += testPrediction_DT / N_FOLDS
    testPredictionsKNN += testPrediction_KNN / N_FOLDS

    print(f'Took {time.time() - t1}')

print('Stage 1 Modelling ended..')
# perform a simple KFold crossvalidation with out of fold prediction

kf = KFold(n_splits = 5, shuffle = True, random_state=RANDOM_SEED)

X_train = np.hstack([trainPredictionsDT.reshape(-1, 1), trainPredictionsKNN.reshape(-1, 1)])
X_test  = np.hstack([testPredictionsDT.reshape(-1, 1), testPredictionsKNN.reshape(-1, 1)])
y_train = train['target']


treeParams = {
    'criterion': 'gini',
    'splitter':'best',
    'max_depth': 7,
    'min_samples_leaf': 800,
    'random_state': RANDOM_SEED,
    'max_leaf_nodes': 100,
}


model = DecisionTreeClassifier(**treeParams)

kFoldScores = {}
testPredictions = []
for index, (tr_index, val_index) in enumerate(kf.split(X_train)):
    t1 = time.time()
    X_te = X_test
    X_tr, X_val = X_train[tr_index, :], X_train[val_index, :]
    y_tr, y_val = y_train.iloc[tr_index], y_train.iloc[val_index]
    
    model.fit(X_tr, y_tr)
    
    
    y_pred = model.predict_proba(X_val)[:, 1]

    testPrediction = model.predict_proba(X_te)
    testPredictions.append(testPrediction[:, 1])
    kFoldScores[index] = log_loss(y_val, y_pred)
    print(f'Fold {index} LogLoss:  {kFoldScores[index]} Took {time.time() - t1}')

print(f'Mean LogLoss {np.mean(list(kFoldScores.values()))}')

prediction = np.mean(np.array(testPredictions), axis = 0)
submission['target'] = prediction
submission.to_csv('submit.csv', index = None)



