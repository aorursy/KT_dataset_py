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
# reading train and test data 

application_train = pd.read_csv('../input/application_train.csv')

application_test = pd.read_csv('../input/application_test.csv')
#displaying shape

print('train shape:', application_train.shape)

print('test shape:', application_test.shape)
application_train.describe()
bureau = pd.read_csv('../input/bureau.csv')

bureau_balance = pd.read_csv('../input/bureau_balance.csv')
print('bureau shape:', bureau.shape)

print('bureau balance shape:', bureau_balance.shape)

bureau.describe()
print('total entry in bureau:', bureau.shape[0])

print('unique values in Application train:', len(application_train['SK_ID_CURR'].unique()), 'unique value in bureau:', len(bureau['SK_ID_CURR'].unique()))

print('avg number of loan taken by person', bureau.shape[0] / len(bureau['SK_ID_CURR'].unique()) )
applicationColumn = application_train.columns.values

bureauColumn = bureau.columns.values

print(set(applicationColumn) & set(bureauColumn))
bureau_numeric = bureau.select_dtypes(include=["number"])

bureau_numeric = bureau_numeric.drop('SK_ID_BUREAU', axis = 1)

bureau_numeric.head()
bureau_categorical = pd.get_dummies(bureau.select_dtypes(exclude = ["number"]))

bureau_categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']

bureau_categorical.head()
def splitData(dataframe):

    df_numeric_data =dataframe.select_dtypes(include=["number"])

    df_categorical_data = dataframe.select_dtypes(exclude = ["number"])

    return df_numeric_data, df_categorical_data
def handleNumericData(df, withColumn):

    df.fillna(0,inplace = True)

    ag = df.groupby(withColumn).agg(['count', 'mean'])

    ag.columns = ag.columns.map('_'.join).str.strip('_')

    ag = ag.reset_index()

    return ag





def handleCategoricalData(df, withColumn):

    ag = df.groupby(withColumn).agg(['sum', 'mean'])

    ag.columns = ag.columns.map('_'.join).str.strip('_')

    ag = ag.reset_index()

    return ag



bureau_categorical = handleCategoricalData(bureau_categorical, 'SK_ID_CURR')

bureau_numeric = handleNumericData(bureau_numeric, 'SK_ID_CURR')
bureau_updated = bureau_numeric.merge(bureau_categorical, on = 'SK_ID_CURR')
bureau_updated.head()
# 



bureau_balance_numeric = bureau_balance.select_dtypes(include=["number"])

bureau_balance_numeric = handleNumericData(bureau_balance_numeric, 'SK_ID_BUREAU')



bureau_balance_categorical = pd.get_dummies(bureau_balance.select_dtypes(exclude=["number"]))

bureau_balance_categorical['SK_ID_BUREAU'] = bureau_balance['SK_ID_BUREAU']

bureau_balance_categorical = handleCategoricalData(bureau_balance_categorical, 'SK_ID_BUREAU')

bureau_balance = bureau_balance_numeric.merge(bureau_balance_categorical, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')

bureau_balance = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_balance, on = 'SK_ID_BUREAU', how = 'left')

bureau_balance_by_client = handleNumericData(bureau_balance.drop(columns = ['SK_ID_BUREAU']),'SK_ID_CURR')
bureau_balance_by_client.head()
bureau_combined = bureau_updated.merge(bureau_balance_by_client, on = 'SK_ID_CURR')

print('bureau_combined shape:', bureau_combined.shape)

bureau_combined.head()
application_train_updated = application_train.merge(bureau_combined, on = 'SK_ID_CURR')

application_test_updated = application_test.merge(bureau_combined, on = 'SK_ID_CURR')



print('application_train_updated shape:', application_train_updated.shape)

print('application_test_updated shape:', application_test_updated.shape)



application_train_updated.head()

application_train_updated.head(20)
#starting with reading bureau and bureau balance



bureau = pd.read_csv('../input/bureau.csv')

bureau_balance = pd.read_csv('../input/bureau_balance.csv')



bureau_combined = bureau.merge(bureau_balance,on = 'SK_ID_BUREAU', how = 'outer')
import gc

del [[bureau,bureau_balance]]

gc.collect()
bureau_combined.head()
corr_matrix = bureau_combined.corr().abs()

# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]

#print(to_drop)

bureau_combined.drop(to_drop, axis=1, inplace=True)
import gc

del [[upper,corr_matrix]]

gc.collect()
bureau_combined.head()
import scipy.stats

def splitData(dataframe, idColumn):

    df_numeric_data =dataframe.select_dtypes(include=["number"])

    df_categorical_data = pd.get_dummies(dataframe.select_dtypes(exclude = ["number"]))

    df_categorical_data[idColumn] = dataframe[idColumn]

    return df_numeric_data, df_categorical_data



def handleNumericData(df, withColumn):

    df.fillna(0,inplace = True)

    ag = df.groupby(withColumn).agg(['mean', 'min', 'max'])

    ag.columns = ag.columns.map('_'.join).str.strip('_')

    ag = ag.reset_index()

    return ag





def handleCategoricalData(df, withColumn):

    

    '''maxCategorical = categorical.groupby(withColumn).agg(lambda x: scipy.stats.mode(x)[0][0])

    maxCategorical = maxCategorical.reset_index()

    #maxCategorical.drop(withColumn,inplace = True)'''

    

    ag = df.groupby(withColumn).agg(['sum'])

    ag.columns = ag.columns.map('_'.join).str.strip('_')

    ag = ag.reset_index()

    #ag = pd.concat([ag, maxCategorical], axis = 1)

    #ag['Max'] = ag[list(ag.columns[1:])].max(axis=1)



    return ag



def treatDataFrame(dataframe, idColumn):

    numerical, categorical = splitData(dataframe, idColumn)

    #del [[df_numeric_data,df_categorical_data]]

    gc.collect()    

    

    numerical = handleNumericData(numerical, idColumn)

    #del [[ag]]

    gc.collect()    

    

    categorical = handleCategoricalData(categorical, idColumn)

    #del [[ag]]

    gc.collect()

    

    categorical.drop(idColumn,axis = 1, inplace= True)

    combined_data = pd.concat([numerical, categorical], axis = 1)

    del [[numerical,categorical]]

    gc.collect()

    return combined_data

    
# handleCategoricalData(categorical.iloc[:10,:],'SK_ID_BUREAU')

# #categorical.head()

#bureau_balance.head()

bureau_combined.drop('SK_ID_BUREAU', axis = 1, inplace = True)

bureau_combined = treatDataFrame(bureau_combined, 'SK_ID_CURR')

bureau_combined.head()
bureau_combined.to_csv('../bureau_combined.csv')
sk_id = bureau['SK_ID_CURR']

bureau.drop('SK_ID_CURR',axis = 1, inplace = True)



bureau = treatDataFrame(bureau, 'SK_ID_BUREAU')

bureau['SK_ID_CURR'] = sk_id

bureau.head()
import scipy.stats



maxCategorical.head()
categorical_dummies = pd.get_dummies(categorical)

categorical_dummies['SK_ID_BUREAU'] = bureau_balance['SK_ID_BUREAU']

categorical_dummies.head()
#ag = categorical_dummies.groupby('SK_ID_BUREAU').agg(['sum'])

#agMode = categorical.groupby('SK_ID_BUREAU').agg(lambda x: x.value_counts().index[0])

handleCategoricalData(categorical_dummies, 'SK_ID_BUREAU').head()





# ag.columns = ag.columns.map('_'.join).str.strip('_')

# ag = ag.reset_index()

# ag['Max'] = ag[list(ag.columns[1:])].max(axis=1)

# ag.head()
ag.head()
handleCategoricalData(categorical, 'SK_ID_BUREAU')
train = pd.read_csv('../input/train-all/train_all.csv')

bureau_combined = pd.read_csv('../input/bureau/bureau_combined.csv')

print(train.shape)

print(bureau_combined.shape)
train.head()
target = train['TARGET']

train.drop(['SK_ID_CURR', 'TARGET', 'Unnamed: 0'],axis = 1, inplace = True)
number = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum() / train.isnull().count() * 100).sort_values(ascending = False)



missing_application_test = pd.concat([number , percent] , axis = 1 , keys = ['Total' , 'Percent'])

print(missing_application_test.shape)

print(missing_application_test.head(20))
train_full = train.loc[:, train.isnull().mean() < .5]
print(train.shape)

print(train_full.shape)
train_full.fillna(train_full.mean(),inplace = True)
train_full.shape
y_predict = rdf.predict(X_train)

from sklearn.metrics import precision_recall_fscore_support

print('precision_recall_fscore_support score:', precision_recall_fscore_support(y_predict, y_train))
import gc

test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')

numerical, categorical = splitData(test, 'SK_ID_CURR')

print(test.shape)

test  = pd.merge(numerical,categorical, on = 'SK_ID_CURR', how = 'inner')

test.head()
test  = pd.merge(test,bureau_combined, on = 'SK_ID_CURR', how = 'left')

print(test.shape)
common_columns = list(set(test.columns) & set(train_full.columns)) 

train_common = train_full[common_columns]
print(train_common.shape)

print(target.shape)
train_common.head()
from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(train_common, target.ravel())
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
from sklearn.ensemble import RandomForestClassifier

rdf = RandomForestClassifier(n_estimators = 100 , max_depth = 5, min_samples_leaf = 4, max_features = 0.5)

rdf.fit(X_train_res , y_train_res)

print(len(test['SK_ID_CURR']))

test.fillna(test.mean(),inplace = True)

test[common_columns].head()
#X_train_res = sm.fit_sample(test[common_columns])

y_pred_sample_score = rdf.predict(test[common_columns])

len(y_pred_sample_score)
df = pd.DataFrame()

df['SK_ID_CURR'] = test['SK_ID_CURR']

df['TARGET'] = y_pred_sample_score

df.to_csv('sample_submission.csv', index=False)
df.head()
sub = pd.read_csv('../input/home-credit-default-risk/sample_submission.csv')

sub['TARGET'] = y_pred_sample_score

sub.to_csv('submission2.csv',index = False)
sub.head()