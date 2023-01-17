import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import random

from sklearn.preprocessing import MinMaxScaler



# Import H2O AutoML

import h2o

from h2o.automl import H2OAutoML

h2o.init(max_mem_size='16G')
raw = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

raw_eval = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')



# Concatenate train & test datasets

data = pd.concat([raw, raw_eval], sort=False)

data.head()
# Summary stat

def summary(df):

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name', 'dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    return summary



summary(data)
# One hot encoding

bin_vars = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

nom_vars1 = ['nom_0', 'nom_1','nom_2', 'nom_3', 'nom_4']

ord_vars1 = [ 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4'] 



# Label encoding

nom_vars2 = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

ord_vars2 = ['ord_5']

other_vars = ['day', 'month']
# Prepare X,y

X = data.copy()

y = data.target



# One hot encoding for binary and ordinal vars

for var in bin_vars+nom_vars1+ord_vars1:

    dummy = pd.get_dummies(X[var], prefix=var, prefix_sep='_')

    X = pd.concat([X,dummy],sort=False, axis=1)

    X.drop(columns=[var], inplace=True)



# Target Encoding for nominal and ordinal vars

for var in nom_vars2+ord_vars2:

    df_target = X.groupby(var).mean()['target'].to_frame().reset_index()

    df_target.columns=[var,var+'_value']

    X = pd.merge(X,df_target, on=var, how='left')

    X.drop(columns=var, inplace=True)



# Rescale day & month vars

scaler = MinMaxScaler()

X['day_scaled'] = 0

X['month_scaled'] = 0

X[['day_scaled', 'month_scaled']] = scaler.fit_transform(X[['day', 'month']])



for var in other_vars:

    df_target = X.groupby(var).mean()['target'].to_frame().reset_index()

    df_target.columns=[var,var+'_value']

    X = pd.merge(X,df_target, on=var, how='left')

    #X.drop(columns=var, inplace=True)



# Drop id column

X.drop(columns='id', inplace=True)



# Handle missing value

for var in X.columns:

    if var!='target':

        X[var].fillna((X[var].mean()), inplace=True)
# Split dataset to train and eval set

train_index = [not i for i in np.isnan(y)]

eval_index = list(np.isnan(y))



# Make train set

X_train = X[train_index].copy()

y_train = y[train_index].copy()



# Make eval set

X_eval = X[eval_index].copy()

X_eval.drop(columns=['target'], inplace=True)
# # Train AutoML Model

# H2O_train = h2o.H2OFrame(X_train)

# x =H2O_train.columns

# y ='target'

# x.remove(y)



# #H2O_train[y] = H2O_train[y].asfactor()



# aml = H2OAutoML(max_runtime_secs=30000)

# aml.train(x=x, y=y, training_frame=H2O_train)



# # Print AutoML leaderboard

# aml.leaderboard



# # Save Model

# h2o.save_model(model=aml.leader)
# Load AutoML model

aml = h2o.load_model('/kaggle/input/model3/StackedEnsemble_AllModels_AutoML_20200208_231228')
# Get train data accuracy

pred = aml.predict(h2o.H2OFrame(X_train))

pred = pred.as_data_frame()['predict'].tolist()

accuracy = sum(1 for x,y in zip(np.round(pred),X_train.target) if x == y) / len(X_train.target)

print('accuracy:',accuracy)
pred = aml.predict(h2o.H2OFrame(X_eval))

pred = pred.as_data_frame()['predict'].tolist()
output = pd.DataFrame({'id': raw_eval.id, 'target': pred})

output.to_csv('my_submission_20200209.csv', index=False)

print("Your submission was successfully saved!")