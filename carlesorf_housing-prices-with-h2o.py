# And then load the libraries you'll use in this notebook

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline



import h2o

from h2o.automl import H2OAutoML



import warnings

warnings.simplefilter(action='ignore')
model = pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col='Id')

pred = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')
model.head(6)
model['YrSold'] = model['YrSold'].astype(str)

model['GarageYrBlt'] = model['GarageYrBlt'].astype(str)

model['YearRemodAdd'] = model['YearRemodAdd'].astype(str)

model['YearBuilt'] = model['YearBuilt'].astype(str)



pred['YrSold'] = pred['YrSold'].astype(str)

pred['GarageYrBlt'] = pred['GarageYrBlt'].astype(str)

pred['YearRemodAdd'] = pred['YearRemodAdd'].astype(str)

pred['YearBuilt'] = pred['YearBuilt'].astype(str)
model.isna().sum().sort_values( ascending=False)[0:20]
pred.isna().sum().sort_values( ascending=False)[0:20]
import seaborn as sns



num_col = model.select_dtypes(exclude='object').columns

columns = len(num_col)/4+1



fg, ax = plt.subplots(figsize=(20, 25))



for i, col in enumerate(num_col):

    fg.add_subplot(columns, 4, i+1)

    sns.boxplot(model.select_dtypes(exclude='object')[col])

    plt.xlabel(col)

    plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
model.shape
dfs = model



for i in dfs.columns:

    if dfs[i].dtype == 'int64' or dfs[i].dtype == 'float64':

        if dfs[i].value_counts().count() > 30:

            Q1 = dfs[i].quantile(0.05)

            Q3 = dfs[i].quantile(0.95)

            IQR = Q3 - Q1

            dfs = dfs[~((dfs[i] < (Q1 - 1.5 * IQR)) | (dfs[i] > (Q3 + 1.5 * IQR)))]

            dfs[i] = dfs[i].fillna(dfs[i].mean())

            print (i, '......Numeric Values, DROP OUTFITS, Q1:',(Q1 - 1.5 * IQR).round(2),'Q3:',(Q3 + 1.5 * IQR).round(2), '/ NAN: Mean')

        else:

            print (i, '......Numeric Categorical / NAN: 0')

            dfs[i] = dfs[i].fillna('0')

    else: 

        if dfs[i].value_counts().count() < 30:

            print (i, '......String Categorical / NAN: NA')

            dfs[i] = dfs[i].fillna('NA')  

        else:

            dfs[i] = dfs[i].fillna(dfs[i].mode().iloc[0]) #most frequent value

            print (i, '......String Non Categorical / NAN: Most frequent value')

            

model = dfs

model.shape
dfs = pred



for i in dfs.columns:

    if dfs[i].dtype == 'int64' or dfs[i].dtype == 'float64':

        if dfs[i].value_counts().count() > 30:

            dfs[i] = dfs[i].fillna(dfs[i].mean())

            print (i, '......Numeric Values / NAN: mean')

        else:

            print (i, '......Numeric Categorical / NAN: 0')

            dfs[i] = dfs[i].fillna('0')

    else: 

        if dfs[i].value_counts().count() < 30:

            print (i, '......String Categorical / NAN: NA')

            dfs[i] = dfs[i].fillna('NA')

        else:

            dfs[i] = dfs[i].fillna(dfs[i].mode().iloc[0]) #most frequent value

            print (i, '......String Non Categorical / NAN: Most frequent value')

            

pred = dfs

print (dfs.shape)
for i in [model,pred]:

    print(i.isna().sum().sort_values(ascending=True)[0:3])
import seaborn as sns



num_col = model.select_dtypes(exclude='object').columns

columns = len(num_col)/4+1



fg, ax = plt.subplots(figsize=(20, 25))



for i, col in enumerate(num_col):

    fg.add_subplot(columns, 4, i+1)

    sns.boxplot(model.select_dtypes(exclude='object')[col])

    plt.xlabel(col)

    plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
model['SalePrice'] = np.log1p(model['SalePrice'])
#model = pd.concat([model, pd.get_dummies(model['OverallQual'])], axis=1)

#pred = pd.concat([pred, pd.get_dummies(pred['OverallQual'])], axis=1)

#print(model.shape, pred.shape)

# Initialize your cluster

h2o.init()
model=h2o.H2OFrame(model)

pred=h2o.H2OFrame(pred)
# Identify predictors and response

x = [col for col in model.columns if col not in ['Id','SalePrice']]

y = 'SalePrice'



test_id = h2o.import_file('../input/home-data-for-ml-course/test.csv')

test_id = test_id['Id']
aml = H2OAutoML(max_models = 30, max_runtime_secs=300, seed = 1, stopping_metric = 'RMSLE')

aml.train(x = x, y = y, training_frame = model)
lb = aml.leaderboard; lb
aml.leader
aml.leader.varimp_plot()
preds = aml.leader.predict(pred)
# Convert results back(they had been transformed using log, remember?) and save them in a csv format.

result = preds.expm1()

sub = test_id.cbind(result)

sub.columns = ['Id','SalePrice']

sub = sub.as_data_frame()

sub.to_csv('submission.csv', index = False)