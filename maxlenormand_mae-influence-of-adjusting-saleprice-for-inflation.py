import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Imputer

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



from xgboost import XGBRegressor



import os
# List of all the functions that we are going to use



def get_rmse(y_predicted,y_real):

    return np.mean(np.sqrt((np.log(y_predicted)-np.log(y_real))**2))
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.columns
train_data.head(10)
#We want to visualize the dsitribution of our target, here the SalePrice:



plt.hist(train_data.SalePrice, bins=100)
#Data seems very scattered, so we will look at the log ot the Sale Price ditribution:



logged_price = np.log(train_data.SalePrice)

plt.hist(logged_price, bins=100)
plt.hist(train_data.YearBuilt)
plt.hist(train_data.YrSold)
plt.hist(test_data.YrSold)
inflation = pd.DataFrame(dict(value_by_1860_usd=[24.29, 24.98, 25.94, 25.85, 26.27],

                          inflation_percent=[3.23, 2.85, 3.84, -0.36, 1.64]) ,

                      index = ['2006', '2007', '2008', '2009', '2010'])

inflation
#Let's start by creating a new data set for each year:

infl_df_2006 = train_data.loc[train_data['YrSold'] == 2006]

infl_df_2007 = train_data.loc[train_data['YrSold'] == 2007]

infl_df_2008 = train_data.loc[train_data['YrSold'] == 2008]

infl_df_2009 = train_data.loc[train_data['YrSold'] == 2009]

infl_df_2010 = train_data.loc[train_data['YrSold'] == 2010] #this one will not be changed, just used for final concat



#Since we tried adjusting prices for inflation, and the MAE turned out ot be worse, 

#we are going to try at different % of inflation correction: 25%, 50%, 75% and 100%



#25%

infl_df_2006_25 = infl_df_2006

infl_df_2007_25 = infl_df_2007

infl_df_2008_25 = infl_df_2008

infl_df_2009_25 = infl_df_2009

infl_df_2010_25 = infl_df_2010



#50%

infl_df_2006_50 = infl_df_2006

infl_df_2007_50 = infl_df_2007

infl_df_2008_50 = infl_df_2008

infl_df_2009_50 = infl_df_2009

infl_df_2010_50 = infl_df_2010



#75%

infl_df_2006_75 = infl_df_2006

infl_df_2007_75 = infl_df_2007

infl_df_2008_75 = infl_df_2008

infl_df_2009_75 = infl_df_2009

infl_df_2010_75 = infl_df_2010



infl_df_2006_25.head()

#We get the value of the USD at a given year

usd_2006 = inflation.loc['2006','value_by_1860_usd']

usd_2007 = inflation.loc['2007','value_by_1860_usd']

usd_2008 = inflation.loc['2008','value_by_1860_usd']

usd_2009 = inflation.loc['2009','value_by_1860_usd']



usd_2010 = inflation.loc['2010','value_by_1860_usd']
#We then get a factor of the value of the USD in a year compared to the 2010 USD:

divid_factor_06 = usd_2010 / usd_2006

divid_factor_07 = usd_2010 / usd_2007

divid_factor_08 = usd_2010 / usd_2008

divid_factor_09 = usd_2010 / usd_2009



print('divide factor for 2006 is {} '.format(divid_factor_06))

print('divide factor for 2007 is {} '.format(divid_factor_07))

print('divide factor for 2008 is {} '.format(divid_factor_08))

print('divide factor for 2009 is {} '.format(divid_factor_09))



#25%:

divid_factor_06_25 = divid_factor_06/4

divid_factor_07_25 = divid_factor_07/4

divid_factor_08_25 = divid_factor_08/4

divid_factor_09_25 = divid_factor_09/4



#50%

divid_factor_06_50 = divid_factor_06/2

divid_factor_07_50 = divid_factor_07/2

divid_factor_08_50 = divid_factor_08/2

divid_factor_09_50 = divid_factor_09/2



#75%

divid_factor_06_75 = divid_factor_06/(4/3)

divid_factor_07_75 = divid_factor_07/(4/3)

divid_factor_08_75 = divid_factor_08/(4/3)

divid_factor_09_75 = divid_factor_09/(4/3)



float(divid_factor_06)
#We need to multiply the values of SalePrice by divid_factor



#25%:

infl_df_2006_25['SalePrice'] = infl_df_2006_25['SalePrice'].apply(lambda x: x*float(divid_factor_06_25));

infl_df_2007_25['SalePrice'] = infl_df_2007_25['SalePrice'].apply(lambda x: x*divid_factor_07_25);

infl_df_2008_25['SalePrice'] = infl_df_2008_25['SalePrice'].apply(lambda x: x*divid_factor_08_25);

infl_df_2009_25['SalePrice'] = infl_df_2009_25['SalePrice'].apply(lambda x: x*divid_factor_09_25);



#50%:

infl_df_2006_50['SalePrice'] = infl_df_2006['SalePrice'].apply(lambda x: x*divid_factor_06_50);

infl_df_2007_50['SalePrice'] = infl_df_2007['SalePrice'].apply(lambda x: x*divid_factor_07_50);

infl_df_2008_50['SalePrice'] = infl_df_2008['SalePrice'].apply(lambda x: x*divid_factor_08_50);

infl_df_2009_50['SalePrice'] = infl_df_2009['SalePrice'].apply(lambda x: x*divid_factor_09_50);



#75%:

infl_df_2006_75['SalePrice'] = infl_df_2006['SalePrice'].apply(lambda x: x*divid_factor_06_75);

infl_df_2007_75['SalePrice'] = infl_df_2007['SalePrice'].apply(lambda x: x*divid_factor_07_75);

infl_df_2008_75['SalePrice'] = infl_df_2008['SalePrice'].apply(lambda x: x*divid_factor_08_75);

infl_df_2009_75['SalePrice'] = infl_df_2009['SalePrice'].apply(lambda x: x*divid_factor_09_75);



#100%:

infl_df_2006['SalePrice'] = infl_df_2006['SalePrice'].apply(lambda x: x*divid_factor_06);

infl_df_2007['SalePrice'] = infl_df_2007['SalePrice'].apply(lambda x: x*divid_factor_07);

infl_df_2008['SalePrice'] = infl_df_2008['SalePrice'].apply(lambda x: x*divid_factor_08);

infl_df_2009['SalePrice'] = infl_df_2009['SalePrice'].apply(lambda x: x*divid_factor_09);
#Now we need to concat all the sub DataFrames in one bigger one

#25%:

frames_25 = [infl_df_2006_25, infl_df_2007_25, infl_df_2008_25, infl_df_2009_25, infl_df_2010_25]

infl_train_data_25 = pd.concat(frames_25)



#50%:

frames_50 = [infl_df_2006_50, infl_df_2007_50, infl_df_2008_50, infl_df_2009_50, infl_df_2010_50]

infl_train_data_50 = pd.concat(frames_50)



#75%:

frames_75 = [infl_df_2006_75, infl_df_2007_75, infl_df_2008_75, infl_df_2009_75, infl_df_2010_75]

infl_train_data_75 = pd.concat(frames_75)



#100:

frames = [infl_df_2006, infl_df_2007, infl_df_2008, infl_df_2009, infl_df_2010]

infl_train_data = pd.concat(frames)





print('size of train data : ',train_data.size)

print('size of train data adjusted for inflation : ', infl_train_data.size)
print('nb of missing data = {0}'.format(train_data.isnull().sum().max()))
#Missing values:

total = train_data.isnull().sum().sort_values(ascending=False)

percent=(train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)



missing_data=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
correlation = train_data.corr()

print('Most correlated columns to {0} are: '.format('SalePrice'),'\n', correlation['SalePrice'].sort_values(ascending = False)[:10])

print('Least correlated columns to {0} are: '.format('SalePrice'),'\n', correlation['SalePrice'].sort_values(ascending = False)[-10:])
corrmat = train_data.corr()

f, ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax = 0.8, square=True)
#Scatterplot

sns.set()

cols_scat = ['SalePrice','OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train_data[cols_scat])

plt.show()
#Now, we get rid of variables that are missing too much data (here, everything that is missing more than 1 variable)

train_data = train_data.drop((missing_data[missing_data['Total']>1]).index, 1)

infl_train_data = infl_train_data.drop((missing_data[missing_data['Total']>1]).index, 1)

infl_train_data_25 = infl_train_data_25.drop((missing_data[missing_data['Total']>1]).index, 1)

infl_train_data_50 = infl_train_data_50.drop((missing_data[missing_data['Total']>1]).index, 1)

infl_train_data_75 = infl_train_data_75.drop((missing_data[missing_data['Total']>1]).index, 1)



#We will delete the entry containing the missing data in the Electrical variable: 

train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)

infl_train_data = infl_train_data.drop(infl_train_data.loc[infl_train_data['Electrical'].isnull()].index)

infl_train_data_25 = infl_train_data_25.drop(infl_train_data_25.loc[infl_train_data_25['Electrical'].isnull()].index)

infl_train_data_50 = infl_train_data_50.drop(infl_train_data_50.loc[infl_train_data_50['Electrical'].isnull()].index)

infl_train_data_75 = infl_train_data_75.drop(infl_train_data_75.loc[infl_train_data_75['Electrical'].isnull()].index)
print('Missing values in train data : ', train_data.isnull().sum().max())

print('Missing values in train data adjusted for inflation : ', infl_train_data.isnull().sum().max())
var = 'GrLivArea'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#We want to see the two points with highest GrLivArea, they seem ourliers

train_data.sort_values(by = 'GrLivArea', ascending = False)[:2] #(from kernel found online)



#I just checked if another technique worked, just for pratcise, it gives the same result (which is good news)

#High_GrLivArea = train_data[train_data['GrLivArea']>4500]

#High_GrLivArea.head()
train_data = train_data.drop(train_data[train_data["Id"] == 524].index)

train_data = train_data.drop(train_data[train_data["Id"] == 1299].index)



infl_train_data = infl_train_data.drop(infl_train_data[infl_train_data['Id']==524].index)

infl_train_data = infl_train_data.drop(infl_train_data[infl_train_data['Id']==1299].index)



infl_train_data_25 = infl_train_data_25.drop(infl_train_data_25[infl_train_data_25['Id']==524].index)

infl_train_data_25 = infl_train_data_25.drop(infl_train_data_25[infl_train_data_25['Id']==1299].index)



infl_train_data_50 = infl_train_data_50.drop(infl_train_data_50[infl_train_data_50['Id']==524].index)

infl_train_data_50 = infl_train_data_50.drop(infl_train_data_50[infl_train_data_50['Id']==1299].index)



infl_train_data_75 = infl_train_data_75.drop(infl_train_data_75[infl_train_data_75['Id']==524].index)

infl_train_data_75 = infl_train_data_75.drop(infl_train_data_75[infl_train_data_75['Id']==1299].index)





data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Train data')



infl_data = pd.concat([infl_train_data['SalePrice'], infl_train_data[var]], axis=1)

infl_data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Adjusted train data')
#One-hot encoding (using categorical data)



cols_with_missing = [col for col in train_data.columns 

                                 if train_data[col].isnull().any()]                                  

candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)

candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)



# "cardinality" means the number of unique values in a column.

# We use it as our only way to select categorical columns here. This is convenient, though

# a little arbitrary.

low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].nunique() < 10 and

                                candidate_train_predictors[cname].dtype == "object"]

numeric_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

train_predictors = candidate_train_predictors[my_cols]

test_predictors = candidate_test_predictors[my_cols]



one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)



#Same for adjusted set:

infl_candidate_train_predictors = infl_train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)



infl_train_predictors = infl_candidate_train_predictors[my_cols]



infl_one_hot_encoded_training_predictors = pd.get_dummies(infl_train_predictors)

infl_final_train, final_test = infl_one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)



#25:

infl_candidate_train_predictors_25 = infl_train_data_25.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)



infl_train_predictors_25 = infl_candidate_train_predictors_25[my_cols]



infl_one_hot_encoded_training_predictors_25 = pd.get_dummies(infl_train_predictors_25)

infl_final_train_25, final_test = infl_one_hot_encoded_training_predictors_25.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)



#50:

infl_candidate_train_predictors_50 = infl_train_data_50.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)



infl_train_predictors_50 = infl_candidate_train_predictors_50[my_cols]



infl_one_hot_encoded_training_predictors_50 = pd.get_dummies(infl_train_predictors_50)

infl_final_train_50, final_test = infl_one_hot_encoded_training_predictors_50.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)



#75:

infl_candidate_train_predictors_75 = infl_train_data_75.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)



infl_train_predictors_75 = infl_candidate_train_predictors_75[my_cols]



infl_one_hot_encoded_training_predictors_75 = pd.get_dummies(infl_train_predictors_75)

infl_final_train_75, final_test = infl_one_hot_encoded_training_predictors_75.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)
X = np.array(final_train)

y = train_data.SalePrice



train_X, val_X, train_y, val_y = train_test_split(X, y)



#We are going to compare the original data set, with the adjusted to inflation one:

infl_X = np.array(infl_final_train)

infl_y = infl_train_data.SalePrice



infl_train_X, infl_val_X, infl_train_y, infl_val_y = train_test_split(infl_X, infl_y)



#25%:

infl_X_25 = np.array(infl_final_train_25)

infl_y_25 = infl_train_data_25.SalePrice



infl_train_X_25, infl_val_X_25, infl_train_y_25, infl_val_y_25 = train_test_split(infl_X_25, infl_y_25)



#50%:

infl_X_50 = np.array(infl_final_train_50)

infl_y_50 = infl_train_data_50.SalePrice



infl_train_X_50, infl_val_X_50, infl_train_y_50, infl_val_y_50 = train_test_split(infl_X_50, infl_y_50)



#75%:

infl_X_75 = np.array(infl_final_train_75)

infl_y_75 = infl_train_data_75.SalePrice



infl_train_X_75, infl_val_X_75, infl_train_y_75, infl_val_y_75 = train_test_split(infl_X_75, infl_y_75)
best_learn_rate = 0.2

best_nb_est = 50 

#These come from a previous Kernel, but for the purpose of just testing this, I will keep those values for now



#we extract the year at which the predicted SalePrice have been sold

year_train = train_X[:,[32]]

year_test = val_X[:,[32]]



my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=best_nb_est, learning_rate = best_learn_rate))



my_pipeline.fit(X,y)

train_y_predicted = my_pipeline.predict(train_X)

val_y_predicted = my_pipeline.predict(val_X)



print('Score on training set:',get_rmse(train_y_predicted,train_y))

print('Score on validation set:',get_rmse(val_y_predicted,val_y))



#Adjusted dataset:

my_pipeline.fit(infl_X,infl_y)

infl_train_y_predicted = my_pipeline.predict(infl_train_X)

infl_val_y_predicted = my_pipeline.predict(infl_val_X)



#25% adjusted dataset:

my_pipeline.fit(infl_X_25,infl_y_25)

infl_train_y_predicted_25 = my_pipeline.predict(infl_train_X_25)

infl_val_y_predicted_25 = my_pipeline.predict(infl_val_X_25)



#50% adjusted dataset:

my_pipeline.fit(infl_X_50,infl_y_50)

infl_train_y_predicted_50 = my_pipeline.predict(infl_train_X_50)

infl_val_y_predicted_50 = my_pipeline.predict(infl_val_X_50)



#75% adjusted dataset:

my_pipeline.fit(infl_X_75,infl_y_75)

infl_train_y_predicted_75 = my_pipeline.predict(infl_train_X_75)

infl_val_y_predicted_75 = my_pipeline.predict(infl_val_X_75)



#Now we need to divide the predictions 'infl_train_y_predicted' by the divid_factor of each year accordingly:

#start by making a copy of the predicted price dataset, just in case:

y_predict = infl_train_y_predicted

y_predict[year_train[:,0] == 2009] = y_predict[year_train[:,0] == 2009]/divid_factor_09

y_predict[year_train[:,0] == 2008] = y_predict[year_train[:,0] == 2008]/divid_factor_08

y_predict[year_train[:,0] == 2007] = y_predict[year_train[:,0] == 2007]/divid_factor_07

y_predict[year_train[:,0] == 2006] = y_predict[year_train[:,0] == 2006]/divid_factor_06

infl_train_y_predicted = y_predict



val_y_predict = infl_val_y_predicted

val_y_predict[year_test[:,0] == 2009] = val_y_predict[year_test[:,0] == 2009]/divid_factor_09

val_y_predict[year_test[:,0] == 2008] = val_y_predict[year_test[:,0] == 2008]/divid_factor_08

val_y_predict[year_test[:,0] == 2007] = val_y_predict[year_test[:,0] == 2007]/divid_factor_07

val_y_predict[year_test[:,0] == 2006] = val_y_predict[year_test[:,0] == 2006]/divid_factor_06

infl_val_y_predicted = val_y_predict



print('Score on adjusted training set:',get_rmse(infl_train_y_predicted,infl_train_y))

print('Score on adjusted validation set:',get_rmse(infl_val_y_predicted,infl_val_y))



#25%:

y_predict_25 = infl_train_y_predicted_25

y_predict_25[year_train[:,0] == 2009] = y_predict_25[year_train[:,0] == 2009]/divid_factor_09_25

y_predict_25[year_train[:,0] == 2008] = y_predict_25[year_train[:,0] == 2008]/divid_factor_08_25

y_predict_25[year_train[:,0] == 2007] = y_predict_25[year_train[:,0] == 2007]/divid_factor_07_25

y_predict_25[year_train[:,0] == 2006] = y_predict_25[year_train[:,0] == 2006]/divid_factor_06_25

infl_train_y_predicted_25 = y_predict_25



val_y_predict_25 = infl_val_y_predicted_25

val_y_predict_25[year_test[:,0] == 2009] = val_y_predict_25[year_test[:,0] == 2009]/divid_factor_09_25

val_y_predict_25[year_test[:,0] == 2008] = val_y_predict_25[year_test[:,0] == 2008]/divid_factor_08_25

val_y_predict_25[year_test[:,0] == 2007] = val_y_predict_25[year_test[:,0] == 2007]/divid_factor_07_25

val_y_predict_25[year_test[:,0] == 2006] = val_y_predict_25[year_test[:,0] == 2006]/divid_factor_06_25

infl_val_y_predicted_25 = val_y_predict_25



print('Score on 25% adjusted training set:',get_rmse(infl_train_y_predicted_25,infl_train_y_25))

print('Score on 25% adjusted validation set:',get_rmse(infl_val_y_predicted_25,infl_val_y_25))
predictions = my_pipeline.predict(final_test)

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': predictions})



output.to_csv('submission.csv', index=False)