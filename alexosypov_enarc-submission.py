import pandas as pd

import numpy as np



# these are our test and train dataframes

test_df = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')

train_df = pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv')



col = train_df.columns

col_test = test_df.columns
# https://machinelearningmastery.com/handle-missing-data-python/

from sklearn.impute import SimpleImputer



# we reform the 'na' entries in the input csv into np.nan for compliance with the sklearn imputer

train_df = train_df.replace('na', np.nan)  # NoNaN :(

# This uses the median strategy for imputation

imp_mean = SimpleImputer(strategy='median')



# Run the imputer on the training data to replace values

imp_mean.fit(train_df)

imputed_train_df = imp_mean.transform(train_df)

imputed_train_df = pd.DataFrame(imputed_train_df, columns=col)



# Display the first 3 feature columns and their distributions

imputed_train_df.plot(x ='id', y='sensor1_measure', kind = 'line', figsize=(10,7))

imputed_train_df.plot(x ='id', y='sensor2_measure', kind = 'line', figsize=(10,7))

imputed_train_df.plot(x ='id', y='sensor3_measure', kind = 'line', figsize=(10,7))

    

# Handling the test data

# we reform the 'na' entries in the input csv into np.nan for compliance with the sklearn imputer

test_df = test_df.replace('na', np.nan)

# This uses the median strategy for imputation

imp_mean_test = SimpleImputer(strategy='median')



# Run the imputer on the test data to replace values

imp_mean_test.fit(test_df)

imputed_test_df = imp_mean_test.transform(test_df)

imputed_test_df = pd.DataFrame(imputed_test_df, columns=col_test) 
# https://machinelearningmastery.com/handle-missing-data-python/

from sklearn.impute import SimpleImputer



# we reform the 'na' entries in the input csv into np.nan for compliance with the sklearn imputer

train_df = train_df.replace('na', np.nan)  # NoNaN :(

# This uses the mean strategy for imputation

imp_mean = SimpleImputer(strategy='mean')



# Run the imputer on the training data to replace values

imp_mean.fit(train_df)

imputed_train_df = imp_mean.transform(train_df)

imputed_train_df = pd.DataFrame(imputed_train_df, columns=col)



# Display the first 3 feature columns and their distributions

imputed_train_df.plot(x ='id', y='sensor1_measure', kind = 'line', figsize=(10,7))

imputed_train_df.plot(x ='id', y='sensor2_measure', kind = 'line', figsize=(10,7))

imputed_train_df.plot(x ='id', y='sensor3_measure', kind = 'line', figsize=(10,7))

    

    



# Handling the test data

# we reform the 'na' entries in the input csv into np.nan for compliance with the sklearn imputer

test_df = test_df.replace('na', np.nan)

# This uses the mean strategy for imputation

imp_mean_test = SimpleImputer(strategy='mean')



# Run the imputer on the test data to replace values

imp_mean_test.fit(test_df)

imputed_test_df = imp_mean_test.transform(test_df)

imputed_test_df = pd.DataFrame(imputed_test_df, columns=col_test) 
import xgboost as xgb

from sklearn.model_selection import train_test_split



trainSet, validationSet = train_test_split(imputed_train_df, test_size=0.5)



dtrain = xgb.DMatrix(trainSet.iloc[:, 2:], label=trainSet['target'])

dtest = xgb.DMatrix(validationSet.iloc[:, 2:], label=validationSet['target'])

true_test = xgb.DMatrix(imputed_test_df.iloc[:, 1:])



# Binary:logistic is the objective function chosen for its output classifications with values [0,1]

param = {'max_depth':7, 'eta':0.25, 'verbosity':0, 'objective':'binary:logistic', 'alpha':0.5}

param['nthread'] = 6

param['eval_metric'] = ['aucpr', 'map', 'rmse']

num_rounds = 100

evallist = [(dtest, 'eval'), (dtrain, 'train')]



# train the data given these set of parameters

model = xgb.train(param, dtrain, num_rounds, evallist)

prediction = model.predict(true_test)



print("Mean of predictions: " + str(np.mean(prediction)))

print("Median of predicitons: " + str(np.median(prediction)))



# visualize the feature importance

xgb.plot_importance(model, max_num_features=20)

xgb.to_graphviz(model)



# XGB returns a probability rather than binary 1/0, so we will perform a split on 0.5

for count in range(len(prediction)):

    if(prediction[count] > .5):

      prediction[count] = 1

    else:

      prediction[count] = 0
import xgboost as xgb

from sklearn.model_selection import train_test_split



trainSet, validationSet = train_test_split(imputed_train_df, test_size=0.5)



dtrain = xgb.DMatrix(trainSet.iloc[:, 2:], label=trainSet['target'])

dtest = xgb.DMatrix(validationSet.iloc[:, 2:], label=validationSet['target'])

true_test = xgb.DMatrix(imputed_test_df.iloc[:, 1:])



# Binary:logistic is the objective function chosen for its output classifications with values [0,1]

param = {'max_depth':7, 'eta':0.2, 'verbosity':1, 'objective':'binary:logistic', 'alpha':0.5}

param['nthread'] = 6



#The parameters for evaluation that I chose to include were aucpr (area under PR curve, the closest)

#evaluation we could get to the F1 score that we were aiming for, map (mean average precision, once

# again useful to predict what the precision of our model is based on training data), and rmse (which

# root mean square error, which measures the differences between the values if an estimate is used, like

# in the classifier here). All of the metrics are then outputed to see the differences between the rounds.

param['eval_metric'] = ['aucpr', 'map', 'rmse']

num_rounds = 100

evallist = [(dtest, 'eval'), (dtrain, 'train')]



model = xgb.train(param, dtrain, num_rounds, evallist)

prediction = model.predict(true_test)



print("Mean of predictions: " + str(np.mean(prediction)))

print("Median of predicitons: " + str(np.median(prediction)))

# print("FSCORE: " + str(model.get_fscore()))

xgb.plot_importance(model, max_num_features=20)

xgb.to_graphviz(model)



# XGB returns a probability rather than binary 1/0, so we will perform a split on 0.5

for count in range(len(prediction)):

    if(prediction[count] >= 0.5):

      prediction[count] = 1

    else:

      prediction[count] = 0
count = 1

temp = [] # to help reform the 1D predictions list into a pandas dataframe

for item in prediction:

  temp.append([count, int(item)])

  count+=1



predictions_df = pd.DataFrame(temp, columns=['id','target'])
# Create & upload a file, to use the model later on

predictions_df.to_csv('output.csv', index=False, header=True)

print(predictions_df) # display the final DataFrame

model.save_model('0001.model')