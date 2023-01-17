#install h2o
!pip install h2o
import h2o
from h2o.automl import H2OAutoML

#initialize h2o server
h2o.init()
#load the data into h2o frames
data = h2o.import_file('https://raw.githubusercontent.com/dphi-official/Datasets/master/Census_Income/Training_set_census.csv')
val = h2o.import_file('https://raw.githubusercontent.com/dphi-official/Datasets/master/Census_Income/Testing_set_census.csv')

#split data into train and test datasets with 80:20 ratio
train, test = data.split_frame(ratios=[0.8])
#remove target variable from features
x = train.columns
x.remove('income_level')

#assign target variable column name to y
y = 'income_level'

#for binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

#specify maximum number of models to try and the seed.
aml = H2OAutoML(max_models = 20, seed = 1)

#train the model
aml.train(x=x, y=y, training_frame=train)
#predict dependent variable values for test dataset
y_pred = aml.predict(test)
#get automl leaderboard to view model_id and its evaluation scores. 
lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
lb
#predict dependent variable values for new test dataset - val
submission_pred = aml.predict(val)
#convert the dependent variable prediction values of new test dataset - val into pandas DataFrame
import pandas as pd
submit =pd.DataFrame({'income_level':submission_pred.as_data_frame(use_pandas=True)['predict'].tolist()})
#convert the pandas DataFrame to csv file and store it on /kaggle/working folder
submit.to_csv('/kaggle/working/sample_submission.csv',index=False)