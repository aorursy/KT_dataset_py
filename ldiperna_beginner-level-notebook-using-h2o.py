# import all the required packages:

# import H2O and the estimators to build your models

import h2o

from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# import plotting methods and set them to appear inside the notebook

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# launch your H2O cluster. H20 runs locally, unless you are connected to additional servers,

# and uses all CPUs, by default, to run its algos in parallel.

h2o.init()



# import the train and test datasets

# replace "../input/train.csv" and "../input/test.csv" with the correct paths

# if you have these files saved in a different location

training = h2o.import_file("../input/train.csv")

test = h2o.import_file("../input/test.csv")
# specify the features and response column:



# set the predictors (remove the response and ID column)

predictors = training.columns[1:-1]

# set the response column

response = 'SalePrice'



# convert MSSubClass (which identifies the type of dwelling involved in the sale.) from a numeric feature

# to a categorical feature so that we don't assume there is an inherent order in these numbers

training['MSSubClass'] = training['MSSubClass'].asfactor()





# take log(1 + x) transform of the SalesPrice (where x is the sales price) to match kaggle's

# evaluation metric - the rsme between the logarithm of the predicted value and the logarithm of 

# the observed sales price

# we use log(1 + x) instead of log() so that we don't have to worry about taking log(0) 



# store the original SalePrice column for plotting purposes

plot_salesprice = training['SalePrice']

training['SalePrice'] = training['SalePrice'].log1p()



# Note: we will take the inverse of log(1 + x),

# to get the original sales prices back before submitting to Kaggle





# take a look at how the log transformation changed the SalePrice distribution:

# plot histograms of the response column before and after it is transformed

# overlain with a density and rug plot

sns.distplot(plot_salesprice.as_data_frame(), hist=True, rug=True)

plt.title("Untransformed Data")

plt.show()



sns.distplot(training['SalePrice'].as_data_frame(), hist=True, rug=True)

plt.title("Log Transformed Data") 

plt.show()
# check for missing values in the traininging and testing data sets

# Note: H2O interprets the 'NA' in the categorical columns as missing for a reaons (its own type of level)

# instead of interpreting the 'NA's as missing values, as pandas does.

# In this case 'NA' does mean missing for a reason (like

# No Alley, or No Basement) so we don't need to do any additional work.



print("missing values in training:", training.isna().sum())

print("missing values in test:", test.isna().sum())

print("")

mis_col = []



# list which columns have missing values

for column in training.columns:

    if training[column].isna().isin(1).sum() > 0:

        mis_col.append(column)

print('mis_col:', mis_col)       

print("")

na_col = []



# list which columns have "NA" or similar

for column in training.columns:

    if training[column].isin("NA").sum() > 0:

        na_col.append(column)

print('na_col:', na_col)

print("")



# is there any missing data in the response column? ans: No

print("missing values in training response:", training[-1].isna().sum())
# split the training dataset into a new train and validation dataset

# this will allow us to perform cross validation (in other words we can train a model on the 'train'

# dataset and then see how it performs on the 'valid' dataset), which gives us a sense of how the 

# algorithm's predictions will perform on the real 'test' dataset that you submit to kaggle.



# use 80% of the training dataset to train the model, and 20% to validate the model. 

# set a seed so the split is reproducible

train, valid = training.split_frame(ratios = [.8], seed = 1234)



# build a GLM: first initialize the estimator, then train the model

glm = H2OGeneralizedLinearEstimator()

glm.train(x = predictors, y = response, training_frame = train, validation_frame = valid)





# build a GBM: first initialize the estimator, then train the model

# set a seed so that parameters that include random initiation are reproducible

gbm = H2OGradientBoostingEstimator(seed = 1234)

gbm.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

# print the RMSE for the GLM and GBM (using default values) on the train and valid datasets



# for GLM: 

print('GLM train rmse:', glm.rmse(train = True))

print('GLM valid rmse', glm.rmse(valid = True))



# for GBM

print('GBM train rmse:', gbm.rmse(train = True))

print('GBM valid rmse', gbm.rmse(valid = True))
# predict on the test data using the GLM and GBM 

GLM_predictions = glm.predict(test)

GBM_predictions = gbm.predict(test)
# take a look at the submission results and update the column headers

print(GLM_predictions.head())

print(GBM_predictions)

print("")



# take the inverse of log(1+x) with exp(x) - 1 (`using expm1`), to get back the original range of sales price values

GLM_predictions['predict'] = GLM_predictions['predict'].expm1()

GBM_predictions['predict'] = GBM_predictions['predict'].expm1()



# take a look at the transformed  values

print('Transformed Back Values:')

print(GLM_predictions.head())

print(GBM_predictions)

print("")
# create a new frame with the Id column and SalePrice prediction value

# use .columns to reset the column names

glm_submission = test['Id'].concat(GLM_predictions)

glm_submission.columns = ['Id', 'SalePrice']



gbm_submission = test['Id'].concat(GBM_predictions)

gbm_submission.columns = ['Id', 'SalePrice']



# take a look at the output

gbm_submission.head()
# create submission files for the GLM and GBM



# export the csv to the folder where this notebook lives

h2o.export_file(glm_submission, 'glm_submission.csv')

h2o.export_file(gbm_submission, 'gbm_submission.csv')