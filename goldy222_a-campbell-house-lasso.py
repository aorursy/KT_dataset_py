# Import the required modules

import pandas as pd

from sklearn import linear_model

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline



 

# import training data to Pandas DataFrame, get dummies, and replace NaN with 0

dummies_df = pd.get_dummies(pd.read_csv('../input/house-lasso/train.csv')).fillna(0)



# Put SalePrice in the 1 position, the 0 position has ID which is good

cols = dummies_df.columns.tolist()

cols.insert(1, cols.pop(cols.index('SalePrice')))

dummies_df = dummies_df.reindex(columns= cols)



# Get feature and target variables

X = dummies_df.iloc[:,2:]

y = dummies_df['SalePrice']



# Use GridSearchCV to determine the best Lasso alpha

gridsearch_steps = [('lasso', linear_model.Lasso())]

gridsearch_pipeline = Pipeline(gridsearch_steps)

parameters = {'lasso__alpha':[.0001,.001,.01,.1,1,10,100,1000,10000]}

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

gm_cv = GridSearchCV(gridsearch_pipeline,parameters)

gm_cv.fit(X_train,y_train)



# Compute and print the metrics

r2 = gm_cv.score(X_test, y_test)

print("Tuned Lasso Alpha: {}".format(gm_cv.best_params_))

print("Tuned Lasso R squared: {}".format(r2))



# Setup the pipeline steps

steps = [('lasso', linear_model.Lasso(alpha=100))]



# Create and fit the pipeline

pipeline = Pipeline(steps)

pipeline.fit(X,y)



# Import the test data to a Pandas DataFrame, get dummies, replace NaN with 0, and exclude the 0 column which is ID

test_df = pd.get_dummies(pd.read_csv('../input/house-lasso/test.csv')).fillna(0)

test_final_df = test_df.iloc[:,1:]



# Find out if training data and test data have the same number of columns, may not due to get dummies

X_list = X.columns.values

test_final_df_list = test_final_df.columns.values



# Give the test data the columns it's missing from the training data

new_cols = (list(set(X_list) - set(test_final_df_list)))

test_ready = pd.concat([test_final_df, pd.DataFrame(columns=new_cols)], axis=1)



# Our new columns need to have NaN replaced with 0

test_ready = test_ready.fillna(0)



# Make sure test data has columns in same order as training data

test_ready = test_ready[X.columns]



# Now we can predict using the test data

y_pred2 = pipeline.predict(test_ready)



# Create our submission DataFrame with the ID column

submit_df = pd.DataFrame(test_df['Id'])



# Create the SalePrice column and populate with our predictions

submit_df['SalePrice'] = y_pred2



# Export our final DataFrame with predictions to a csv file so it can be uploaded

submit_df.to_csv('../working/submit_final.csv')