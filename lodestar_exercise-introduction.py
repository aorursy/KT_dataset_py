# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex1 import *

print("Setup Complete")
#LodeStar: THIS IS VERSION 2 WITH SOME CHANGES WHILE WORKING TOWARDS A BETTER RESULT ! 



import pandas as pd

from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 100)



# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Obtain target and predictors

y = X_full.SalePrice



#LodeStar: LET'S TRY TO CHANGE SOME OF THE FEATURES THAT HAVE NON-ZERO AND NON-MISSING VALUES (SINCE I HAVEN'T LEARNT HANDLING MISSING VALUES YET) 

#LodeStar: Below "features" list is the one given originally for the exercise

#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']



#LodeStar: TotalBsmtSF, GarageCars seem to have NaN or infinity values - will deal with them in future ! 

#features = ['OverallCond', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'GarageCars', 'PoolArea']



#LodeStar: Here is a list that works, does not have NaN and gives lower MAE ! 

features = ['OverallCond', 'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'PoolArea']



X = X_full[features].copy()

X_test = X_test_full[features].copy()



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
X_full.describe()
X_train.head()
from sklearn.ensemble import RandomForestRegressor



# Define the models

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)



models = [model_1, model_2, model_3, model_4, model_5]
from sklearn.metrics import mean_absolute_error



# Function for comparing different models

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d MAE: %d" % (i+1, mae))
# Fill in the best model

best_model = model_3



# Check your answer

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Define a model. # Put your code here



#LodeStar: Below is my intitial call of the RFR

#my_model = RandomForestRegressor(n_estimators=50, random_state=1)



#LodeStar: With that, I am at a MAE of 21757.3, I want to see if I can reduce it further. 

#LodeStar: Change n_estimators to 100: MAE is 21563.0 

#LodeStar: Change n_estimators to 200: MAE is 21647.3 !

#LodeStar: Add criterion='mae': MAE is 21887.4 !

#LodeStar: Change to random_state=0: MAE is 21592 

#LodeStar: Change to random_state=50: MAE is 21979.9 ! 

#LodeStar: Add max_depth=7: MAE is 21610.7 ! 



#LodeStar: OK, so for now, I'll stick with this for now and move to studying missing values, etc. for imporvement ! 

my_model = RandomForestRegressor(n_estimators=100, random_state=1)



# Check your answer

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Fit the model to the training data

my_model.fit(X, y)



# Generate test predictions

preds_test = my_model.predict(X_test)



my_mae = score_model(my_model) 

 

print(my_mae)



# Save predictions in format used for competition scoring

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
output.describe()
output.head()