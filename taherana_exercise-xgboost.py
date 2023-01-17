# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex6 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
from xgboost import XGBRegressor



# Define the model

my_model_1 = XGBRegressor(random_state=0) # Your code here



# Fit the model

my_model_1.fit(X_train, y_train) # Your code here



# Check your answer

step_1.a.check()
# Lines below will give you a hint or solution code

#step_1.a.hint()

# step_1.a.solution()
from sklearn.metrics import mean_absolute_error



# Get predictions

predictions_1 = my_model_1.predict(X_valid) # Your code here



# Check your answer

step_1.b.check()
# Lines below will give you a hint or solution code

#step_1.b.hint()

# step_1.b.solution()
# Calculate MAE

mae_1 = mean_absolute_error(y_true=y_valid, y_pred=predictions_1) # Your code here



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_1)



# Check your answer

step_1.c.check()
# Lines below will give you a hint or solution code

#step_1.c.hint()

#step_1.c.solution()
# Define the model

my_model_2 = XGBRegressor(n_estimators=500, learning_rate=0.1) # Your code here



# Fit the model

my_model_2.fit(X_train, y_train) # Your code here



# Get predictions

predictions_2 = my_model_2.predict(X_valid) # Your code here



# Calculate MAE

mae_2 = mean_absolute_error(y_pred=predictions_2, y_true=y_valid) # Your code here



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_2)



# Check your answer

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

# step_2.solution()
# Define the model

my_model_3 = XGBRegressor(n_estimators=1)



# Fit the model

my_model_3.fit(X_train, y_train) # Your code here



# Get predictions

predictions_3 = my_model_3.predict(X_valid)



# Calculate MAE

mae_3 = mean_absolute_error(y_pred=predictions_3, y_true=y_valid)



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_3)



# Check your answer

step_3.check()
# Lines below will give you a hint or solution code

#step_3.hint()

# step_3.solution()
n_esitmators = list(range(100, 1001, 100))

print('n_esitmators', n_esitmators)

learning_rates = [x / 100 for x in range(5, 101, 5)]

print('learning_rates', learning_rates)
import warnings

warnings.filterwarnings('ignore')



results = pd.DataFrame({'n_estimator': [], 'learning_rate':[], 'mae': []})

i = 0

for n in n_esitmators:

    for l in learning_rates:

        my_model = XGBRegressor(n_estimators=n, learning_rate=l, n_jobs=4)

        # Fit the model

        my_model.fit(X_train, y_train) # Your code here



        # Get predictions

        predictions = my_model.predict(X_valid)



        # Calculate MAE

        mae = mean_absolute_error(y_pred=predictions, y_true=y_valid)



        # Uncomment to print MAE

        results.loc[i] = [n, l, mae]

        #   print(results.loc[i])

        print(n, l, mae)

        i += 1

        



print('done')     
# that is because I have mistype mae in the code above

# results.rename(columns={'mea': 'mae'}, inplace=True)
results['mae'].min()
results.loc[results['mae'].idxmin()]
results['mae'].max()
results.loc[results['mae'].idxmax()]
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.figure(figsize=(30, 8))

# plt.plot(x=results[['n_estimator', 'learning_rate']], y=results['mae'])

sns.lineplot(x=results['learning_rate'], y=results['mae'], hue=results['n_estimator'])

plt.show()
results2 = results.loc[:100]

results2.tail()
plt.figure(figsize=(30, 20))

# plt.plot(x=results[['n_estimator', 'learning_rate']], y=results['mae'])

sns.lineplot(x=results2['learning_rate'], y=results2['mae'], 

             hue=results2['n_estimator'], legend='full')

plt.show()
results3 = results.loc[:40]

results3.tail()
plt.figure(figsize=(30, 20))

sns.lineplot(x=results3['learning_rate'], y=results3['mae'], 

             hue=results3['n_estimator'], legend='full')

plt.show()
results.loc[results['mae'].idxmin()]
final_model = XGBRegressor(n_estimators=400, learning_rate=0.1, n_jobs=4)
final_model.fit(X_train, y_train)
preds_test = final_model.predict(X_test)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)

print('done')