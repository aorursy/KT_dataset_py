# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.chdir('/kaggle/input/predicting-energy-rating-from-raw-data/')
df = pd.read_csv('train_rating_eu.csv')
print(df.shape)
df.head()
df.rating.unique()
df.rating.value_counts(normalize=True)
regr =  df.copy()
regr['rating'].unique()
orig = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
values = ['12.5', '38.0', '63.0', '88.0', '113.0', '138.0', '163.0']
regr['rating'] = regr['rating'].replace(orig, values)
regr.rating.unique()
regr['rating'] = regr['rating'].astype('float64')
regr['rating'].unique()
regr = regr.drop(['Unnamed: 0', 'building_id', 'site_id'], axis=1)
y = regr['rating']
X = regr.drop('rating', axis=1)
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
# creates function to display results
def display_results(results):
    results_df  = pd.DataFrame(results).T
    results_cols = results_df.columns
    for col in results_df:
        results_df[col] = results_df[col].apply(np.mean)
    return results_df
RESULTS = {} # creates an empty dictionary
# defining a function that trains a model using "cross validate" on any "estimator" or algorithm
def evaluate_model(estimator):
    cv_results = cross_validate(estimator,
                    X=X, # X = independant variables
                    y=y, # y = target variable (aka answer)
                    scoring="neg_root_mean_squared_error", # what error metric to use to compare
                    cv=3, #number of folds in our cv
                    return_train_score=True) #Any guesses what this is doing? How could we check?
    return pd.DataFrame(cv_results).abs().mean().to_dict()
from sklearn.linear_model import LinearRegression

RESULTS["lm"] = evaluate_model(LinearRegression())

pd.DataFrame.from_dict(RESULTS).T
from sklearn.linear_model import ElasticNet, Lasso, Ridge

RESULTS["elasticnet"] = evaluate_model(ElasticNet()) 
RESULTS["lasso"] = evaluate_model(Lasso())  
RESULTS["ridge"] = evaluate_model(Ridge())

pd.DataFrame.from_dict(RESULTS).T
from sklearn.tree import DecisionTreeRegressor

RESULTS["tree"] = evaluate_model(DecisionTreeRegressor())
pd.DataFrame.from_dict(RESULTS).T
RESULTS["tree_4"] = evaluate_model(DecisionTreeRegressor(max_depth=4))
RESULTS["tree_3"] = evaluate_model(DecisionTreeRegressor(max_depth=3))
RESULTS["tree_2"] = evaluate_model(DecisionTreeRegressor(max_depth=2))
pd.DataFrame.from_dict(RESULTS).T
tree_4 = DecisionTreeRegressor(max_depth=4)
tree_4.fit(X,y)
predictions = tree_4.predict(X)
pred_df = regr.copy()
pred_df['prediction'] = predictions
pred_df.head()
letter_predictions = []
for prediction in predictions:
    if prediction <= 25.0:
        letter_predictions.append('A')
    elif (prediction > 25.0) and (prediction <= 50.0):
        letter_predictions.append('B')
    elif (prediction > 50.0) and (prediction <= 75.0):
        letter_predictions.append('C')
    elif (prediction > 75.0) and (prediction <= 100.0):
        letter_predictions.append('D')
    elif (prediction > 100.0) and (prediction <= 125.0):
        letter_predictions.append('E')
    elif (prediction > 125.0) and (prediction <= 150.0):
        letter_predictions.append('F')
    else:
        letter_predictions.append('G')
        
pred_df['letter_prediction'] = letter_predictions
pred_df.head()
pred_df['letter_rating'] = df.rating
pred_df
pred_df = pred_df[['letter_rating', 'letter_prediction']]
pred_df.head()
pred_df[pred_df.letter_rating == pred_df.letter_prediction].shape[0]/df.shape[0]
def getAccuracy(predictions, df) :
    pred_df = df.copy()
    pred_df['prediction'] = predictions
    letter_predictions = []
    for prediction in predictions:
        if prediction <= 25.0:
            letter_predictions.append('A')
        elif (prediction > 25.0) and (prediction <= 50.0):
            letter_predictions.append('B')
        elif (prediction > 50.0) and (prediction <= 75.0):
            letter_predictions.append('C')
        elif (prediction > 75.0) and (prediction <= 100.0):
            letter_predictions.append('D')
        elif (prediction > 100.0) and (prediction <= 125.0):
            letter_predictions.append('E')
        elif (prediction > 125.0) and (prediction <= 150.0):
            letter_predictions.append('F')
        else:
            letter_predictions.append('G')
        
    pred_df['letter_prediction'] = letter_predictions
    pred_df['letter_rating'] = pred_df['rating'].replace([12.5, 38.0, 63.0, 88.0, 113.0, 138.0, 163.0],
                                                         ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    pred_df = pred_df[['letter_rating', 'letter_prediction']]
    return pred_df[pred_df.letter_rating == pred_df.letter_prediction].shape[0]/pred_df.shape[0]
accuracy_results = {}
#test function on the one we already tried the long way
accuracy_results['tree_4'] = getAccuracy(predictions, regr)
accuracy_results
tree = DecisionTreeRegressor()
tree.fit(X,y)
accuracy_results['tree'] = getAccuracy(tree.predict(X), regr)
accuracy_results
tree_3 = DecisionTreeRegressor(max_depth=3)
tree_3.fit(X,y)
accuracy_results['tree_3'] = getAccuracy(tree_3.predict(X), regr)

tree_2 = DecisionTreeRegressor(max_depth=2)
tree_2.fit(X,y)
accuracy_results['tree_2'] = getAccuracy(tree_2.predict(X), regr)
accuracy_results
lasso = Lasso()
lasso.fit(X,y)
accuracy_results['lasso'] = getAccuracy(lasso.predict(X), regr)
accuracy_results
results = pd.DataFrame.from_dict(accuracy_results, orient='index')
results = results.rename(columns={0: 'accuracy'})
results