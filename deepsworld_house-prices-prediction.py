import pandas as pd

# extract data into dataframe
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# print columns
train.columns
# prints shape which is (1460 * 81) 
train.shape
# prints first few values of the datset with their columns
train.head()
# description
train.describe()
# target 
y_train = train.SalePrice

# drops the Id and SalePrice columns
X_train = train.drop(['Id','SalePrice'], axis= 1)

# drops Id column from test
X_test = test.drop(['Id'], axis= 1)

# code to encode object data(text data) using one-hot encoding(commonly used)
one_hot_encoded_training_data = pd.get_dummies(X_train)
one_hot_encoded_testing_data = pd.get_dummies(X_test)

# align command make sure that the columns in both the datasets are in same order
final_train, final_test = one_hot_encoded_training_data.align(one_hot_encoded_testing_data,join='inner',axis=1)
# to check the increased number of columns
final_train.shape
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import numpy as np

# RandomForestRegressor:
for n in range(10,200,10):
    # define pipeline
    pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor(max_leaf_nodes=n,random_state=1))
    # cross validation score
    scores = cross_val_score(pipeline, final_train, y_train, scoring= 'neg_mean_absolute_error')
    print(n,scores)

# XGBRegressor:
# define pipeline
pipeline = make_pipeline(SimpleImputer(), XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                            nthread = -1, random_state=1))
# cross validation score
scores = cross_val_score(pipeline,final_train, y_train, scoring= 'neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * scores.mean()))
#Validation function


# GradientBoostingRegressor:(just another model)
# define pipeline
my_pipeline = make_pipeline(SimpleImputer(), GradientBoostingRegressor())
# cross validation score
score = cross_val_score(my_pipeline,final_train, y_train, scoring= 'neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * score.mean()))

# fit and make predictions
pipeline.fit(final_train,y_train)
predictions= pipeline.predict(final_test)

print(predictions)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
def get_some_data():
    cols_to_use = ['YearBuilt', 'TotRmsAbvGrd', 'LotArea']
    data = pd.read_csv('../input/train.csv')
    y = data.SalePrice
    X = data[cols_to_use]
    my_imputer = SimpleImputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y


# get_some_data is defined in hidden cell above.
X, y = get_some_data()
# scikit-learn originally implemented partial dependence plots only for Gradient Boosting models
# this was due to an implementation detail, and a future release will support all model types.
my_model = GradientBoostingRegressor()
# fit the model as usual
my_model.fit(X, y)
# Here we make the plot
my_plots = plot_partial_dependence(my_model,       
                                   features=[0,2], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names=['YearBuilt', 'TotRmsAbvGrd', 'LotArea'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis