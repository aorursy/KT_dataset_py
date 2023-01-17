# Extreme Gradient Boost approach on planning analysis

#setup
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import seaborn as sns

#name data
dproject= pd.read_csv('../input/1 - Project.csv')
dcost= pd.read_csv('../input/2 - Cost.csv')
dcostp= pd.read_csv('../input/2B - Cost and Project.csv')
dplan= pd.read_csv('../input/3 - Planning.csv')
dplanp= pd.read_csv('../input/3B - Planning and Project.csv')
dmile= pd.read_csv('../input/4 - Milestone.csv')
dmilep= pd.read_csv('../input/4B - Milestone and Project.csv')

#pull data into target (y) and predictors (X)
#turn success values into binary
dplanp['Success?']= dplanp['Success?'].map({'Yes':1, 'No':0})
y= dplanp['Success?']
predictor_cols= ['Cumul_Planned','Period_Planned','Cumul_Earned','Period_Earned']
#Create training predictors data
X= dplanp[predictor_cols]

#split data into test and training sets
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

#imputate missing values
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
#create model with learning rate= 0.05 and n-est cycles= 1000, with early stop= 5 rounds
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
#partial dependence plots to show correlation
def get_some_data():
    predictor_cols= ['Cumul_Planned','Period_Planned','Cumul_Earned','Period_Earned']
    dplanp= pd.read_csv('../input/3B - Planning and Project.csv')
    dplanp['Success?']= dplanp['Success?'].map({'Yes':1, 'No':0})
    y= dplanp['Success?']
    X= dplanp[predictor_cols]
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y


from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import Imputer

X, y = get_some_data()
my_model = GradientBoostingRegressor()
my_model.fit(X, y)
# make the plot
my_plots = plot_partial_dependence(my_model,       
                                   features=[0,2], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names=['Cumul_Planned','Period_Planned','Cumul_Earned','Period_Earned'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis
# make predictions and determine mean absolute error of model
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

from sklearn.metrics import mean_squared_error
print("Mean Squared Error : " + str(mean_squared_error(predictions, test_y)))