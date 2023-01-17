import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
iowa_data = pd.read_csv(iowa_file_path) 

##Separating data in feature predictors, and target to predict
iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

### Separate dataset samples into train and test data 
X_train, X_test, y_train, y_test = train_test_split(iowa_predictors, 
                                                    iowa_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

### Getting OHE train and test datasets
X_train_OHE = pd.get_dummies(X_train)
X_test_OHE = pd.get_dummies(X_test)

### Getting aligned train & test datasets for both model evaluations
finalX_train, finalX_test = X_train_OHE.align(X_test_OHE,
                                            join='left', 
                                            axis=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
my_pipeline.fit(finalX_train, y_train)
predictions = my_pipeline.predict(finalX_test)

from sklearn.metrics import mean_absolute_error

### Calculate the Mean absolute error of the model
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))