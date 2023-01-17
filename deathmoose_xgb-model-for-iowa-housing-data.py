# Setting up the environment

import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Reading in Iowa housing data from csv file
training_file_path = '../input/train.csv' # path to csv file of Iowa housing data
data = pd.read_csv(training_file_path)

# Reading in the test Iowa data
test_file_path = "../input/test.csv"
test = pd.read_csv(test_file_path)
## Preprocess the data
# Identifying labels and features in training data
training_labels = data.SalePrice
features = data.drop(['SalePrice', 'Id'], axis=1)
features['label'] = 'train'
test_features = test.drop(['Id'], axis=1)
test_features['label'] = 'test'
combined_features = pd.concat([features, test_features])
one_hot_features = pd.get_dummies(combined_features)
training_features = one_hot_features[one_hot_features.label_train == 1]
training_features = training_features.drop(['label_train', 'label_test'], axis=1)
test_features = one_hot_features[one_hot_features.label_test == 1]
test_features = test_features.drop(['label_train', 'label_test'], axis=1)
# Creating functions to evaluate a model and determine optimal hyperparameters

def evaluate_XGB(model_learning_rate):
    """
    A function for evaluating cross validation

    Args:
        model_learning_rate: float, the learning rate of the model being assessed
    Returns:
        mean_absolute_error: float, the average error of the folds
    """
    # Build pipeline
    my_pipeline = make_pipeline(SimpleImputer(), XGBRegressor(learning_rate=model_learning_rate, n_estimators=1000))
    # Fit pipeline
    my_pipeline.fit(training_features, training_labels)
    # Get cross validation scores    
    scores = cross_val_score(my_pipeline, 
                         training_features, 
                         training_labels,
                         scoring='neg_mean_absolute_error')
    # return the negative mean of the scores, the mean absolute error
    mean_absolute_error = -1 * scores.mean()
    return mean_absolute_error


def compare_XGB(learning_rate_list):
    """
    A function for comparing the effects of different learning rates in XGB models
    
    Args:
        learning_rate_list: a list of different learning rates to test
    Returns:
        errors_list: list, contains mean absolute errors of the models being compared
    """
    errors_list = []
    for model_learning_rate in learning_rate_list:
        mean_absolute_error = evaluate_XGB(model_learning_rate)
        print("learning rate: %s MAE: %s" %(model_learning_rate, mae))
        errors_list.append(mean_absolute_error)
    return errors_list

def optimize_learning_rate(learning_rate, step):
    """
    A function for selecting the optimal learning rate for an XGB model
    
    Args:
        learning_rate: the initial learning rate that will be tested
        step: the step size between learning_rates
        max_recursion: int, the maximum amount of times the function will be allowed to
            execute before halting, prevents infinite loops when model is not converging
        
    Returns:
        learning_rate: float, the determined optimal learning_rate from the function
    """
    def recursive_optimization(learning_rate=learning_rate,
                               step=step,
                               mae0 = evaluate_XGB(learning_rate),
                               mae1 = evaluate_XGB(learning_rate + step),
                               mae2 = evaluate_XGB(learning_rate + 2*step),
                               mae3 = evaluate_XGB(learning_rate + 3*step),
                               mae4 = evaluate_XGB(learning_rate + 4*step),
                               mae5 = evaluate_XGB(learning_rate + 5*step),
                               mae6 = evaluate_XGB(learning_rate + 6*step),
                               mae7 = evaluate_XGB(learning_rate + 7*step),
                               mae8 = evaluate_XGB(learning_rate + 8*step),
                               mae9 = evaluate_XGB(learning_rate + 9*step),
                               mae10 = evaluate_XGB(learning_rate + 10*step),
                               max_recursions=100, 
                               recursions=0):
        if mae0 < min([mae1, mae2, mae3, mae4, mae5, mae6, mae7, mae8, mae9, mae10]):
        #mae1 and mae0 < mae2 and mae0 < mae3 and mae0 < mae4 and mae0 < mae5 and mae0 < mae6 and mae0 < mae7 and mae0 < mae8 and mae0 < mae9 and mae0 < mae10:
            print("optimal learning_rate: %s \nMAE: %s" %(learning_rate, mae0))
        else:
            while recursions < max_recursions:        
                recursions += 1
                return(recursive_optimization(learning_rate + step, 
                                              step, 
                                              mae0=mae1,
                                              mae1=mae2,
                                              mae2=mae3,
                                              mae3=mae4,
                                              mae4=mae5,
                                              mae5=mae6,
                                              mae6=mae7,
                                              mae7=mae8,
                                              mae8=mae9,
                                              mae9=mae10,
                                              mae10 = evaluate_XGB(learning_rate + 11*step),
                                              recursions = recursions))
            print("Possibly entering infinite loop. Select new learning rate and step size.")
        return learning_rate
    learning_rate = recursive_optimization()
    return learning_rate
    
# Determine the optimal learning rate
#optimal_learning_rate = optimize_learning_rate(0.02, 0.001)
# The optimal learning rate was calculated to be 0.051 with MAE of 15280.773702057026
## Creating output for submision

# Define pipeline
my_pipeline = make_pipeline(SimpleImputer(), 
                            XGBRegressor(learning_rate=0.016, n_estimators=1000))

# Train pipeline
my_pipeline.fit(training_features, training_labels)

# Generate predictions
test_predictions = my_pipeline.predict(test_features)

# Output predicitons to a csv
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': test_predictions})
submission.to_csv('submission.csv', index=False)