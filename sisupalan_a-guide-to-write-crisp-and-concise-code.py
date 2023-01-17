# Selective library imports
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Selectively import functions
from math import sqrt
from IPython.display import display
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report, confusion_matrix
# Disable warnings. This is not a good practice generally, but for the sake of aesthetics we are disabling this :D
warnings.filterwarnings("ignore")

# Suppress scientific notation
pd.options.display.float_format = '{:20,.2f}'.format

# Set plot sizes
sns.set(rc={'figure.figsize':(11.7,8.27)})

# Set plotting style
plt.style.use('fivethirtyeight')
def plot_dist(df, var, target, var_type='num'):
    
    '''Function helper to facet on target variable'''
    
    if var_type == 'num':
        sns.distplot(df.query('target == 1')[var].tolist() , color="red", label="{} for target == 1".format(var))
        sns.distplot(df.query('target == 0')[var].tolist() , color="skyblue", label="{} for target == 0".format(var))
        plt.legend()
        
    else:
        fig, ax = plt.subplots(1,2)
        sns.countplot(data=df.query('target == 1') , color="salmon", x=var, label="{} for target == 1".format(var), ax=ax[0])
        sns.countplot(data=df.query('target == 0') , color="skyblue", x=var, label="{} for target == 0".format(var), ax=ax[1])
        fig.legend()
        fig.show()
def process_data(df, test_size=0.3, random_state=1, scale=True, scaler=MinMaxScaler(), feature_selection=True, k=10):
    
    '''Function helper to generate train and test datasets and apply transformations if any'''
    
    
    # Dummify columns
    dummy_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    df = pd.get_dummies(df, columns=dummy_cols)
    
    
    # All the columns
    cols = df.columns.tolist()
    
    # X cols
    cols = [col for col in cols if 'target' not in col] 
    
    # Subset x and y
    X = df[cols]
    y = df['target']
    
    # Feature selection
    if feature_selection == True:
        
        k_best = SelectKBest(score_func=chi2, k=k)
        selector = k_best.fit(X, y)
        selection_results = pd.DataFrame({'feature' : cols, 'selected' : selector.get_support()})
        selected_features = list(selection_results.query('selected == True')['feature'])
        X = X[selected_features]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    
    # Make a copy to apply on. Else Set-copy warning will be displayed
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    # Scale columns if needed
    if scale == True:
        scale_cols = ['age', 
                      'trestbps', 
                      'chol', 
                      'thalach', 
                      'oldpeak']
        
        # If any features are dropped from feature selection we need to account for that
        scale_cols = list(set(selected_features) & set(scale_cols))
        
        # Define scaler to use
        scaler = scaler

        # Apply scaling
        X_train_copy.loc[:, scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test_copy.loc[:, scale_cols] = scaler.transform(X_test[scale_cols])
      
    # Return train and tests
    return X_train_copy, X_test_copy, y_train, y_test
def select_model(X_train, y_train, cv=3, nruns=3, scorer='recall'):
    
    '''Function helper to automate selection of best baseline model without hyperparameter tuning'''

    record_scorer = []
    iter_scorer = []
    model_name = []
    model_accuracy = []

    # Specify estimators
    estimators = [('logistic_regression' , LogisticRegression()), 
                  ('random_forest' , RandomForestClassifier(n_estimators=100)),
                  ('lightgbm' , LGBMClassifier(n_estimators=100)), 
                  ('xgboost' , XGBClassifier(n_estimators=100))]


    scorer = scorer
    
    # Iterate through the number of runs. Default is 3.
    for run in range(nruns):
        print('Running iteration %s with %s as scoring metric' % ((run + 1), scorer))

        for name, estimator in estimators:

            print('Fitting %s model' % name)

            # Run cross validation
            cv_results = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=scorer)

            # Append all results in list form which will be made into a dataframe at the end.
            iter_scorer.append((run + 1))
            record_scorer.append(scorer)
            model_name.append(name)
            model_accuracy.append(cv_results.mean())

        print()

    # Use ordered dictionary to set the dataframe in the exact order of columns declared.
    results = pd.DataFrame(OrderedDict({'Iteration' : iter_scorer, 
                                        'Scoring Metric' : record_scorer, 
                                        'Model' : model_name, 
                                        'Model Accuracy' : model_accuracy}))
    
    # Pivot to view results in a more aesthetic form
    results_pivot = results.pivot_table(index=['Iteration', 'Scoring Metric'], columns=['Model'])
    
    # Display the results
    print('\nFinal results : ')
    display(results_pivot)

    # Get the mean performance
    performance = results_pivot.apply(np.mean, axis=0)
    performance = performance.reset_index()
    performance.columns = ['metric', 'model', 'performance']
    
    # Get the mean performance
    performance = results_pivot.apply(np.mean, axis=0)
    performance = performance.reset_index()
    performance.columns = ['metric', 'model', 'performance']
    best_model = performance.loc[performance['performance'].idxmax()]['model']

    # Return the pivot 
    return results_pivot, best_model
def tune_model(X_train, X_test, y_train, y_test, best_model, scorer='recall'):
    
    # Define parameters for each model
    grid = {'logistic_regression' : {'model' : LogisticRegression(class_weight='balanced', random_state=42), 
                                    'params' : {'C' : [0.01, 0.1, 1, 10, 100]}},

            'random_forest' : {'model' : RandomForestClassifier(class_weight='balanced', random_state=42), 
                            'params' : {'n_estimators' : [100, 200, 300], 
                                        'max_depth' : [3, 5, 7], 
                                        'max_features' : ['log2', 5, 'sqrt']}},

            'lightgbm' : {'model' : LGBMClassifier(class_weight='balanced', random_state=42), 
                        'params' : {'n_estimators' : [100, 200, 300], 
                                    'max_depth' : [3, 5, 7], 
                                    'boosting_type' : ['gbdt', 'dart', 'goss']}},

            'xgboost' : {'model' : XGBClassifier(nthread=-1), 
                        'params' : {'n_estimators' : [100, 200, 300], 
                                    'max_depth' : [3, 5, 7], 
                                    'scale_pos_weight' : [5, 10, 20]}}                        
                                
        }

    # Select the best model
    model = grid[best_model]['model']

    # Define the grid
    params = grid[best_model]['params']

    # 3 Fold Cross Validation
    grid = GridSearchCV(model, cv=3, param_grid=params, scoring=scorer, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    

    return(grid)
def model_performance(X_train, X_test, y_train, y_test, grid):
      
    # Select the model with the best paramters
    model = grid.best_estimator_

    # Fit the model on the data
    model.fit(X_train, y_train)
    

    # Get the training predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test) 
    
    # Get the train and test probabilities
    train_probabilities = model.predict_proba(X_train)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    # Get the accuracy for train and test
    print('Accuracy score for training is : %s' % accuracy_score(y_train, train_predictions))
    print('Accuracy score for testing is : %s' % accuracy_score(y_test, test_predictions))
    
    # Get the classification report for train and test
    print('\nClassification report for training is : \n%s' % classification_report(y_train, train_predictions))
    print('Classification report for testing is : \n%s' % classification_report(y_test, test_predictions))
    
    # Get the confusion matrix for train and test
    print('\nConfusion matrix for training is : \n%s' % confusion_matrix(y_train, train_predictions))
    print('Confusion matrix for testing is : \n%s' % confusion_matrix(y_test, test_predictions))
    
    # Get the ROC AUC for train and test
    print('\nROC AUC score for training is : %s' % roc_auc_score(y_train, train_probabilities))
    print('ROC AUC score for testing is : %s' % roc_auc_score(y_test, test_probabilities))
df = pd.read_csv('../input/heart.csv')
df.shape
# Summary statistic
df.describe()
# Data types of columns
df.dtypes
# Distribution of target
df['target'].value_counts([0])
df.isnull().sum()
# first few rows of data
df.head()
plot_dist(df, 'age', 'target')
plot_dist(df, 'sex', 'target', 'cat')
plot_dist(df, 'cp', 'target', 'cat')
plot_dist(df, 'trestbps', 'target')
plot_dist(df, 'chol', 'target')
plot_dist(df, 'fbs', 'target', 'cat')
plot_dist(df, 'restecg', 'target', 'cat')
plot_dist(df, 'thalach', 'target')
plot_dist(df, 'exang', 'target', 'cat')
plot_dist(df, 'oldpeak', 'target')
plot_dist(df, 'slope', 'target', 'cat')
plot_dist(df, 'ca', 'target', 'cat')
plot_dist(df, 'thal', 'target', 'cat')
# Get the train and test datasets
X_train, X_test, y_train, y_test = process_data(df, test_size=0.3, random_state=100, scale=True, scaler=MinMaxScaler(), feature_selection=True, k=10)
# View the train dataset
X_train.head()
# View the test dataset
X_test.head()
# Display results of each model
results_pivot, best_model = select_model(X_train, y_train, cv=5, nruns=10, scorer='balanced_accuracy')
results_pivot
# Print and tune the best model
print('Tuning model for {}'.format(best_model))
grid = tune_model(X_train, X_test, y_train, y_test, best_model, scorer='balanced_accuracy')
grid
# Display model performance
model_performance(X_train, X_test, y_train, y_test, grid)