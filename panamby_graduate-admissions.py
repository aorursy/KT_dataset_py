# I will use in this Kernel the step-by-step process of Will Koehrsen.
# I won't use everything, but most of them.
# This project at in GitHub repository: https://github.com/WillKoehrsen/machine-learning-project-walkthrough
# Let's import the main libraries that I will use in this dataset.

# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Display up to 9 columns of a dataframe
pd.set_option('display.max_columns', 9)

# Matplotlib visualization
import matplotlib.pyplot as plt
%matplotlib inline

# Set default font size
plt.rcParams['font.size'] = 24

# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Splitting data into training and testing
from sklearn.model_selection import train_test_split
# I will check the two CSV files to see what the difference between them.

graduate_first = pd.read_csv('../input/Admission_Predict.csv')
graduate_first.head()
graduate_first.shape
graduate_second = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
graduate_second.head()
graduate_second.shape
# I saw that maybe the second one is a recent version, so I will use this one.
graduate = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

# I will drop the 'Serial No' because it's not important for our model.
graduate.drop(labels='Serial No.', axis=1, inplace=True)
# See the column data types and non-missing values.
graduate.info()

# Apparently we don't have any missing values;
# We don't have any 'object' column to convert to 'float' or 'int'.
# Statistics for each column

graduate.describe()
# with the pourpuse to be sure about no missing values in our dataset. I will code a function which will help us on it.

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
missing_values_table(graduate)

# Great! Now we now that for sure we don't have any missing values.
# Let's start now the Exploratory Data Analysis (EDA) to understand better our dataset and see the correlations among the variables.

# Fist I will see the name of the columns. The goal here is read the name of the columns and realize some strange names or even errors.
# Sometimes is a good practice to rename some of them to easy manipulate.

graduate.columns

# We can see below that some column names has a space in the end, is good to rename them droping these spaces.
# To manipule better the columns, I will chance the name of some of them as weel.
graduate.rename(columns = {'Serial No.': 'SerialNo', 'GRE Score': 'GRE', 'TOEFL Score': 'TOEFL', 'University Rating': 'UniversityRating', 'LOR ': 'LOR', 'Chance of Admit ': 'Chance'}, inplace=True)
graduate.columns
# First of all, I will see the correlation between any variable with the target.

# I will drop the 'SerialNo' and 'Research' columns because the serial number just identify the student and the Research has a boolean value and I will 
# use in the 'hue' parameter.

fig = plt.figure(figsize=(30,20))
fig.subplots_adjust(hspace=0.3, wspace=0.2)
for i in range(1, 7):
    ax = fig.add_subplot(3, 3, i)
    sns.scatterplot(x=graduate['Chance'], y= graduate.iloc[:,i], hue=graduate.Research)
    plt.xlabel('Chance of Admit')
    plt.ylabel(graduate.columns[i])
    
# Conclusions:
#    - The better graph of the features 'UniversityRating', 'SOP', 'LOR' and 'Research' is not scatterplot;
#    - 'GRE', 'TOEFL' and 'CGPA' graphs have a linear behavior;
#    - The tendency which we can see is, as higher as the 'GRE', 'TOEFL' and 'CGPA' higher is the chance of admission;
#    - The other tendency that we can see is if the person has a research has more probability to be admitted.
fig = plt.figure(figsize=(20,8))
fig.subplots_adjust(hspace=0.1, wspace=0.3)
for i in range(1, 4):
    ax = fig.add_subplot(1, 3, i)
    sns.lineplot(x= graduate.iloc[:,i+2], y= graduate['Chance'], hue=graduate.Research)
    plt.xlabel(graduate.columns[i+2])
    plt.ylabel('Chance of Admit')
    
# Conclusion:
#    - Here we can see again a linear correlation between these variables and the target;
#    - The tendency which we can see is, as higher as the 'UniversityRating', 'SOP' and 'LOR' higher is the chance of admission;
#    - The other tendency that we can see is if the person has a research has more probability to be admitted.
# Now we will remove the outliers

# I will use a stats concept (formula) to figure out the outliers that maybe can there are in my dataset.

for i in graduate.columns:
    # Calculate first and third quartile
    first_quartile = graduate[i].describe()['25%']
    third_quartile = graduate[i].describe()['75%']

    # Interquartile range
    iqr = third_quartile - first_quartile

    # Remove outliers
    graduate = graduate[(graduate[i] > (first_quartile - 3 * iqr)) & (graduate[i] < (third_quartile + 3 * iqr))]
# Let's quantify the correlations between the features with the target and see what variables have more impact in the admisson.

# Find all correlations and sort 
correlations_data = graduate.corr()['Chance'].sort_values(ascending=False)

# Print the correlations
print(correlations_data)

# Conclusions:
#    - We have basic three groups of influencers: high(CGPA, GRE and TOEFL), intermediary(University rating, SOP and LOR) and low(Research);
#    - All of them have a positive influence.
# # # Split Into Training and Testing Sets

# Separate out the features and targets
features = graduate.drop(columns='Chance')
targets = pd.DataFrame(graduate['Chance'])

# Split into 70% training and 30% testing set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# # # Establish a Baseline

# # Metric: Mean Absolute Error

# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))
# Now we can make the median guess and evaluate it on the test set.
baseline_guess = np.median(y_train)

print('The baseline guess is a score of %0.2f' % baseline_guess)
print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))
# # # Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# # # Evaluating and Comparing Machine Learning Models

# Imputing missing values and scaling values
from sklearn.preprocessing import Imputer, MinMaxScaler

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
# Create an imputer object with a median filling strategy
imputer = Imputer(strategy='median')

# Train on the training features
imputer.fit(X_train)

# Transform both training data and testing data
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
# Convert y to one-dimensional array (vector)
y_train = np.array(y_train).reshape((-1, ))
y_test = np.array(y_test).reshape((-1, ))
# # # Models to Evaluate

# We will compare five different machine learning models:

# 1 - Linear Regression
# 2 - Support Vector Machine Regression
# 3 - Random Forest Regression
# 4 - Gradient Boosting Regression
# 5 - K-Nearest Neighbors Regression

# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    
    # Return the performance metric
    return model_mae
# # Linear Regression

lr = LinearRegression()
lr_mae = fit_and_evaluate(lr)

print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)
# # SVM

svm = SVR(C = 1000, gamma = 0.1)
svm_mae = fit_and_evaluate(svm)

print('Support Vector Machine Regression Performance on the test set: MAE = %0.4f' % svm_mae)
# # Random Forest

random_forest = RandomForestRegressor(random_state=60)
random_forest_mae = fit_and_evaluate(random_forest)

print('Random Forest Regression Performance on the test set: MAE = %0.4f' % random_forest_mae)
# # Gradiente Boosting Regression

gradient_boosted = GradientBoostingRegressor(random_state=60)
gradient_boosted_mae = fit_and_evaluate(gradient_boosted)

print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' % gradient_boosted_mae)
# # KNN

knn = KNeighborsRegressor(n_neighbors=10)
knn_mae = fit_and_evaluate(knn)

print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f' % knn_mae)
# Now, to better understand the results, I will show in a graph the model that has the better MEAN (closer to original MEAN)

plt.style.use('fivethirtyeight')
figsize(8, 6)

# Dataframe to hold the results
model_comparison = pd.DataFrame({'model': ['Linear Regression', 'Support Vector Machine',
                                           'Random Forest', 'Gradient Boosted',
                                            'K-Nearest Neighbors'],
                                 'mae': [lr_mae, svm_mae, random_forest_mae, 
                                         gradient_boosted_mae, knn_mae]})

# Horizontal bar chart of test mae
model_comparison.sort_values('mae', ascending = False).plot(x = 'model', y = 'mae', kind = 'barh',
                                                           color = 'red', edgecolor = 'black')

# Plot formatting
plt.ylabel(''); plt.yticks(size = 14); plt.xlabel('Mean Absolute Error'); plt.xticks(size = 14)
plt.title('Model Comparison on Test MAE', size = 20);
# # # Model Optimization

# # Hyperparameter

# Hyperparameter Tuning with Random Search and Cross Validation

# Here we will implement random search with cross validation to select the optimal hyperparameters for the gradient boosting regressor. 
# We first define a grid then peform an iterative process of: randomly sample a set of hyperparameters from the grid, evaluate the hyperparameters using 4-fold cross-validation, 
# and then select the hyperparameters with the best performance.

# Loss function to be optimized
loss = ['ls', 'lad', 'huber']

# Number of trees used in the boosting process
n_estimators = [100, 500, 900, 1100, 1500]

# Maximum depth of each tree
max_depth = [2, 3, 5, 10, 15]

# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 10]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}
# In the code below, we create the Randomized Search Object passing in the following parameters:

#    estimator: the model
#    param_distributions: the distribution of parameters we defined
#    cv the number of folds to use for k-fold cross validation
#    n_iter: the number of different combinations to try
#    scoring: which metric to use when evaluating candidates
#    n_jobs: number of cores to run in parallel (-1 will use all available)
#    verbose: how much information to display (1 displays a limited amount)
#    return_train_score: return the training score for each cross-validation fold
#    random_state: fixes the random number generator used so we get the same results every run
# The Randomized Search Object is trained the same way as any other scikit-learn model. 
# After training, we can compare all the different hyperparameter combinations and find the best performing one.

# Create the model to use for hyperparameter tuning
model = GradientBoostingRegressor(random_state = 42)

# Set up the random search with 4-fold cross validation
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=4, n_iter=25, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)
# Fit on the training data
random_cv.fit(X_train, y_train)
# Scikit-learn uses the negative mean absolute error for evaluation because it wants a metric to maximize. 
# Therefore, a better score will be closer to 0. We can get the results of the randomized search into a dataframe, and sort the values by performance.

# Get all of the cv results and sort by the test performance
random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)

random_results.head(10)
random_cv.best_estimator_
# The best gradient boosted model has the following hyperparameters:

# loss = lad
# n_estimators = 500
# max_depth = 2
# min_samples_leaf = 8
# min_samples_split = 6
# max_features = None 
# I will focus on a single one, the number of trees in the forest (n_estimators).
# By varying only one hyperparameter, we can directly observe how it affects performance. 
# In the case of the number of trees, we would expect to see a significant affect on the amount of under vs overfitting.

# Here we will use grid search with a grid that only has the n_estimators hyperparameter. 
# We will evaluate a range of trees then plot the training and testing performance to get an idea of what increasing the number of trees does for our model. 
# We will fix the other hyperparameters at the best values returned from random search to isolate the number of trees effect.
# Create a range of trees to evaluate
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}

model = GradientBoostingRegressor(loss = 'lad', max_depth = 2,
                                  min_samples_leaf = 8,
                                  min_samples_split = 6,
                                  max_features = None,
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)
# Fit the grid search
grid_search.fit(X_train, y_train)
# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize=(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_n_estimators'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_n_estimators'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean Abosolute Error'); plt.legend();
plt.title('Performance vs Number of Trees');

# There will always be a difference between the training error and testing error (the training error is always lower) but if there is a significant difference, 
# we want to try and reduce overfitting, either by getting more training data or reducing the complexity of the model through hyperparameter tuning or regularization.

# For now, we will use the model with the best performance and accept that it may be overfitting to the training set.
results.sort_values('mean_test_score', ascending = False).head(5)
# # # Evaluate Final Model on the Test Set

# We will use the best model from hyperparameter tuning to make predictions on the testing set.

# For comparison, we can also look at the performance of the default model. The code below creates the final model, trains it (with timing), and evaluates on the test set.

# Default model
default_model = GradientBoostingRegressor(random_state = 42)

# Select the best model
final_model = grid_search.best_estimator_

final_model
%%timeit -n 1 -r 5
default_model.fit(X_train, y_train)
%%timeit -n 1 -r 5
final_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)
final_pred = final_model.predict(X_test)

print('Default model performance on the test set: MAE = %0.4f.' % mae(y_test, default_pred))
print('Final model performance on the test set:   MAE = %0.4f.' % mae(y_test, final_pred))

# The model have the very good performace!!!
# To get a sense of the predictions, we can plot the distribution of true values on the test set and the predicted values on the test set.

# Train the model.
lr.fit(X_train, y_train)
    
# Make predictions and evalute.
model_pred = lr.predict(X_test)
    
figsize=(8, 8)

# Density plot of the final predictions and the test values.
sns.kdeplot(model_pred, label = 'Predictions')
sns.kdeplot(y_test, label = 'Values')

# Label the plot.
plt.xlabel('Chance of Admission'); plt.ylabel('Density');
plt.title('Test Values and Predictions');

# The distribution looks to be nearly the same.
# Another diagnostic plot is a histogram of the residuals. 
# Ideally, we would hope that the residuals are normally distributed, meaning that the model is wrong the same amount in both directions (high and low).

figsize = (6, 6)

# Calculate the residuals 
residuals = model_pred - y_test

# Plot the residuals in a histogram
plt.hist(residuals, color = 'red', bins = 20,
         edgecolor = 'black')
plt.xlabel('Error'); plt.ylabel('Count')
plt.title('Distribution of Residuals');

# The residuals are close to normally disributed, with a few noticeable outliers on the low end. 
# These indicate errors where the model estimate was far below that of the true value.
model.fit(X_train, y_train)
# # # Interprete the Model

# # Feature Importances

# Extract the feature importances into a dataframe
graduate_features = graduate.drop(labels='Chance', axis=1)
feature_results = pd.DataFrame({'feature': list(graduate_features.columns), 
                                'importance': model.feature_importances_})

# Show the top 10 most important
feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)

feature_results.head(10)
# Let's graph the feature importances to compare visually.

figsize=(12, 10)
plt.style.use('fivethirtyeight')

# Plot the 10 most important features in a horizontal bar chart
feature_results.loc[:9, :].plot(x = 'feature', y = 'importance', 
                                 edgecolor = 'k',
                                 kind='barh', color = 'blue');
plt.xlabel('Relative Importance', size = 20); plt.ylabel('')
plt.title('Feature Importances', size = 30);
# # Use Feature Importances for Feature Selection

# Let's try using only the 10 most important features in the linear regression to see if performance is improved.
# We can also limit to these features and re-evaluate the random forest.

# Extract the names of the most important features
most_important_features = feature_results['feature'][:10]

# Find the index that corresponds to each feature name
indices = [list(graduate_features.columns).index(x) for x in most_important_features]

# Keep only the most important features
X_train_reduced = X_train[:, indices]
X_test_reduced = X_test[:, indices]

print('Most important training features shape: ', X_train_reduced.shape)
print('Most important testing  features shape: ', X_test_reduced.shape)
lr = LinearRegression()

# Fit on full set of features
lr.fit(X_train, y_train)
lr_full_pred = lr.predict(X_test)

# Fit on reduced set of features
lr.fit(X_train_reduced, y_train)
lr_reduced_pred = lr.predict(X_test_reduced)

# Display results
print('Linear Regression Full Results: MAE =    %0.4f.' % mae(y_test, lr_full_pred))
print('Linear Regression Reduced Results: MAE = %0.4f.' % mae(y_test, lr_reduced_pred))

# Well, reducing the features did not improve the linear regression results! 
# It turns out that the extra information in the features with low importance do actually improve performance.
# Let's look at using the reduced set of features in the gradient boosted regressor.

# Create the model with the same hyperparamters
model_reduced = GradientBoostingRegressor(loss='lad', max_depth=2, max_features=None,
                                  min_samples_leaf=8, min_samples_split=6, 
                                  n_estimators=800, random_state=42)

# Fit and test on the reduced set of features
model_reduced.fit(X_train_reduced, y_train)
model_reduced_pred = model_reduced.predict(X_test_reduced)

print('Gradient Boosted Reduced Results: MAE = %0.4f' % mae(y_test, model_reduced_pred))

# The model results are slightly worse with the reduced set of features and we will keep all of the features for the final model
# # Locally Interpretable Model-agnostic Explanations

# We will look at using LIME to explain individual predictions made the by the model. 
#LIME is a relatively new effort aimed at showing how a machine learning model thinks by approximating the region around a prediction with a linear model.

# We will look at trying to explain the predictions on an example the model gets very wrong and an example the model gets correct. 
#We will restrict ourselves to using the reduced set of 10 features to aid interpretability. 
#The model trained on the 10 most important features is slightly less accurate, but we generally have to trade off accuracy for interpretability!
# Find the residuals
residuals = abs(model_reduced_pred - y_test)
    
# Exact the worst and best prediction
wrong = X_test_reduced[np.argmax(residuals), :]
right = X_test_reduced[np.argmin(residuals), :]
# Create a lime explainer object
# LIME for explaining predictions
import lime 
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(training_data = X_train_reduced, 
                                                   mode = 'regression',
                                                   training_labels = y_train,
                                                   feature_names = list(most_important_features))
# Display the predicted and true value for the wrong instance
print('Prediction: %0.4f' % model_reduced.predict(wrong.reshape(1, -1)))
print('Actual Value: %0.4f' % y_test[np.argmax(residuals)])

# Explanation for wrong prediction
wrong_exp = explainer.explain_instance(data_row = wrong, 
                                       predict_fn = model_reduced.predict)

# Plot the prediction explaination
wrong_exp.as_pyplot_figure();
plt.title('Explanation of Prediction', size = 28);
plt.xlabel('Effect on Prediction', size = 22);

# In this example, our gradient boosted model predicted a score of 0.7288 and the actual value was 0.45.

# The plot from LIME is showing us the contribution to the final prediction from each of the features for the example.

# We can see that the GRE singificantly increased the prediction when we comparing with the others. 
# The Research on the other hand, decreased the prediction when we comparing with the others.
# Now we can go through the same process with a prediction the model got correct.

# Display the predicted and true value for the wrong instance
print('Prediction: %0.4f' % model_reduced.predict(right.reshape(1, -1)))
print('Actual Value: %0.4f' % y_test[np.argmin(residuals)])

# Explanation for wrong prediction
right_exp = explainer.explain_instance(right, model_reduced.predict, num_features=10)
right_exp.as_pyplot_figure();
plt.title('Explanation of Prediction', size = 28);
plt.xlabel('Effect on Prediction', size = 22);

# The correct value for this case was 0.8899 which our gradient boosted model got almost right on!

# The plot from LIME again shows the contribution to the prediciton of each of feature variables for the example.

# Observing break down plots like these allow us to get an idea of how the model makes a prediction. 
# This is probably most valuable for cases where the model is off by a large amount as we can inspect the errors and perhaps engineer better features or adjust the hyperparameters of the model 
# to improve predictions for next time. The examples where the model is off the most could also be interesting edge cases to look at manually.
# A process such as this where we try to work with the machine learning algorithm to gain understanding of a problem seems much better than simply letting the model make predictions
# and completely trusting them! Although LIME is not perfect, it represents a step in the right direction towards explaining machine learning models.
# Good job with this project!
# See you in the next one!!!
# I will use in this Kernel the step-by-step process of Will Koehrsen.
# I won't use everything, but most of them.
# This project at in GitHub repository: https://github.com/WillKoehrsen/machine-learning-project-walkthrough