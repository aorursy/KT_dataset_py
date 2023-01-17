# Special thanks to Dan S. Becker for his amazing Machine Learning tutorial in the Kaggle Learn tab.
# The information below is mostly a more compact/reordered/generalized version of his work.
# NOTE: There will be errors if you try to run this due to lack of data. This is meant to be a reference-sheet.
# ************IMPORTING AND LOOKING AT DATA************
import pandas as pd
# You'll always start a project by importing data.  Find the file path (in the data tab) and then save it as a DataFrame using pandas
my_dataframe = pd.read_csv('../input/train.csv')
# For the amount of rows/columns:
my_dataframe.shape
# If you want summarized info concerning all your numerical columns, use the following.  Or choose a specific col (even categorical).
print(my_dataframe.describe())
# If you want a list of all of the columns, use one of the following lines of code:
print(my_dataframe.columns)
list(my_dataframe.columns.values)
list(my_dataframe)
# For a quick sneak-peek of some sample data:
my_dataframe.head() # By default, prints 5 rows.  Place an int in the parenthesis to change that value i.e. data.head(20)
# ************HOW TO REMOVE/DETECT MISSING DATA************
# Most sci-kit learn libraries give you an error if you build a model using data with missing values.  Let's learn to eliminate it.
data_without_missing_values = original_data.dropna(axis=1) # axis=1 will drop the whole column, 0 will drop the row if it spots a NaN
# If you drop columns from your training data but not your testing data, you'll get errors.  
# You can detect which columns should be dropped and then just apply it to both datasets.  This stores the columns into a list:
cols_with_missing = [col for col in original_data.columns 
                                 if original_data[col].isnull().any()]
# If you would rather not drop anything, you can impute it.  That will be discussed far below in the cell titled "IMPUTATION..."
# ************SELECTING AND SEPARATING DATA************
# Pull the target variable out of your dataframe and store it.  Pandas accepts two syntaxes:
target_variable = my_dataframe.Predictor
target_variable = my_dataframe['Predictor']
# Now you'll need to decide what data to work with as predictors.  You need to AT LEAST take the target variable out.
# But let's say that there's a column that you just know is going to be useless.  Let's wipe that one out, too.
my_dataframe_without_target_variable = my_dataframe.drop(['Predictor', 'UselessCol'], axis=1) # axis=0 is a row, axis=1 is a column
# Alternatively, you might want to just use a few columns.  In that case, it makes more sense to select them directly.
my_dataframe_hand_picked_predictors = my_dataframe['Column1', 'Column2', 'Column3']
# You can select or drop something based on the type of data.  For example, to drop categorical data:
data_without_categoricals = my_dataframe.select_dtypes(exclude=['object']) # include/exclude will often be used with 'number', too.
# ************ONE-HOT ENCODING: CONVERTING CATEGORICAL DATA INTO NUMERIC DATA************
# One-hot coding takes every answer to a column and creates new columns for each of those answers.  The value is 0 or 1.
one_hot_encoded_data = pd.get_dummies(my_dataframe) # get_dummies() ignored numerical data on its own.  It is smart.
# ************X, y, AND SPLITTING INTO TEST/TRAIN************
# By convention, the predictors are called X and the target variable (prediction target, etc) is called y.
X = my_dataframe_without_target_variable
y = target_variable
# It is a really bad idea to fit a model and test the model against its own data.
# If you want to check whether your model works but don't have testing data, you should split what you have:
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25) # test_size is what percent is reserved for testing
# train/predictors: the data you train your model with.  Do not test the model using train data.  There'd be too much overfitting.
# test/validation/val: the data you reserve for testing your model.  Does not affect your model.
# If you do not like this method, or if your dataset is relatively small, use "Cross-Validation" (explanation in a cell far below).
# You may need to use X.as_matrix() or y.as_matrix() in some situations.
# ************IMPUTATION: ALTERNATIVE TO DROPPING MISSING DATA************
# Imputing data fills in the missing values.  There are sophisticated methods of imputation, but they rarely garner better results.
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(train_X)
imputed_X_test = my_imputer.transform(test_X)
# fit_transform() saves the parameters used to calculate the imputed values.  tranform() uses the same parameters to be consistent.
# You might get a warning telling you that this method won't be around forever.  In that case, use this:
from sklearn.impute import SimpleImputer
# This only seems to be a problem when you try to use pipelines, discussed in "PIPELINES" below.
# ************LINING DATA UP************
# If for any reason your train/test data isn't lined up the same way (after drop, one-hot, etc), fix it like this:
alligned_train, alligned_test = train_X.align(test_X, join='left', axis=1)
# ************MODELS: FITTING AND PREDICTING************
from sklearn.tree import DecisionTreeRegressor # A decision tree regressor.  Nothing fancy.
from sklearn.ensemble import RandomForestRegressor # A random forest is just a bunch of decision trees voting on the right answer.
my_model_dt = DecisionTreeRegressor(max_leaf_nodes=500) # It's a good to test a few values for max_leaf_nodes. Not required field.
my_model_rf = RandomForestRegressor() # All you would have to do is switch my_model_dt with my_model_rf below to use rf instead.
my_model_dt.fit(train_X, test_y)
# Great, the model now thinks it can guess predictors.  You can give it a whirl with the following:
predictions = my_model_dt.predict(test_X)
# Cool, we have guesses.  We need to check how they compare to the real values.  One way to do this is to check the MAE
from sklearn.metrics import mean_absolute_error
mean_absolute_error(test_y, predictions)
# ************PIPELINES************
# You can merge the imputation and modelling steps by using a pipeline:
from sklearn.pipeline import make_pipeline
my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor()) # A pipeline must start with a tranfomer and end with a model.
# SimpleImputer() is the tranformer (after fitting a tranformer, you apply it with the 'tranform' command)
# RandomForestRegressor() is the model (after fitting a model, you apply it with the 'predict' command)
my_pipeline.fit(train_X, test_y)
predictions = my_pipeline.predict(test_X) # As you can see, this is virtually the same thing.  It's just cleaner.
# ************CROSS VALIDATION: ALTERNATIVE TO TEST/TRAIN SPLIT************
# When you have little data to work with, cross validation will help you avoid overfitting that test/train split would cause. 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
# sklearn has a convention where higher numbers are ALWAYS good, hence the never-before-used negative MAE.
# If you want to get the mean of all of your cross-validation scores (default 3), do this:
'Mean Absolute Error %2f' %(-1 * scores.mean())
# ************SAVING THE RESULTS************
# If you made some guesses and want to save a csv of the output, do this.  When you run a kernel, this will save in the Output tab.
my_submission = pd.DataFrame({'Id': test.Id, 'Predictor': predictions})
my_submission.to_csv('submission.csv', index=False)
# ************VISUALIZATION WITH PARTIAL DEPENDENCE PLOTS************
# A partial dependence plot requires a fitted model to work, as well as some variables of interest.
# It will take a row of data, change the variable-of-interest, predict the target variable, and then compare it to the real answer.
# It will do this for every row with many variations of the VOI and return the average difference for each variation used.
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
my_plots = plot_partial_dependence(my_model,       # self-explanatory
                                   features=[0, 2],# Selects feature names to display.  In this case, 'Column2' would not be plotted.
                                   X=X,            # raw predictors data.
                                   feature_names=['Column1', 'Column2', 'Column3'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis
# ************XGBOOST: LEADING MODEL FOR WORKING WITH TABULAR DATA************
# XGBoost requires a 'naive' model, and then builds a model that predicts the errors of that model.
# It then continues to build this 'ensemble' model by using all of the old models to predict for the new one.  This repeats a lot.
# Using XGBoost is rather similar to what's been done above.  You'll need to use a tranformer and fit a model:
from xgboost import XGBRegressor 
# Every other step is the same so far.  Split your data, use an imputer--do what you need to do until you hit the model.
my_xgb_model = XGBRegressor(n_estimators=1000) # 1000 n estimators won't actually be used.  This is just the theoretical mzximum.
my_model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False)
# early_stopping_rounds is how many models in a row need to decrease your score for it to stop using more n estimators.
xgb_predictions = my_model.predict(test_X)
# At this point, you'd probably use MAE like before to see how well the model is perforing.