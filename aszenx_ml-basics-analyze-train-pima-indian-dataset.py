import pandas as pd # pandas is a dataframe library
import matplotlib.pyplot as plt # plots data
import numpy as np # provides n-dimensional object support

# do plotting inline instead of in a seprate windows
# jupyter window magic function
%matplotlib inline 
df = pd.read_csv('../input/pimaindian/pima-data.csv') # load pima data 
df.shape # gives the structure of the data: rows and cols
df.head() # returns by default the top 5 data rows of the data frame
df.head(3) # returns just 3 rows
df.tail(2) # returns the last two rows of the data frame 
df.isnull().values.any() # if any nulls are found any() returns true
def plot_corr(df, size = 16):
    """
    Function plots a graphical correlation matrix for each pair of columns in the database.
        Input:
           df: pandas dataframe to check
           size: vertical and horizontal size of the plot
        Displays:
            matrix of correlation b/w cols.
            Blue-Cyan-Yellow-Red-DarkRed => less to more correlated
            Expect a darkred line running from top left to bottom right
    """
    
    corr = df.corr() # data frame correlation function.
    fig, ax = plt.subplots(figsize = (size, size))
    cax = ax.matshow(corr) # color code the rectange by correlation value
    # pass the no of columns and their names for labeling x & y cells 
    plt.xticks(range(len(corr.columns)), corr.columns) # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns) # draw y tick marks
    plt.title('Diabetes feature correlation')
plot_corr(df)
df.corr()
del df['skin'] # removes the skin col from our data frame
# verify the col is dropped by checking the head again
df.head()
# lets now verify again that there are no correlated cols in the plot
plot_corr(df)
df.head()
# define a mapping dictionary
diabetes_map = {True: 1, False: 0}
# Then we use the map method to change the values from True to 1 and False to 0.
df['diabetes'] = df['diabetes'].map(diabetes_map)
# lets check to see the results now
df.head()
num_true = len(df.loc[df['diabetes'] == 1])
num_false = len(df.loc[df['diabetes'] == 0])
percentage_true = (num_true / (num_true + num_false)) * 100
percentage_false = (num_false / (num_true + num_false)) * 100
print('Number of true cases: {0} ({1:2.2f}%)'.format(num_true, percentage_true))
print('Number of false cases: {0} ({1:2.2f}%)'.format(num_false, percentage_false))
# scikit-learn contains train_test_split method which makes it easy to split the data 


from sklearn.model_selection import train_test_split # import training test split method from sklearn

# next we define the feature cols and predicted col
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

# split our data into two data frames one containing the features cols and other with the diabetes result
X = df[feature_col_names].values # predictor feature cols (8)
Y = df[predicted_class_names].values # predicated class (1=True, 0=False) col (1)

split_test_size = 0.30 # define the train_test split ratio 30%

# These data frames and the split size are passed to the function which then return four numpy arrays of data
# the arrays contain the values of test and training feature cols and the test and train diabetes results
# Since the splitting process must be random we pass the random_state any value
# random_state sets the seed for the random no generator used as part of the splitting process
# setting the seed to a constant ensures that if the function is run again the split will be identical
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split_test_size, random_state = 42)
#  here df.index is the whole data frame
print('{0:0.2f}% in training set'.format((len(X_train) / len(df.index)) * 100))
print('{0:0.2f}% in test set'.format((len(X_test) / len(df.index)) * 100))
total_length = len(df.index) # no of observarions (rows) in the whole data frame
original_true = len(df.loc[df['diabetes'] == 1]) # no of observations with diabetes 
original_true_perct = (original_true / total_length) * 100 # ratio of true to whole in the full set
original_false = len(df.loc[df['diabetes'] == 0]) # no of observations without diabetes 
original_false_perct = (original_false / total_length) * 100 # ratio of false to whole in the full set
# printing the result
print('Original set True : {0} ({1: 0.2f}%)'.format(original_true, original_true_perct)) 
print('Original set False : {0} ({1: 0.2f}%)'.format(original_false, original_false_perct))

training_true = len(Y_train[Y_train[:] == 1]) # no of observations with diabetes in the train set
training_true_perct = (training_true / len(Y_train)) * 100 # ratio of true to whole in train set
training_false = len(Y_train[Y_train[:] == 0]) # no of observations without diabetes in the train set
training_false_perct = (training_false / len(Y_train)) * 100 # ratio of false to whole in train set
# printing the result
print('Train set True : {0} ({1: 0.2f}%)'.format(training_true, training_true_perct))
print('Train set False : {0} ({1: 0.2f}%)'.format(training_false, training_false_perct))

testing_true = len(Y_test[Y_test[:] == 1]) # no of observations with diabetes in the test set
testing_true_perct = (testing_true / len(Y_test)) * 100 # ratio of true to whole in test set
testing_false = len(Y_test[Y_test[:] == 0]) # no of observations without diabetes in the test set
testing_false_perct = (testing_false / len(Y_test)) * 100 # ratio of false to whole in test set
# printing the result
print('Test set True : {0} ({1: 0.2f}%)'.format(testing_true, testing_true_perct))
print('Test set False : {0} ({1: 0.2f}%)'.format(testing_false, testing_false_perct))

# Sometimes null values can be hiding in plain sight.
df.head()
def printZeroValues(df):
    print('rows in dataframe: {}'.format(len(df)))
    print('rows missing glucose_conc: {}'.format(len(df.loc[df['glucose_conc'] == 0])))
    print('rows missing diastolic_bp: {}'.format(len(df.loc[df['diastolic_bp'] == 0])))
    print('rows missing thickness: {}'.format(len(df.loc[df['thickness'] == 0])))
    print('rows missing insulin: {}'.format(len(df.loc[df['insulin'] == 0])))
    print('rows missing bmi: {}'.format(len(df.loc[df['bmi'] == 0])))
    print('rows missing diab_pred: {}'.format(len(df.loc[df['diab_pred'] == 0])))
    print('rows missing age: {}'.format(len(df.loc[df['age'] == 0])))

printZeroValues(df)

# scikit contains an impute class which makes it very easy to impute data
# import the Imputer class
from sklearn.preprocessing import Imputer

# Impute with mean all 0 readings for all values on the axis 0 which is col
fill_0 = Imputer(missing_values = 0, strategy = 'mean', axis = 0)

# use fit_transform function to create a new numpy array with any feature value of 0 replaced by the mean for the col
# do this for both train and test feature values
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

# import Naive Bayes algorithm from the library
# In case of naive_bayes there are multiple implementations 
# we are using the gaussian algo that assumes that the feature data is distributed in a gaussian 
from sklearn.naive_bayes import GaussianNB

# create Gaussian Naive Bayes model object and train it with data
nb_model = GaussianNB() # our model object

# call the fit method to create a model trained with the training data 
# numpy.ravel returns a contiguous flattened array
nb_model.fit(X_train, Y_train.ravel())
# pass feature data to the models predict function
# the predict function will return 1's and 0's representing True and False
# lets first predict against the training data 
# X_train is the data we used to train the model
nb_predict_train = nb_model.predict(X_train)

# to see the accuracy we load the scikit metrics library
# metrics has methods that let us get the statistics on the models predictive performance
from sklearn import metrics
# Accuracy
# accuracy_score takes two parameters: the actual output data and the predicted output
train_accuracy = metrics.accuracy_score(Y_train, nb_predict_train) # will be b/w 0 & 1
print('Accuracy(%) on training data itself: {0: .4f}'.format(train_accuracy * 100))

# Now lets predict against the testing data
# X_test is the data we kept aside for testing
nb_predict_test = nb_model.predict(X_test)
# Y_test is the actual output and nb_predict_test is the predicted one 
test_accuracy = metrics.accuracy_score(Y_test, nb_predict_test)
print('Accuracy(%) on test data: {0: .4f}'.format(test_accuracy * 100))
# Confusion matrix provides the True Positive, False Positive, False Negative & True Negative.
print('Confusion Matrix')
# the labels are for 1 = True to upper left and 0 = False to lower right
print(metrics.confusion_matrix(Y_test, nb_predict_test, labels=[1, 0]))
print('')

# the classification report generates statistics based on the values shown in the confusion matrix.
print('Classification report')
print(metrics.classification_report(Y_test, nb_predict_test, labels = [1, 0]))
# import random forest from scikit
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state = 54) # Create random forest object
rf_model.fit(X_train, Y_train.ravel())
rf_predict_train = rf_model.predict(X_train)
# training metrics
print('Accuracy: {0:.4f}'.format(metrics.accuracy_score(Y_train, rf_predict_train)))
rf_predict_test = rf_model.predict(X_test)
# training metrics
print('Accuracy: {0:.4f}'.format(metrics.accuracy_score(Y_test, rf_predict_test)))
print(metrics.confusion_matrix(Y_test, rf_predict_test, labels=[1, 0]))
print('')
print('Classification report')
print(metrics.classification_report(Y_test, rf_predict_test, labels=[1, 0]))
# import the algo from sci-kit learn linear model module
from sklearn.linear_model import LogisticRegression
# set up the model
# C - regulatization hyperparameter 
lr_model = LogisticRegression(C=0.7, random_state=54) # set c to 0.7 initially
# train the algo
lr_model.fit(X_train, Y_train.ravel())
# evaluate against the training data
lr_predict_train = lr_model.predict(X_train)
# training metrics
print('Logistic Regression -> Accuracy on training data: {0:.4f}'.format(metrics.accuracy_score(Y_train, lr_predict_train)))
lr_predict_test = lr_model.predict(X_test)
# training metrics
print('Logistic Regression -> Accuracy on test data: Accuracy: {0:.4f}'.format(metrics.accuracy_score(Y_test, lr_predict_test)))
print(metrics.confusion_matrix(Y_test, lr_predict_test, labels=[1, 0]))
print('')
print('Classification report')
print(metrics.classification_report(Y_test, lr_predict_test, labels=[1, 0]))
# try C values from 0 to 4.9 in increments of 0.1
# for each C value a logistic regression object is created
# then trained with the training data and then used to predict the test results
# Each test recall score is computed and the highest recall score is recorded,
# the score is used to get the C value that produced the highest recall score. 
# see which C value results in the best recall score
C_start = 0.1
C_end = 5
C_inc = 0.1
C_values, recall_scores = [], [] # used to hold the C_values and their corresponding recall_scores

C_val = C_start
best_recall_score = 0
# this while loop will try C values from 0.1 to 4.9
while C_val < C_end:
    # add this C_val to the array
    C_values.append(C_val)
    
    # create LR object using current C value
    lr_model_loop = LogisticRegression(C=C_val, random_state = 35)
    # Train the algo 
    lr_model_loop.fit(X_train, Y_train.ravel())
    # Predict using test data
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    # Get the recall score
    recall_score = metrics.recall_score(Y_test, lr_predict_loop_test)
    
    # add the recall_score to the array
    recall_scores.append(recall_score)
    
    # if current recall_score is greater than the best so far then update the best scores
    if recall_score > best_recall_score:
        best_recall_score = recall_score # update the best_recall_score
        best_lr_predict_test = lr_predict_loop_test # also the predictions for best score
    
    # increment the C_val
    C_val += C_inc 

# get the best C_val from the array
best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print('recall max value of {0:.3f} occured at C={1:.3f}'.format(best_recall_score, best_score_C_val))
  
# we also plot the recall scores vs regularization value
# so as to get an idea of how recall changes with different regularization values  
plt.plot(C_values, recall_scores, '-')
plt.xlabel('Regularization parameter C')
plt.ylabel('Recall Score')
plt.title('Regularization parameter C vs Recall Score for unbalanced class weight\n')
# everythings remains the same except for the class_weight hyperparameter set in the LR object.

C_start = 0.1
C_end = 5
C_inc = 0.1
C_values, recall_scores = [], [] # used to hold the C_values and their corresponding recall_scores

C_val = C_start
best_recall_score = 0
# this while loop will try C values from 0.1 to 4.9
while C_val < C_end:
    # add this C_val to the array
    C_values.append(C_val)
    
    # create LR object using current C value and hyperparamter class_weight set to balanced
    lr_model_loop = LogisticRegression(C=C_val, class_weight='balanced', random_state = 35)
    # Train the algo 
    lr_model_loop.fit(X_train, Y_train.ravel())
    # Predict using test data
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    # Get the recall score
    recall_score = metrics.recall_score(Y_test, lr_predict_loop_test)
    
    # add the recall_score to the array
    recall_scores.append(recall_score)
    
    # if current recall_score is greater than the best so far then update the best scores
    if recall_score > best_recall_score:
        best_recall_score = recall_score # update the best_recall_score
        best_lr_predict_test = lr_predict_loop_test # also the predictions for best score
    
    # increment the C_val
    C_val += C_inc 

# get the best C_val from the array
best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print('recall max value of {0:.3f} occured at C={1:.3f} with balanced classes'.format(best_recall_score, best_score_C_val))
# we also plot the recall scores vs regularization value
# so as to get an idea of how recall changes with different regularization values  
plt.plot(C_values, recall_scores, '-')
plt.xlabel('Regularization parameter C')
plt.ylabel('Recall Score')
plt.title('Regularization parameter C vs Recall Score for balanced class weight\n')
# now we can train our model with the best hyperparameter C_value  and class_weigh balanced
best_lr_model = LogisticRegression(class_weight='balanced', C=best_score_C_val, random_state=35)
best_lr_model.fit(X_train, Y_train.ravel())
best_lr_predict_train = best_lr_model.predict(X_train)
best_lr_predict_test = best_lr_model.predict(X_test)
# and calculate the performance metrics
# training metrics on training data
print('LR model with best C value & balanced weights -> Accuracy on training data: {0:.4f}'.format(metrics.accuracy_score(Y_train, best_lr_predict_train)))
print('')
# training metrics on test data
print('LR model with best C value & balanced weights -> Accuracy on test data: {0:.4f}'.format(metrics.accuracy_score(Y_test, best_lr_predict_test)))
print('')
# performance metrics on test data
print('The confusion matrix for LR model with best C value and balanced weights is: ')
print(metrics.confusion_matrix(Y_test, best_lr_predict_test, labels=[1, 0]))
print('')
print('Classification report for LR model with best C value and balanced weights: ')
print(metrics.classification_report(Y_test, best_lr_predict_test, labels=[1, 0]))
## scikit-learn cross validation library to access cross validation methods
## that make it easy to perform k-fold cross validation 

# scikit-learn provides special ensemble versions of the algorithms that contain the code
# to determine the optimal hyperparamter value and set the model to that value

# AlgorithmCV built in Variants
    ## Can be used just like normal algorithms
    ## Take little longer to run 
    ## They begin with the base class name and end with CV
    ## Expose fit(), predict(),...
    ## Use the parameters in the constructor to specify things such as no of folds 
    ## The algorithm in then run k times
    ## When fit is then run k-fold validation is run with k folds on the training data
    ## the other parameters in the constructor let you define how the optimal value for hyperparameters is determined (mean, median)
    ## Algorithm + Cross Validation = AlgorithmCV
# scikit learn has an ensemble algorithm that combines logistic regression with cross validation called LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV

# n_jobs -> use all the cores resources on the system
# cv -> no of folds
# Cs -> smaller values specify stronger regularization
# solver -> algorithm to use in the optimization problem (default - 'lbfgs')
lr_cv_model = LogisticRegressionCV(n_jobs=-1, Cs=3, refit=True, max_iter=100, solver='liblinear', cv=10, class_weight='balanced')
lr_cv_model.fit(X_train, Y_train.ravel())
lr_cv_predict_train = lr_cv_model.predict(X_train)

# training metrics
print('Accuracy on training data: {0:.4f}'.format(metrics.accuracy_score(Y_train, lr_cv_predict_train)))
print("")
print('The confusion matrix for LR CV model with cross validation and balanced weights on training data is: ')
print(metrics.confusion_matrix(Y_train, lr_cv_predict_train, labels=[1, 0]))
print('')
print('Classification report for LR CV model with cross validation and balanced weights on training data is: ')
print(metrics.classification_report(Y_train, lr_cv_predict_train, labels=[1, 0]))
print(' 1 -> Diabetic\n 0 -> Not Diabetic')
lr_cv_predict_test = lr_cv_model.predict(X_test)

# training metrics
print('Accuracy on testing data: {0:.4f}'.format(metrics.accuracy_score(Y_test, lr_cv_predict_test)))
print("")
print('The confusion matrix for LR CV model with cross validation and balanced weights on training data is: ')
print(metrics.confusion_matrix(Y_test, lr_cv_predict_test, labels=[1, 0]))
print('')
print('Classification report for LR CV model with cross validation and balanced weights on training data is: ')
print(metrics.classification_report(Y_test, lr_cv_predict_test, labels=[1, 0]))
print(' 1 -> Diabetic\n 0 -> Not Diabetic')
