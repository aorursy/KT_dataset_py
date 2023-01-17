%matplotlib inline 
import math
import random
import matplotlib
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import MinMaxScaler

# Reading random number of rows (used by performing test with reduced versions of dataset)
n = 10  # every 10th line = 10% of the lines
h1b_final = pd.read_csv('../input/h-1b-visa/h1b_kaggle.csv', skiprows=lambda i: i % n != 0, index_col=0)

# Loading data
#h1b_final = pd.read_csv('../input/h-1b-visa/h1b_kaggle.csv',  nrows=500000, index_col=0)
print('Loading h1b_kaggle.csv file..')
print(len(h1b_final))
h1b_final.head(1)

# Removing rejected, invalidated and pending review applications
h1b_final = h1b_final[h1b_final['CASE_STATUS'] != 'REJECTED']  
h1b_final = h1b_final[h1b_final['CASE_STATUS'] != 'INVALIDATED']  
h1b_final = h1b_final[h1b_final['CASE_STATUS'] != 'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED']  

# Restart the index after deleting rows and limiting to remove outlier wages above $1 billion. 
h1b_final.reset_index(drop=True, inplace=True)
h1b_final = h1b_final[(h1b_final['PREVAILING_WAGE'] < 1000000)]

# Features & outcome 
outcome_raw = h1b_final['CASE_STATUS']
features_raw = h1b_final.drop(['CASE_STATUS','EMPLOYER_NAME','JOB_TITLE', 'lon', 'lat', 'WORKSITE'], axis = 1)
features_raw.head(1)
# Counting applications by case status
case_status = h1b_final['CASE_STATUS'].value_counts()
print('\nNumber of applications grouped by case status:\n\n', case_status)
# Plot applications grouped by case status
ax = case_status.plot(kind='bar', stacked=True, title='Applications grouped by case status', color='lightblue')
ax.set_xlabel("Case status")
ax.set_ylabel('Number of applications')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Counting applications by year
year = h1b_final['YEAR'].value_counts()
print('\nNumber of applications grouped by year:\n\n', year)

# Plot applications grouped by year
ax = year.head(10).plot(kind='bar', stacked=True, title='Applications grouped by year', color='lightblue')
ax.set_xlabel("Year")
ax.set_ylabel('Number of applications')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Counting applications by worksite
worksite = h1b_final['WORKSITE'].value_counts()
print('\nNumber of applications grouped by worksite:\n\n', worksite.head(5))

# Plot applications grouped by worksite
ax = worksite.head(10).plot(kind='bar', stacked=True, title='Applications grouped by worksite', color='lightblue')
ax.set_xlabel("Worksite")
ax.set_ylabel('Number of applications')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Counting applications by job title
job_title = h1b_final['JOB_TITLE'].value_counts()
print('\nNumber of applications grouped by job title:\n\n', job_title.head(5))

# Plot applications grouped by job title
ax = job_title.head(10).plot(kind='bar', stacked=True, title='Applications grouped by job title', color='lightblue')
ax.set_xlabel("Job title")
ax.set_ylabel('Number of applications')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Counting applications by prevailing wage
prevailing_wage = h1b_final['PREVAILING_WAGE'].value_counts()
print('\nNumber of applications grouped by prevailing wage:\n\n', prevailing_wage.head(5))

# Plot applications grouped by prevailing wage
_, ax = plt.subplots()
ax.hist(h1b_final['PREVAILING_WAGE'], bins=100, facecolor='lightblue', edgecolor='w')
ax.set_xlabel("Prevailing wage")
ax.set_ylabel('Number of applications')
plt.xlim([10000, 180000])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Counting applications grouped by employer name
employer_name = h1b_final['EMPLOYER_NAME'].value_counts()
print('\nNumber of applications grouped by job title:\n\n', employer_name.head(5))

# Plot applications grouped by employer name
ax = employer_name.head(10).plot(kind='bar', stacked=True, title='Applications grouped by employer name', color='lightblue')
ax.set_xlabel("Employer name")
ax.set_ylabel('Number of applications')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Show the calculated statistics
print("Statistics for H1-B Visa Applications:\n")
print("Minimum wage: ${:,.2f}".format(min(features_raw['PREVAILING_WAGE'])))
print("Maximum wage: ${:,.2f}".format(max(features_raw['PREVAILING_WAGE'])))
print("Mean wage: ${:,.2f}".format(np.mean(features_raw['PREVAILING_WAGE'])))
print("Median wage ${:,.2f}".format(np.median(features_raw['PREVAILING_WAGE'])))
print("Standard deviation of wage: ${:,.2f}".format(np.std(features_raw['PREVAILING_WAGE'])))
### Prepare the data
import scipy
from sklearn.preprocessing import MinMaxScaler

# Normalize numerical features by initializing a scaler and applying it to the features
scaler = MinMaxScaler()
numerical = ['YEAR', 'PREVAILING_WAGE']
features_raw[numerical] = scaler.fit_transform(features_raw[numerical])

# Visualize skewed data and original data 
display(features_raw.head(1))
#vs.distribution(features_raw) #Check and confirm there is no need to treat skew data

# Check 'normality' of the features with a quantile-quantile (q-q) plot
scipy.stats.probplot(features_raw['PREVAILING_WAGE'], plot=plt)
plt.title('Prevailing wage')
plt.show()
### Data Preprocessing 
time0 = time()
outcome_raw = outcome_raw.apply(lambda x: 1 if x == 'CERTIFIED' else x)
outcome_raw = outcome_raw.apply(lambda x: 2 if x == 'CERTIFIED-WITHDRAWN' else x)
outcome_raw = outcome_raw.apply(lambda x: 3 if x == 'DENIED' else x)
outcome = outcome_raw.apply(lambda x: 4 if x == 'WITHDRAWN' else x)

features = pd.get_dummies(features_raw)

encoded = list(features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Show processing time in h:m:s
m, s = divmod(time()-time0, 60)
h, m = divmod(m, 60)
print("Time elapsed: %d:%02d:%02d" % (h, m, s))
### Evaluating Model performance with Naive predictor
import warnings 
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_score, fbeta_score, recall_score, classification_report

predictions_naive = pd.Series(np.ones(len(outcome), dtype = int))

# Naive performance using average = weighted for multiclass 
accuracy = accuracy_score(outcome, predictions_naive)
fscore = fbeta_score(outcome, predictions_naive, beta=0.5, average='weighted')
precision = precision_score(outcome, predictions_naive, average='weighted')

print('Accuracy score:', accuracy)
print('F-score (weighted):', fscore)
print('Precision (weighted):', precision)
print('Recall (weighted):', recall_score(outcome, predictions_naive, average='weighted'))
print('\nClassification Report:\n\n', classification_report(outcome, predictions_naive))
### Shuffle and split the data into training and testing subsets
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size=0.2, random_state=11)

print("Training and testing split was successful.")
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
### Creating a Training and Predicting Pipeline (by Udacity)
from sklearn.metrics import accuracy_score, fbeta_score, f1_score, precision_score, recall_score
def train_predict(learner, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - X_train: features training set
       - y_train: outcome training set
       - X_test: features testing set
       - y_test: outcome testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size'
    start = time() 
    learner = learner.fit(X_train, y_train)
    end = time() 
    
    # Calculate the training time
    results['train_time'] = end - start
        
    # Get the predictions on the test set & on the first 10% of the training samples - X_train[:239831]
    start = time() 
    predictions_test = clf.predict(X_test)
    predictions_train = clf.predict(X_train)
    end = time()
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
    
    # Compute precision on the training samples
    results['precision_train'] = precision_score(y_train, predictions_train, average='weighted')
    
    # Compute precision on test set
    results['precision_test'] = precision_score(y_test, predictions_test, average='weighted')
    
    # Compute F-score on the training samples
    results['f_train'] = fbeta_score(y_train, predictions_train, beta=0.5, average='weighted')
    
    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5, average='weighted')

    # Print all
    print("Training {}".format(learner.__class__.__name__))
    print("Precision result {}".format(results['precision_test'], learner.__class__.__name__))
    print("F-score result {}".format(results['f_test'], learner.__class__.__name__))
    print("Recall result {}".format(recall_score(y_test, predictions_test, average='weighted'), learner.__class__.__name__))

    # Return the results
    return results
    
print('Done')
### Initial Model Evaluation: Import the supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
time0 = time()

# Initialize the three models
clf_A = LogisticRegression(penalty='l2', random_state=24)
clf_B = RandomForestClassifier(n_estimators=10, random_state=16)
clf_C = DecisionTreeClassifier(random_state=41)

# Collect results on the learners
results = {}

for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    results[clf_name] = train_predict(clf, X_train, y_train, X_test, y_test)

# Show processing time in h:m:s
m, s = divmod(time()-time0, 60)
h, m = divmod(m, 60)
print("Time elapsed: %d:%02d:%02d" % (h, m, s))
### Plot training & prediction times for all classifiers 
ind = 1
width = 0.4       
plt.xticks(range(0,1)) # Show no bar-labels
plt.xlabel('Models (left = prediction, right = training)')
plt.ylabel('Time (in seconds)')

for key, data_dict in results.items():
    x = data_dict.keys() 
    y = list(data_dict.values()) #list used for Python3
    if key == 'LogisticRegression':
        plt.bar(ind, y[1], color='#A00000', align='center', width = 0.3, label = 'Logit') # Prediction 
        plt.bar(ind+width, y[0], color='#A00000', align='center', width = 0.3, ) # Training
    elif key == 'RandomForestClassifier': 
        plt.bar(ind+width*2, y[1], color='#00A000', align='center', width = 0.3,  label = "RF") # Prediction 
        plt.bar(ind+width*3, y[0], color='#00A000', align='center', width = 0.3) # Training
    elif key == 'DecisionTreeClassifier': 
        plt.bar(ind+width*4, y[1], color='#00A0A0', align='center', width = 0.3, label = "DT") # Prediction 
        plt.bar(ind+width*5, y[0], color='#00A0A0', align='center', width = 0.3) # Training

#print(results.values())
plt.suptitle("Prediction & Training Times", fontsize = 16, x = 0.53, y = .95)
plt.legend(loc = 'upper left')
### Plot precision for all classifiers 
ind = 1
width = 0.4       
plt.xticks(range(0,1)) # Show no bar-labels
plt.xlabel('Models')
plt.ylabel('Scores')

for key, data_dict in results.items():
    if key == 'LogisticRegression':
        x = data_dict.keys() 
        y = list(data_dict.values()) #list used for Python3
        plt.bar(ind, y[3], color='#A00000', align='center', width = 0.2, label = 'Logit') # y[1] Precision - y[2] F-Score
    elif key == 'DecisionTreeClassifier': 
        x = data_dict.keys() 
        y = list(data_dict.values()) 
        plt.bar(ind+width, y[3], color='#00A0A0', align='center', width = 0.2, label = 'DT')  
    elif key == 'RandomForestClassifier': 
        x = data_dict.keys() 
        y = list(data_dict.values()) 
        plt.bar(ind+width*2, y[3], color='#00A000', align='center', width = 0.2, label = 'RF')  

plt.axhline(y = precision, linewidth = 1, color = 'k', linestyle = 'dashed')
plt.suptitle("Precision scores", fontsize = 16, x = 0.53, y = .95)
plt.legend(loc = 'lower left')
### Plot F-score for all classifiers 
ind = 1
width = 0.4       
plt.xticks(range(0,1)) # Show no bar-labels
plt.xlabel('Models')
plt.ylabel('Scores')

for key, data_dict in results.items():
    if key == 'LogisticRegression':
        x = data_dict.keys() 
        y = list(data_dict.values()) #list used for Python3
        plt.bar(ind, y[5], color='#A00000', align='center', width = 0.2, label = 'Logit') # y[1] Precision - y[2] F-Score
    elif key == 'DecisionTreeClassifier': 
        x = data_dict.keys() 
        y = list(data_dict.values())
        plt.bar(ind+width, y[5], color='#00A0A0', align='center', width = 0.2, label = 'DT')  
    elif key == 'RandomForestClassifier': 
        x = data_dict.keys() 
        y = list(data_dict.values())
        plt.bar(ind+width*2, y[5], color='#00A000', align='center', width = 0.2, label = 'RF') 

plt.axhline(y = fscore, linewidth = 1, color = 'k', linestyle = 'dashed')
plt.suptitle("F-scores", fontsize = 16, x = 0.53, y = .95)
plt.legend(loc = 'lower left')
### Model Tuning for Logistic Regression classifiers
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

time2 = time()

# Initialize the classifier
clf = LogisticRegression(random_state=82)

# Create the parameters list you wish to tune
param_grid = {'C': [1, 10, 100, 1000]}
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5, average='weighted')

# Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, param_grid, scoring = scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores 
print("Unoptimized model\n------")
print("Precision score: {:.4f}".format(precision_score(y_test, predictions, average='weighted')))
print("F-score: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5, average='weighted')))
print("\nOptimized Model\n------")
print("Final precision score: {:.4f}".format(precision_score(y_test, best_predictions, average='weighted')))
print("Final F-score: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5, average='weighted')))

# Show processing time in h:m:s
m, s = divmod(time() - time2, 60)
h, m = divmod(m, 60)
print("\nTime elapsed to tune classifier: %d:%02d:%02d" % (h, m, s))
### Model Tuning Decision Tree classifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer 
time0 = time()

# Initialize the classifier
clf = DecisionTreeClassifier(random_state=27)

# Create the parameters list you wish to tune
parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 
              'max_depth': [1,3,5,10,15], 'max_leaf_nodes': [2,5,10,15,30,50,100]}

# Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5, average='weighted')

# Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Precision score on testing data: {:.4f}".format(precision_score(y_test, predictions, average='weighted')))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5, average='weighted')))
print("\nOptimized Model\n------")
print("Final precision score on the testing data: {:.4f}".format(precision_score(y_test, best_predictions, average='weighted')))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5, average='weighted')))

# Show processing time in h:m:s
m, s = divmod(time()-time0, 60)
h, m = divmod(m, 60)
print("Time elapsed: %d:%02d:%02d" % (h, m, s))
### Model Tuning Random Forest classifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer 
time0 = time()

# Initialize the classifier
clf = RandomForestClassifier(n_estimators=10)

clf.get_params().keys()

# Create the parameters list you wish to tune
parameters = {'max_leaf_nodes': [5,10,50,100,200,500]}

# Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5, average='weighted')

# Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Precision score on testing data: {:.4f}".format(precision_score(y_test, predictions, average='weighted')))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5, average='weighted')))
print("\nOptimized Model\n------")
print("Final precision score on the testing data: {:.4f}".format(precision_score(y_test, best_predictions, average='weighted')))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5, average='weighted')))

# Show processing time in h:m:s
m, s = divmod(time()-time0, 60)
h, m = divmod(m, 60)
print("Time elapsed: %d:%02d:%02d" % (h, m, s))
### Make predictions using the final models
predictions_LR = clf_A.predict(X_test)
predictions_RF = clf_B.predict(X_test)
predictions_DT = clf_C.predict(X_test)

# Report the scores for the models
print("Random Forest:\n------")
print("Precision score on testing data: {:.4f}".format(precision_score(y_test, predictions_RF, average='weighted')))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions_RF, beta = 0.5, average='weighted')))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, predictions_RF, average='weighted')))
print("\nDecision Tree:\n------")
print("Precision score on the testing data: {:.4f}".format(precision_score(y_test, predictions_DT, average='weighted')))
print("F-score on the testing data: {:.4f}".format(fbeta_score(y_test, predictions_DT, beta = 0.5, average='weighted')))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, predictions_DT, average='weighted')))
print("\nLogistic Regression:\n------")
print("Precision score on the testing data: {:.4f}".format(precision_score(y_test, predictions_LR, average='weighted')))
print("F-score on the testing data: {:.4f}".format(fbeta_score(y_test, predictions_LR, beta = 0.5, average='weighted')))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, predictions_LR, average='weighted')))
### Identifying most relevant features by using a supervised model with 'feature_importances_'

# Extract the feature importances using Decision Tree Classifier (clf_B for Random Forest)
importances = clf_C.feature_importances_

# Visualization (from Udacity's rep)
indices = np.argsort(importances)[::-1]
columns = X_train.columns.values[indices[:5]]
values = importances[indices][:5]

# Creat the plot
fig = plt.figure(figsize = (20,8))
plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
plt.bar(np.arange(5), values, width = 0.3, align="center", color = '#00A000', \
      label = "Feature Weight")
plt.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
      label = "Cumulative Feature Weight")
plt.xticks(np.arange(5), columns)
plt.xlim((-0.5, 4.5))
plt.ylabel("Weight", fontsize = 12)
plt.xlabel("Feature", fontsize = 12)
plt.legend(loc = 'upper center')
plt.tight_layout()
plt.show()  
### Validate the robutsness of the Logistic Regression model using cross-validation
from sklearn.model_selection import cross_val_score

cross_score = cross_val_score(clf_A, X_test, y_test, scoring=scorer, cv=5)
print('Logit cross validation score:', cross_score)
score = best_clf.score(X_test, y_test)
print('R^2 score:', score)
print('Mean score:', cross_score.mean())
### Validate the robutsness of the Random Forest Tree classifier using cross-validation
from sklearn.model_selection import cross_val_score

cross_score = cross_val_score(clf_B, X_test, y_test, scoring=scorer, cv=5)
print('RF cross validation score:', cross_score)
score = best_clf.score(X_test, y_test)
print('R^2 score:', score)
print('Mean score:', cross_score.mean())
### Validate the robutsness of the Decision Tree classifier using cross-validation
from sklearn.model_selection import cross_val_score

cross_score = cross_val_score(clf_C, X_test, y_test, scoring=scorer, cv=5)
print('DT cross validation score:', cross_score)
score = best_clf.score(X_test, y_test)
print('R^2 score:', score)
print('Mean score:', cross_score.mean())