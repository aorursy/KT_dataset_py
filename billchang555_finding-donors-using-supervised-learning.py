# Import libraries necessary for this project

import numpy as np

import pandas as pd

from time import time

import matplotlib.pyplot as plt

import seaborn as sns



# Import the supervised learning models form sklearn

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import fbeta_score, accuracy_score, roc_auc_score, make_scorer





# Load the Census dataset

data = pd.read_csv("../input/udacity-mlcharity-competition/census.csv")



# Display the first five records

data.head()
# gain statistics insight

data.describe()
# TODO: Total number of records

n_records = data.shape[0]



# TODO: Number of records where individual's income is more than $50,000

n_greater_50k = len(data[data['income']=='>50K'])



# TODO: Number of records where individual's income is at most $50,000

n_at_most_50k = len(data[data['income']=='<=50K'])



# TODO: Percentage of individuals whose income is more than $50,000

greater_percent = n_greater_50k/n_records



# Print the results

print("Total number of records: {}".format(n_records))

print("Individuals making more than $50,000: {}".format(n_greater_50k))

print("Individuals making at most $50,000: {}".format(n_at_most_50k))

print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent*100))
# split features and target

income_raw = data['income']

feature_raw = data.drop('income', axis=1)
# continues features

continues = list(feature_raw.describe().columns)

print(continues)
for col in feature_raw.columns:

    if col not in continues:

        values = list(data[col].value_counts().index)

        print('{}: {}'.format(col, ', '.join(values)))

        print('\n')
# check the distributions of those continues features

for col in continues:

    a = sns.FacetGrid(feature_raw, height=8, aspect=2)

    a.map(sns.distplot, col, kde_kws={'bw': 25})

    a.add_legend

    print('{} skew: {}'.format(col, feature_raw[col].skew()))
# Log-transform the skewed features

skewed = ['capital-gain', 'capital-loss']

features_log_transformed = pd.DataFrame(data = feature_raw)

features_log_transformed[skewed] = feature_raw[skewed].apply(lambda x: np.log(x + 1))
for col in skewed:

    print('{} skew: {}'.format(col, features_log_transformed[col].skew()))
# show the distributions after log-transform

sns.set()

fig = plt.figure(figsize=(11,5))

fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", fontsize=16)



for i, feature in enumerate(skewed):

    ax = fig.add_subplot(1, 2, i+1)

    ax.hist(features_log_transformed[feature], bins=25)

    ax.set_title("{} Feature Distribution".format(feature), fontsize=14)

    ax.set_xlabel("Value")

    ax.set_ylabel("Number of Records")

    ax.set_ylim((0,2000))

    ax.set_yticks([0, 500, 1000, 1500, 2000])

    ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])
# Import sklearn.preprocessing.StandardScaler

from sklearn.preprocessing import MinMaxScaler



# Initialize a scaler, then apply it to the features

scaler = MinMaxScaler()



features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)

features_log_minmax_transform[continues] = scaler.fit_transform(features_log_transformed[continues])



# Show an example of a record with scaling applied

features_log_minmax_transform.head()
# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()

features_final = pd.get_dummies(features_log_minmax_transform)



# Encode the 'income_raw' data to numerical values

income = income_raw.map({"<=50K":0, ">50K":1})



# Print the number of features after one-hot encoding

encoded = list(features_final.columns)

print("{} total features after one-hot encoding.".format(len(encoded)))

features_final.head()
# Split the 'features' and 'income' data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(features_final, 

                                                    income, 

                                                    test_size = 0.2, 

                                                    random_state = 0)



# Show the results of the split

print("Training set has {} samples.".format(X_train.shape[0]))

print("Testing set has {} samples.".format(X_test.shape[0]))
# create a training and predicting pipeline



def train_predict(learner, X_train, y_train, X_test, y_test): 

    '''

    inputs:

       - learner: the learning algorithm to be trained and predicted on

       - X_train: features training set

       - y_train: income training set

       - X_test: features testing set

       - y_test: income testing set

    '''

    

    results = {}

    

    # Fit the learner to the training data 

    start = time() # Get start time

    learner = learner.fit(X_train, y_train)

    end = time() # Get end time

    

    # Calculate the training time

    results['train_time'] = end - start

        

    # Get the predictions on the test set(X_test),

    # then get predictions on the first 300 training samples

    start = time() # Get start time

    predictions_test = learner.predict(X_test)

    predictions_train = learner.predict(X_train[:300])

    end = time() # Get end time

    

    # Calculate the total prediction time

    results['pred_time'] = end - start

            

    # Compute accuracy on the first 300 training samples which is y_train[:300]

    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

        

    # Compute accuracy on test set using accuracy_score()

    results['acc_test'] = accuracy_score(y_test, predictions_test)

    

    # Compute F-score on the the first 300 training samples using fbeta_score()

    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)

        

    # Compute F-score on the test set which is y_test

    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

    

    # Compute roc_auc_score on the first 300 training samples which is y_train[:300]

    results['roc_auc_score_train'] = roc_auc_score(y_train[:300], learner.predict_proba(X_train[:300])[:, 1])

    

    # Compute roc_auc_score on the test set which is y_test

    results['roc_auc_score_test'] = roc_auc_score(y_test, learner.predict_proba(X_test)[:, 1])

    

    print('{} trained on {} samples'.format(learner.__class__.__name__, len(X_train)))

    print('\n'+'-'*10)

    # Return the results

    return results
# TODO: Initialize the four models

clf_A = MultinomialNB()

clf_B = RandomForestClassifier()

clf_C = AdaBoostClassifier()

clf_D = GradientBoostingClassifier()





# Collect results on the learners

results = {}

for clf in [clf_A, clf_B, clf_C, clf_D]:

    clf_name = clf.__class__.__name__

    results[clf_name] = train_predict(clf, X_train, y_train, X_test, y_test)

    
# create a dataframe for those metrics

metrics_frame = pd.DataFrame(data=results).transpose().reset_index()

metrics_frame = metrics_frame.rename(columns={'index': 'models'})

metrics_frame
# show the visulization

# create shape(4,2) grouped bar plots, it displays metrics of both train and test on each row.

fig, ax = plt.subplots(4,2, figsize=(20, 30))

    

# column list for metrics

metrics_col = list(metrics_frame.columns[1:])

i=0

j=0

for col in range(int(len(metrics_col)/2)):

    

    sns.barplot(x='models', y=metrics_col[2*col], data=metrics_frame, ax=ax[i, j])

    j+=1

    sns.barplot(x='models', y=metrics_col[2*col+1], data=metrics_frame, ax=ax[i, j])

    i+=1

    j-=1

    if i==4 and j==0:

        break

        

    # set ylim(0,1) for the three metrics(accuracy, fbeta_score, roc_au_score)

    ax[i,j].set_ylim((0, 1))

    ax[i, j+1].set_ylim((0,1))



plt.suptitle("Performance Metrics for Supervised Learning Models", fontsize = 25, y = 1.10)

plt.tight_layout()

# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



# TODO: Initialize the classifier

clf = AdaBoostClassifier()



# TODO: Create the parameters list you wish to tune, using a dictionary if needed.

# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}

parameters = {'n_estimators':[50, 200, 400, 1000], 'learning_rate':[0.01, 0.1, 0.5, 1]}



# TODO: Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, parameters, scoring=scorer, n_jobs=-1)



# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_test)

best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores

print("Unoptimized model\n------")

print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))

print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))

print("Roc_au_score on testing data: {:.4f}".format(roc_auc_score(y_test, clf.fit(X_train, y_train).predict_proba(X_test)[:,1])))

print("\nOptimized Model\n------")

print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))

print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))

print("Roc_au_score on testing data: {:.4f}".format(roc_auc_score(y_test, best_clf.predict_proba(X_test)[:,1])))
submission = pd.read_csv('../input/udacity-mlcharity-competition/test_census.csv')

submission.head()
submission.info()
# fill missing values with mean for continues features

for col in ['age', 'education-num', 'hours-per-week']:

    submission[col] = submission[col].fillna(submission[col].mean())

    

# fill missing values with median for skewed features

for col in skewed:

    submission[col] = submission[col].fillna(submission[col].median())

    

# fill missing values with most frequent values for categorical features

for col in submission.columns:

    if col not in continues:

        most_frequent_values = submission[col].value_counts().sort_values().index[-1]

        submission[col] = submission[col].fillna(most_frequent_values)        
# check again

submission.info()
# log transform skewed features

for col in skewed:

    submission[col] = submission[col].apply(lambda x: np.log(x+1))
# normalize numerical features

scaler = MinMaxScaler()

submission[continues] = scaler.fit_transform(submission[continues])
# one-hot encoding for categorical features

submission = pd.get_dummies(submission)

print('{} total features after one_hot encoding.'.format(len(submission.columns)))
# drop first columns

submission_final = submission.drop('Unnamed: 0', 1)
# predict the results using the tuned model and output the file

submission['id'] = np.arange(len(submission))

submission['income'] = (best_clf.predict_proba(submission_final))[:, 1]

submission[['id', 'income']].to_csv('submission.csv', index=False)
!!jupyter nbconvert *.ipynb