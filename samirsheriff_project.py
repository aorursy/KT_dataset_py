###########################################

# Suppress matplotlib user warnings

# Necessary for newer version of matplotlib

import warnings

warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

#

# Display inline matplotlib plots with IPython

from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

###########################################



import matplotlib.pyplot as pl

import matplotlib.patches as mpatches

import numpy as np

import pandas as pd

from time import time

from sklearn.metrics import f1_score, accuracy_score





def distribution(data, transformed = False):

    """

    Visualization code for displaying skewed distributions of features

    """

    

    # Create figure

    fig = pl.figure(figsize = (11,5));



    # Skewed feature plotting

    for i, feature in enumerate(['capital-gain','capital-loss']):

        ax = fig.add_subplot(1, 2, i+1)

        ax.hist(data[feature], bins = 25, color = '#00A0A0')

        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)

        ax.set_xlabel("Value")

        ax.set_ylabel("Number of Records")

        ax.set_ylim((0, 2000))

        ax.set_yticks([0, 500, 1000, 1500, 2000])

        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])



    # Plot aesthetics

    if transformed:

        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \

            fontsize = 16, y = 1.03)

    else:

        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \

            fontsize = 16, y = 1.03)



    fig.tight_layout()

    fig.show()





def evaluate(results, accuracy, f1):

    """

    Visualization code to display results of various learners.

    

    inputs:

      - learners: a list of supervised learners

      - stats: a list of dictionaries of the statistic results from 'train_predict()'

      - accuracy: The score for the naive predictor

      - f1: The score for the naive predictor

    """

  

    # Create figure

    fig, ax = pl.subplots(2, 4, figsize = (15,9))



    # Constants

    bar_width = 0.1

    colors = ['#A00000','#00A0A0','#00A000', '#000000', '#F7DC6F', '#BB8FCE', '#EC7063']

    

    # Super loop to plot four panels of data

    for k, learner in enumerate(results.keys()):

        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):

            for i in np.arange(3):

                

                # Creative plot code

                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])

                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])

                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])

                ax[j//3, j%3].set_xlabel("Training Set Size")

                ax[j//3, j%3].set_xlim((-0.1, 3.0))

    

    # Add unique y-labels

    ax[0, 0].set_ylabel("Time (in seconds)")

    ax[0, 1].set_ylabel("Accuracy Score")

    ax[0, 2].set_ylabel("F-score")

    ax[1, 0].set_ylabel("Time (in seconds)")

    ax[1, 1].set_ylabel("Accuracy Score")

    ax[1, 2].set_ylabel("F-score")

    

    # Add titles

    ax[0, 0].set_title("Model Training")

    ax[0, 1].set_title("Accuracy Score on Training Subset")

    ax[0, 2].set_title("F-score on Training Subset")

    ax[1, 0].set_title("Model Predicting")

    ax[1, 1].set_title("Accuracy Score on Testing Set")

    ax[1, 2].set_title("F-score on Testing Set")

    

    # Add horizontal lines for naive predictors

    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    

    # Set y-limits for score panels

    ax[0, 1].set_ylim((0, 1))

    ax[0, 2].set_ylim((0, 1))

    ax[1, 1].set_ylim((0, 1))

    ax[1, 2].set_ylim((0, 1))



    # Set additional plots invisibles

    ax[0, 3].set_visible(False)

    ax[1, 3].axis('off')



    # Create legend

    for i, learner in enumerate(results.keys()):

        pl.bar(0, 0, color=colors[i], label=learner)

    pl.legend()

    

    # Aesthetics

    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)

    pl.tight_layout()

    pl.show()

    



def feature_plot(importances, X_train, y_train):

    

    # Display the five most important features

    indices = np.argsort(importances)[::-1]

    columns = X_train.columns.values[indices[:5]]

    values = importances[indices][:5]



    # Creat the plot

    fig = pl.figure(figsize = (9,5))

    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)

    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \

          label = "Feature Weight")

    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \

          label = "Cumulative Feature Weight")

    pl.xticks(np.arange(5), columns)

    pl.xlim((-0.5, 4.5))

    pl.ylabel("Weight", fontsize = 12)

    pl.xlabel("Feature", fontsize = 12)

    

    pl.legend(loc = 'upper center')

    pl.tight_layout()

    pl.show()  
# Import libraries necessary for this project

import numpy as np

import pandas as pd

from time import time

from IPython.display import display # Allows the use of display() for DataFrames

import seaborn as sns

import matplotlib.pyplot as plt



# Pretty display for notebooks

%matplotlib inline



# Load the Census dataset

data = pd.read_csv("../input/census.csv")



# Success - Display the first record

display(data.head())
# TODO: Total number of records

n_records = len(data)



# TODO: Number of records where individual's income is more than $50,000

n_greater_50k = (data['income'] == '>50K').sum()



# TODO: Number of records where individual's income is at most $50,000

n_at_most_50k = (data['income'] == '<=50K').sum()



# TODO: Percentage of individuals whose income is more than $50,000

greater_percent = n_greater_50k / n_records * 100.0



# Print the results

print("Total number of records: {}".format(n_records))

print("Individuals making more than $50,000: {}".format(n_greater_50k))

print("Individuals making at most $50,000: {}".format(n_at_most_50k))

print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
fig, ax = plt.subplots(figsize=(15,2))

sns.countplot(x='age', hue='sex', data=data, ax=ax)
fig, ax = plt.subplots(figsize=(15,2))

sns.countplot(x='age', hue='income', data=data, ax=ax)
fig, ax = plt.subplots(figsize=(15,2))

sns.countplot(x='education_level', hue='income', data=data, ax=ax)
fig, ax = plt.subplots(figsize=(25,2))

sns.countplot(x='occupation', hue='income', data=data, ax=ax)
fig, ax = plt.subplots(1, 2, figsize=(15, 4))

sns.countplot(x='income', hue='sex', data=data, ax=ax[0])

sns.countplot(x='income', hue='marital-status', data=data, ax=ax[1])
fig, ax = plt.subplots(1, 2, figsize=(15, 4))

sns.countplot(x='race', hue='income', data=data, ax=ax[0])

sns.countplot(x='relationship', hue='income', data=data, ax=ax[1])
fig, ax = plt.subplots(figsize=(30,2))

sns.countplot(x='native-country', hue='income', data=data, ax=ax)
fig, ax = plt.subplots(1, 2, figsize=(15, 4))

sns.boxplot(x='income', y='hours-per-week', data=data, ax=ax[0])

sns.boxplot(x='income', y='age', data=data, ax=ax[1])
fig, ax = plt.subplots(1, 2, figsize=(15, 4))

new_data = pd.DataFrame(data[['income', 'capital-gain', 'capital-loss']])

new_data['capital-gain-log'] = new_data['capital-gain'].apply(lambda x: np.log(x + 1))

new_data['capital-loss-log'] = new_data['capital-loss'].apply(lambda x: np.log(x + 1))

sns.violinplot(x='income', y='capital-gain-log', data=new_data, ax=ax[0])

sns.violinplot(x='income', y='capital-loss-log', data=new_data, ax=ax[1])
# Split the data into features and target label

income_raw = data['income']

features_raw = data.drop('income', axis = 1)



# Visualize skewed continuous features of original data

distribution(data)
# Log-transform the skewed features

skewed = ['capital-gain', 'capital-loss']

features_log_transformed = pd.DataFrame(data = features_raw)

features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))



# Visualize the new log distributions

distribution(features_log_transformed, transformed = True)
# Import sklearn.preprocessing.StandardScaler

from sklearn.preprocessing import MinMaxScaler



# Initialize a scaler, then apply it to the features

scaler = MinMaxScaler() # default=(0, 1)

numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']



features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)

features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])



# Show an example of a record with scaling applied

display(features_log_minmax_transform.head(n = 5))
features_log_minmax_transform.head()
# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()

categorical = ['workclass', 'education_level', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

features_final = pd.get_dummies(features_log_minmax_transform, columns=categorical)



# TODO: Encode the 'income_raw' data to numerical values

income = income_raw.apply(lambda x:  0 if x == '<=50K' else 1)



# Print the number of features after one-hot encoding

encoded = list(features_final.columns)

print("{} total features after one-hot encoding.".format(len(encoded)))



# Uncomment the following line to see the encoded feature names

encoded
# Import train_test_split

from sklearn.model_selection import train_test_split



# Split the 'features' and 'income' data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(features_final, 

                                                    income, 

                                                    test_size = 0.2, 

                                                    random_state = 0)



# Show the results of the split

print("Training set has {} samples.".format(X_train.shape[0]))

print("Testing set has {} samples.".format(X_test.shape[0]))
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 

                    # encoded to numerical values done in the data preprocessing step.

FP = income.count() - TP # Specific to the naive case



TN = 0 # No predicted negatives in the naive case

FN = 0 # No predicted negatives in the naive case



# TODO: Calculate accuracy, precision and recall

accuracy = TP / (TP + FP)

recall = TP / (TP + FN)

precision = TP / (TP + FP)



# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.

beta = 0.5

fscore = (1 + beta**2) * precision * recall / ((beta**2 * precision)  + recall)



# Print the results 

print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score

from sklearn.metrics import fbeta_score, accuracy_score



def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 

    '''

    inputs:

       - learner: the learning algorithm to be trained and predicted on

       - sample_size: the size of samples (number) to be drawn from training set

       - X_train: features training set

       - y_train: income training set

       - X_test: features testing set

       - y_test: income testing set

    '''

    

    results = {}

    

    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])

    start = time() # Get start time

    learner.fit(X_train[:sample_size], y_train[:sample_size])

    end = time() # Get end time

    

    # TODO: Calculate the training time

    results['train_time'] = end - start

        

    # TODO: Get the predictions on the test set(X_test),

    #       then get predictions on the first 300 training samples(X_train) using .predict()

    start = time() # Get start time

    predictions_test = learner.predict(X_test)

    predictions_train = learner.predict(X_train[:300])

    end = time() # Get end time

    

    # TODO: Calculate the total prediction time

    results['pred_time'] = end - start

            

    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]

    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

        

    # TODO: Compute accuracy on test set using accuracy_score()

    results['acc_test'] = accuracy_score(y_test, predictions_test)

    

    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()

    results['f_train'] = fbeta_score(y_train[:300], predictions_train, 0.5)

        

    # TODO: Compute F-score on the test set which is y_test

    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)

       

    # Success

    print("{} trained on {} samples \t with accuracy of {:0.2f}, {:0.2f} and F-score of {:0.2f}, {:0.2f}".format(

        learner.__class__.__name__, sample_size, results['acc_train'], results['acc_test'], 

        results['f_train'], results['f_test']))

        

    # Return the results

    return results
def save_max_result(clf_name, metric, result, max_results, type='max'):

    if type == 'max' and result[metric] > max_results[metric][1]:

        max_results[metric] = (clf_name, result[metric])

    elif type == 'min' and result[metric] < max_results[metric][1]:

        max_results[metric] = (clf_name, result[metric])



def save_max_results(clf_name, result, max_results, type='max'):

    if len(max_results) == 0:

        max_results['train_time'] = (clf_name, result['train_time'])

        max_results['pred_time'] = (clf_name, result['pred_time'])

        max_results['acc_train'] = (clf_name, result['acc_train'])

        max_results['acc_test'] = (clf_name, result['acc_test'])

        max_results['f_train'] = (clf_name, result['f_train'])

        max_results['f_test'] = (clf_name, result['f_test'])

    else:

        save_max_result(clf_name, 'train_time', result, max_results, type)

        save_max_result(clf_name, 'pred_time', result, max_results, type)

        save_max_result(clf_name, 'acc_train', result, max_results, type)

        save_max_result(clf_name, 'acc_test', result, max_results, type)

        save_max_result(clf_name, 'f_train', result, max_results, type)

        save_max_result(clf_name, 'f_test', result, max_results, type)

        

def display_max_results(max_results):

    for k, v in max_results.items():

        print(k, '\t', v)
# TODO: Import the three supervised learning models from sklearn

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



# TODO: Initialize the three models

clf_A = AdaBoostClassifier(random_state=42)

clf_B = SGDClassifier(random_state=42)

clf_C = LogisticRegression(random_state=42)

clf_D = RandomForestClassifier(random_state=42)

clf_E = GradientBoostingClassifier(random_state=42)

clf_F = DecisionTreeClassifier(random_state=42)

clf_G = BaggingClassifier(random_state=42)



# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data

# HINT: samples_100 is the entire training set i.e. len(y_train)

# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)

# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)

samples_100 = len(y_train)

samples_10 = int(0.1 * len(y_train))

samples_1 = int(0.01 * len(y_train))



# Collect results on the learners

results = {}

max_results = {}

min_results = {}

for clf in [clf_A, clf_B, clf_C, clf_D, clf_E, clf_F, clf_G]:

    clf_name = clf.__class__.__name__

    results[clf_name] = {}

    for i, samples in enumerate([samples_1, samples_10, samples_100]):

        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

    save_max_results(clf_name, results[clf_name][2], max_results, 'max')

    save_max_results(clf_name, results[clf_name][2], min_results, 'min')



print()

print('Max Results: ')

display_max_results(max_results)

print()

print('Min Results: ')

display_max_results(min_results)



# Run metrics visualization for the three supervised learning models chosen

evaluate(results, accuracy, fscore)
# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.ensemble import AdaBoostClassifier



# TODO: Initialize the classifier

clf = AdaBoostClassifier(random_state=42)



# TODO: Create the parameters list you wish to tune, using a dictionary if needed.

# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}

parameters = {

    'n_estimators' : [10, 50, 100],

    'learning_rate' : [1.0, 0.1]

}



# TODO: Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(fbeta_score, beta=0.5)



# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, parameters, scorer)



# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_

print("Best Estimator = ", best_clf)



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_test)

best_predictions = best_clf.predict(X_test)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))

print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))

print("\nOptimized Model\n------")

print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))

print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
# TODO: Import a supervised learning model that has 'feature_importances_'

from sklearn.ensemble import AdaBoostClassifier



# TODO: Train the supervised model on the training set using .fit(X_train, y_train)

model = AdaBoostClassifier()

model.fit(X_train, y_train)



# TODO: Extract the feature importances using .feature_importances_ 

importances = model.feature_importances_

print(importances)



# Plot

feature_plot(importances, X_train, y_train)
fig, ax = plt.subplots(1, 2, figsize=(15, 4))

ax[0].scatter(data['capital-gain'], data['income'])

ax[0].set_xlabel('capital-gain')

ax[0].set_ylabel('income')

ax[1].scatter(data['capital-loss'], data['income'])

ax[1].set_xlabel('capital-loss')

ax[1].set_ylabel('income')
# Import functionality for cloning a model

from sklearn.base import clone



# Reduce the feature space

X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]

X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]



# Train on the "best" model found from grid search earlier

start_time = time()

clf = (clone(best_clf)).fit(X_train, y_train)

print("Time for final Model to be trained on full data", (time() - start_time))



# Make new predictions

start_time = time()

reduced_predictions = clf.predict(X_test)

print("Time for final model trained on full data to predict", (time() - start_time))



# Train on the "best" model found from grid search earlier

start_time = time()

clf = (clone(best_clf)).fit(X_train_reduced, y_train)

print("Time for final Model to be trained on reduced data", (time() - start_time))



# Make new predictions

start_time = time()

reduced_predictions = clf.predict(X_test_reduced)

print("Time for final model trained on reduced data to predict", (time() - start_time))



# Report scores from the final model using both versions of data

print("Final Model trained on full data\n------")

print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))

print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))

print("\nFinal Model trained on reduced data\n------")

print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))

print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))
test_data = pd.read_csv('../input/test_census.csv', index_col=0)

test_data.head()
# Split the data into features and target label

features_raw = test_data
skewed = ['capital-gain', 'capital-loss']

features_log_transformed = pd.DataFrame(data = features_raw)

features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))



# Visualize the new log distributions

distribution(features_log_transformed, transformed = True)
scaler = MinMaxScaler() # default=(0, 1)

numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']



features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)

features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

features_log_minmax_transform.head()
categorical = ['workclass', 'education_level', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

features_final = pd.get_dummies(features_log_minmax_transform, columns=categorical)



# TODO: Encode the 'income_raw' data to numerical values

income = income_raw.apply(lambda x:  0 if x == '<=50K' else 1)



# Print the number of features after one-hot encoding

encoded = list(features_final.columns)

print("{} total features after one-hot encoding.".format(len(encoded)))



# Uncomment the following line to see the encoded feature names

encoded
features_final.head()
features_final.isnull().sum()
[col for col in test_data.columns if test_data[col].isnull().any()]
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

new_test_data = pd.DataFrame(imputer.fit_transform(features_final))

new_test_data.columns = features_final.columns

new_test_data.head()
new_test_data.isnull().sum()
preds = best_clf.predict(new_test_data)
test_pred_df = pd.DataFrame(preds, columns=['income']).reset_index().rename(columns={

    'index': 'id'

})
test_pred_df.head()
test_pred_df.to_csv('submission.csv', index=False)