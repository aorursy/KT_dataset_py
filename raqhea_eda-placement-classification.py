import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualisation

import seaborn as sns # data visualisation

sns.set_style('darkgrid')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = "/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv"
data = pd.read_csv(path, index_col = "sl_no")

data.head()
data.describe()
for feature in data.columns:

    print("There are {} null values for {} feature ".format( sum(data[feature].isnull()), feature ))
# So all the null values are because of the status feature. We don't need to replace it with something else.

data[data['status'] == 'Not Placed'].count()
#  First of all, does the dataset has a balanced distribution over the placement?

sns.countplot(x = 'status', hue = 'status', data = data)

plt.show()
# Is there a relationship between score percentages and placement?

score_p_cols = ['ssc_p', 'hsc_p', 'mba_p', 'degree_p']

score_cols_descs = ['Secondary School', 'Higher Secondary School', 'Masters of Business Administration', 'Under-grad Degree']

plt.figure(figsize = (12,12))



for s in range(len(score_p_cols)):

    plt.subplot(2,2, s + 1)

    plt.title("Graph 1.{}: ")

    sns.boxplot(x = data['status'], y = data[score_p_cols[s]], data = data)

    plt.title("Graph 1.{}: ".format(s) + score_cols_descs[s])

    plt.ylabel("")

plt.show()
# Are there any correlation between employment test scores and placement?

sns.boxplot(x = 'status', y = 'etest_p', data = data)

sns.swarmplot(x = 'status', y = 'etest_p', data = data, color = ".2")

plt.title('Graph 2: Employment Test Score Distribution by Status')

plt.ylabel('')

plt.show()



print("Placed students average score on employment test: {:.2f}".format(data[data['status'] == 'Placed'].etest_p.mean()))

print("Placed students standard deviation on employment test: {:.2f}".format(data[data['status'] == 'Placed'].etest_p.std()))

print("Not placed students average score on employment test: {:.2f}".format(data[data['status'] == 'Not Placed'].etest_p.mean()))

print("Not placed students standard deviation on employment test: {:.2f}".format(data[data['status'] == 'Not Placed'].etest_p.std()))
sns.countplot(x = 'status', hue = 'workex', data = data)

plt.title('Graph 3: Does having work experience is a strong factor for recruitment?')

plt.show()



# Let's look at some numbers

w_workex = data[data['workex'] == 'Yes'].status.value_counts().values

wo_workex = data[data['workex'] == 'No'].status.value_counts().values

print("Students with work experience:\n Total: {} \n Placed: {} \n Not Placed: {}".format(sum(w_workex),w_workex[0], w_workex[1]))

print("{:.2f}% of students with work experience is placed while {:.2f}% of them couldn't get placed".format(w_workex[0]/sum(w_workex) * 100, w_workex[1]/sum(w_workex) * 100))

print("\n Students without work experience:\n Total: {} \n Placed: {} \n Not Placed: {}".format(sum(wo_workex), wo_workex[0], wo_workex[1]))

print("{:.2f}% of students with work experience is placed while {:.2f}% of them couldn't get placed".format(wo_workex[0]/sum(wo_workex) * 100, wo_workex[1]/sum(wo_workex) * 100))
# Does an undergrad degree is a factor for recruitment?

sns.countplot(x = 'degree_t', hue = 'status', data = data)

plt.title('Graph 4.0')
sns.countplot(x = 'hsc_s', hue = 'status', data = data)

plt.title('Graph 4.1')
sns.countplot(x = 'specialisation', hue = 'status', data = data)

plt.title('Graph 4.2')
# Is there any correlation between secondary and higher secondary education?

sns.scatterplot(data['ssc_p'], data['hsc_p'])

plt.title('Graph 5.0')
# Let's look at the best fitting line

sns.lmplot(x = 'ssc_p', y = 'hsc_p', data = data)

plt.title('Graph 5.1')
# drop the unnecessary features

classification_data = data.drop(['salary', 'gender', 'ssc_b', 'hsc_b'], axis = 1)

# classification_data = data.drop(['salary'], axis = 1)

classification_data.head(10)
# One-hot-encode the categorical data we have and drop the first column in order get rid of dummy variable trap

classification_data = pd.get_dummies(classification_data, drop_first = True)

classification_data.shape
# We should split the dataset to feature/target subsets, then we'll split it to training and test set



# features

X = classification_data.values[:,:-1]

# target is "status" column

y = classification_data.values[:,-1]



# Split the dataset into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, shuffle = True, random_state = 42)
# Scale the data using StandardScaler 

# I'll just apply scaling to continuous variables.

from sklearn.preprocessing import StandardScaler

continuous_vars_train = X_train[:, :5]

continuous_vars_test = X_test[:, :5]



ss = StandardScaler()



continuous_vars_train = ss.fit_transform(continuous_vars_train)

continuous_vars_test = ss.transform(continuous_vars_test)



X_train[:, :5] = continuous_vars_train

X_test[:, :5] = continuous_vars_test
pd.DataFrame(X_train).describe()
# Import the classification models 

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

# I'll be testing the models using different metrics.

from sklearn.metrics import accuracy_score, fbeta_score, average_precision_score, roc_auc_score, roc_curve

from sklearn.metrics import confusion_matrix

from sklearn.metrics import make_scorer

# In order to do a proper test on the models, we should use KFold cross-validation

from sklearn.model_selection import StratifiedKFold, KFold

# After testing different algorithms, we need to optimize the algorithm further.

# I'll try to do this by using Grid Search

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Evaluation with cross validation offers more generalized results

from sklearn.model_selection import cross_val_score
# the most basic approach would be picking the majority class all the time. 

simple_baseline = np.ones(len(y_test))

accuracy_score(y_test, simple_baseline)
# Utility function to calculate the geometric mean of sensitivity (recall) and specificity (TNR)

def g_mean(y_true, y_preds):

    cm = confusion_matrix(y_true, y_preds)

    tn, fp, fn, tp = cm.ravel()

    sensivity = tp / (tp + fn)

    specifity = tn / (tn + fp)

    g_mean_score = np.sqrt(sensivity * specifity)

    return g_mean_score

g_mean_scorer = make_scorer(g_mean)
# I'll store the models in baseline_models list.

baseline_models = [LogisticRegression(), SVC(probability = True), 

                   KNeighborsClassifier(), GaussianNB(), 

                   DecisionTreeClassifier(), RandomForestClassifier()]



baseline_model_names = ['Logistic Regression', 'Support Vector Classifier', 

                        'K Nearest Neighbors', 'Gaussian Naive Bayes', 

                        'Decision Tree', 'Random Forest']



metric_names = ['Accuracy', 'F1 Score', 'G-Mean', 'ROC AUC Score']



# I'll apply 5 fold cross-validation to get a better inspection about the models

n_splits = 5



# Store the results so we can compare the models

results = np.zeros((len(baseline_models), len(metric_names)))



for i in range(len(baseline_models)):

    # Initialize an array to store the fold results

    model_results = np.zeros((n_splits, len(metric_names)))

    

    # Initialize the StratifiedKFold object with 5 splits.    

    skf = KFold(n_splits)

    skf.get_n_splits(X_train, y_train)

    

    # Get the model

    model = baseline_models[i]

    

    for fold_iter, (train_index, test_index) in enumerate(skf.split(X_train,y_train)):

        # Get the training and test folds

        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]

        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        

        # Fit the data to the model 

        model.fit(X_train_fold, y_train_fold)

        # Evaluate the test fold

        model_predictions = model.predict(X_test_fold)

        model_pred_probs = model.predict_proba(X_test_fold)

        # Get the evaluation scores 

        acc = accuracy_score(y_test_fold, model_predictions)

        f1_score = fbeta_score(y_test_fold, model_predictions, beta = 1)

        g_mean_score = g_mean(y_test_fold, model_predictions)

        roc_auc = roc_auc_score(y_test_fold, model_pred_probs[:,1])

        # Store the results in the model_results array

        model_results[fold_iter] = [acc, f1_score, g_mean_score, roc_auc]

        

    # The final result for the model is going to be the average of stratified cross-validation results

    final_result = model_results.mean(axis = 0)

    # store the results along with the model name in "results" list

    results[i] = final_result



# print out the results as a dataframe

skf_results_df = pd.DataFrame(results, 

                              columns = ["Average %s for %d Folds"%(metric, n_splits) for metric in metric_names],

                              index = baseline_model_names)

skf_results_df
# estimator = LogisticRegression()

# parameters = {

#     'penalty': ['l1', 'l2', 'elasticnet', 'none'],

#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

#     'tol': [0.5, 0.3, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],

#     'C': np.logspace(-4, 4, 32),

#     'max_iter': np.linspace(50, 2000, 40),

# }



# best_estimator = RandomizedSearchCV(estimator, parameters, n_iter = 3000, scoring = 'accuracy', n_jobs = -1)

# best_estimator.fit(X_train, y_train)

# print('Best parameters: ', best_estimator.best_params_)

# print('Best score: ', best_estimator.best_score_)





# I've found these parameters while optimizing hyper-parameters with the Random Search algorithm.

# You can also try different parameter options to apply random search, just uncomment the code above.

fine_tuned_params = {'tol': 0.3, 'solver': 'sag', 'penalty': 'none', 'max_iter': 1850.0, 'C': 0.0019512934226359622}

# fine_tuned_params = best_estimator.best_params_



tuned_estimator = LogisticRegression(**fine_tuned_params) # pick the best estimator

tuned_estimator.fit(X_train, y_train);
# make predictions

final_preds = tuned_estimator.predict(X_test)

final_acc_cv = cross_val_score(LogisticRegression(**fine_tuned_params),

                            X_test, y_test, cv = 5,

                            scoring = 'accuracy')

final_acc = accuracy_score(y_test, final_preds)

final_probas = tuned_estimator.predict_proba(X_test)

print('Final accuracy: %.4f | Final average accuracy (5 fold): %.4f'%(final_acc, final_acc_cv.mean()))



# Get the confusion matrix

cm = confusion_matrix(y_test, final_preds)

tn, fp, fn, tp = cm.ravel()

# plot it

sns.heatmap([[tn, fp], [fn, tp]], cmap = 'Blues', annot = True)

plt.xlabel('predictions')

plt.ylabel('actual values')

plt.show()



# my final visualization for the results is the ROC curve. 

# As a "no-skill" baseline approach, I'll use picking the majority method.

final_roc_auc_score = cross_val_score(LogisticRegression(**fine_tuned_params),

                                    X_test, y_test, 

                                    scoring = 'roc_auc')

baseline_roc_auc_score = roc_auc_score(y_test, simple_baseline)

print("Average ROC AUC Score:", final_roc_auc_score.mean())

print("Baseline ROC AUC Score:", baseline_roc_auc_score)





ns_fpr, ns_tpr, _ = roc_curve(y_test, simple_baseline)

lr_fpr, lr_tpr, _ = roc_curve(y_test, final_probas[:, 1])

sns.lineplot(ns_fpr, ns_tpr , label = 'Baseline (no-skill)')

sns.lineplot(lr_fpr, lr_tpr, label = 'Logistic Regression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("ROC Curve")

plt.legend()

plt.show()