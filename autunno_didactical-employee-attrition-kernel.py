import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

dataset.head()
# number of entries

print("Number of entries: " + str(len(dataset.index)))
dataset.isnull().any()
# Define a set of graphs, 3 by 5, usin the matplotlib library

f, axes = plt.subplots(5, 3, figsize=(24, 36), sharex=False, sharey=False)



# Define a few seaborn graphs, which for the most part only need the "dataset", the "x and "y" axis and the position. 

# You can also show a third value and expand your analysis by setting the "hue" property.

sns.swarmplot(x="EducationField", y="MonthlyIncome", data=dataset, hue="Gender", ax=axes[0,0])

axes[0,0].set( title = 'Monthly income against Educational Field')



sns.pointplot(x="PerformanceRating", y="JobSatisfaction", data=dataset, hue="Gender", ax=axes[0,1])

axes[0,1].set( title = 'Job satisfaction against Performance Rating')



sns.barplot(x="NumCompaniesWorked", y="PerformanceRating", data=dataset, ax=axes[0,2])

axes[0,2].set( title = 'Number of companies worked against Performance rating')



sns.barplot(x="JobSatisfaction", y="EducationField", data=dataset, ax=axes[1,0])

axes[1,0].set( title = 'Educational Field against Job Satisfaction')



sns.barplot(x="YearsWithCurrManager", y="JobSatisfaction", data=dataset, ax=axes[1,1])

axes[1,1].set( title = 'Years with current Manager against Job Satisfaction')



sns.pointplot(x="JobSatisfaction", y="MonthlyRate", data=dataset, ax=axes[1,2])

axes[1,2].set( title = 'Job Satisfaction against Monthly rate')



sns.barplot(x="WorkLifeBalance", y="DistanceFromHome", data=dataset, ax=axes[2,0])

axes[2,0].set( title = 'Distance from home against Work life balance')



sns.pointplot(x="OverTime", y="WorkLifeBalance", hue="Gender", data=dataset, jitter=True, ax=axes[2,1])

axes[2,1].set( title = 'Work life balance against Overtime')



sns.pointplot(x="OverTime", y="RelationshipSatisfaction", hue="Gender", data=dataset, ax=axes[2,2])

axes[2,2].set( title = 'Overtime against Relationship satisfaction')



sns.pointplot(x="MaritalStatus", y="YearsInCurrentRole", hue="Gender", data=dataset, ax=axes[3,0])

axes[3,0].set( title = 'Marital Status against Years in current role')



sns.pointplot(x="Age", y="YearsSinceLastPromotion", hue="Gender", data=dataset, ax=axes[3,1])

axes[3,1].set( title = 'Age against Years since last promotion')



sns.pointplot(x="OverTime", y="PerformanceRating", hue="Gender", data=dataset, ax=axes[3,2])

axes[3,2].set( title = 'Performance Rating against Overtime')



sns.barplot(x="Gender", y="PerformanceRating", data=dataset, ax=axes[4,0])

axes[4,0].set( title = 'Performance Rating against Gender')



sns.barplot(x="Gender", y="JobSatisfaction", data=dataset, ax=axes[4,1])

axes[4,1].set( title = 'Job satisfaction against Gender')



sns.countplot(x="Attrition", data=dataset, ax=axes[4,2])

axes[4,2].set( title = 'Attrition distribution')
if 'EmployeeNumber' in dataset:

    del dataset['EmployeeNumber']

    

if 'EmployeeCount' in dataset:

    del dataset['EmployeeCount']
features = dataset[dataset.columns.difference(['Attrition'])]

output = dataset.iloc[:, 1]
categorical = []

for col, value in features.iteritems():

    if value.dtype == 'object':

        categorical.append(col)



# Store the numerical columns in a list numerical

numerical = features.columns.difference(categorical)



print(categorical)
# get the categorical dataframe, and one hot encode it

features_categorical = features[categorical]

features_categorical = pd.get_dummies(features_categorical, drop_first=True)

features_categorical.head()
# get the numerical dataframe

features_numerical = features[numerical]



# concatenate the features

features = pd.concat([features_numerical, features_categorical], axis=1)



features.head()
labelencoder = LabelEncoder()

output = labelencoder.fit_transform(output)



print(output)
features_train, features_test, attrition_train, attrition_test = train_test_split(features, output, test_size = 0.3, random_state = 0)
random_forest = RandomForestClassifier(n_estimators = 800, criterion = 'entropy', random_state = 0)

random_forest.fit(features_train, attrition_train)
# Get the prediction array, 

attrition_pred = random_forest.predict(features_test)



# Get the accuracy %

print("Accuracy: " + str(accuracy_score(attrition_test, attrition_pred) * 100) + "%") 
# Making the Confusion Matrix

rf_cm = confusion_matrix(attrition_test, attrition_pred)



# building a graph to show the confusion matrix results

rf_cm_plot = pd.DataFrame(rf_cm, index = [i for i in {"Attrition", "No Attrition"}],

                  columns = [i for i in {"No attrition", "Attrition"}])

plt.figure(figsize = (6,5))

sns.heatmap(rf_cm_plot, annot=True, vmin=5, vmax=90.5, cbar=False, fmt='g')
def plot_feature_importances(importances, features):

    # get the importance rating of each feature and sort it

    indices = np.argsort(importances)



    # make a plot with the feature importance

    plt.figure(figsize=(12,14), dpi= 80, facecolor='w', edgecolor='k')

    plt.grid()

    plt.title('Feature Importances')

    plt.barh(range(len(indices)), importances[indices], height=0.8, color='mediumvioletred', align='center')

    plt.axvline(x=0.03)

    plt.yticks(range(len(indices)), list(features))

    plt.xlabel('Relative Importance')

    plt.show()



plot_feature_importances(random_forest.feature_importances_, features)
new_features = features.filter(['OverTime_Yes', 'MaritalStatus_Single', 'MaritalStatus_Married', 'JobRole_Sales Representative', 'JobRole_Sales Executive', 'JobRole_Research Scientist', 'JobRole_Research Director', 'JobRole_Manufacturing Director', 'JobRole_Manager', 'JobRole_Laboratory Technician', 'JobRole_Human Resources', 'Gender_Male', 'YearsWithCurrManager', 'YearsSinceLastPromotion', 'YearsAtCompany'])

new_features.head()
features_train_new, features_test_new, attrition_train_new, attrition_test_new = train_test_split(new_features, output, test_size = 0.3, random_state = 0)
# Build the new classifier

random_forest_new = RandomForestClassifier(n_estimators = 800, criterion = 'entropy', random_state = 0)

random_forest_new.fit(features_train_new, attrition_train_new)



# Get the prediction array, 

attrition_pred_new = random_forest_new.predict(features_test_new)



# Get the accuracy %

print("Accuracy: " + str(accuracy_score(attrition_test_new, attrition_pred_new) * 100) + "%") 
# Define a set of graphs, 1 by 2, to show the previous and the new confusion matrix side by side

f, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

plt.figure(figsize = (6,5))



# building a graph to show the old confusion matrix results

sns.heatmap(rf_cm_plot, annot=True, vmin=5, vmax=90.5, cbar=False, fmt='g', ax=axes[0])

axes[0].set( title = 'Random Forest CM (All features)')



# Making the new Confusion Matrix

rf_cm_new = confusion_matrix(attrition_test_new, attrition_pred_new)

rf_cm_new_plot = pd.DataFrame(rf_cm_new, index = [i for i in {"Attrition", "No Attrition"}],

                  columns = [i for i in {"No attrition", "Attrition"}])



# building a graph to show the new confusion matrix results

sns.heatmap(rf_cm_new_plot, annot=True, vmin=10, vmax=90.5, cbar=False, fmt='g', ax=axes[1])

axes[1].set( title = 'Random Forest CM (Selected features)')
plot_feature_importances(random_forest_new.feature_importances_, new_features)
sc = StandardScaler()

features_train_scaled = sc.fit_transform(features_train)

features_test_scaled = sc.transform(features_test)

features_train_new_scaled = sc.fit_transform(features_train_new)

features_test_new_scaled = sc.transform(features_test_new)
def plot_two_confusion_matrix(cm_1, title_1, cm_2, title_2):

    # Define a set of graphs, 1 by 2, to show the previous and the new confusion matrix side by side

    f, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    plt.figure(figsize = (6,5))



    # Builds the first CM plot

    cm_1_plot = pd.DataFrame(cm_1, index = [i for i in {"Attrition", "No Attrition"}],

                      columns = [i for i in {"No attrition", "Attrition"}])

    sns.heatmap(cm_1_plot, annot=True, vmin=5, vmax=90.5, cbar=False, fmt='g', ax=axes[0])

    axes[0].set(title = title_1)



    # Builds the second CM plot

    cm_2_plot = pd.DataFrame(cm_2, index = [i for i in {"Attrition", "No Attrition"}],

                      columns = [i for i in {"No attrition", "Attrition"}])

    sns.heatmap(cm_2_plot, annot=True, vmin=10, vmax=90.5, cbar=False, fmt='g', ax=axes[1])

    axes[1].set( title = title_2)



# We could have built a generic method to plot N number of confusion matrix, but we won't need it for this notebook

def plot_four_confusion_matrix(cm_1, title_1, cm_2, title_2, cm_3, title_3, cm_4, title_4):

    # Define a set of graphs, 2 by 2, to show the previous and the new confusion matrix side by side

    f, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=False, sharey=False)

    plt.figure(figsize = (6,5))



    # Builds the first CM plot

    cm_1_plot = pd.DataFrame(cm_1, index = [i for i in {"Attrition", "No Attrition"}],

                      columns = [i for i in {"No attrition", "Attrition"}])

    sns.heatmap(cm_1_plot, annot=True, vmin=5, vmax=90.5, cbar=False, fmt='g', ax=axes[0,0])

    axes[0,0].set(title = title_1)



    # Builds the second CM plot

    cm_2_plot = pd.DataFrame(cm_2, index = [i for i in {"Attrition", "No Attrition"}],

                      columns = [i for i in {"No attrition", "Attrition"}])

    sns.heatmap(cm_2_plot, annot=True, vmin=10, vmax=90.5, cbar=False, fmt='g', ax=axes[0,1])

    axes[0,1].set( title = title_2)



    # Builds the third CM plot

    cm_3_plot = pd.DataFrame(cm_3, index = [i for i in {"Attrition", "No Attrition"}],

                      columns = [i for i in {"No attrition", "Attrition"}])

    sns.heatmap(cm_3_plot, annot=True, vmin=10, vmax=90.5, cbar=False, fmt='g', ax=axes[1,0])

    axes[1,0].set( title = title_3)



    # Builds the fourth CM plot

    cm_4_plot = pd.DataFrame(cm_4, index = [i for i in {"Attrition", "No Attrition"}],

                      columns = [i for i in {"No attrition", "Attrition"}])

    sns.heatmap(cm_4_plot, annot=True, vmin=10, vmax=90.5, cbar=False, fmt='g', ax=axes[1,1])

    axes[1,1].set( title = title_4)

    

# Fit the classifier, get the prediction array and print the accuracy

def fit_and_pred_classifier(classifier, X_train, X_test, y_train, y_test):

    # Fit the classifier to the training data

    classifier.fit(X_train, y_train)



    # Get the prediction array

    y_pred = classifier.predict(X_test)

    

    # Get the accuracy %

    print("Accuracy with selected features: " + str(accuracy_score(y_test, y_pred) * 100) + "%") 

    

    return y_pred



# Run grid search, get the prediction array and print the accuracy and best combination

def fit_and_pred_grid_classifier(classifier, param_grid, X_train, X_test, y_train, y_test, scoring = "f1", folds = 5):

    # Apply grid search with F1 Score to help balance the results (avoid bias on "no attrition")

    grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid, cv = folds, scoring = scoring, n_jobs = -1, verbose = 0)

    grid_search.fit(X_train, y_train)

    best_accuracy = grid_search.best_score_

    best_parameters = grid_search.best_params_



    # Get the prediction array

    grid_search_pred = grid_search.predict(X_test)



    # Print the accuracy and best parameter combination

    print(scoring + " score: " + str(best_accuracy * 100) + "%")

    print("Accuracy: " + str(accuracy_score(y_test, grid_search_pred) * 100) + "%") 

    print("Best parameter combination: " + str(best_parameters)) 

    

    return grid_search_pred, grid_search_pred
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)



# Get the prediction array

knn_pred = fit_and_pred_classifier(knn, features_train_scaled, features_test_scaled, attrition_train, attrition_test)
knn_new = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)



# Get the prediction array

knn_pred_new = fit_and_pred_classifier(knn_new, features_train_new_scaled, features_test_new_scaled, attrition_train_new, attrition_test_new)
param_grid = {

    'n_neighbors': [1,2,4,5],

    'weights': ['distance', 'uniform'],

    'algorithm': ['ball_tree', 'kd_tree', 'brute'],

    'metric': ['minkowski','euclidean','manhattan'], 

    'p': [1, 2]

}
# Run grid search, print the results and get the prediction array and model

knn_grid_search_pred, knn_grid = fit_and_pred_grid_classifier(knn, param_grid, features_train_scaled, features_test_scaled, attrition_train, attrition_test)
# Run grid search, print the results and get the prediction array and model

knn_grid_search_pred_new, knn_grid_new = fit_and_pred_grid_classifier(knn_new, param_grid, features_train_new_scaled, features_test_new_scaled, attrition_train_new, attrition_test_new)
# Build the Confusion Matrix with all features

knn_cm = confusion_matrix(attrition_test, knn_pred)



# Build the Confusion Matrix with the selected features

knn_cm_new = confusion_matrix(attrition_test_new, knn_pred_new)



# Build the Confusion Matrix with all features and grid search

knn_grid_cm = confusion_matrix(attrition_test, knn_grid_search_pred)



# Build the Confusion Matrix with the selected features and grid search

knn_grid_cm_new = confusion_matrix(attrition_test_new, knn_grid_search_pred_new)



# Plot the four Coufusion Matrix

plot_four_confusion_matrix(knn_cm, 'K-NN (All features)', 

                           knn_cm_new, 'K-NN (Selected Features)',

                           knn_grid_cm, 'K-NN (Grid Search + All features)', 

                           knn_grid_cm_new, 'K-NN (Grid Search + Selected Features)')
# Build the model and fit it to the training data

svc = SVC(kernel = 'rbf', random_state = 0)



# Get the prediction array

svc_pred = fit_and_pred_classifier(svc, features_train_scaled, features_test_scaled, attrition_train, attrition_test)
# Build the model and fit it to the training data

svc_new = SVC(kernel = 'rbf', random_state = 0)



# Get the prediction array

svc_pred_new = fit_and_pred_classifier(svc_new, features_train_new_scaled, features_test_new_scaled, attrition_train_new, attrition_test_new)
param_grid = [

    {

        'C': [1, 2, 4, 5], 

        'kernel': ['linear']

    }, 

    {

        'C': [10, 11, 12, 13, 14, 15], 

        'kernel': ['rbf', 'sigmoid'], 

        'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]

    },

]
# Run grid search, print the results and get the prediction array and model

svc_grid_search_pred, svc_grid = fit_and_pred_grid_classifier(svc, param_grid, features_train_scaled, features_test_scaled, attrition_train, attrition_test, scoring = "recall")
# Run grid search, print the results and get the prediction array and model

svc_grid_search_pred_new, svc_grid_new = fit_and_pred_grid_classifier(svc, param_grid, features_train_new_scaled, features_test_new_scaled, attrition_train_new, attrition_test_new, scoring = "recall")
# Build the Confusion Matrix with all features

svc_cm = confusion_matrix(attrition_test, svc_pred)



# Build the Confusion Matrix with the selected features

svc_cm_new = confusion_matrix(attrition_test_new, svc_pred_new)



# Build the Confusion Matrix with all features and grid search

svc_grid_cm = confusion_matrix(attrition_test, svc_grid_search_pred)



# Build the Confusion Matrix with the selected features and grid search

svc_grid_cm_new = confusion_matrix(attrition_test_new, svc_grid_search_pred_new)



# Plot the four Coufusion Matrix

plot_four_confusion_matrix(svc_cm, 'Kernel SVM (All features)', 

                           svc_cm_new, 'Kernel SVM (Selected Features)',

                           svc_grid_cm, 'Kernel SVM (Grid Search + All features)', 

                           svc_grid_cm_new, 'Kernel SVM (Grid Search + Selected Features)')
# Build the Naive Bayes classifier with all the features

naive_bayes = GaussianNB()



# Fit the model and get the prediction array

attrition_pred_nb = fit_and_pred_classifier(naive_bayes, features_train, features_test, attrition_train, attrition_test)
# Build the Naive Bayes classifier with the selected features

naive_bayes_new = GaussianNB()



# Fit the model and get the prediction array

attrition_pred_nb_new = fit_and_pred_classifier(naive_bayes_new, features_train_new, features_test_new, attrition_train_new, attrition_test_new)
# Build the Confusion Matrix with all features

nb_cm = confusion_matrix(attrition_test, attrition_pred_nb)



# Build the Confusion Matrix with the selected features

nb_new_cm = confusion_matrix(attrition_test_new, attrition_pred_nb_new)



# Plot all Confusion Matrix

plot_two_confusion_matrix(nb_cm, 'Naive Bayes CM (All features)', nb_new_cm, 'Naive Bayes CM (Selected Features)')
param_grid = {

    'bootstrap': [True],

    'max_depth': [30, 40, 50],

    'max_features': [6, 7, 8],

    'min_samples_leaf': [1, 2, 3],

    'min_samples_split': [8, 9, 10],

    'n_estimators': [200, 300, 400, 500]

}
# Run grid search, print the results and get the prediction array and model

rf_grid_search_pred, rf_grid = fit_and_pred_grid_classifier(random_forest, param_grid, features_train, features_test, attrition_train, attrition_test)
# Run grid search, print the results and get the prediction array and model

rf_grid_search_pred_new, rf_grid_new = fit_and_pred_grid_classifier(random_forest, param_grid, features_train_new, features_test_new, attrition_train_new, attrition_test_new)
# Build the Confusion Matrix with all features and grid search

rf_grid_cm = confusion_matrix(attrition_test, rf_grid_search_pred)



# Build the Confusion Matrix with the selected features and grid search

rf_grid_cm_new = confusion_matrix(attrition_test_new, rf_grid_search_pred_new)



# Plot the four Coufusion Matrix

plot_four_confusion_matrix(rf_cm_plot, 'Random Forest CM (All features)', 

                           rf_cm_new_plot, 'Random Forest CM (Selected Features)',

                           rf_grid_cm, 'Random Forest CM (Grid Search + All features)', 

                           rf_grid_cm_new, 'Random Forest CM (Grid Search + Selected Features)')
# Ensemble 1 models

rf_ens_1 = RandomForestClassifier(max_depth = 30, max_features = 7, min_samples_leaf = 2, min_samples_split = 9, n_estimators = 400, random_state = 0)

kn_ens_1 = KNeighborsClassifier(algorithm = 'ball_tree', metric = 'minkowski', n_neighbors = 1, p = 1, weights = 'distance')



# Ensemble 2 models

rf_ens_2 = RandomForestClassifier(max_depth = 30, max_features = 7, min_samples_leaf = 1, min_samples_split = 9, n_estimators = 300, random_state = 0)

kn_ens_2 = KNeighborsClassifier(algorithm = 'brute', metric = 'minkowski', n_neighbors = 5, p = 2, weights = 'distance')
# Helper method to fit multiple estimators at once

def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):

    return  [clf.fit(X, y) if sample_weights is None else clf.fit(X, y, sample_weights) for clf, X in zip([clf for _, clf in classifiers], X_list)]



# Helper method to act as an ensemble

def predict_from_multiple_estimator(estimators, X_list, weights = None):

    # Predict 'soft' voting with probabilities

    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])

    pred2 = np.average(pred1, axis=0, weights=weights)

    

    # Return the prediction

    return np.argmax(pred2, axis=1)
# Build the estimator list and the applicable feature list in correct order

classifiers = [('knn', kn_ens_1), ('rf', rf_ens_1), ('nb', naive_bayes)]

train_list = [features_train_scaled, features_train, features_train]

test_list = [features_test_scaled, features_test, features_test]



# Fit the ensemble

fitted_estimators_1 = fit_multiple_estimators(classifiers, train_list, attrition_train)



# Run a prediction for our ensemble

ensemble_1_pred = predict_from_multiple_estimator(fitted_estimators_1, test_list, weights=[1,2,5])



# Get the accuracy %

print("Accuracy with selected features: " + str(accuracy_score(attrition_test, ensemble_1_pred) * 100) + "%") 
# Build the estimator list and the applicable feature list in correct order

classifiers = [('knn', kn_ens_2), ('rf', rf_ens_2), ('nb', naive_bayes)]

train_list = [features_train_scaled, features_train, features_train]

test_list = [features_test_scaled, features_test, features_test]



# Fit the ensemble

fitted_estimators_2 = fit_multiple_estimators(classifiers, train_list, attrition_train)



# Run a prediction for our ensemble

ensemble_2_pred = predict_from_multiple_estimator(fitted_estimators_2, test_list, weights=[2,3,5])



# Get the accuracy %

print("Accuracy with selected features: " + str(accuracy_score(attrition_test, ensemble_2_pred) * 100) + "%") 
# Build the Confusion Matrix for ensemble 1

ensemble_1_cm = confusion_matrix(attrition_test, ensemble_1_pred)



# Build the Confusion Matrix for ensemble 2

ensemble_2_cm = confusion_matrix(attrition_test_new, ensemble_2_pred)



# Plot the four Coufusion Matrix

plot_two_confusion_matrix(ensemble_1_cm, 'Ensemble 1 CM',  ensemble_2_cm, 'Ensemble 2 CM')
plot_four_confusion_matrix(svc_grid_cm, 'Linear SVM (All features)', 

                           rf_grid_cm_new, 'Random Forest (Grid Search + Selected Features)',

                           nb_cm, 'Naive Bayes (All features)', 

                           ensemble_2_cm, 'Ensemble 2')
# Add the encoded attrition to the original dataset, to allow it being combined plotted with categorical features

encoded_dataset = dataset

encoded_dataset['Attrition'] = output



# Define a set of graphs, 2 by 3, usin the matplotlib library

f, axes = plt.subplots(2, 3, figsize=(24, 16), sharex=False, sharey=False)



sns.boxplot(x="Attrition", y="YearsAtCompany", data=encoded_dataset, ax=axes[0,0])

axes[0,0].set( title = 'Years at company against Attrition')



sns.boxplot(x="Attrition", y="YearsSinceLastPromotion", data=encoded_dataset, ax=axes[0,1])

axes[0,1].set( title = 'Years since last promotion against Attrition')



sns.boxplot(x="Attrition", y="YearsWithCurrManager", data=encoded_dataset, ax=axes[0,2])

axes[0,2].set( title = 'Years with current manager against Attrition')



sns.violinplot(x="Attrition", y="JobRole", data=encoded_dataset, ax=axes[1,0])

axes[1,0].set( title = 'Job role against against Attrition')



sns.violinplot(x="Attrition", y="Gender", data=encoded_dataset, ax=axes[1,1])

axes[1,1].set( title = 'Gender against against Attrition')



sns.violinplot(x="Attrition", y="OverTime", data=encoded_dataset, ax=axes[1,2])

axes[1,2].set( title = 'Overtime against against Attrition')