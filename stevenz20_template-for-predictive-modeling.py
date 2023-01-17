import os



dir = "../input/titanic"

print("List of file provided: ")

print("-"*25)

for file in os.listdir(dir):

    location = os.path.join(dir, file)

    print(str(os.path.getsize(location)) + " bytes\t" + file) 

print("-"*25)


import warnings

warnings.filterwarnings("ignore")

import time



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_palette(["#30a2da", "#fc4f30", "#e5ae38", "#6d904f", "#8b8b8b"])



import sklearn

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn import linear_model, ensemble, neighbors, svm, tree, naive_bayes, gaussian_process, gaussian_process, discriminant_analysis

from sklearn import model_selection

from xgboost import XGBClassifier



# Set global random seed.

np.random.seed(42)



# Load train and validation datasets. The original train dataset will be splitted into train and test datasets.

df_raw = pd.read_csv("../input/titanic/train.csv")

df_val = pd.read_csv("../input/titanic/test.csv")



# Make a copy of the train dataset as base dataset in data cleaning process.

df_ref = df_raw.copy()



# Make a list of the dataset reference for batch cleaning.

data_list = [df_raw, df_val]



print(df_raw.info())

df_raw.head()
# Extract title from name.

for data in data_list:

    data['Title'] = df_raw['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



print("Unique titles in training dataset before cleaning: \n")

print(df_raw['Title'].unique())

print("-"*25)    

    

# There are some titles with low count. Combine those titles into one category "Misc".

# Filter rows with uncommon titles (total < 10).

uncommon_titles = df_raw['Title'].value_counts() < 10

df_raw['Title'] = df_raw['Title'].apply(lambda title: 'Misc' if uncommon_titles.loc[title] == True else title)

final_titles = df_raw['Title'].unique()



print("Title count for training dataset after cleaning: \n")

print(df_raw['Title'].value_counts())

print("-"*25)



# Clean up validation dataset.

print("Unique titles in validation dataset before cleaning: \n")

print(df_val['Title'].unique())

print("-"*25)



# Map the titles in the validation dataset to the existing titles in training set.

df_val['Title'] = df_raw['Title'].apply(lambda title: 'Misc' if title not in final_titles else title)



print("Title count for testing dataset after cleaning: \n")

print(df_val['Title'].value_counts())

print("-"*25)
print("Train/test data missing values per column:")

print(df_raw.isnull().sum())

print("-"*25)



print("Validation data missing values per column:")

print(df_val.isnull().sum())

print("-"*25)
# Fill in the missing Age with median of same Title, Sex and Marriage status.

for data in data_list:

    data['Age'].fillna(value=df_raw.groupby(['Sex', 'Title', 'SibSp'])['Age'].transform('median'), inplace=True)

    # If still missing, fill based on Median of same sex.

    data['Age'].fillna(value=df_raw.groupby(['Sex'])['Age'].transform('median'), inplace=True)



    # Fill in the missing Fare with median of the same Pclass.

    data['Fare'].fillna(value=df_raw.groupby(['Pclass'])['Fare'].transform('median'), inplace=True)

    

    # Fill in the missing Embarked with mode.

    data['Embarked'].fillna(value=df_raw['Embarked'].mode()[0], inplace=True)



# Too many missing value for Cabin. Add it to feature to drop.

column_to_drop = ['Ticket', 'Cabin', 'PassengerId']



# Verify all missing values are filled in except for Cabin, which will be deleted later.

print("Columns with missing values in training dataset: ")

print(df_raw.columns[df_raw.isnull().any()].tolist())



print("Columns with missing values in testing dataset: ")

print(df_val.columns[df_val.isnull().any()].tolist())
# Overview of the continuous features

sns.set()

sns.distplot(df_raw['Age'])

plt.show()



sns.distplot(df_raw['Fare'])

plt.show()
# Create bins for continuous variables.



# Bin the Fare based on quantile. (Testing dataset should use the bins created by the training dataset.)

df_raw['FareBin'], fare_bins = pd.qcut(df_raw['Fare'], q=4, retbins=True, labels=False)

df_val['FareBin'] = pd.cut(df_val['Fare'], bins=fare_bins, include_lowest=True, labels=False)



# Bin the Age using cut.

df_raw['AgeBin'], age_bins = pd.cut(df_raw['Age'].astype(int), bins=4, retbins=True, labels=False)

df_val['AgeBin'] = pd.cut(df_val['Age'], bins=age_bins, include_lowest=True, labels=False)
# Convert categorical features to numerical features.



# An overview of all the non-numerical features in the dataset.

print("Non numerical features in the dataset: ")

print(df_raw.select_dtypes(include=['object']).columns.tolist())

print('-'*25)



# Encode categorical data.

label_encoder = LabelEncoder()

for data in data_list:

    data['SexLabel'] = label_encoder.fit_transform(data['Sex'])



# Add additional feature as needed.

for data in data_list:

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    data['IsAlone'] = data['FamilySize'].apply(lambda x: 1 if x <= 1 else 0)
# Select necessary features for further study.

target = ['Survived']

features = ['Pclass', 'SexLabel', 'AgeBin', 'FamilySize', 'IsAlone', 'FareBin', 'Embarked', 'Title']



df_clean = df_raw[features]

df_clean_val = df_val[features]



print("Selected features: " + str(len(features)))

print(features)

print('-'*25)

print("Remaining categorical features in the dataset: ")

print(df_clean.select_dtypes(include=['object']).columns.tolist())

print('-'*25)



X_all = pd.get_dummies(df_clean)

y_all = df_raw[target]

X_val = pd.get_dummies(df_clean_val)



final_features = X_all.columns.tolist()

print("Features after one hot encoding: " + str(len(final_features)))

print(final_features)

print('-'*25)



X_all.head()
# Final check of data types and missing values.

print("Overview of training data: \n")

print(X_all.info())



print("\nOverview of validation data: \n")

print(X_val.info())
# Initial candidate models and add to MLA list.

MLA = [

    # Linear model.

    linear_model.LogisticRegression(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    # Basic Tree model.

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    # Ensemble model.

    ensemble.AdaBoostClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    ensemble.BaggingClassifier(),

    

    # Naive Bayes

    naive_bayes.GaussianNB(),

    naive_bayes.BernoulliNB(),

    

    # SVM

    svm.LinearSVC(),

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    

    # Clustering

    neighbors.KNeighborsClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    

    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #xgboost

    XGBClassifier()

    

]



# Create table to store MLA comparison info.

columns = [

    'Estimator',

    'Parameters',

    'Mean Train Acc',

    'Mean Test Acc',

    'Mean Test 3-Sigma',

    'Runtime'

]

MLA_summary = pd.DataFrame(columns=columns)
# Create table to compare MLA predictions

prediction = pd.DataFrame()



# Restructure the MLA with pair of [Model, rank] for future selection. Initialize the rank to 0.

for (index, model) in enumerate(MLA):

    MLA[index] = [model, 0]

# Iterate through MLA list and save performance metrics to summary table.

row_index = 0

for entry in MLA:

    model = entry[0]

    

    # Write name and parameters

    model_name = model.__class__.__name__

    MLA_summary.loc[row_index, 'Estimator'] = model_name

    MLA_summary.loc[row_index, 'Parameters'] = str(model.get_params())

    

    # Create Pipeline.

    pipeline = Pipeline(steps=[('scaler', StandardScaler()), (model_name, model)])

    

    # Run estimation with cross validation.

    cv_result = model_selection.cross_validate(pipeline, X_all, y_all, cv=5, scoring='accuracy', return_train_score=True)

    

    # Write model performance metrics.

    MLA_summary.loc[row_index, 'Mean Train Acc'] = cv_result['train_score'].mean()

    MLA_summary.loc[row_index, 'Mean Test Acc'] = cv_result['test_score'].mean()

    

    # Update rank with accuracy.

    entry[1] = cv_result['test_score'].mean()

    

    # If this is a non-bias random sample, then +/-3 standard deviations (std) from the mean.

    # It should statistically capture 99.7% of the subsets.

    MLA_summary.loc[row_index, 'Mean Test 3-Sigma'] = cv_result['test_score'].std() * 3  

    MLA_summary.loc[row_index, 'Runtime'] = cv_result['fit_time'].mean()

    

    # Save model predictions for future usage.

    pipeline.fit(X_all, y_all)

    prediction[model_name] = pipeline.predict(X_val)

    

    row_index += 1
MLA_summary.sort_values(by=['Mean Test Acc'], ascending=False, inplace=True)

MLA_summary
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html

fig, ax = plt.subplots(figsize=(10, 10))

sns.barplot(x='Mean Test Acc', y = 'Estimator', data = MLA_summary, color = 'b')



#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

plt.title('Machine Learning Algorithm Accuracy on Test Data \n')

plt.xlabel('Test Accuracy')

plt.ylabel('Model')
MLA.sort(reverse=True, key=lambda x: x[1])

MLA = MLA[:10]



print("Top 10 models for further optimization: ")

print('-'*25)

for entry in MLA:

    print(entry[0].__class__.__name__)

print('-'*25)
# General parameters used by multiple estimators.

grid_n_estimator = [10, 50, 100, 300]

grid_ratio = [.1, .25, .5, .75, 1.0]

grid_learning_rate = [.01, .03, .05, .1, .25]

grid_kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

grid_penalty = ['l1', 'l2']

grid_max_depth = [2, 4, 6, 8, 10, None]

grid_max_samples = [5, 10, .03, .05, .10]

grid_criterion = ['gini', 'entropy']

grid_bool = [True, False]



# Set grid params for all estimators in MLA list, matching the index.

grid_param = [

    # RidgeClassifierCV

    [{

        'RidgeClassifierCV__cv': [3, 5, 10]

    }],

    

    # LinearDiscriminantAnalysis

    [{

        'LinearDiscriminantAnalysis__solver': ['svd', 'lsqr', 'eigen']

    }],

    

    # LinearSVC

    [{

        'LinearSVC__penalty': grid_penalty,

        'LinearSVC__C': grid_ratio

    }],

    

    # LogisticRegression

    [{

        'LogisticRegression__penalty': grid_penalty,

        'LogisticRegression__C': grid_ratio

    }],

    

    # NuSVC

    [{

        'NuSVC__kernel': grid_kernel,

        'NuSVC__degree': [2, 3, 5]

    }],

    

    # SVC

    [{

        'SVC__C': grid_ratio,

        'SVC__kernel': grid_kernel,

        'SVC__degree': [2, 3, 5]

    }],

    

    # RandomForestClassifier

    [{

        'RandomForestClassifier__n_estimators': grid_n_estimator,

        'RandomForestClassifier__criterion': grid_criterion,

        'RandomForestClassifier__max_depth': grid_max_depth,

        'RandomForestClassifier__oob_score': grid_bool

    }],

    

    # GaussianProcessClassifier

    [{

        'GaussianProcessClassifier__optimizer': ['fmin_l_bfgs_b']

    }],

    

    # BaggingClassifier

    [{

        'BaggingClassifier__n_estimators': grid_n_estimator,

        'BaggingClassifier__oob_score': grid_bool,

        'BaggingClassifier__max_samples': grid_max_samples,

        'BaggingClassifier__max_features': grid_max_samples

    }],

    

    # GradientBoostingClassifier

    [{

        'GradientBoostingClassifier__learning_rate': grid_learning_rate,

        'GradientBoostingClassifier__max_depth': grid_max_depth,

        'GradientBoostingClassifier__n_estimators': grid_n_estimator,

    }],

]

# Create table to compare MLA predictions

prediction = pd.DataFrame()



# Create table to store MLA comparison info.

columns = [

    'Estimator',

    'Parameters',

    'Train Acc',

    'Test Acc',

    'Mean Test 3-Sigma',

    'Runtime',

    'Test ROC-AUC'

]

MLA_summary_2nd = pd.DataFrame(columns=columns)



# Iterate through MLA list and update performance metrics to summary table.

start_total = time.perf_counter()

row_index = 0



# Due to feature name mismatch issue between XGBoost and pandas dataframe. Convert input to numpy array.

X_train_np = X_all.values

X_val_np = X_val.values



MLA_ver_2 = MLA[:]

for entry in MLA_ver_2:

    model = entry[0]

    

    # Write name.

    model_name = model.__class__.__name__

    MLA_summary_2nd.loc[row_index, 'Estimator'] = model_name

    

    # Create pipeline.

    pipeline = Pipeline(steps=[('scaler', StandardScaler()), (model_name, model)])

    

    #

    start = time.perf_counter()

    grid_search = model_selection.GridSearchCV(estimator=pipeline, 

                                               param_grid=grid_param[row_index], 

                                               cv=5,

                                               refit='accuracy',

                                               return_train_score=True,

                                               scoring=['accuracy', 'roc_auc'])

    grid_search.fit(X_train_np, y_all)

    runtime = time.perf_counter() - start

    

    # Update the best parameters

    best_param = grid_search.best_params_

    MLA_summary_2nd.loc[row_index, 'Parameters'] = str(best_param)

    best_model = grid_search.best_estimator_    

    

    # Write model performance metrics.

    result = grid_search.cv_results_

    best_model_index = grid_search.best_index_

    MLA_summary_2nd.loc[row_index, 'Train Acc'] = result['mean_train_accuracy'][best_model_index]

    MLA_summary_2nd.loc[row_index, 'Test Acc'] = result['mean_test_accuracy'][best_model_index]

    MLA_summary_2nd.loc[row_index, 'Test ROC-AUC'] = result['mean_test_roc_auc'][best_model_index]

    MLA_summary_2nd.loc[row_index, 'Mean Test 3-Sigma'] = result['std_test_roc_auc'][best_model_index] * 3

    MLA_summary_2nd.loc[row_index, 'Runtime'] = result['mean_fit_time'][best_model_index]



    # Save model predictions for future usage.

    best_model.fit(X_train_np, y_all)

    prediction[model_name] = model.predict(X_val_np)

    

    # Update MLA list.

    entry[0] = best_model

    entry[1] = result['mean_test_accuracy'][best_model_index]

    

    row_index += 1

    

    print('The best parameter for {} is {}.'.format(model_name, best_param))

    print("Mean test accuracy: {:.5f}.".format(result['mean_test_accuracy'][best_model_index]))

    print("Grid search runtime: {:.2f} seconds.".format(runtime))

    print('-'*25)

    

run_total = time.perf_counter() - start_total

print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-'*25)

MLA_summary_2nd
# Create vote estimator input list.

vote_estimators = []

for entry in MLA_ver_2:

    vote_estimators.append((entry[0][1].__class__.__name__, entry[0]))

    

# Hard vote

vote_hard = ensemble.VotingClassifier(estimators=vote_estimators, voting='hard')

vote_hard.fit(X_train_np, y_all)

vote_hard_cv = model_selection.cross_validate(vote_hard, 

                                              X_train_np, 

                                              y_all, 

                                              cv=5, 

                                              scoring='accuracy',

                                              return_train_score=True)





# Cross validation result.

print("Performance of Hard Voting in cross validation:")

print('-'*25)

print("Train Accuracy {:.5f} \t|| Test Accuracy {:.5f} \t|| Train Time {:.3f}"\

      .format(vote_hard_cv['train_score'].mean(), 

              vote_hard_cv['test_score'].mean(), 

              vote_hard_cv['fit_time'].mean()))



    
# Prediction.

submit = pd.DataFrame()

submit['PassengerId'] = df_val['PassengerId']

submit['Survived'] = vote_hard.predict(X_val.values)



# Save file for submission.

submit.to_csv("submission.csv", index=False)



print('Validation Data Distribution: \n', submit['Survived'].value_counts(normalize = True))

print('-'*25)

submit.sample(5)

!pip install watermark
%reload_ext watermark

%watermark -a 'S.Zhao' -nvm -p numpy,pandas,sklearn,matplotlib,seaborn,xgboost