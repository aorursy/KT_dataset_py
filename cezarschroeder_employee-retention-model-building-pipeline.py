import pandas as pd

import numpy as np

import datetime

import time

import os



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Configuring Plot Appearance



%config InlineBackend.figure_format='retina'

sns.set() # Revert to matplotlib defaults

plt.rcParams['figure.figsize'] = (9, 6)

plt.rcParams['axes.labelpad'] = 10

sns.set_style("darkgrid")
%load_ext version_information

%version_information pandas, numpy, matplotlib, seaborn, sklearn
hr_data_df = pd.read_csv('hr_data.csv')
hr_data_df.columns
hr_data_df.head()
hr_data_df.tail()
hr_data_df.left.value_counts()
# How is the target distributed?

hr_data_df.left.value_counts().plot('barh')
# Is data missing?

hr_data_df.left.isnull().sum()
# Which of the features are numerical (continuous or discrete) and which are categorical?

hr_data_df.dtypes
# Plotting Feature Distributions



for f in hr_data_df.columns:

    fig = plt.figure()

    s = hr_data_df[f]

    if s.dtype in ('float', 'int'): # histograms for numerical features

        num_bins = min((30, len(hr_data_df[f].unique())))

        s.hist(bins=num_bins)

    else:                           # bar plots for categorical features

        s.value_counts().plot.bar()    

    plt.xlabel(f)
# Percentage of Missing Values (NaNs) for each Feature

hr_data_df.isnull().sum() / len(hr_data_df) * 100
# Removing 'is_smoker' since it is full of missing values

del hr_data_df['is_smoker']
# Filling 'time_spend_company' Missing Values with Its Median (More Adequate for Discrete Numerical Features)

hr_data_df.time_spend_company = hr_data_df.time_spend_company.fillna(hr_data_df.time_spend_company.median())
# Checking that the values were correctly filled in

hr_data_df.isnull().sum() / len(hr_data_df) * 100
# Trying to take advantage of the relation between average_montly_hours and number_project

sns.boxplot(x='number_project', y='average_montly_hours', data=hr_data_df)

plt.savefig('employee-retention-hours-num-proj-boxplot.png', bbox_inches='tight', dpi=300)
# Calculating fill values for average_montly_hours given a number of projects

# This will result in more accurate fill values



mean_per_number_project = hr_data_df.groupby('number_project').average_montly_hours.mean()

mean_per_project = dict(mean_per_number_project)

mean_per_number_project
# Fill in average_monthly_hours with the appropriate values



hr_data_df.average_montly_hours = hr_data_df.average_montly_hours.fillna(hr_data_df.number_project.map(mean_per_number_project))
# Checking that the values were correctly filled in

hr_data_df.isnull().sum() / len(hr_data_df) * 100
# Converting categorical features to binary integer representation and to one-hot encoding



hr_data_df.left = hr_data_df.left.map({'no': 0, 'yes': 1})

hr_data_df = pd.get_dummies(hr_data_df)
# Verifying the final processed dataset



hr_data_df.columns
# Saving the preprocessed dataset to a file, which will then be used as input to the learning algorithm

hr_data_df.to_csv('hr_data_preprocessed.csv', index=False)
# Loading the processed data



hr_data_df = pd.read_csv('hr_data_preprocessed.csv')
# The two features we'll use for training in this section

# These features are going to be used only for illustration purposes, to show how the different models work.

# At the end of this section, we will build a learning model based on all the features.



sns.jointplot('satisfaction_level', 'last_evaluation', data=hr_data_df, kind='hex')

plt.savefig('employee-retention-satisfaction-evaluation-jointplot.png', bbox_inches='tight', dpi=300)
# Segmenting the plot by the target variable



fig, ax = plt.subplots()

plot_args = dict(shade=True, shade_lowest=False)



for i, c in zip((0, 1), ('Reds', 'Blues')):

    sns.kdeplot(hr_data_df.loc[hr_data_df.left==i, 'satisfaction_level'],

                hr_data_df.loc[hr_data_df.left==i, 'last_evaluation'],

                cmap=c, **plot_args)



ax.text(0.05, 1.05, 'left = 0', size=16, color=sns.color_palette('Reds')[-2])

ax.text(0.35, 1.05, 'left = 1', size=16, color=sns.color_palette('Blues')[-2])

plt.savefig('employee-retention-satisfaction-evaluation-bivariate-segmented.png', bbox_inches='tight', dpi=300)
# Splitting the dataset into training and testing sets



from sklearn.model_selection import train_test_split



features = ['satisfaction_level', 'last_evaluation']

X_train, X_test, y_train, y_test = train_test_split(

    hr_data_df[features].values, hr_data_df['left'].values,

    test_size=0.3, random_state=1)
# Scale the data for SVMs and K-Nearest Neighbors

# We should do this scaling operation always AFTER splitting the dataset

# Test data should not be influenced by operations on training data



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
# Train a support vector machine classifier

# The "very silly" qualifier of our initial SVM model comes from the rather naïve linear kernel for a nonlinear problem



from sklearn.svm import SVC



svm = SVC(kernel='linear', C=1, random_state=1)

svm.fit(X_train_scaled, y_train)
# Determining its classification accuracy



from sklearn.metrics import accuracy_score



y_pred = svm.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)

print('accuracy = {:.1f}%'.format(acc*100))
# Determining its accuracy by class

# Here, we clearly see the "obvious guessing" decision process of our silly model



from sklearn.metrics import confusion_matrix



print('percent accuracy score per class:')

cmat = confusion_matrix(y_test, y_pred)

scores = cmat.diagonal() / cmat.sum(axis=1) * 100

print('left = 0 : {:.2f}%'.format(scores[0]))

print('left = 1 : {:.2f}%'.format(scores[1]))
# Plot the resulting decision regions

# Note that all samples are being classified as left = 0



from mlxtend.plotting import plot_decision_regions



N_samples = 200

X, y = X_train_scaled[:N_samples], y_train[:N_samples]

plot_decision_regions(X, y, clf=svm);
# Training a more adequate SVM model for a nonlinear problem by means of radial basis functions



svm = SVC(kernel='rbf', C=1, random_state=1)

svm.fit(X_train_scaled, y_train)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from IPython.display import display

from mlxtend.plotting import plot_decision_regions



def check_model_fit(clf, X_test, y_test):

    # Print overall test-set accuracy

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred, normalize=True) * 100

    print('total accuracy = {:.1f}%'.format(acc))

    

    # Print confusion matrix

    cmat = confusion_matrix(y_test, y_pred)

    cols = pd.MultiIndex.from_tuples([('predictions', 0), ('predictions', 1)])

    indx = pd.MultiIndex.from_tuples([('actual', 0), ('actual', 1)])

    display(pd.DataFrame(cmat, columns=cols, index=indx))

    print()

    

    # Print test-set accuracy grouped by the target variable 

    print('percent accuracy score per class:')

    cmat = confusion_matrix(y_test, y_pred)

    scores = cmat.diagonal() / cmat.sum(axis=1) * 100

    print('left = 0 : {:.2f}%'.format(scores[0]))

    print('left = 1 : {:.2f}%'.format(scores[1]))

    print()

    

    # Plot decision regions

    fig = plt.figure(figsize=(8, 8))

    N_samples = 200

    X, y = X_test[:N_samples], y_test[:N_samples]

    plot_decision_regions(X, y, clf=clf)

    

    plt.xlabel('satisfaction_level')

    plt.ylabel('last_evaluation')

    plt.legend(loc='upper left')
check_model_fit(svm, X_test_scaled, y_test)

plt.savefig('employee-retention-svm-rbf.png', bbox_inches='tight', dpi=300)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train_scaled, y_train)



check_model_fit(knn, X_test_scaled, y_test)

plt.savefig('employee-retention-knn-overfit.png', bbox_inches='tight', dpi=300)
# Increasing the number of "nearest neighbors" to reduce overfitting



knn = KNeighborsClassifier(n_neighbors=25)

knn.fit(X_train_scaled, y_train)



check_model_fit(knn, X_test_scaled, y_test)

plt.savefig('employee-retention-knn.png', bbox_inches='tight', dpi=300)
from sklearn.ensemble import RandomForestClassifier



# Please, remember to limit max_depth in order to reduce overfitting

forest = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=1)

forest.fit(X_train, y_train)



check_model_fit(forest, X_test, y_test)

plt.xlim(-0.1, 1.2)

plt.ylim(0.2, 1.2)

plt.savefig('employee-retention-forest.png', bbox_inches='tight', dpi=300)
from sklearn.tree import export_graphviz

import graphviz



dot_data = export_graphviz(

                forest.estimators_[0],

                out_file=None, 

                feature_names=features,  

                class_names=['no', 'yes'],  

                filled=True, rounded=True,  

                special_characters=True)

graph = graphviz.Source(dot_data)

graph
# Stratified k-fold cross validation

# Model's predictive accuracy calculation



from sklearn.model_selection import cross_val_score



X = hr_data_df[features].values

y = hr_data_df.left.values



# Instantiate the model

clf = RandomForestClassifier(n_estimators=100, max_depth=5)



np.random.seed(1) # for reproducibility purposes

scores = cross_val_score(estimator=clf, X=X, y=y, cv=10)



print('accuracy = {:.3f} +/- {:.3f}'.format(scores.mean(), scores.std()))
# Custom function for class accuracy calculation



from sklearn.model_selection import StratifiedKFold



def cross_val_class_score(clf, X, y, cv=10):

    kfold = StratifiedKFold(n_splits=cv).split(X, y)



    class_accuracy = []

    for k, (train, test) in enumerate(kfold):

        clf.fit(X[train], y[train])

        y_test = y[test]

        y_pred = clf.predict(X[test])

        cmat = confusion_matrix(y_test, y_pred)

        class_acc = cmat.diagonal()/cmat.sum(axis=1)

        class_accuracy.append(class_acc)

        print('fold: {:d} accuracy: {:s}'.format(k+1, str(class_acc)))

        

    return np.array(class_accuracy)
# Stratified k-fold cross validation

# This time, including class accuracy calculation



np.random.seed(1) # for reproducibility purposes

scores = cross_val_class_score(clf, X, y)



print('accuracy = {} +/- {}'.format(scores.mean(axis=0), scores.std(axis=0)))
# Calcualte a validation curve



from sklearn.model_selection import validation_curve



clf = RandomForestClassifier(n_estimators=10)

max_depths = np.arange(3, 16, 3)



train_scores, test_scores = validation_curve(

            estimator=clf,

            X=X,

            y=y,

            param_name='max_depth',

            param_range=max_depths,

            cv=10);
# Function to draw the validation curve



def plot_validation_curve(train_scores, test_scores,

                          param_range, xlabel='', log=False):

    '''

    This code is from scikit-learn docs:

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    '''

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)



    fig = plt.figure()

    

    # Accuracy on training set

    

    plt.plot(param_range, train_mean, 

             color=sns.color_palette('Set1')[1], marker='o', 

             markersize=5, label='training accuracy')



    plt.fill_between(param_range, train_mean + train_std,

                     train_mean - train_std, alpha=0.15,

                     color=sns.color_palette('Set1')[1])



    # Accuracy on testing set

    

    plt.plot(param_range, test_mean, 

             color=sns.color_palette('Set1')[0], linestyle='--', 

             marker='s', markersize=5, 

             label='validation accuracy')



    plt.fill_between(param_range, 

                     test_mean + test_std,

                     test_mean - test_std, 

                     alpha=0.15, color=sns.color_palette('Set1')[0])



    if log:

        plt.xscale('log')

    plt.legend(loc='lower right')

    if xlabel:

        plt.xlabel(xlabel)

    plt.ylabel('Accuracy')

    plt.ylim(0.9, 1.0)

    return fig
# The max_depth parameter to be used is determined by the "elbow" in validation curves

# In this case, max_depth would equal 6



plot_validation_curve(train_scores, test_scores, max_depths, xlabel='max_depth')

plt.ylim(0.9, 0.95)

plt.savefig('employee-retention-validation-curve-overfitting.png', bbox_inches='tight', dpi=300)
features = ['satisfaction_level', 'last_evaluation', 'number_project',

       'average_montly_hours', 'time_spend_company', 'work_accident',

       'promotion_last_5years', 'department_IT', 'department_RandD',

       'department_accounting', 'department_hr', 'department_management',

       'department_marketing', 'department_product_mng', 'department_sales',

       'department_support', 'department_technical', 'salary_high',

       'salary_low', 'salary_medium']



X = hr_data_df[features].values

y = hr_data_df.left.values
# Jupyter magic function to assess CPU usage time

%time



# Calculating a validation curve for max_depth using a Random Forest classifier



np.random.seed(1)

clf = RandomForestClassifier(n_estimators=20)

max_depths = [3, 4, 5, 6, 7,

              9, 12, 15, 18, 21]

print('Training {} models ...'.format(len(max_depths)))

train_scores, test_scores = validation_curve(

            estimator=clf,

            X=X,

            y=y,

            param_name='max_depth',

            param_range=max_depths,

            cv=5);
# Drawing the validation curve

# Note the "elbow" at max_depth = 6



plot_validation_curve(train_scores, test_scores, max_depths, xlabel='max_depth')

plt.xlim(3, 21)

plt.savefig('employee-retention-max-depth-val.png', bbox_inches='tight', dpi=300)
# Performing a k-fold cross validation for the selected model:

# a random forest with max_depth = 6 and n_estimators = 200



np.random.seed(1)

clf = RandomForestClassifier(n_estimators=200, max_depth=6)

scores = cross_val_class_score(clf, X, y)



print('accuracy = {} +/- {}'.format(scores.mean(axis=0), scores.std(axis=0)))
# Box plot of result

# Note the higher uncertainty for 'left = 1' class (class imbalance)



fig = plt.figure(figsize=(5, 7))

sns.boxplot(data=pd.DataFrame(scores, columns=[0, 1]), palette=sns.color_palette('Set1'))

plt.xlabel('Left')

plt.ylabel('Accuracy')

plt.savefig('employee-retention-full-acc-wo-pca.png', bbox_inches='tight', dpi=300)
# Visualizing the feature importances

# Useful for extracting from the model knowledge about why employees are leaving, in accordance with business needs



pd.Series(clf.feature_importances_, name='Feature Importance', index=hr_data_df[features].columns).sort_values().plot.barh()

plt.xlabel('Feature Importance')

plt.savefig('employee-retention-full-feature-importance.png', bbox_inches='tight', dpi=300)
from sklearn.decomposition import PCA



pca_features = ['work_accident', 'salary_low', 'salary_high', 'salary_medium',

       'promotion_last_5years', 'department_RandD', 'department_hr',

       'department_technical', 'department_support',

       'department_management', 'department_sales',

       'department_accounting', 'department_IT', 'department_product_mng',

       'department_marketing']



X_reduce = hr_data_df[pca_features]



pca = PCA(n_components=3)

pca.fit(X_reduce)

X_pca = pca.transform(X_reduce)
# Adding principal components to hr_data_df



hr_data_df['first_principle_component'] = X_pca.T[0]

hr_data_df['second_principle_component'] = X_pca.T[1]

hr_data_df['third_principle_component'] = X_pca.T[2]
# Selecting reduced-dimension feature set



features = ['satisfaction_level', 'number_project', 'time_spend_company',

            'average_montly_hours', 'last_evaluation',

            'first_principle_component',

            'second_principle_component',

            'third_principle_component']



X = hr_data_df[features].values

y = hr_data_df.left.values
# Performing a (k=10)-fold cross validation for the selected model with reduced dimensionality:

# a random forest with max_depth = 6 and n_estimators = 200



np.random.seed(1)

clf = RandomForestClassifier(n_estimators=200, max_depth=6)

scores = cross_val_class_score(clf, X, y)



print('accuracy = {} +/- {}'.format(scores.mean(axis=0), scores.std(axis=0)))
# Box plot of result

# Note the higher accuracy and lower standard deviation for 'left = 1' class

# However, class imbalance still plays a quite significant role in the accuracy values



fig = plt.figure(figsize=(5, 7))

sns.boxplot(data=pd.DataFrame(scores, columns=[0, 1]), palette=sns.color_palette('Set1'))

plt.xlabel('Left')

plt.ylabel('Accuracy')

plt.savefig('employee-retention-full-acc-pca.png', bbox_inches='tight', dpi=300)
# IMPORTANT FINAL STEP: Train the final model on all the samples



np.random.seed(1)

clf = RandomForestClassifier(n_estimators=200, max_depth=6)

clf.fit(X, y)
# Saving the model for use in an external application



from sklearn.externals import joblib

joblib.dump(clf, 'random-forest-trained.pkl')
# Load model from pkl file



clf = joblib.load('random-forest-trained.pkl')

clf
# Example of using the model for a specific employee



sandra = hr_data_df.iloc[573]

X = sandra[features]

X
# Predict the class label for Sandra

# She would LEAVE the job



clf.predict([list(X.values)])
# Predict the probability of class labels for Sandra

# P('left = 0') = 0.06576239

# P('left = 1') = 0.93423761



clf.predict_proba([X])