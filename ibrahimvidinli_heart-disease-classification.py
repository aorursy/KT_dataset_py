# Importing the tools that we'll use

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Models

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# Model evaluations

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import plot_roc_curve



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
df['target'].value_counts()
df['target'].value_counts().plot(kind='bar', color=['salmon', 'lightblue'], xlabel='Target', ylabel='Count');
# Percentage of patients that have heart disease

df['target'].value_counts(normalize=True) * 100
df.info()
# Check to see if we have any missing values

df.isna().sum()
df.describe()
# Heart Disease Frequency for Sex

pd.crosstab(df['target'], df['sex']).plot(kind='bar', figsize=(10, 6), color=['salmon', 'lightblue'])

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('0: No Disease, 1: Disease')

plt.ylabel('Amount')

plt.legend(['Female', 'Male'])

plt.xticks(rotation=0)

plt.show()
plt.figure(figsize=(10,6))



# Positve examples

plt.scatter(df.age[df.target==1], 

            df.thalach[df.target==1], 

            c="salmon")



# Negative examples

plt.scatter(df.age[df.target==0], 

            df.thalach[df.target==0], 

            c="lightblue")



plt.title("Heart Disease in function of Age and Max Heart Rate")

plt.xlabel("Age")

plt.legend(["Disease", "No Disease"])

plt.ylabel("Max Heart Rate")

plt.show()
pd.crosstab(df.cp, df.target).plot(kind="bar", figsize=(10, 6), color=["lightblue", "salmon"]);

plt.title("Heart Disease Frequency Per Chest Pain Type")

plt.xlabel("Chest Pain Type")

plt.ylabel("Amount")

plt.legend(["No Disease", "Disease"])

plt.xticks(rotation=0);
# Correlation Matrix

corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(15, 10))

ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu");
# Turning Categorical variables into Dummy variables

cp = pd.get_dummies(df['cp'], prefix = "cp")

thal = pd.get_dummies(df['thal'], prefix = "thal")

slope = pd.get_dummies(df['slope'], prefix = "slope")

restecg = pd.get_dummies(df['restecg'], prefix = "restecg")

frames = [df, cp, thal, slope, restecg]

df = pd.concat(frames, axis = 1)

df = df.drop(columns = ['cp', 'thal', 'slope', 'restecg'])

df.head()
# Splitting the data into X and y and to train and test 

X = df.drop('target', axis=1)

y = df['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Put models in a dictionary

models = {"Logistic Regression": LogisticRegression(),

         "KNN": KNeighborsClassifier(),

         "Random Forest": RandomForestClassifier()}



# Create a function to fit and score models

def fit_and_score(models, X_train, X_test, y_train, y_test):

    """

    Fits and evaluatest given machine learning models. 

    models: a dict of different Scikit-Learn machine learning models

    X_train: training data (no labels)

    X_test: testing data (no labels)

    y_train: training labels

    y_test: testing labels

    """

    # Set random seed 

    np.random.seed(42)

    

    # Make a dictionary to keep model scores

    model_scores = {}

    

    for name, model in models.items():

        model.fit(X_train, y_train)

        model_scores[name] = model.score(X_test, y_test)

    return model_scores
model_scores = fit_and_score(models=models, 

                             X_train=X_train, 

                             X_test=X_test, 

                             y_train=y_train, 

                             y_test=y_test)

model_scores
# Model comparison

model_compare = pd.DataFrame(model_scores, index=["accuracy"])

model_compare.T.plot.bar();
# Let's tuning KNeighborsClassifier



train_scores = []

test_scores = []



# Create a list of different values for n_neighbors

neighbors = range(1, 21)



# Setup KNN instance

knn = KNeighborsClassifier()



# Loop through different n_neighbors 

for i in neighbors:

    knn.set_params(n_neighbors=i)

    # Fit the algorithm

    knn.fit(X_train, y_train)

    # Update the training scores list

    train_scores.append(knn.score(X_train, y_train))

    # Update the test scores list 

    test_scores.append(knn.score(X_test, y_test))
# Visualizing the KNN Scores

plt.plot(neighbors, train_scores, label="Train scores")

plt.plot(neighbors, test_scores, label="Test scores")

plt.xticks(np.arange(1,21,1))

plt.xlabel("Number of neighbors")

plt.ylabel("Model score")

plt.legend()



print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%");
# Hyperparameter grid for LogisticRegression

logReg_grid = {"C": np.logspace(-4, 4, 20),

                "solver": ["liblinear"]}



# Hyperparameter grid for RandomForestClassifier

randomForest_grid = {"n_estimators": np.arange(10, 1000, 50),

           "max_depth": [None, 3, 5, 10], 

           "min_samples_split": np.arange(2, 20, 2), 

           "min_samples_leaf": np.arange(1, 20, 2)}
# Tune LogisticsRegression

np.random.seed(42)



# Setup random hyperparameter search for LogsiticRegression

rs_logReg = RandomizedSearchCV(LogisticRegression(), 

                                param_distributions=logReg_grid, 

                                cv=5, n_iter=20, 

                                verbose=True)



# Fit random hyperparameter search model for LogisticRegression

rs_logReg.fit(X_train, y_train)
# Best parameters for LogisticRegression in RandomizedSearchCV

rs_logReg.best_params_
# Score after RandomizedSearchCV Hyperparameter Tuning

rs_logReg.score(X_test, y_test)
# Score before RandomizedSearchCV Hyperparameter Tuning

model_scores['Logistic Regression']
# Tune RandomForestClassifier

np.random.seed(42)



# Setup randdom hyperparameter search for RandomForestClassifier

rs_rf = RandomizedSearchCV(RandomForestClassifier(), 

                           param_distributions=randomForest_grid, 

                           cv=5, 

                           n_iter=20, 

                           verbose=True)



# Fit random hyperparameter search model for RandomForestClassifier

rs_rf.fit(X_train, y_train)
# Best Parameters for RandomForestClassifier in RandomizedSearchCV

rs_rf.best_params_
# Score after RandomizedSearchCV Hyperparameter Tuning

rs_rf.score(X_test, y_test)
# Score before RandomizedSearchCV Hyperparameter Tuning

model_scores['Random Forest']
# Different hyperparameters for our LogisticRegression model

log_reg_grid = {"C": np.logspace(-4, 4, 30),

                "solver": ["liblinear"]}



# Setup grid hyperparameter search for LogisticRegression

gs_log_reg = GridSearchCV(LogisticRegression(), 

                          param_grid=log_reg_grid, 

                          cv=5,  

                          verbose=True)



# Fit grid hyperparameter search model

gs_log_reg.fit(X_train, y_train)
# Best parameters

gs_log_reg.best_params_
# Evaluating the grid search LogisticRegression model

gs_log_reg.score(X_test, y_test)
# Make predictions with tuned model

y_preds = gs_log_reg.predict(X_test)

y_preds
# Plot ROC curve and calculate AUC metric

plot_roc_curve(gs_log_reg, X_test, y_test);
# Confusion Matrix

sns.set(font_scale=1.5)



fig, ax = plt.subplots(figsize=(3, 3))

ax = sns.heatmap(confusion_matrix(y_test, y_preds),

                    annot=True,

                    cbar=False)

plt.xlabel("True Label")

plt.ylabel("Predicted Label")
# Classification Method

print(classification_report(y_test, y_preds))
# Check best hyperparameters

gs_log_reg.best_params_
# Create a new classifier with best parameters

clf = LogisticRegression(C=0.20433597178569418, solver='liblinear')
# Cross-validated accuracy

cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

cv_acc = np.mean(cv_acc)

cv_acc
# Cross-validated precision

cv_precision = cross_val_score(clf, X, y, cv=5, scoring="precision")

cv_precision = np.mean(cv_precision)

cv_precision
# Cross-validated recall

cv_recall = cross_val_score(clf, X, y, cv=5, scoring="recall")

cv_recall = np.mean(cv_recall)

cv_recall
# Cross-validated f1-score

cv_f1 = cross_val_score(clf, X, y, cv=5, scoring="f1")

cv_f1 = np.mean(cv_f1)

cv_f1
# Visualize the cross-validated metrics

cv_metrics = pd.DataFrame({"Accuracy": cv_acc, 

                           "Precision": cv_precision,

                           "Recall": cv_recall,

                           "F1": cv_f1}, index=[0])



cv_metrics.T.plot.bar(title="Cross-validated classification metrics", 

                      legend=False);