# Import all tools we need



# Regular EDA (exploratory data analysis) and plotting libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# we want our plots to appear inside the notebook

%matplotlib inline 



# Models from Scikit-Learn

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# Model Evaluations

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import plot_roc_curve
df = pd.read_csv('/kaggle/input/heartdisease/data/heart-disease.csv', delimiter=',', nrows=None)

df.shape 
df.head()
df.tail()
# Let's find out how many of each classes there

df.target.value_counts()
df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"]);
df.info()
# Are there any missing values?

df.isna().sum()
df.describe()
df.sex.value_counts()
# Compare target column with sex column  

pd.crosstab(df.target, df.sex)
# Create a plot of crosstab

pd.crosstab(df.target, df.sex).plot(kind="bar", 

                                    color=["salmon", "lightblue"],

                                    figsize=(10, 6));



plt.title("Heart Disease Frequency for Sex")

plt.xlabel("0 = No Disease, 1 = Disease")

plt.ylabel("Amount")

plt.legend(["Female", "Male"]);

plt.xticks(rotation=0)
df.head()
# Creating another figure

plt.figure(figsize=(10, 6))



# Scatter with positive examples

plt.scatter(df.age[df.target == 1],

            df.thalach[df.target == 1],

            color = "salmon")



# Scatter with negetive examples

plt.scatter(df.age[df.target == 0],

            df.thalach[df.target == 0],

            color = "lightblue")
df.age[df.target == 1]
# Check the distribution of age column with histogram

df.age.plot.hist();
pd.crosstab(df.cp, df.target)
# Make the crosstab more visual 

pd.crosstab(df.cp, df.target).plot(kind="bar",

                                   figsize=(10, 6),

                                   color = ["red","blue"])



# Add some communication

plt.title('Heart Disease Freq per chest pain type')

plt.xlabel('Chest pain type')

plt.ylabel("Amount")

plt.legend(["No Disease", "Disease"])

plt.xticks(rotation = 0);
df.head()
# Make a correlation matrix

df.corr()
# Let's make our correlation matrix more prettier

corrMatrix = df.corr()

fig, ax = plt.subplots(figsize=(15, 10))

ax = sns.heatmap(corrMatrix,

                 annot = True,

                 linewidths = 0.5,

                 fmt = ".2f",

                 cmap = "YlGnBu")

df.head()
# Split data to X & y

X = df.drop("target", axis=1)



y  = df.target
X
y
# Split data to train & test sets

np.random.seed(44)



# Split into train & test set

X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.2)
X_train
y_train, len(y_train)
# Put models in a dictionary

models = {"Logistic Regression":LogisticRegression(),

          "KNN":KNeighborsClassifier(),

         "Random Forest":RandomForestClassifier()}



# Create a function to fit and scroe models

def fit_and_score(models, X_train, X_test, y_train, y_test):

    """

    Fits and evaluates given machine learning models.

    models: a dict of different Scikit-Learn machine learning models

    X_train: training data (no labels)

    X_test: testing data(no labels)

    y_train: training labels

    y_test: test labels

    """

    # Set random seed

    np.random.seed(44)

    # Make a dictionary to keep model scores

    model_scores = {}

    # Loop through models

    for name, model in models.items():

        # Fit the model to the data

        model.fit(X_train, y_train)

        # Evaluate the model and append its score to model_scores

        model_scores[name] = model.score(X_test, y_test)

    return model_scores
model_scores = fit_and_score(models=models,

                             X_train=X_train,

                             X_test=X_test,

                             y_train=y_train,

                             y_test=y_test)



model_scores
models_compare = pd.DataFrame(model_scores, index=["accuracy"])

models_compare.T.plot(kind="bar")



plt.xticks(rotation =0);
# Let's tune KNN



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

    

    # Update training scores list

    train_scores.append(knn.score(X_train, y_train))

    

    # Update test scores list

    test_scores.append(knn.score(X_test, y_test))
train_scores
test_scores
plt.plot(neighbors, train_scores, label="Train score")

plt.plot(neighbors, test_scores, label="Test score")

plt.xticks(np.arange(1, 21, 1))

plt.xlabel("No of neighbors")

plt.ylabel("Model score")

plt.legend();



print(f'Maximum KNN score on test data:{max(test_scores)* 100:.2f}%')
# Create a hyperparameter grid for LogisticRegression

log_reg_grid ={"C":np.logspace(-4, 4, 20),

              "solver": ["liblinear"]}



# Create a hyperparameter grid for RandomForestClassifier

rf_grid ={"n_estimators": np.arange(10, 1000, 50),

          "max_depth":[None, 3, 5, 10],

          "min_samples_split":np.arange(2, 20, 2),

          "min_samples_leaf": np.arange(1, 20, 2)}
# Tune LogisticRegression



np.random.seed(34)



# Setup random hyperparameter search for LogisticRegression

rs_log_reg = RandomizedSearchCV(LogisticRegression(),

                                param_distributions=log_reg_grid,

                                cv=5,

                                n_iter=20,

                                verbose=True)



# Fit random hyperparameter search model for LogisticRegression

rs_log_reg.fit(X_train, y_train)
rs_log_reg.best_params_
rs_log_reg.score(X_test, y_test)
# Setup random seed

np.random.seed(44)



# Setup hyperparameter search for RandomForestClassifier

rs_rf = RandomizedSearchCV(RandomForestClassifier(),

                           param_distributions = rf_grid,

                           cv=5,

                           n_iter=2,

                           verbose=True)



# Fit random hyperparamete search for RandomForestClassifier

rs_rf.fit(X_train, y_train)
# Find the best hyperparameters

rs_rf.best_params_
# Evaluate the randomized search RandomForestClassifier model

rs_rf.score(X_test, y_test)
model_scores
# Different hyperparameter for LogisticRegression model

log_reg_grid = {"C": np.logspace(-4, 4, 30),

               "solver": ['liblinear']}



# Setup grid hyperparameter search for LogisticRegression

gs_log_reg = GridSearchCV(LogisticRegression(),

                          param_grid=log_reg_grid,

                          cv=5,

                          verbose=True)



# Fit grid hyperparameter search model

gs_log_reg.fit(X_train, y_train);
# Check the best parameters

gs_log_reg.best_params_
# Evaluate the grid search LogisticRegression model

gs_log_reg.score(X_test, y_test)
# Make predictions with tuned model

y_preds = gs_log_reg.predict(X_test)
y_preds
y_test
# Plot ROC curve and calculate and AUC metric

plot_roc_curve(gs_log_reg, X_test, y_test);
# Confusion matrix

confusion_matrix(y_test,y_preds)
sns.set(font_scale=1.5)



def plot_conf_mat(y_test, y_preds):

    """

    Plots a nice looking confusion matrix using Seaborn's heatmap()

    """

    fig, ax = plt.subplots(figsize=(3,3))

    ax = sns.heatmap(confusion_matrix(y_test, y_preds),

                     annot=True,

                     cbar=True)

    plt.xlabel("True label")

    plt.ylabel("Predicted label")

    

plot_conf_mat(y_test, y_preds)
print(classification_report(y_test, y_preds))
# Check best hyperparameters

gs_log_reg.best_params_
# Create a new classifier with best parameters

clf = LogisticRegression(C = 4.893900918477489,

                        solver = "liblinear")
# Cross-validated accuracy

cv_acc = cross_val_score(clf,

                         X,

                         y,

                         cv=5,

                         scoring = "accuracy")

cv_acc
cv_acc = np.mean(cv_acc)

cv_acc
# Cross-validated precision

cv_precision = cross_val_score(clf,

                               X,

                               y,

                               cv=5,

                               scoring = "precision")

cv_precision = np.mean(cv_precision)

cv_precision
# Cross-validated recall

cv_recall = cross_val_score(clf,

                            X,

                            y,

                            cv=5,

                            scoring = "recall")

cv_recall = np.mean(cv_recall)

cv_recall
# Cross-validated f1-score

cv_f1 = cross_val_score(clf,

                        X,

                        y,

                        cv=5,

                        scoring = "f1")

cv_f1 = np.mean(cv_f1)

cv_f1
# Visualize cross-validated metrics

cv_metrics = pd.DataFrame({"Accuracy": cv_acc,

                           "Precision": cv_precision,

                           "Recall":cv_recall,

                           "F1":cv_f1},

                         index= [0])



cv_metrics.T.plot(kind="bar",

                  title= "Cross-validated classification metrics",

                  legend=False)

plt.xticks(rotation=0);
# Fit an instance of LogisticRegression

clf = LogisticRegression(C = 4.893900918477489,

                        solver='liblinear')

clf.fit(X_train, y_train);
# Check  coef_

clf.coef_
X
X.head()
# Match coef's of features to columns

feature_dict = dict(zip(df.columns, list(clf.coef_[0])))

feature_dict
# Visualize feature impoertance

feature_df = pd.DataFrame(feature_dict, index=[0])

feature_df.T.plot(kind="bar",

                  title="Feature Importance",

                  figsize=(6,6),

                 legend = False)

plt.xticks(rotation=90);
pd.crosstab(df.sex, df.target)
pd.crosstab(df.slope, df.target)