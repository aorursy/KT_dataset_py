

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings('ignore')





from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier





from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import plot_roc_curve



%matplotlib inline
heart_dataset = pd.read_csv('../input/heart-disease-uci/heart.csv')

heart_dataset.head()
heart_dataset.target.value_counts()
heart_dataset.target.value_counts(normalize=True)
heart_dataset.target.value_counts().plot(kind = 'bar', color = ["blue", 'red'])

plt.tight_layout()

plt.show()
pd.crosstab(heart_dataset.target, heart_dataset.sex)
pd.crosstab(heart_dataset.target, heart_dataset.sex).plot(kind ='bar',

                                                         figsize = (10, 8),

                                                         color = ["red","blue"])

plt.title("Hear Disease Frequency for Sex attribute")

plt.xlabel("0 = No Disease \t 1 = Disease")

plt.ylabel("Target Amount")

plt.legend(["Male", "Female"])

plt.tight_layout()

plt.show()
# Create another figure

plt.figure(figsize=(10,6))



# Start with positve examples

plt.scatter(heart_dataset.age[heart_dataset.target==1], 

            heart_dataset.thalach[heart_dataset.target==1], 

            c="salmon") # define it as a scatter figure



# Now for negative examples, we want them on the same plot, so we call plt again

plt.scatter(heart_dataset.age[heart_dataset.target==0], 

            heart_dataset.thalach[heart_dataset.target==0], 

            c="lightblue") # axis always come as (x, y)



# Add some helpful info

plt.title("Heart Disease in function of Age and Max Heart Rate")

plt.xlabel("Age")

plt.legend(["Disease", "No Disease"])

plt.ylabel("Max Heart Rate");
# Create a new crosstab and base plot

pd.crosstab(heart_dataset.cp, heart_dataset.target).plot(kind="bar", 

                                   figsize=(10,6), 

                                   color=["blue", "salmon"])



# Add attributes to the plot to make it more readable

plt.title("Heart Disease Frequency Per Chest Pain Type")

plt.xlabel("Chest Pain Type")

plt.ylabel("Frequency")

plt.legend(["No Disease", "Disease"])

plt.xticks(rotation = 0);
corr_matrix = heart_dataset.corr()

plt.figure(figsize=(15, 10))

sns.heatmap(corr_matrix, 

            annot=True, 

            linewidths=0.5, 

            fmt= ".2f", 

            cmap="YlGnBu");
X = heart_dataset.drop('target', axis = 1)

y = heart_dataset.target.values
# Random seed for reproducibility

np.random.seed(42)



# Split into train & test set

X_train, X_test, y_train, y_test = train_test_split(X, # independent variables 

                                                    y, # dependent variable

                                                    test_size = 0.2) # percentage of data to use for test set
# Put models in a dictionary

models = {"KNN": KNeighborsClassifier(),

          "Logistic Regression": LogisticRegression(), 

          "Random Forest": RandomForestClassifier()}



# Create function to fit and score models

def fit_and_score(models, X_train, X_test, y_train, y_test):

    """

    Fits and evaluates given machine learning models.

    models : a dict of different Scikit-Learn machine learning models

    X_train : training data

    X_test : testing data

    y_train : labels assosciated with training data

    y_test : labels assosciated with test data

    """

    # Random seed for reproducible results

    np.random.seed(42)

    # Make a list to keep model scores

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
model_compare = pd.DataFrame(model_scores, index=['accuracy'])

model_compare.T.plot.bar();
# Create a list of train scores

train_scores = []



# Create a list of test scores

test_scores = []



# Create a list of different values for n_neighbors

neighbors = range(1, 21) # 1 to 20



# Setup algorithm

knn = KNeighborsClassifier()



# Loop through different neighbors values

for i in neighbors:

    knn.set_params(n_neighbors = i) # set neighbors value

    

    # Fit the algorithm

    knn.fit(X_train, y_train)

    

    # Update the training scores

    train_scores.append(knn.score(X_train, y_train))

    

    # Update the test scores

    test_scores.append(knn.score(X_test, y_test))
plt.plot(neighbors, train_scores, label="Train score")

plt.plot(neighbors, test_scores, label="Test score")

plt.xticks(np.arange(1, 21, 1))

plt.xlabel("Number of neighbors")

plt.ylabel("Model score")

plt.legend()



print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")
# Different LogisticRegression hyperparameters

log_reg_grid = {"C": np.logspace(-4, 4, 20),

                "solver": ["liblinear"]}



# Different RandomForestClassifier hyperparameters

rf_grid = {"n_estimators": np.arange(10, 1000, 50),

           "max_depth": [None, 3, 5, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf": np.arange(1, 20, 2)}
# Setup random seed

np.random.seed(42)



# Setup random hyperparameter search for LogisticRegression

RandomSearch_model = RandomizedSearchCV(LogisticRegression(),

                                param_distributions=log_reg_grid,

                                cv=5,

                                n_iter=20,

                                verbose=True)



# Fit random hyperparameter search model

RandomSearch_model.fit(X_train, y_train);
RandomSearch_model.best_params_
RandomSearch_model.score(X_test, y_test)
# Setup random seed

np.random.seed(42)



# Setup random hyperparameter search for RandomForestClassifier

RFC = RandomizedSearchCV(RandomForestClassifier(),

                           param_distributions=rf_grid,

                           cv=5,

                           n_iter=20,

                           verbose=True)



# Fit random hyperparameter search model

RFC.fit(X_train, y_train);
RFC.score(X_test, y_test)
# Different LogisticRegression hyperparameters

log_reg_grid = {"C": np.logspace(-4, 4, 20),

                "solver": ["liblinear"]}



# Setup grid hyperparameter search for LogisticRegression

gs_log_reg = GridSearchCV(LogisticRegression(),

                          param_grid=log_reg_grid,

                          cv=5,

                          verbose=True)



# Fit grid hyperparameter search model

gs_log_reg.fit(X_train, y_train);
# Make preidctions on test data

y_preds = gs_log_reg.predict(X_test)
# Import ROC curve function from metrics module

from sklearn.metrics import plot_roc_curve



# Plot ROC curve and calculate AUC metric

plot_roc_curve(gs_log_reg, X_test, y_test);
# Import Seaborn

import seaborn as sns

sns.set(font_scale=1.5) # Increase font size



def plot_conf_mat(y_test, y_preds):

    """

    Plots a confusion matrix using Seaborn's heatmap().

    """

    fig, ax = plt.subplots(figsize=(3, 3))

    ax = sns.heatmap(confusion_matrix(y_test, y_preds),

                     annot=True, # Annotate the boxes

                     cbar=False)

    plt.xlabel("true label")

    plt.ylabel("predicted label")

    

plot_conf_mat(y_test, y_preds)
print(classification_report(y_test, y_preds))
# Import cross_val_score

from sklearn.model_selection import cross_val_score



# Instantiate best model with best hyperparameters (found with GridSearchCV)

clf = LogisticRegression(C=0.23357214690901212,

                         solver="liblinear")
cross_validation_score_accuracy = cross_val_score(clf,

                                                  X,

                                                  y,

                                                  cv = 5,

                                                  scoring = 'accuracy')
cross_validation_score_precision = cross_val_score(clf,

                                                  X,

                                                  y,

                                                  cv = 5,

                                                  scoring = 'precision')
cross_validation_score_recall = cross_val_score(clf,

                                                  X,

                                                  y,

                                                  cv = 5,

                                                  scoring = 'recall')
cross_validation_score_f1 = cross_val_score(clf,

                                                  X,

                                                  y,

                                                  cv = 5,

                                                  scoring = 'f1')
# Visualizing cross-validated metrics

cv_metrics = pd.DataFrame({"Accuracy": cross_validation_score_accuracy,

                            "Precision": cross_validation_score_precision,

                            "Recall": cross_validation_score_recall,

                            "F1": cross_validation_score_f1})

cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False);