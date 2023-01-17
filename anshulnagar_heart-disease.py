# Import all the tools we need 



# Regular EDA (exploratory data analysis) and plotting libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline   

# We want our plots to appear inside the notebook



# Models from SciKit-Learn 

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# Model Evaluations

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score, plot_roc_curve
df = pd.read_csv("../input/heartdisease/heart-disease.csv")

df.shape # (rows,columns)
df.head()
df.tail()
# lets find out how many of each class there

df["target"].value_counts()


plt.style.use("seaborn")

df["target"].value_counts().plot(kind="bar",color=["red" , "green"]);

plt.title("Heart Disease Cases",fontsize=18)

plt.xlabel("0 = No Disease, 1 = Disease")

plt.ylabel("Count");

df.info()
# Are there any mising values ?

df.isna().sum()
df.describe()
df.sex.value_counts() # 1- Male 0- Female
# Compare Target column with sex column



pd.crosstab(df.target, df.sex).plot(kind="bar", color=["pink","blue"]);

plt.title("Heart Disease Frequency for Sex",fontsize=18)

plt.xlabel("0 = No Disease, 1 = Disease")

plt.ylabel("Ammount")

plt.legend(["Female","Male"])

plt.xticks(rotation=0);
df.thalach.value_counts() # there are many different values 
over_age = df[df["age"]>0]

over_age.head()
# Select the Style

plt.style.use('seaborn')

# Subplot of chol,age,thalach

fig, (ax0, ax1) = plt.subplots(nrows=2,

                              ncols=1,

                              figsize=(12,12),

                              sharex=True)



scatter = ax0.scatter(x=over_age["age"],

                      y=over_age["chol"],

                    c=over_age["target"],

                     cmap="winter" )# changes the colour skin

# customize the data

ax0.set(title="Heart Disease and cholestrol Levels",

      ylabel="Cholesterol")

ax0.set_xlim([20,80]) # changes the x axis limits

ax0.set_ylim([20,450]) # changes the y axis limits

# Add a Legend

ax0.legend(title="Target");

# Add a horizontal line

ax0.axhline(over_age["chol"].mean(), linestyle="--", color="red");



scatter = ax1.scatter(x=over_age["age"],

                    y=over_age["thalach"],

                    c=over_age["target"],

                     cmap="plasma")

# customize the data

ax1.set(title="Heart Disease and Max Heart Rate",

      xlabel="age",

      ylabel="Max Heart Rate")

ax1.set_xlim([20,80]) # changes the x axis limits

ax1.set_ylim([20,250]) # changes the y axis limits

# Add a Legend

ax1.legend(title="Target");

# Add a horizontal line

ax1.axhline(over_age["thalach"].mean(), linestyle="--", color="red");

# Add a Title to Figure

fig.suptitle("Heart Disease Analysis", fontsize=22 ,fontweight="bold");
# Cheak the distribution of the age column with a histogram

df.age.plot.hist(bins=20); # Normal distribution

plt.title("Age vs No. of Patients Data",fontsize=18);

plt.xlabel("Age");
over_50 = df[df["age"]>0]

over_50.head()
# Make a correlation matrix

df.corr()
# Let's make our correlation matrix a little prettier

corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(15, 10))

ax = sns.heatmap(corr_matrix,

                 annot=True,

                 linewidths=0.5,

                 fmt=".2f",

                 cmap="jet");

bottom, top = ax.get_ylim()

#ax.set_ylim(bottom + 0.5, top - 0.5) Adjustment

plt.title("Corelation Matrix",fontsize=18);
df.head()
# Split  data into X and Y



X = df.drop("target", axis=1)

Y = df["target"]
X
pd.DataFrame(Y)
# Split Data into Train And Test Sets

np.random.seed(42)



# Split into train and test set

X_train, X_test, Y_train, Y_test=train_test_split(X,

                                                  Y,

                                                  test_size=0.2)
len(X_train), len(Y_train), len(X_test), len(Y_test)
# Put models in a dictionary

models = {"Logistic Regression": LogisticRegression(),

          "KNN": KNeighborsClassifier(),

          "Random Forest": RandomForestClassifier()}



# Create a function to fit and score models

def fit_and_score(models, X_train, X_test, Y_train, Y_test):

    """

    Fits and evaluates given machine learning models.

    models : a dictionary of different Scikit-Learn machine learning models

    X_train : training data (no labels)

    X_test : testing data (no labels)

    y_train : training labels

    y_test : test labels

    """

    # Set random seed

    np.random.seed(42)

    # Make a dictionary to keep model scores

    model_scores = {}

    # Loop through models

    for name, model in models.items():

        # Fit the model to the data

        model.fit(X_train, Y_train)

        # Evaluate the model and append its score to model_scores

        model_scores[name] = model.score(X_test, Y_test)

    return model_scores
model_scores = fit_and_score(models=models, X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test);

model_scores
model_compare = pd.DataFrame(model_scores, index=["accuracy"])

model_compare.T.plot.bar();

plt.title("Model Comparision",fontsize=18)
# Let's tune KNN



train_scores = []

test_scores = []



# Create a list of differnt values for n_neighbors

neighbors = range(1, 21)



# Setup KNN instance

knn = KNeighborsClassifier()



# Loop through different n_neighbors

for i in neighbors:

    knn.set_params(n_neighbors=i)

    

    # Fit the algorithm

    knn.fit(X_train, Y_train)

    

    # Update the training scores list

    train_scores.append(knn.score(X_train, Y_train))

    

    # Update the test scores list

    test_scores.append(knn.score(X_test, Y_test))
train_scores,
test_scores
plt.plot(neighbors, train_scores, label="Train score")

plt.plot(neighbors, test_scores, label="Test score")

plt.xticks(np.arange(1,21,1))

plt.xlabel("Number of neighbors")

plt.ylabel("Model score")

plt.legend();



print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%");
 # Create a hyperparameter grid for LogisticRegression

log_reg_grid = {"C": np.logspace(-4, 4, 20),

                "solver": ["liblinear"]}



# Create a hyperparameter grid for RandomForestClassifier

rf_grid = {"n_estimators": np.arange(10, 1000, 50),

           "max_depth": [None, 3, 5, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf": np.arange(1, 20, 2)}
# Tune LogisticRegression



np.random.seed(42)



# Setup random hyperparameter search for LogisticRegression

rs_log_reg = RandomizedSearchCV(LogisticRegression(),

                                param_distributions=log_reg_grid,

                                cv=5,

                                n_iter=20,

                                verbose=True)



# Fit random hyperparameter search model for LogisticRegression

rs_log_reg.fit(X_train, Y_train)
rs_log_reg.best_params_
rs_log_reg.score(X_test,Y_test)
# Setup random seed

np.random.seed(42)



# Setup random hyperparameter search for RandomForestClassifier

rs_rf = RandomizedSearchCV(RandomForestClassifier(), 

                           param_distributions=rf_grid,

                           cv=5,

                           n_iter=20,

                           verbose=True)



# Fit random hyperparameter search model for RandomForestClassifier()

rs_rf.fit(X_train, Y_train)
# Find the best hyperparameters

rs_rf.best_params_
# Evaluate the randomized search RandomForestClassifier model

rs_rf.score(X_test, Y_test)
model_scores
# Different hyperparameters for our LogisticRegression models

log_reg_grid = {"C": np.logspace(-4,4,30),

                "penalty": ['l1', 'l2', 'elasticnet', 'none'],

                "solver": ["liblinear"],

                "dual":[True,False],

                "max_iter":[100,110,120,130,140]}



# Setup grid hyperparameter search for LogisticRegression

gs_log_reg = GridSearchCV(LogisticRegression(),

                          param_grid=log_reg_grid,

                          cv=5,

                          verbose=True)



# Fit grid hyperparameter search model

gs_log_reg.fit(X_train, Y_train);
# Check the best hyperparmaters

gs_log_reg.best_params_
# Evaluate the grid search LogisticRegression model

gs_log_reg.score(X_test, Y_test)
# Make predictions with tuned model

Y_preds = gs_log_reg.predict(X_test)
Y_preds
np.array(Y_test)
# Plot ROC curve and calculate and calculate AUC metric

plot_roc_curve(gs_log_reg, X_test, Y_test);
# Confusion matrix

print(confusion_matrix(Y_test, Y_preds))
sns.set(font_scale=1.5)



def plot_conf_mat(y_test, y_preds):

    """

    Plots a nice looking confusion matrix using Seaborn's heatmap()

    """

    fig, ax = plt.subplots(figsize=(3, 3))

    ax = sns.heatmap(confusion_matrix(y_test, y_preds),

                     annot=True,

                     cbar=False)

    ax.set_title("Confusion Matrix")

    plt.xlabel("True label")

    plt.ylabel("Predicted label")

    

    bottom, top = ax.get_ylim()

    #ax.set_ylim(bottom + 0.5, top - 0.5)

    

plot_conf_mat(Y_test, Y_preds)
print(classification_report(Y_test, Y_preds))
# Check best hyperparameters

gs_log_reg.best_params_
# Create a new classifier with best parameters

clf = LogisticRegression(C=0.20433597178569418,

                         solver="liblinear",

                         dual=False,

                         max_iter=100)
# Cross-validated accuracy

cv_acc = cross_val_score(clf,

                         X,

                         Y,

                         cv=5,

                         scoring="accuracy")

cv_acc
cv_acc=np.mean(cv_acc)

cv_acc
# Cross-validated precision

cv_precision = cross_val_score(clf,

                         X,

                         Y,

                         cv=5,

                         scoring="precision")

cv_precision=np.mean(cv_precision)

cv_precision
# Cross-validated recall

cv_recall = cross_val_score(clf,

                         X,

                         Y,

                         cv=5,

                         scoring="recall")

cv_recall = np.mean(cv_recall)

cv_recall
# Cross-validated f1-score

cv_f1 = cross_val_score(clf,

                         X,

                         Y,

                         cv=5,

                         scoring="f1")

cv_f1 = np.mean(cv_f1)

cv_f1
# Visualize cross-validated metrics

cv_metrics = pd.DataFrame({"Accuracy": cv_acc,

                           "Precision": cv_precision,

                           "Recall": cv_recall,

                           "F1": cv_f1},

                          index=[0])



cv_metrics.T.plot.bar(title="Cross-validated classification metrics",

                      legend=False);
# Fit an instance of LogisticRegression

clf = LogisticRegression(C=0.20433597178569418,

                         solver="liblinear")



clf.fit(X_train, Y_train);
# Check coef_

clf.coef_
df.head()
# Match coef's of features to columns

feature_dict = dict(zip(df.columns, list(clf.coef_[0])))

feature_dict
# Visualize feature importance

feature_df = pd.DataFrame(feature_dict, index=[0])

feature_df.T.plot.bar(title="Feature Importance", legend=False);
pd.crosstab(df["sex"], df["target"])
pd.crosstab(df["slope"], df["target"])

