from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#plots will be displayed in the notebook



%matplotlib inline



#models from sckikit-learn



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



#model evaluation



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import plot_roc_curve

#plots will be displayed in the notebook



%matplotlib inline



#models from sckikit-learn



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



#model evaluation



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import plot_roc_curve

df = pd.read_csv('/kaggle/input/heart-disease.csv')
df.shape
df.head(10)
##Data exploration(Explaratory data analysis eda)



    #What question(s) are you trying to solve (or prove wrong)? i.e. problem defn

    #What kind of data do you have and how do you treat different types? numerical or categorical

    #What’s missing from the data and how do you deal with it?

    #Where are the outliers and why should you care about them?

    #How can you add, change or remove features to get more out of your data?

df.tail()
#finding how many of each class there , 1 equals Yes(has heart disease) & 0 equlas No

df["target"].value_counts()
# Plot the value counts with a bar graph

df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"]);



df.info() 
#are there any missing values?

df.isna().sum()
df.describe()
#compare our target column with the sex column.

#Note: from the data dictionary for the target column, 1 = heart disease present, 0 = no heart disease. And for sex, 1 = male, 0 = female.

df.sex.value_counts()
#compare target column with sex coulmn



pd.crosstab(df.target, df.sex)
#creating a plot of crosstab

pd.crosstab(df.target, df.sex).plot(kind='bar',

                                  figsize=(15,9),

                                  color=["salmon", "lightblue"])

plt.title("Heart Disease Frequency Of Males vs Females")

plt.xlabel("0= Not a heart Patient, 1= Heart Patient")

plt.ylabel("No. of people")

plt.legend(["Female", "Male"]);

plt.xticks(rotation=0)
#create another figure

plt.figure(figsize=(15, 9))



#scatter plot with heart disease as positive

plt.scatter(df.age[df.target==1],

            df.thalach[df.target==1],

           c='red')



#scatter plot with heart disease as negative

plt.scatter(df.age[df.target==0],

            df.thalach[df.target==0],

           c='blue');



# Add some helpful info

plt.title("Heart Disease in function of Age and Max Heart Rate")

plt.xlabel("Age")

plt.legend(["Disease", "No Disease"])

plt.ylabel("Max Heart Rate");
# Histograms to check the distribution of age variable

df.age.plot.hist()

plt.xlabel("Age");
pd.crosstab(df.cp, df.target)
# Creating a new crosstab and base plot for chest pain

pd.crosstab(df.cp, df.target).plot(kind="bar", 

                                   figsize=(10,6), 

                                   color=["lightblue", "salmon"])



# Add attributes to the plot to make it more readable

plt.title("Heart Disease Frequency Per Chest Pain Type")

plt.xlabel("Chest Pain Type")

plt.ylabel("Frequency")

plt.legend(["No Disease", "Disease"])

plt.xticks(rotation = 0);



#cp - chest pain type 

#0: Typical angina: chest pain related decrease blood supply to the heart 

#1: Atypical angina: chest pain not related to heart 

#2: Non-anginal pain: typically esophageal spasms (non heart related) 

#3: Asymptomatic: chest pain not showing signs of disease
#comparing all of the independent variables in one hit.

#Why?

#Because this may give an idea of which independent variables may or may not have an impact on our target variab





# Finding the correlation between our independent variables

corr_matrix = df.corr()

corr_matrix







# making the matrix look a little prettier

corr_matrix = df.corr()

plt.figure(figsize=(15, 10))

sns.heatmap(corr_matrix, 

            annot=True, 

            linewidths=0.5, 

            fmt= ".2f", 

            cmap="YlGnBu");







df.head(10)



# Everything except target variable

X = df.drop("target", axis=1)



# Target variable

y = df.target.values
# Independent variables (no target column)

X

# Random seed for reproducibility

np.random.seed(42)



# Split into train & test set

X_train, X_test, y_train, y_test = train_test_split(X, # independent variables 

                                                    y, # dependent variable

                                                    test_size = 0.2) # percentage of data to use for test set
X_train.head()
y_train, len(y_train)


X_test.head()

y_test, len(y_test)




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
#KNN's train scores.

train_scores
#plotting KNN scores



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

rs_log_reg = RandomizedSearchCV(LogisticRegression(),

                                param_distributions=log_reg_grid,

                                cv=5,

                                n_iter=20,

                                verbose=True)



# Fit random hyperparameter search model

rs_log_reg.fit(X_train, y_train);
rs_log_reg.best_params_
rs_log_reg.score(X_test, y_test)
# Setup random seed

np.random.seed(42)



# Setup random hyperparameter search for RandomForestClassifier

rs_rf = RandomizedSearchCV(RandomForestClassifier(),

                           param_distributions=rf_grid,

                           cv=5,

                           n_iter=20,

                           verbose=True)



# Fit random hyperparameter search model

rs_rf.fit(X_train, y_train);

# Find the best parameters

rs_rf.best_params_

# Evaluate the randomized search random forest model

rs_rf.score(X_test, y_test)
model_scores
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
# Check the best parameters

gs_log_reg.best_params_

# Evaluate the model

gs_log_reg.score(X_test, y_test)
# Make preidctions on test data

y_preds = gs_log_reg.predict(X_test)
y_preds
y_test
# Import ROC curve function from metrics module

from sklearn.metrics import plot_roc_curve



# Plot ROC curve and calculate AUC metric

plot_roc_curve(gs_log_reg, X_test, y_test);
# Display confusion matrix

print(confusion_matrix(y_test, y_preds))
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
# Show classification report

print(classification_report(y_test, y_preds))
# Check best hyperparameters

gs_log_reg.best_params_
# Import cross_val_score

from sklearn.model_selection import cross_val_score



# Instantiate best model with best hyperparameters (found with GridSearchCV)

clf = LogisticRegression(C=0.23357214690901212,

                         solver="liblinear")
# Cross-validated accuracy score

cv_acc = cross_val_score(clf,

                         X,

                         y,

                         cv=5, # 5-fold cross-validation

                         scoring="accuracy") # accuracy as scoring

cv_acc
#Since there are 5 metrics here, taking the average.

cv_acc = np.mean(cv_acc)

cv_acc





# Cross-validated precision score

cv_precision = np.mean(cross_val_score(clf,

                                       X,

                                       y,

                                       cv=5, # 5-fold cross-validation

                                       scoring="precision")) # precision as scoring

cv_precision







# Cross-validated recall score

cv_recall = np.mean(cross_val_score(clf,

                                    X,

                                    y,

                                    cv=5, # 5-fold cross-validation

                                    scoring="recall")) # recall as scoring

cv_recall







# Cross-validated F1 score

cv_f1 = np.mean(cross_val_score(clf,

                                X,

                                y,

                                cv=5, # 5-fold cross-validation

                                scoring="f1")) # f1 as scoring

cv_f1



# Visualizing cross-validated metrics

cv_metrics = pd.DataFrame({"Accuracy": cv_acc,

                            "Precision": cv_precision,

                            "Recall": cv_recall,

                            "F1": cv_f1},

                          index=[0])

cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False);
# Fit an instance of LogisticRegression (taken from above)

clf.fit(X_train, y_train);
# Check coef_

clf.coef_
# Match features to columns

features_dict = dict(zip(df.columns, list(clf.coef_[0])))

features_dict




# Visualize feature importance

features_df = pd.DataFrame(features_dict, index=[0])

features_df.T.plot.bar(title="Feature Importance", legend=False);



pd.crosstab(df["sex"], df["target"])
# Contrast slope (positive coefficient) with target

pd.crosstab(df["slope"], df["target"])