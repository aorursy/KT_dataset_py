# We importing all the tool we need



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# We want our plots to appear inside the notebook

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
# Importing the file from the folder

# Loading the data

df_test = pd.read_csv(f"/kaggle/input/titanic/test.csv")
df_test
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_train
df_train.columns
df_test.columns
df_test.dtypes
df_train.dtypes
df_train.isnull().sum()
df_test.isnull().sum()
df_test = df_test.drop('Cabin', axis=1)

df_train = df_train.drop("Cabin", axis = 1)
df_test.columns
df_train.drop(["PassengerId", "Name", "Ticket", "Embarked"], axis = 1, inplace = True)

df_test.drop(["PassengerId", "Name", "Ticket", "Embarked"], axis = 1, inplace = True)
df_test
df_train.info()
# Make crosstab visual

pd.crosstab(df_train.Sex, df_train.Survived).plot(kind="bar",

                                           figsize=(10,6),

                                           color=["salmon", "lightblue"])

# Adiing more communications

plt.title("Passenger servived based on gender")

plt.xlabel("Sex of passenger")

plt.ylabel("Number of people")

plt.legend(["Not Survived", "Survived"])

plt.xticks(rotation = 0);
df_train.describe()
# Creating another figure

plt.figure(figsize=(10,6))



# Scatter with positive example

plt.scatter(df_train.Age[df_train.Survived==1],

            df_train.Sex[df_train.Survived==1],

            c="salmon")



# Scatter with negative examples

plt.scatter(df_train.Age[df_train.Survived==0],

            df_train.Sex[df_train.Survived==0],

            c="lightblue")



# Adding some communications

plt.title("Relation of survived passenger based on sex and age")

plt.xlabel("Age")

plt.ylabel("Sex")

plt.legend(["Survived","Death"])
# Now checking the missing values

df_train.info()
df_test.info()
#  Checking the outlier of the Age group

df_train.Age.plot.hist()
df_test.Age.plot.hist()
df_train["Age"].fillna((df_train["Age"].mean()), inplace = True)

df_test["Age"].fillna((df_test["Age"].mean()), inplace = True)
df_train.info()
df_test.info()
df_test.describe()
df_test.Fare.plot.hist()
df_test["Fare"].fillna((df_test["Fare"].mean()), inplace = True)
df_test.info()
# Lets make correlation matrix a little prettier

corr_matrix = df_train.corr()

fig, ax = plt.subplots(figsize=(15, 10))

ax = sns.heatmap(corr_matrix,

                 annot = True,

                 linewidths = 0.5,

                 fmt = ".2f",

                 cmap = "YlGnBu")

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
# Converting the sex into binary number

df_train.Sex[df_train.Sex=='male']=1

df_train.Sex[df_train.Sex=='female']=2
df_train.head()
df_test.Sex[df_test.Sex=='male']=1

df_test.Sex[df_test.Sex=='female']=2
df_train.info()
# Converting column `Sex` column into integer

df_train["Sex"] = df_train["Sex"].astype(int)

df_test["Sex"] = df_test["Sex"].astype(int)
df_train.info()
df_train.head()
df_test.head()
# Now splitting data into X and y

X = df_train.drop("Survived", axis=1)

y = df_train["Survived"]
X.info()
y
# Splitting the data into train and validation data

X_train, X_val, y_train, y_val = train_test_split(X, 

                                                  y, 

                                                  test_size=0.2)
X_train
y_train, y_val
# putting models in dictionary



models = {"Logistics Regression": LogisticRegression(),

            "KNN": KNeighborsClassifier(),

            "Random Forest": RandomForestClassifier()}

# Create a function to fit the score models

def fit_and_score(models, X_train, X_val, y_train, y_val):

    '''

    Fits and evaluate machine learning models.

    models: a dict of different scikit learn machine models

    X_train: training data(no labels)

    X_val : validation data (no labels)

    y_train : training data with labels

    y_val : validation data with labels

    '''

    

    # Make dictionary to keep model scores

    model_scores = {}

    # Loop through the models

    for name, model in models.items():

        

        #Fit the model to data

        model.fit(X_train, y_train)

        

        # Evaluate the model and append the scores to model_scores

        model_scores[name] = model.score(X_val, y_val)

    

    return model_scores
model_scores = fit_and_score(models = models,

                             X_train = X_train,

                             X_val = X_val,

                             y_train = y_train,

                             y_val = y_val)

model_scores
model_compare = pd.DataFrame(model_scores, index=["accuracy"])

model_compare.T.plot.bar()
# Lets tune KNN



train_scores = []

val_scores = []



# Create the list of different values for n_neighbors

neighbors = range(1, 21)



# Setup KNN instance

knn = KNeighborsClassifier()



# Loop through different n_neighbors

for i in neighbors:

    knn.set_params(n_neighbors=i)

    

    # Fit the algorithm 

    knn.fit(X_train, y_train)

    

    # Update training scores

    train_scores.append(knn.score(X_train, y_train))

    

    #Update the val score list

    val_scores.append(knn.score(X_val, y_val))
train_scores
val_scores
# Now plotting the graph for train scores and test scores

plt.plot(neighbors, train_scores, label = "Train score")

plt.plot(neighbors, val_scores, label = "Val Scores")

plt.xticks(np.arange(1, 21, 1))

plt.xlabel("Number of neighbors")

plt.ylabel("Model Score")

plt.legend()



print(f"Maximum Knn Score on validation data :{max(val_scores)*100:.2f}%")
# Creating a hyperparameter grid for LogisticRegression

log_reg_grid = {"C": np.logspace(-4, 4, 20),

                "solver": ["liblinear"]}



# Create a hyperparameter grid for RandomizedClassifier

rf_grid = {"n_estimators": np.arange(10, 1000, 50),

           "max_depth": [None, 3, 5, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf":np.arange(1, 20, 2)}
# Tune LogisticRegression

# Setup random hyerparameters search for LogisticRegression

rs_log_reg = RandomizedSearchCV(LogisticRegression(),

                               param_distributions = log_reg_grid,

                               cv = 5,

                               n_iter = 20,

                               verbose = True)



# Fit random hyperparameters search model for LogisticRegression

rs_log_reg.fit(X_train, y_train)
rs_log_reg.best_params_
rs_log_reg.score(X_val, y_val)
# Setting up hyperparameters for RandomForestClassifier()



rs_rf = RandomizedSearchCV(RandomForestClassifier(),

                          param_distributions = rf_grid,

                          cv = 5,

                          n_iter = 20,

                          verbose = True)



# Fit rondom hyperparameter search model for RandomForestClassifier()

rs_rf.fit(X_train, y_train)
rs_rf.best_params_
# Evaluate the randomized Search RandomForestClassifier()

rs_rf.score(X_val, y_val)
# Make predictions with tuned model

y_preds = rs_rf.predict(df_test)
y_preds
np.array(y_preds).sum()
df_test
y_val
# Plot ROC curve and calculate and calculate AUC matric

plot_roc_curve(rs_rf, X_val, y_val)
np.mean(y_preds), len(y_preds)
# Since I have modified all the data in original test data set.. So i have to import it again to get the passenger ID

test_dataset = pd.read_csv(f"/kaggle/input/titanic/test.csv")
test_dataset
# Formatting the predictions Kaggle is asking for

df_preds = pd.DataFrame()

df_preds["PassengerId"] = test_dataset["PassengerId"]

df_preds["Survived"] = y_preds
# Exporting prediction data

df_preds.to_csv("test.csv", index=False)