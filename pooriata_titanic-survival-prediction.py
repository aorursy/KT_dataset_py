# Import all the tools we need



# Regular EDA (exploratory data analysis) and plotting libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# we want our plots to appear inside the notebook

%matplotlib inline 
# Models from Scikit-Learn

# We are going to test various models to check which one can get the higher accuracy 

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# Model Evaluations

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import plot_roc_curve
# Loading the train dataset

df = pd.read_csv('/kaggle/input/titanic/train.csv')#titanic/train.csv')

df.shape # (rows, columns)
#showing the first ten rows of the train data

df.head(10)
# Describing the features with numerical values

df.describe()
# Info of DataFarame

df.info()
df.isna().sum()
# Make a copy of the original DataFrame to perform edits on

df_edited = df.copy()

df_edited.drop('Cabin', axis=1, inplace=True) # Droping the cabin column

df_edited.info()
# Number of each class of Embarked feature

df_edited.Embarked.value_counts()
# Now we are filling the missing values



# Filling the Age Column

df_edited['Age'].fillna(df_edited['Age'].median(), inplace=True)



#Filling the Embarked Column

df_edited['Embarked'].fillna('S', inplace=True)
df_edited.isna().sum()
# Checking the number of each class in target variable

df.Survived.value_counts()
# Ploting the nuber of each class in bar chart

df["Survived"].value_counts().plot(kind='bar', color=['maroon','forestgreen'])

plt.title("Number of Survived vs. not Survived")

plt.xlabel("0 = Not Survived, 1 = Survived")

plt.ylabel("Number")

plt.xticks(rotation=0);
# Age distribution with histogram

df_edited['Age'].plot.hist()

plt.title("Age Distribution")

plt.xlabel("Age range");
# Create a plot of crosstab

# Survival vs.Sex

pd.crosstab(df_edited.Survived, df_edited.Sex).plot(kind="bar",

                                                    figsize=(10, 6),

                                                    color=["salmon", "lightblue"])



plt.title("Number of Survived vs. not Survived")

plt.xlabel("0 = Not Survived, 1 = Survived")

plt.ylabel("Number")

plt.legend(["Female", "Male"])

plt.xticks(rotation=0);
# Create a plot of crosstab

# Survival vs.Pclass

pd.crosstab(df_edited.Survived, df_edited.Pclass).plot(kind="bar",

                                                    figsize=(10, 6),

                                                    color=["salmon", "lightblue", "lightseagreen"])



plt.title("Number of Survived vs. not Survived")

plt.xlabel("0 = Not Survived, 1 = Survived")

plt.ylabel("Number")

plt.legend(["1st class", "2nd class", "3rd class"])

plt.xticks(rotation=0);
df_edited.head()
# This will turn all of the string value into category values

for label, content in df_edited.items():

    if pd.api.types.is_string_dtype(content):

        df_edited[label] = content.astype("category").cat.as_ordered()
df_edited.info()
# Turn categorical variables into numbers and fill missing

for label, content in df_edited.items():

    if not pd.api.types.is_numeric_dtype(content):

        df_edited[label] = pd.Categorical(content).codes
df_edited.info()
# Splitting the data to X & y

X = df_edited.drop('Survived', axis=1)

y = df_edited['Survived']



#Selected features 

#features = ["Pclass", "Sex", "SibSp", "Parch"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]

X = pd.get_dummies(X[features])



X.shape, y.shape
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
# Put models in a dictionary

models = {"KNN": KNeighborsClassifier(), "Random Forest": RandomForestClassifier()}



# Create a function to fit and score models

def fit_and_score(models, X_train, X_val, y_train, y_val):

    """

    Fits and evaluates given machine learning models.

    models : a dict of differetn Scikit-Learn machine learning models

    X_train : training data (no labels)

    X_val : Validation data (no labels)

    y_train : training labels

    y_val : Validation labels

    """

    # Set random seed

    np.random.seed(42)

    # Make a dictionary to keep model scores

    model_scores = {}

    # Loop through model 

    for name, model in models.items():

        # Fit the model to the data

        model.fit(X_train, y_train)

        # Evaluates the model and append its scores to model_scores

        model_scores[name] = model.score(X_val, y_val)

    return model_scores
model_scores = fit_and_score(models, X_train, X_val, y_train, y_val)

model_scores
# Create a hyperparameter grid for RandomForestClassifier

rf_grid = {"n_estimators": np.arange(200, 1000, 50),

           "max_depth": [None],

           "min_samples_split": np.arange(10, 20, 2),

           "min_samples_leaf": np.arange(1, 10, 2)}
# Setup random seed

np.random.seed(42)



# Setup random hyperparameter search for RandomForestClassifier

rs_rf = RandomizedSearchCV(RandomForestClassifier(), 

                           param_distributions=rf_grid,

                           cv=5,

                           n_iter=20,

                           verbose=True)



# Fit random hyperparameter search model for RandomForestClassifier()

rs_rf.fit(X_train, y_train)
# Find the best hyperparameters

rs_rf.best_params_
# Evaluate the randomized search RandomForestClassifier model

rs_rf.score(X_val, y_val)
# Importing the test dataset an preparing for the prediction

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test.head()
df_test.info()
df_test.isna().sum()
# remving Cabin column

df_test.drop('Cabin', axis=1, inplace=True) # Droping the cabin column

# Filling the empty item of Age and Fare columns

df_test['Age'].fillna(df_test['Age'].median(), inplace=True)

df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)

df_test.head()
df_test.isna().sum()
# Converting the non-numerica column to numeric ones

# This will turn all of the string value into category values

for label, content in df_test.items():

    if pd.api.types.is_string_dtype(content):

        df_test[label] = content.astype("category").cat.as_ordered()
df_test.info()
# Turn categorical variables into numbers and fill missing

for label, content in df_test.items():

    if not pd.api.types.is_numeric_dtype(content):

        df_test[label] = pd.Categorical(content).codes
df_test.info()
#Selected features 

#features = ["Pclass", "Sex", "SibSp", "Parch"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]

X_test = pd.get_dummies(df_test[features])

X_test.shape
# Now it is a time to make a predition with Randomforest model

# X & y 

model = RandomForestClassifier(n_estimators = 500,

                               min_samples_split = 14,

                               min_samples_leaf = 7,

                               max_depth = None, random_state=42)



model.fit(X,y)

y_preds = model.predict(X_test)
y_preds.shape
# Now we can convert our data to desired output.

output = pd.DataFrame({

    "PassengerId": df_test.PassengerId,

    "Survived": y_preds

})



output.to_csv("predicted_submission.csv", index=False)

print("The output CSV file was saved successfully!!")