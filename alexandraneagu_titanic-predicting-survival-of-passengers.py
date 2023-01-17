# Linear algebra

import numpy as np 



# Data processing

import pandas as pd 



# Data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# Evaluating algorithms

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import roc_auc_score



# Files location

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Test dataset

test = pd.read_csv("/kaggle/input/titanic/test.csv")



# Train dataset

train = pd.read_csv("/kaggle/input/titanic/train.csv")
# Look at a sample of the train dataset

train.head()
# Get the size of the train dataset

train.shape
# Get the data types of the columns and check for missing values

train.info()
# Use describe() function to get a summary of the data

train.describe(include = 'all')
# Sum all missing values for each column 

total = train.isnull().sum().sort_values(ascending=False)



# Calculate the percentage of missing values for each column

percent = round(train.isnull().sum()/train.isnull().count()*100, 1).sort_values(ascending=False)



# Create a data frame containing the total number of missing values and the % out of the total number of values

missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])



# Check the first 5 rows of the missing_data data frame

missing_data.head(5)
# Create a copy of the original train dataset to work with further on without PassengerId, Name, Ticket & Cabin 

train_ml = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)



# Replace the 2 missing values for Embarked with the most common value "S" in the dataset

train_ml["Embarked"] = train_ml["Embarked"].fillna("S")



# Replace missing values for Age with random numbers between mean-std and mean+std

# Calculate mean value for Age

mean = train_ml["Age"].mean()

# Calculate standard deviation for Age

std = train_ml["Age"].std()

# Get the size for the randint() function

is_null = train_ml["Age"].isnull().sum()

# Compute random numbers between mean-std and mean+std

rand_age = np.random.randint(mean - std, mean + std, size = is_null)

# Fill NaN values in Age column with random values generated

age_slice = train_ml["Age"].copy()

age_slice[np.isnan(age_slice)] = rand_age

train_ml["Age"] = age_slice



# Check that everything worked as expected using info()

train_ml.info()
# Convert Sex column to integer form

train_ml.loc[train_ml["Sex"] == "male", "Sex"] = 0

train_ml.loc[train_ml["Sex"] == "female", "Sex"] = 1

train_ml["Sex"] = train_ml["Sex"].astype(int)



# Convert Embarked column to integer form

train_ml.loc[train_ml["Embarked"] == "S", "Embarked"] = 0

train_ml.loc[train_ml["Embarked"] == "C", "Embarked"] = 1

train_ml.loc[train_ml["Embarked"] == "Q", "Embarked"] = 2

train_ml["Embarked"] = train_ml["Embarked"].astype(int)



# Check that everything work as expected

train_ml.head()
# CREATE A FUNCTION TO PLOT THE COUNT % PER CATEGORICAL COLUMN



def plot_count_percentage(x, x_axis_label, title, h=5, a=1, y_range=100):

    # Create a new data frame to plot the percentage of passengers per chosen categorical column

    df1 = train_ml[x].value_counts(normalize=True)*100



    # Rename the columns using an appropriate x_axis_label

    df1 = df1.reset_index().rename(columns={"index": x_axis_label, x: "Percentage"})

    

    # If chosen columns are Sex or Embarked convert to text

    if x == "Sex":

        df1.loc[df1[x_axis_label] == 0, x_axis_label] = "male"

        df1.loc[df1[x_axis_label] == 1, x_axis_label] = "female"

    elif x == 'Survived':

        df1.loc[df1[x_axis_label] == 0, x_axis_label] = "No"

        df1.loc[df1[x_axis_label] == 1, x_axis_label] = "Yes"

    elif x == "Embarked":

        df1.loc[df1[x_axis_label] == 1, x_axis_label] = "Cherbourg"

        df1.loc[df1[x_axis_label] == 2, x_axis_label] = "Queenstown"

        df1.loc[df1[x_axis_label] == 0, x_axis_label] = "Southampton"



    # Plot a barplot showing the percentage of passengers per chosen categorical column

    g = sns.catplot(x=x_axis_label,y='Percentage',kind='bar',data=df1, height=h, aspect=a)

    g.ax.set_ylim(0,y_range)

    

    # Set an appropriate title to the bar plot

    g.set(title = title + "\n")



    # Add the bar values to the plot

    for p in g.ax.patches:

        txt = str(p.get_height().round(0)) + '%'

        txt_x = p.get_x() 

        txt_y = p.get_height()

        g.ax.text(txt_x,txt_y,txt)
# CREATE A FUNCTION TO PLOT THE SURVIVED % PER CATEGORICAL COLUMN



def plot_survived_percentage(x, x_axis_label, title, h=5, a=1, y_range=100):

    # Create a new data frame to plot the % of survived/dead passengers vs. chosen categorical column

    df2 = train_ml.groupby(x)['Survived'].value_counts(normalize=True)*100

    df2 = df2.to_frame().rename(columns={'Survived': "Percentage"}).reset_index().rename(columns={x: x_axis_label})



    # Convert the values for survived to text for plotting

    df2.loc[df2["Survived"] == 0, "Survived"] = "dead"

    df2.loc[df2["Survived"] == 1, "Survived"] = "survived"

    

    # If chosen columns are Sex or Embarked convert to text

    if x == "Sex":

        df2.loc[df2[x_axis_label] == 0, x_axis_label] = "male"

        df2.loc[df2[x_axis_label] == 1, x_axis_label] = "female"

    elif x == "Embarked":

        df2.loc[df2[x_axis_label] == 1, x_axis_label] = "Cherbourg"

        df2.loc[df2[x_axis_label] == 2, x_axis_label] = "Queenstown"

        df2.loc[df2[x_axis_label] == 0, x_axis_label] = "Southampton"



    # Plot a barplot showing the % of survived/dead passengers vs. chosen categorical column

    g = sns.catplot(x=x_axis_label,y='Percentage', hue = 'Survived', kind='bar',data=df2, height=h, aspect=a)

    g.ax.set_ylim(0,y_range)

    

    # Set an approriate title

    g.set(title = title+ "\n")

    g._legend.remove()

    g.ax.legend().set_title('')



    # Add the bar values to the plot

    for p in g.ax.patches:

        txt = str(p.get_height().round(0)) + '%'

        txt_x = p.get_x() 

        txt_y = p.get_height()

        g.ax.text(txt_x,txt_y,txt)
# Use the function plot_count_percentage to plot the % of total passengers that survived/died

plot_count_percentage('Survived', "Survived Yes/No", 'Percentage of total passengers that survived/died')
# Use the function plot_count_percentage to plot the % of total passengers per passenger class

plot_count_percentage('Pclass', "Passenger Class", 'Percentage of total passengers per passenger class')
# Use the function plot_survived_percentage to plot the % of survived/dead passengers per passenger class

plot_survived_percentage('Pclass', 'Passenger Class', 'Percentage of survived/dead passengers vs. passenger class')
# Use the function plot_count_percentage to plot the % of total passengers per gender

plot_count_percentage('Sex', "Gender", 'Percentage of total passengers per gender')
# Use the function plot_survived_percentage to plot the % of survived/dead passengers per gender

plot_survived_percentage('Sex', 'Gender', 'Percentage of survived/dead passengers vs. gender')
# Use the function plot_count_percentage to plot the % of total passengers per number of siblings/spouses

plot_count_percentage('SibSp', "Number of siblings/spouses", 'Percentage of total passengers per number of siblings/spouses')
# Use the function plot_survived_percentage to plot the % of survived/dead passengers per number of siblings/spouses

plot_survived_percentage('SibSp', "Number of siblings/spouses", 'Percentage of survived/dead passengers vs. number of siblings/spouses', 8, 1)
# Use the function plot_count_percentage to plot the % of total passengers per number of parents/children

plot_count_percentage('Parch', "Number of parents/children", 'Percentage of total passengers per number of parents/children')
# Use the function plot_survived_percentage to plot the % of survived/dead passengers per number of parents/children"

plot_survived_percentage('Parch', "Number of parents/children", 'Percentage of survived/dead passengers vs. number of parents/children"', 8, 1)
# Use the function plot_count_percentage to plot the % of total passengers per embarkation point

plot_count_percentage('Embarked', "Embarkation point", 'Percentage of total passengers per embarkation point')
# Use the function plot_survived_percentage to plot the % of survived/dead passengers per embarkation point"

plot_survived_percentage('Embarked', "Embarkation point", 'Percentage of survived/dead passengers vs. embarkation point')
# As "Age" is a continous numerical variable is best to first have a look at its histogram

sns.distplot(train_ml.Age, kde=False, bins = 20)
# Create the column age_categories '0-10', '11-17','18-22', '23-26', '27-32', '33-39', '40-49','>=50'

train_ml['age_categories'] = pd.cut(train_ml.Age, [0, 11, 18, 23, 27, 33, 40, 50, np.inf], 

                                    labels=[0, 1, 2, 3, 4, 5, 6,7], include_lowest=True, right=False).astype(int)



# Create age_categories_text for plotting

train_ml['age_categories_text'] = pd.cut(train_ml.Age, [0, 11, 18, 23, 27, 33, 40, 50, np.inf],right=False,

                                    labels=['0-10', '11-17','18-22', '23-26', '27-32', '33-39', '40-49','>=50'], include_lowest=True)
# As "Fare" is a continous numerical variable is best to first have a look at its histogram

sns.distplot(train_ml.Fare, kde=False, bins = 50)
# Create the column fare_categories '0-7.99', '8-13.99','14-30.99', '31-98.99', '99-249.99', '>=250'

train_ml['fare_categories'] = pd.cut(train_ml.Fare, [0, 8, 14, 31, 99, 250, np.inf], 

                                    labels=[0, 1, 2, 3, 4, 5], include_lowest=True, right=False).astype(int)



# Create age_categories_text for plotting

train_ml['fare_categories_text'] = pd.cut(train_ml.Fare, [0, 8, 14, 31, 99, 250, np.inf], right=False,

                                    labels=['0-7.99', '8-13.99','14-30.99', '31-98.99', '99-249.99', '>=250'], include_lowest=True)
# Use the function plot_count_percentage to plot the % of total passengers per age_category

plot_count_percentage('age_categories_text', "Age Categories", 'Percentage of total passengers per age category', h =6, y_range=25)
# Use the function plot_survived_percentage to plot the % of survived/dead passengers per age category"

plot_survived_percentage('age_categories_text', "Age Category", 'Percentage of survived/dead passengers vs. age category', h=8)
# Use the function plot_count_percentage to plot the % of total passengers per fare_category

plot_count_percentage('fare_categories_text', "Fare Categories", 'Percentage of total passengers per fare category', h =6, y_range=35)
# Use the function plot_survived_percentage to plot the % of survived/dead passengers per fare category"

plot_survived_percentage('fare_categories_text', "Fare Category", 'Percentage of survived/dead passengers vs. fare category', h=8)
# Plot the heat map for the correlation matrix calculated using Pearson method

fig, ax = plt.subplots(figsize=(7,7)) 

sns.heatmap(train_ml[['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'age_categories', 'fare_categories']].corr(method='pearson'), 

            annot = True, square=True, fmt='.2g', vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', ax=ax, linewidths=.5, cbar=False)
# Plot fare categories by passenger class vs. survived

g = sns.catplot(x="fare_categories_text", y="Survived", hue="Pclass", kind="point", data=train_ml, height = 6,

                markers=["^", "o", "*"], linestyles=["solid", "dotted", "dashed"])

g.set(title = "Fare categories by Passenger Class vs. Survived"+ "\n")

g._legend.remove()

g.ax.legend().set_title('Passenger Class')

g.set_axis_labels("Fare Categories", "Survival Rate")
# Create a new string column for gender for plotting

train_ml['sex_text'] = train_ml['Sex'] 

train_ml.loc[train_ml['sex_text'] == 0, 'sex_text'] = "male"

train_ml.loc[train_ml['sex_text'] == 1, 'sex_text'] = "female"



# Plot gender by passenger class vs. survived

g = sns.catplot(x="sex_text", y="Survived", hue="Pclass", kind="point", data=train_ml, height = 6,

                markers=["^", "o", "*"], linestyles=["solid", "dotted", "dashed"])

g.set(title = "Gender by Passenger Class vs. Survived"+ "\n")

g._legend.remove()

g.ax.legend().set_title('Passenger Class')

g.set_axis_labels("Gender", "Survival Rate")
# Plot gender by fare categories vs. survived

g = sns.catplot(x="fare_categories_text", y="Survived", hue="sex_text", kind="point", data=train_ml, height = 6,

                markers=["^", "o"], linestyles=["solid", "dotted"])

g.set(title = "Gender by Fare Categories vs. Survived"+ "\n")

g._legend.remove()

g.ax.legend().set_title('Gender')

g.set_axis_labels("Fare Categories", "Survival Rate")
# Create a new string column for gender for plotting

train_ml['embarked_text'] = train_ml['Embarked'] 

train_ml.loc[train_ml['embarked_text'] == 1, 'embarked_text'] = "Cherbourg"

train_ml.loc[train_ml['embarked_text'] == 2, 'embarked_text'] = "Queenstown"

train_ml.loc[train_ml['embarked_text'] == 0, 'embarked_text'] = "Southampton"



# Plot passenger class, gender, embarkation point

g = sns.FacetGrid(train_ml, row='embarked_text', height=4.5, aspect=1.6)

g.map(sns.pointplot, 'Pclass', 'Survived', 'sex_text', palette=None,  order=None, hue_order=None )

g.add_legend()

g.set_axis_labels("Passenger Class", "Survival Rate")

g.set_titles(row_template = 'Embarkation point: {row_name}')
# As we've seen from the correlation matrix there is a relatively high corr between # of siblings/spouces & # of parents/children: 0.41

# We will create a new feature denoted family_members

train_ml['family_members'] = train_ml.SibSp + train_ml.Parch
# Plot Family Members by Gender vs. Survived

g = sns.catplot(x="family_members", y="Survived", hue="sex_text", kind="point", data=train_ml, height = 6,

                markers=["^", "o"], linestyles=["solid", "dotted"])

g.set(title = "Family Members by Gender vs. Survived"+ "\n")

g._legend.remove()

g.ax.legend().set_title('Gender')

g.set_axis_labels("Family Members", "Survival Rate")
# Until now we have only worked with the train dataset

# Lets have a look at a sample of the test dataset

test.head()
# Get the size of the test dataset

test.shape
# Use describe() function to get a summary of the test dataset

test.describe(include = 'all')
# Check for missing values



# Sum all missing values for each column 

total = test.isnull().sum().sort_values(ascending=False)



# Calculate the percentage of missing values for each column

percent = round(test.isnull().sum()/test.isnull().count()*100, 1).sort_values(ascending=False)



# Create a data frame containing the total number of missing values and the % out of the total number of values

missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])



# Check the first 5 rows of the missing_data data frame

missing_data.head(5)
# Create a copy of the original test dataset without Name, Ticket & Cabin

# In this case we keep the PassengerId column as it will be needed in the submission file

test_ml = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)



# Replace missing values for Age with random numbers between mean-std and mean+std

# Calculate mean value for Age

mean = test_ml["Age"].mean()

# Calculate standard deviation for Age

std = test_ml["Age"].std()

# Get the size for the randint() function

is_null = test_ml["Age"].isnull().sum()

# Compute random numbers between mean-std and mean+std

rand_age = np.random.randint(mean - std, mean + std, size = is_null)

# Fill NaN values in Age column with random values generated

age_slice = test_ml["Age"].copy()

age_slice[np.isnan(age_slice)] = rand_age

test_ml["Age"] = age_slice
# Check the passenger class for the 1 missing value of Fare

test_ml[test_ml.isnull().any(axis=1)]



# Replace missing values for Fare with mean value for the Passenger Class 3

test_ml["Fare"] = test_ml["Fare"].fillna(test_ml.groupby('Pclass').mean()['Fare'][3])



# Check that everything worked as expected using info()

test_ml.info()
# Convert Sex column to integer form

test_ml.loc[test_ml["Sex"] == "male", "Sex"] = 0

test_ml.loc[test_ml["Sex"] == "female", "Sex"] = 1

test_ml["Sex"] = test_ml["Sex"].astype(int)



# Convert Embarked column to integer form

test_ml.loc[test_ml["Embarked"] == "S", "Embarked"] = 0

test_ml.loc[test_ml["Embarked"] == "C", "Embarked"] = 1

test_ml.loc[test_ml["Embarked"] == "Q", "Embarked"] = 2

test_ml["Embarked"] = test_ml["Embarked"].astype(int)
# Create the column age_categories '0-10', '11-17','18-22', '23-26', '27-32', '33-39', '40-49','>=50'

test_ml['age_categories'] = pd.cut(test_ml.Age, [0, 11, 18, 23, 27, 33, 40, 50, np.inf], 

                                    labels=[0, 1, 2, 3, 4, 5, 6,7], include_lowest=True, right=False).astype(int)



# Create the column fare_categories '0-7.99', '8-13.99','14-30.99', '31-98.99', '99-249.99', '>=250'

test_ml['fare_categories'] = pd.cut(test_ml.Fare, [0, 8, 14, 31, 99, 250, np.inf], 

                                    labels=[0, 1, 2, 3, 4, 5], include_lowest=True, right=False).astype(int)
# We will create a new feature denoted family_members

test_ml['family_members'] = test_ml.SibSp + test_ml.Parch
# Now the test dataset is ready to use

# Let's select only the relevant columns

test_final = test_ml[['PassengerId', 'Pclass', 'Sex', 'Embarked', 'age_categories', 'fare_categories', 'family_members']]

test_final.head()
# Test dataset features to use in the ML algorithms

X_test = test_final.drop('PassengerId', axis=1)
# Train dataset features to use in the ML algorithms

X_train = train_ml[['Pclass', 'Sex', 'Embarked', 'age_categories', 'fare_categories', 'family_members']]



# Target variable

Y_train = train_ml.Survived
# Apply logistic regression model to the train dataset

log_reg = LogisticRegression()

log_reg.fit(X_train, Y_train)



# Perform K-Fold Cross Validation to get an accuracy score

log_reg_acc = round(cross_val_score(log_reg, X_train, Y_train, cv=10, scoring = "accuracy").mean()*100, 4)



# Predictions

predictions = log_reg.predict(X_test)
# Apply K-nearest Neighbors model to the train dataset

knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)



# Perform K-Fold Cross Validation to get an accuracy score

knn_acc = round(cross_val_score(knn, X_train, Y_train, cv=10, scoring = "accuracy").mean()*100, 4)



# Predictions

predictions = knn.predict(X_test)
# Apply Naïve Bayes model to the train dataset

gaussian_nb = GaussianNB()

gaussian_nb.fit(X_train, Y_train)



# Perform K-Fold Cross Validation to get an accuracy score

gaussian_nb_acc = round(cross_val_score(gaussian_nb, X_train, Y_train, cv=10, scoring = "accuracy").mean()*100, 4)



# Predictions

predictions = gaussian_nb.predict(X_test)
# Apply Decision Tree Classification model to the train dataset

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)



# Perform K-Fold Cross Validation to get an accuracy score

decision_tree_acc = round(cross_val_score(decision_tree, X_train, Y_train, cv=10, scoring = "accuracy").mean()*100, 4)



# Predictions

predictions = decision_tree.predict(X_test)
# Apply Random Forest model to the train dataset

random_forest = RandomForestClassifier()

random_forest.fit(X_train, Y_train)



# Perform K-Fold Cross Validation to get an accuracy score

random_forest_acc = round(cross_val_score(random_forest, X_train, Y_train, cv=10, scoring = "accuracy").mean()*100, 4)



# Predictions

predictions = random_forest.predict(X_test)
# Create a data frame with the name of the model used and the score

model_score = pd.DataFrame({'Model': ['Logistic regression', 'K-nearest Neighbors', 'Gaussian Naïve Bayes', 'Decision Tree', 'Random Forest'],

                            'Score': [log_reg_acc, knn_acc, gaussian_nb_acc, decision_tree_acc, random_forest_acc]})



# Sort by Score

model_score.sort_values(by='Score', ascending=False).reset_index().drop('index', axis=1)
# Create a data frame showing the feature importance as given by the random forest model

feature_importance = pd.DataFrame({'Feature':X_train.columns,'Importance':np.round(random_forest.feature_importances_,3)})

feature_importance.sort_values('Importance',ascending=False).reset_index().drop('index', axis=1)
# Lets have a look at the parameters that the random forest model is using

random_forest.get_params()
# We will use RandomizedSearchCV for hyperparameter tuning so first we need to create a parameter grid



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]



# Number of features to consider at every split

max_features = ['auto', 'sqrt', 'log2']



# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)



# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]



# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]



# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

random_grid
# Random search of parameters using 3 fold cross validation and 100 different combinations

random_forest_random = RandomizedSearchCV(estimator = random_forest, param_distributions = random_grid, n_iter = 100, verbose=2, cv = 3, random_state=42, n_jobs = -1)



# Fit the random search model

random_forest_random.fit(X_train, Y_train)
# Let's have a look at the best parameters from fitting the random search

random_forest_random.best_params_
# Apply Random Forest model to the train dataset using the best parameters

random_forest_2 = random_forest_random.best_estimator_

random_forest_2.fit(X_train, Y_train)



# Perform K-Fold Cross Validation to get an accuracy score

random_forest_acc_2 = round(cross_val_score(random_forest_2, X_train, Y_train, cv=10, scoring = "accuracy").mean()*100, 4)



# Let's compare the 2 Random Forest models

# Add the accuracy score to the model_score data frame

model_score = model_score.append(pd.DataFrame({'Model': ['Random Forest 2'], 'Score': [random_forest_acc_2]})) 



# Sort by Score

model_score.sort_values(by='Score', ascending=False).reset_index().drop('index', axis=1)
# We are gonna use GridSearchCV to evaluate all combinations we define



# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [20, 40, 60, 80, 100],

    'max_features': ['log2'],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [4, 5, 6],

    'n_estimators': [100, 200, 300, 700, 1000]

}



# Define the grid search model

random_forest_grid_search = GridSearchCV(estimator = random_forest, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)



# Fit the grid search model

random_forest_grid_search.fit(X_train, Y_train)
# Let's at the best parameters from teh grid search

random_forest_grid_search.best_params_
# Apply Random Forest model to the train dataset using the best parameters

random_forest_3 = random_forest_grid_search.best_estimator_

random_forest_3.fit(X_train, Y_train)



# Perform K-Fold Cross Validation to get an accuracy score

random_forest_acc_3 = round(cross_val_score(random_forest_3, X_train, Y_train, cv=10, scoring = "accuracy").mean()*100, 4)



# Let's compare the 2 Random Forest models

# Add the accuracy score to the model_score data frame

model_score = model_score.append(pd.DataFrame({'Model': ['Random Forest 3'], 'Score': [random_forest_acc_3]})) 



# Sort by Score

model_score.sort_values(by='Score', ascending=False).reset_index().drop('index', axis=1)
## Use confusion matrix to get a matrix of true negatives & false positives predictions

predictions = cross_val_predict(random_forest_3, X_train, Y_train, cv=3)

confusion_matrix(Y_train, predictions)
# Let's now look at precision and recall

print("Precision:", precision_score(Y_train, predictions))

print("Recall:",recall_score(Y_train, predictions))
# Getting the probabilities of our predictions

y_scores = random_forest_3.predict_proba(X_train)

y_scores = y_scores[:,1]



# Computing the roc_auc_score

r_a_score = roc_auc_score(Y_train, y_scores)

print("ROC-AUC-Score:", r_a_score)
# Use the final random_forest_3 model to make predictions on the test dataset

final_predictions = random_forest_3.predict(X_test)



output = pd.DataFrame({'PassengerId': test_final.PassengerId, 'Survived': final_predictions})

output.to_csv('random_forest_model.csv', index=False)