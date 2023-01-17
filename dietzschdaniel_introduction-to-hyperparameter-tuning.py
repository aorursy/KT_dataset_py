# Data Processing

import numpy as np 

import pandas as pd 





# Data Visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





# Modeling

from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier





# Hyperparameter Tuning

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV
# Loading the data

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")





# Handling NaN values

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())



df_train['Cabin'] = df_train['Cabin'].fillna("Missing")

df_test['Cabin'] = df_test['Cabin'].fillna("Missing")



df_train = df_train.dropna()



df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())



# Cleaning the data

df_train = df_train.drop(columns=['Name'], axis=1)

df_test = df_test.drop(columns=['Name'], axis=1)



sex_mapping = {

    'male': 0,

    'female': 1

}

df_train.loc[:, "Sex"] = df_train['Sex'].map(sex_mapping)

df_test.loc[:, "Sex"] = df_test['Sex'].map(sex_mapping)



df_train = df_train.drop(columns=['Ticket'], axis=1)

df_test = df_test.drop(columns=['Ticket'], axis=1)



df_train = df_train.drop(columns=['Cabin'], axis=1)

df_test = df_test.drop(columns=['Cabin'], axis=1)



df_train = pd.get_dummies(df_train, prefix_sep="__",

                              columns=['Embarked'])

df_test = pd.get_dummies(df_test, prefix_sep="__",

                              columns=['Embarked'])
df_train.head()
df_test.head()
# Everything except the target variable

X = df_train.drop("Survived", axis=1)



# Target variable

y = df_train['Survived'].values
# Random seed for reproducibility

np.random.seed(42)



# Splitting the data into train & test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Setting up KNeighborsClassifier()

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn.score(X_test, y_test)
# Setting up RandomForestClassifier()

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

rfc.score(X_test, y_test)
# List of train scores

train_scores = []



# List of test scores

test_scores = []



# List of different values for n_neighbors

neighbors = range(1, 51) # 1 to 50



# Setting up the classifier

knn = KNeighborsClassifier()



# Loop through different neighbors values

for i in neighbors:

    knn.set_params(n_neighbors = i) # set neighbors value

    

    # Fitting the algorithm

    knn.fit(X_train, y_train)

    

    # Append the training scores

    train_scores.append(knn.score(X_train, y_train))

    

    # Append the test scores

    test_scores.append(knn.score(X_test, y_test))
# Plotting the Train and Test scores

plt.figure(figsize=(20,10))

plt.plot(neighbors, train_scores, label="Train score")

plt.plot(neighbors, test_scores, label="Test score")

plt.xticks(np.arange(1, 51, 1))

plt.xlabel("Number of neighbors")

plt.ylabel("Model score")

plt.legend()





print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")

print(f"Number of Neighbours with Maximum KNN: {test_scores.index(max(test_scores)) + 1}")
# Setting up dictionary with RandomForestClassifier hyperparameters

rfc_rs_grid = {"n_estimators": np.arange(10, 1000, 50),

               "max_depth": [None, 3, 5, 10],

               "min_samples_split": np.arange(2, 20, 2),

               "min_samples_leaf": np.arange(1, 20, 2)}
# Import RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV



# Setting random seed

np.random.seed(42)



# Setting random hyperparameter search for RandomForestClassifier

rs_rfc = RandomizedSearchCV(RandomForestClassifier(),

                           param_distributions=rfc_rs_grid,

                           cv=5,

                           n_iter=20,

                           verbose=True)



# Fitting random hyperparameter search model

rs_rfc.fit(X_train, y_train);
# Finding the best parameters

rs_rfc.best_params_
# Evaluate the model

rs_rfc.score(X_test, y_test)
# Setting up dictionary with RandomForestClassifier hyperparameters

rfc_gs_grid = {"n_estimators": np.arange(10, 1010, 100),

               "max_depth": [None, 5, 10]}
# Import GridSearchCV

from sklearn.model_selection import GridSearchCV





# Setting grid hyperparameter search for RandomForestClassifier

gs_rfc = GridSearchCV(RandomForestClassifier(),

                          param_grid=rfc_gs_grid,

                          cv=5,

                          verbose=True)



# Fitting grid hyperparameter search model

gs_rfc.fit(X_train, y_train);
# Finding the best parameters

gs_rfc.best_params_
# Evaluate the model

gs_rfc.score(X_test, y_test)