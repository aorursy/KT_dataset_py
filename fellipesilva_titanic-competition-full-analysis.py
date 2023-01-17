# --------------------------------------------------------------------------------------------------

# All the code

# --------------------------------------------------------------------------------------------------

# Importing Inicial Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

#print(os.listdir("../input"))



import warnings  

warnings.filterwarnings('ignore')



# --------------------------------------------------------------------------------------------------

# First Explorations



# Reading training data to X_full

X_full = pd.read_csv('../input/train.csv', index_col='PassengerId')



# Reading test data to X_test_full

X_test_full = pd.read_csv('../input/test.csv', index_col='PassengerId')



#X_full.shape

# There is 891 passengers(rows) in this dataset and 11 features(columns)



# Looking at some training data

#X_full.head()



# --------------------------------------------------------------------------------------------------

# Understanding each feature(Column):

# i-> PassengerId: Sequential Credential for each passenger.

# t-> Survived: 0=deceased -- 1=survived (THIS IS OUR TARGET, so I will remove from the X_full avoiding Data Leakage).

# f-> Pclass: Passenger Class. Important feature to train(fit) the model.

# e-> Name: Name of the passengers. Not important to train, neither to predict. (ELIMINATE)

# f-> Sex: Sex of each passsenger. Important feature for survival in Titanic.

# f-> Age: Age of each passenger. Important feature for survival in Titanic.

# e-> SibSp: Number of siblings / spouses aboard the Titanic.

# e-> Parch: Number of parents / children aboard the Titanic.

# e-> Ticket: Ticket Number.

# f-> Fare: Passenger fare (tarifa).

# e-> Cabin: Cabin number.

# e-> Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)



# Legend: f = feature chosen

#         i = index

#         t = target

#         e = eliminated

# --------------------------------------------------------------------------------------------------



# Target

X_full.dropna(axis=0, subset=['Survived'])

y = X_full.Survived



# Features

features = ['Pclass', 'Sex', 'Age', 'Fare']



X = X_full[features]

X_test = X_test_full[features]



from sklearn.model_selection import train_test_split

# Separating training and validation data from X and y

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)



# Looking at some data after selecting the 'features'

#X.head(2)

#X_train.loc[141]



# --------------------------------------------------------------------------------------------------

# Dealing with Categorical Values using Label Encoding



# It's important to do this unique() function on both(X_train and X_valid) because Label Encoding won't work if there is different labels on both

#print(X_train['Sex'].unique())

#print(X_valid['Sex'].unique())

# They have the same unique values, no need to separete good_labels from bad_labels in object_cols



from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

#good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]

#bad_label_cols = list(set(object_cols) - set(good_label_cols))



label_X_train = X_train

label_X_valid = X_valid

label_X_test = X_test



for col in set(object_cols):

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])

    label_X_test[col] = label_encoder.transform(X_test[col])



# Looking at some data after Label Encoding

#label_X_train.head(5)

#X.loc[141]

#X_train.head(5)

#label_X_train.loc[141]



# --------------------------------------------------------------------------------------------------

# Dealing with missing values(NaN) in feature 'Age' using Imputer



# Looking at how much and which column has missing values

#mis_val_count_by_col = X_train.isnull().sum()

#print(mis_val_count_by_col[mis_val_count_by_col > 0])

# Only Age has NaN values, missing values.

# 141 of 891

# I'll use imputation to fill these values



from sklearn.impute import SimpleImputer



# Creating object

my_imputer = SimpleImputer(strategy='most_frequent')



# Imputing values in 'Age' and fitting

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(label_X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(label_X_valid))

imputed_X_test = pd.DataFrame(my_imputer.transform(label_X_test))



# Imputer removed columns and index, here they come back

imputed_X_train.columns = label_X_train.columns

imputed_X_valid.columns = label_X_valid.columns

imputed_X_test.columns = label_X_test.columns

imputed_X_train.index = label_X_train.index

imputed_X_valid.index = label_X_valid.index

imputed_X_test.index = label_X_test.index



# Looking at some data after Imputer

#imputed_X_train.head(5)

#label_X_train.head(5)



# --------------------------------------------------------------------------------------------------

# Normalizing names to apply Machine Learning

X_train_final = imputed_X_train

X_valid_final = imputed_X_valid

X_test_final = imputed_X_test



# Looking at some data after Normalization

#X_train_final.head(5)



# --------------------------------------------------------------------------------------------------

# Machine Learning Model 1 - XGBoost

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



# Define the model (com parâmetros -> LER A LISTA DE PARÂMETROS NAS MINHAS ANOTAÇÕES!!!)

my_model_1 = XGBRegressor(n_estimators=1000, learning_rate=0.03, n_jobs=4, random_state=0)



# Fit the model

my_model_1.fit(X_train_final, y_train)

#my_model_1.fit(X_train_final, y_train, early_stopping_rounds=15, eval_set=[(X_valid_final, y_valid)], verbose=False)



# Get predictions

predictions_1 = my_model_1.predict(X_valid_final)



# Calculate MAE

mae_1 = mean_absolute_error(y_valid, predictions_1)



# Print MAE

print("Mean Absolute Error - Model 1: ", mae_1)



# --------------------------------------------------------------------------------------------------

# Machine Learning Model 2 - Random Forest

from sklearn.ensemble import RandomForestRegressor



# Define the model

my_model_2 = RandomForestRegressor(random_state=0)



# Fit the model

my_model_2.fit(X_train_final, y_train)



# Get Preditions

predictions_2 = my_model_2.predict(X_valid_final)



# Calculate MAE

mae_2 = mean_absolute_error(y_valid, predictions_2)



# Print MAE

print('Mean Absolute Error - Model 2: ', mae_2)



# --------------------------------------------------------------------------------------------------

# Machine Learning Model 3 - Decision Tree

from sklearn.tree import DecisionTreeRegressor



# Function to get ideal max_leaf_nodes so it would not underfitt or overfitt

def ideal_leaf_nodes(train_X, val_X, train_y, val_y):

    # Creating various nodes from 5 to 85

    possible_leaf_nodes = []

    for i in range(5,90,5):

        possible_leaf_nodes.append(i)

    

    # Fit, Predict, MAE

    mae_choice = 1

    lf_choice = 1

    for lf in possible_leaf_nodes:

        model = DecisionTreeRegressor(max_leaf_nodes=lf, random_state=0)

        model.fit(train_X, train_y)

        pred = model.predict(val_X)

        mae = mean_absolute_error(val_y, pred)

        if mae < mae_choice:

            mae_choice = mae

            lf_choice = lf

                

    #print("Mae: %.6f \t lf: %d" %(mae_choice, lf_choice))

    return(lf_choice)

# Function End



# Calling ideal_leaf_nodes to obtain the best tree size

max_leaf_nodes_choice = ideal_leaf_nodes(X_train_final, X_valid_final, y_train, y_valid)



# Define model

my_model_3 = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes_choice, random_state=0)



# Fit the model

my_model_3.fit(X_train_final, y_train)



# Get Predictions

predictions_3 = my_model_3.predict(X_valid_final)



# Calculate MAE

mae_3 = mean_absolute_error(y_valid, predictions_3)



# Print MAE

print('Mean Absolute Error - Model 3: ', mae_3)



# --------------------------------------------------------------------------------------------------

# Generate and Save test predictions to file

# WROOOONG output

#pred_test = my_model_3.predict(X_test_final)



#output = pd.DataFrame({'PassengerId': X_test_final.index,

#                       'Survived': pred_test})

#output.to_csv('submission.csv', index=False)
# After all this, I realized I made a Huge Mistake!!!

# I looked at submission.csv and all my predictions are in DECIMALS! =(

# What is wrong?



# It's a Classification Problem, I mean, my final result must be 0 or 1. The XGBRegressor 

# is not the right Machine Learning Model for this problem, the ideal is the XGBClassifier, 

# so my output will be 0 or 1.



# Other important point is the verification, Mean Absolute Error will compare my decimals,

# it need to be accuracy in this case, so the 'accuracy_score' will perform perfectly.



# With all said, let's get it done!
# --------------------------------------------------------------------------------------------------

# All the CORRECT code

# --------------------------------------------------------------------------------------------------

# Importing Inicial Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

#print(os.listdir("../input"))



import warnings  

warnings.filterwarnings('ignore')



# --------------------------------------------------------------------------------------------------

# First Explorations



# Reading training data to X_full

X_full = pd.read_csv('../input/train.csv', index_col='PassengerId')



# Reading test data to X_test_full

X_test_full = pd.read_csv('../input/test.csv', index_col='PassengerId')



#X_full.shape

# There is 891 passengers(rows) in this dataset and 11 features(columns)



# Looking at some training data

#X_full.head()



# --------------------------------------------------------------------------------------------------

# Understanding each feature(Column):

# i-> PassengerId: Sequential Credential for each passenger.

# t-> Survived: 0=deceased -- 1=survived (THIS IS OUR TARGET, so I will remove from the X_full avoiding Data Leakage).

# f-> Pclass: Passenger Class. Important feature to train(fit) the model.

# e-> Name: Name of the passengers. Not important to train, neither to predict. (ELIMINATE)

# f-> Sex: Sex of each passsenger. Important feature for survival in Titanic.

# f-> Age: Age of each passenger. Important feature for survival in Titanic.

# e-> SibSp: Number of siblings / spouses aboard the Titanic.

# e-> Parch: Number of parents / children aboard the Titanic.

# e-> Ticket: Ticket Number.

# f-> Fare: Passenger fare (tarifa).

# e-> Cabin: Cabin number.

# e-> Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)



# Legend: f = feature chosen

#         i = index

#         t = target

#         e = eliminated

# --------------------------------------------------------------------------------------------------



# Target

X_full.dropna(axis=0, subset=['Survived'])

y = X_full.Survived



# Features

features = ['Pclass', 'Sex', 'Age', 'Fare']



X = X_full[features]

X_test = X_test_full[features]



from sklearn.model_selection import train_test_split

# Separating training and validation data from X and y

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)



# Looking at some data after selecting the 'features'

#X.head(2)

#X_train.loc[141]



# --------------------------------------------------------------------------------------------------

# Dealing with Categorical Values using Label Encoding



# It's important to do this unique() function on both(X_train and X_valid) because Label Encoding won't work if there is different labels on both

#print(X_train['Sex'].unique())

#print(X_valid['Sex'].unique())

# They have the same unique values, no need to separete good_labels from bad_labels in object_cols



from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

#good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]

#bad_label_cols = list(set(object_cols) - set(good_label_cols))



label_X_train = X_train

label_X_valid = X_valid

label_X_test = X_test



for col in set(object_cols):

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])

    label_X_test[col] = label_encoder.transform(X_test[col])



# Looking at some data after Label Encoding

#label_X_train.head(5)

#X.loc[141]

#X_train.head(5)

#label_X_train.loc[141]



# --------------------------------------------------------------------------------------------------

# Dealing with missing values(NaN) in feature 'Age' using Imputer



# Looking at how much and which column has missing values

#mis_val_count_by_col = X_train.isnull().sum()

#print(mis_val_count_by_col[mis_val_count_by_col > 0])

# Only Age has NaN values, missing values.

# 141 of 891

# I'll use imputation to fill these values



from sklearn.impute import SimpleImputer



# Creating object

my_imputer = SimpleImputer(strategy='most_frequent')



# Imputing values in 'Age' and fitting

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(label_X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(label_X_valid))

imputed_X_test = pd.DataFrame(my_imputer.transform(label_X_test))



# Imputer removed columns and index, here they come back

imputed_X_train.columns = label_X_train.columns

imputed_X_valid.columns = label_X_valid.columns

imputed_X_test.columns = label_X_test.columns

imputed_X_train.index = label_X_train.index

imputed_X_valid.index = label_X_valid.index

imputed_X_test.index = label_X_test.index



# Looking at some data after Imputer

#imputed_X_train.head(5)

#label_X_train.head(5)



# --------------------------------------------------------------------------------------------------

# Normalizing names to apply Machine Learning

X_train_final = imputed_X_train

X_valid_final = imputed_X_valid

X_test_final = imputed_X_test



# Looking at some data after Normalization

#X_train_final.head(5)



# --------------------------------------------------------------------------------------------------

# Machine Learning Model 1 - XGBClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score



# Define the model (com parâmetros -> LER A LISTA DE PARÂMETROS NAS MINHAS ANOTAÇÕES!!!)

my_model_1 = XGBClassifier(n_estimators=100, learning_rate=0.03, n_jobs=4, random_state=0)



# Fit the model

my_model_1.fit(X_train_final, y_train)

#my_model_1.fit(X_train_final, y_train, early_stopping_rounds=15, eval_set=[(X_valid_final, y_valid)], verbose=False)



# Get predictions

predictions_1 = my_model_1.predict(X_valid_final)



# Calculate Accuracy

acc_1 = accuracy_score(y_valid, predictions_1)



# Print Accuracy

print("Accuracy - Model 1: ", acc_1)



# --------------------------------------------------------------------------------------------------

# Machine Learning Model 2 - Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier



# Function to automatic get ideal max_leaf_nodes so it would not underfitt or overfitt

def ideal_leaf_nodes(train_X, val_X, train_y, val_y):

    # Creating various nodes from 5 to 200

    possible_leaf_nodes = []

    for i in range(10,205,10):

        possible_leaf_nodes.append(i)

    

    # Fit, Predict, ACC

    acc_choice = 0

    n_choice = 0

    for n in possible_leaf_nodes:

        model = RandomForestClassifier(max_leaf_nodes=n, random_state=0)

        model.fit(train_X, train_y)

        pred = model.predict(val_X)

        acc = accuracy_score(val_y, pred)

        if acc > acc_choice:

            acc_choice = acc

            n_choice = n

        #print("ACC: %.6f \t n: %d" %(acc, n))

    #print("ACC: %.6f \t n: %d" %(acc_choice, n_choice))

    return(n_choice)

# Function End



# Calling ideal_leaf_nodes to obtain the best tree size

max_leaf_nodes_choice = ideal_leaf_nodes(X_train_final, X_valid_final, y_train, y_valid)



# Define the model

my_model_2 = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes_choice, random_state=0)



# Fit the model

my_model_2.fit(X_train_final, y_train)



# Get Preditions

predictions_2 = my_model_2.predict(X_valid_final)



# Calculate Accuracy

acc_2 = accuracy_score(y_valid, predictions_2)



# Print Accuracy

print("Accuracy - Model 2: ", acc_2)



# --------------------------------------------------------------------------------------------------

# Machine Learning Model 3 - Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier



# Function to automatic get ideal max_leaf_nodes so it would not underfitt or overfitt

def ideal_leaf_nodes(train_X, val_X, train_y, val_y):

    # Creating various nodes from 5 to 200

    possible_leaf_nodes = []

    for i in range(5,205,5):

        possible_leaf_nodes.append(i)

    

    # Fit, Predict, ACC

    acc_choice = 0

    n_choice = 0

    for n in possible_leaf_nodes:

        model = DecisionTreeClassifier(max_leaf_nodes=n, random_state=0)

        model.fit(train_X, train_y)

        pred = model.predict(val_X)

        acc = accuracy_score(val_y, pred)

        if acc > acc_choice:

            acc_choice = acc

            n_choice = n

                

    #print("ACC: %.6f \t n: %d" %(acc_choice, n_choice))

    return(n_choice)

# Function End



# Calling ideal_leaf_nodes to obtain the best tree size

max_leaf_nodes_choice = ideal_leaf_nodes(X_train_final, X_valid_final, y_train, y_valid)



# Define model

my_model_3 = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes_choice, random_state=0)



# Fit the model

my_model_3.fit(X_train_final, y_train)



# Get Predictions

predictions_3 = my_model_3.predict(X_valid_final)



# Calculate Accuracy

acc_3 = accuracy_score(y_valid, predictions_3)



# Print Accuracy

print("Accuracy - Model 3: ", acc_3)



# --------------------------------------------------------------------------------------------------

# Machine Learning Model 4 - LogisticRegression

from sklearn.linear_model import LogisticRegression



# Define model

my_model_4 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')



# Fit the model

my_model_4.fit(X_train_final, y_train)



# Get Predictions

predictions_4 = my_model_4.predict(X_valid_final)



# Calculate Accuracy

acc_4 = accuracy_score(y_valid, predictions_4)



# Print Accuracy

print("Accuracy - Model 4: ", acc_4)



# --------------------------------------------------------------------------------------------------

# Generate and Save test predictions to file



pred_test = my_model_1.predict(X_test_final)



output = pd.DataFrame({'PassengerId': X_test_final.index,

                       'Survived': pred_test})

output.to_csv('submission.csv', index=False)