# Import standard librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix



# Input data files are available in the read-only "../input/" directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def load_data(dir_data_name):

    """

    Loads the data into the corresponding data frames

    

    Arguments:

    dir-data_name -- directory containing the csv files

    

    Returns:

    train  -- a DataFrame containing the training data

    test   -- a DataFrame containing the test data

    gender -- a DataFrame containing the gender_submission data

    """

    

    train = pd.read_csv(os.path.join(dir_data_name, "train" + ".csv"))

    test  = pd.read_csv(os.path.join(dir_data_name, "test" + ".csv"))

    gender = pd.read_csv(os.path.join(dir_data_name, "gender_submission" + ".csv"))

    

    return train.drop(["Survived", "PassengerId"], axis=1), test.drop("PassengerId", axis=1), gender, train['Survived'], test["PassengerId"]
# We load the data



train, test, gender, train_labels, test_PassengerId = load_data("/kaggle/input/titanic")
test
train
train.info()
train.describe()
# This show the histogram on train data of the numerical attributes

%matplotlib inline

train.hist(bins=50, figsize=(15,10))



plt.show()
# Plots the scatter matrix over numerical attributes

attributes = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

scatter_matrix(train[attributes], figsize=(12,8))
def print_percentage_NaN_values(data):

    """

    Prints the percentage of NaN values for each column of the dataset

    

    Arguments:

    data -- the DataFrame to analyze (here either train or test)

    

    Returns:

    No return value

    """

    

    for key, item in data.iteritems():

        print(key, " : ", round(sum(pd.isnull(item))/len(item)*100, 2), "% are NaN")
print_percentage_NaN_values(train)
# Create pclass_pipeline

# Classify NaN as 0 for numerical value + OneHotEncoder



pclass_pipeline = Pipeline([

    ('handle_NaN', SimpleImputer(fill_value=0, strategy="constant")),

    ('one_hot', OneHotEncoder(handle_unknown = 'ignore')),

])

# Test pclass_pipeline with the 'Pclass' column of the DataFrame

train_Pclass = train[["Pclass"]]

essai = pclass_pipeline.fit_transform(train_Pclass)

essai.toarray()
# Create category_pipeline

# Classify NaN as "'Unknown' + OneHotEncoder



category_pipeline = Pipeline([

    ('handle_NaN', SimpleImputer(fill_value='Unknown', strategy="constant")),

    ('one_hot', OneHotEncoder(handle_unknown = 'ignore')),

])

# Test category_pipeline with the 'Sex' and 'Embarked' columns of the DataFrame



train_Sex_Embarked= train[['Sex', 'Embarked']]

essai = category_pipeline.fit_transform(train_Sex_Embarked)

essai.toarray()
train
# create category_with_ceiling_pipeline

# Categorize 'everything > 1' in the same category + OneHotEncoder: will be applied to SibSp, Parch



# replace by 2 when element is bigger than 1

def ceil_above_one(X):

    return np.where(X > 1, 2,X)



category_with_ceiling_pipeline = Pipeline([

    ('ceiling', FunctionTransformer(ceil_above_one)),

    ('one_hot', OneHotEncoder(handle_unknown = 'ignore')),

])
# Test category_with_ceiling_pipeline with the 'SibSp' and 'Parch' columns of the DataFrame



train_SibSp_Parch= train[['SibSp', 'Parch']]

essai = category_with_ceiling_pipeline.fit_transform(train_SibSp_Parch)

essai.toarray()
# Transform function categorize_name

# We replace every name containing man attribute with Mr., Don.,... 

# every name containing Mrs. attributes with Mrs.

# and every name containing Miss. attributes with Miss.

# The string str_to_replace contains the regex permitting to select the name with the correct attribute

# The string str_value is the replacement string, which replaces the whole name

def categorize_name(X):

    str_to_replace = []

    str_value = []

    str_to_replace.append('.*Mr\..*'); str_value.append('Mr.')

    str_to_replace.append('.*Sir\..*'); str_value.append('Mr.')

    str_to_replace.append('.*Master\..*'); str_value.append('Master.')

    str_to_replace.append('.*Rev\..*'); str_value.append('Mr.')

    str_to_replace.append('.*Dr\..*'); str_value.append('Mr.')

    str_to_replace.append('.*Don\..*'); str_value.append('Mr.')

    str_to_replace.append('.*Major\..*'); str_value.append('Mr.')

    str_to_replace.append('.*Col\..*'); str_value.append('Mr.')

    str_to_replace.append('.*Capt\..*'); str_value.append('Mr.')

    str_to_replace.append('.*Mrs\..*'); str_value.append('Mrs.')

    str_to_replace.append('.*Ms\..*'); str_value.append('Mrs.')

    str_to_replace.append('.*Lady\..*'); str_value.append('Mrs.')

    str_to_replace.append('.*Mme\..*'); str_value.append('Mrs.')

    str_to_replace.append('.*Countess\..*'); str_value.append('Mrs.')

    str_to_replace.append('.*Miss\..*'); str_value.append('Miss.')

    str_to_replace.append('.*Mlle\..*'); str_value.append('Miss.')

    str_to_replace.append('.*\,.*'); str_value.append('Unknown')



    X = X.replace(to_replace=str_to_replace, value=str_value, regex=True)

    print(X.value_counts())

    

    return X
# Create name_pipeline

# Classify according to Mrs, Miss,... and classify remaining as Unknown + OneHotEncoder: Name



name_pipeline = Pipeline([

    ('name category', FunctionTransformer(categorize_name)),

    ('one_hot', OneHotEncoder(handle_unknown = 'ignore')),

])
# Test name_pipeline with the 'Name' columns of the DataFrame



train_Name= train[['Name']]

essai = name_pipeline.fit_transform(train_Name)

essai.toarray()
# Create age_pipeline

# classify NaN as median value + normalize between 0 and 1: applied to Age

# we will have to watch and pay attention with test data



age_pipeline = Pipeline([

    ('handle_NaN', SimpleImputer(strategy="median")),

    ('min_max_scaler', MinMaxScaler()),

])
# Test age_pipeline with the 'Age' columns of the DataFrame



train_Age = train[['Age']]

essai = age_pipeline.fit_transform(train_Age)

essai[0]
# categorize according to letters and to numbers on the ticket

# Element is replaced with With_Letter if it starts with a letter

# Element is replaced by Big_Number if there are from 6 to 8 digits

# Element is replaced with Small_Number if there are less than 6 digits

def categorize_with_letters(X):

    str_to_replace = []

    str_value = []

    str_to_replace.append('^[A-Z].*'); str_value.append('With_Letter')

    str_to_replace.append('^[0-9]{6,8}.*'); str_value.append('Big_Number')

    str_to_replace.append('^[0-9]{1,5}.*'); str_value.append('Small_Number')



    X = X.replace(to_replace=str_to_replace, value=str_value, regex=True)

    print(X.value_counts())

    return X
# Create ticket_pipeline

# Categorize according to presence or not of letters + OneHotEncoder: Ticket



ticket_pipeline = Pipeline([

    ('categorize_with_letters', FunctionTransformer(categorize_with_letters)),

    ('one_hot', OneHotEncoder(handle_unknown = 'ignore')),

])
# Test ticket_pipeline with the 'Ticket' columns of the DataFrame



train_Ticket= train[['Ticket']]

essai = ticket_pipeline.fit_transform(train_Ticket)

essai.shape
# Create fare_pipeline

# Set data above 150 to 150 + normalize between 0 and 1: Fare



# replace by 150 when element is bigger than 150

def ceil_above_150(X):

    return np.where(X > 150., 150.,X)



fare_pipeline = Pipeline([

    ('ceil_above_150', FunctionTransformer(ceil_above_150)),

    ('handle_NaN', SimpleImputer(strategy='median')),

    ('min_max_scaler', MinMaxScaler()),

])
# Test fare_pipeline with the 'Fare' columns of the DataFrame



train_Fare= train[['Fare']]

essai = fare_pipeline.fit_transform(train_Fare)

essai.shape
# Create cabin_pipeline

# Categorize according to cabin letter and classify NaN as NoCabin + OneHotEncoder: Cabin



# replace cabin number with cabin letter

def classify_according_to_cabin_letter(X):

    str_to_replace = []

    str_value = []

    str_to_replace.append('^A.*'); str_value.append('A')

    str_to_replace.append('^B.*'); str_value.append('B')

    str_to_replace.append('^C.*'); str_value.append('C')

    str_to_replace.append('^D.*'); str_value.append('D')

    str_to_replace.append('^E.*'); str_value.append('E')

    str_to_replace.append('^[F-Z].*'); str_value.append('F')



    X = X.replace(to_replace=str_to_replace, value=str_value, regex=True)

    print(X.value_counts())

    return X



cabin_pipeline = Pipeline([

    ('classify_according_to_cabin_letter', FunctionTransformer(classify_according_to_cabin_letter)),

    ('handle_NaN', SimpleImputer(strategy="constant", fill_value='NoCabin')),

    ('one_hot', OneHotEncoder(handle_unknown = 'ignore')),

])
# Test cabin_pipeline with the 'Cabin' columns of the DataFrame



train_Cabin= train[['Cabin']]

essai = cabin_pipeline.fit_transform(train_Cabin)

essai.shape
full_pipeline = ColumnTransformer([

        ("Pclass", pclass_pipeline, ["Pclass"]),

        ("Category", category_pipeline, ["Sex", "Embarked"]),

        ("With Ceiling", category_with_ceiling_pipeline, ["SibSp", "Parch"]),

        ("Name", name_pipeline, ["Name"]),

        ("Age", age_pipeline, ["Age"]),

        ("Ticket", ticket_pipeline, ["Ticket"]),

        ("Fare", fare_pipeline, ["Fare"]),

        ("Cabin", cabin_pipeline, ["Cabin"]),

    ])
train_prepared = full_pipeline.fit_transform(train)
Pclass_cat = full_pipeline.named_transformers_["Pclass"]

print(list(Pclass_cat))
print(type(train_prepared))
# We first use the model using a Random Forest Regressor with reasonable values

forest_reg = RandomForestClassifier(n_estimators=100, max_depth=5, max_features='log2', random_state=1)

forest_reg.fit(train_prepared, train_labels)

train_predictions = forest_reg.predict(train_prepared)

forest_mse = accuracy_score(train_labels, train_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
# We make a cross-validation by subduvudung the training sets in 10 folds and testing on eahc of the fold after having trained the model over the 9 other folds

# This permits to see if the previuos training overfits the training set

forest_scores = cross_val_score(forest_reg, train_prepared, train_labels, scoring="accuracy", cv=10)

print(forest_scores.mean())
test_prepared = full_pipeline.transform(test)
test.info()
test_predictions = forest_reg.predict(test_prepared)
test_predictions
output = pd.DataFrame({'PassengerId': test_PassengerId, 'Survived': test_predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")