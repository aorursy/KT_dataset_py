# Numerical example

x = 5

print (x)
# Text example

x = "hello"

print (x)
# int variable

x = 5

print (x)

print (type(x))



# float variable

x = 5.0

print (x)

print (type(x))



# text variable

x = "5" 

print (x)

print (type(x))



# boolean variable

x = True

print (x)

print (type(x))
# int variables

a = 5

b = 3

print (a + b)



# string variables

a = "5"

b = "3"

print (a + b)
# Making a list

list_x = [3, "hello", 1]

print (list_x)



# Adding to a list

list_x.append(7)

print (list_x)



# Accessing items at specific location in a list

print ("list_x[0]: ", list_x[0])

print ("list_x[1]: ", list_x[1])

print ("list_x[2]: ", list_x[2])

print ("list_x[-1]: ", list_x[-1]) # the last item

print ("list_x[-2]: ", list_x[-2]) # the second to last item



# Slicing

print ("list_x[:]: ", list_x[:])

print ("list_x[2:]: ", list_x[2:])

print ("list_x[1:3]: ", list_x[1:3])

print ("list_x[:-1]: ", list_x[:-1])



# Length of a list

len(list_x)



# Replacing items in a list

list_x[1] = "hi"

print (list_x)



# Combining lists

list_y = [2.4, "world"]

list_z = list_x + list_y

print (list_z)
# Creating a tuple

tuple_x = (3.0, "hello")

print (tuple_x)



# Adding values to a tuple

tuple_x = tuple_x + (5.6,)

print (tuple_x)



# Trying to change a tuples value (you can't)

tuple_x[1] = "world"
# Creating a dictionary

arun = {"name": "Arun",

        "eye_color": "brown"}

print (arun)

print (arun["name"])

print (arun["eye_color"])



# Changing the value for a key

arun["eye_color"] = "black"

print (arun)



# Adding new key-value pairs

arun["age"] = 24

print (arun)



# Length of a dictionary

print (len(arun))
# If statement

x = 4

if x < 1:

    score = "low"

elif x <= 4:

    score = "medium"

else:

    score = "high"

print (score)



# If statment with a boolean

x = True

if x:

    print ("it worked")
# For loop

x = 1

for i in range(3): # goes from i=0 to i=2

    x += 1 # same as x = x + 1

    print ("i={0}, x={1}".format(i, x)) # printing with multiple variables
# While loop

x = 3

while x > 0:

    x -= 1 # same as x = x - 1

    print (x)
# Create a function

def add_two(x):

    x += 2

    return x



# Use the function

score = 0

score = add_two(x=score)

print (score)
# Function with multiple inputs

def join_name(first_name, last_name):

    joined_name = first_name + " " + last_name

    return joined_name



# Use the function

first_name = "Arunkumar"

last_name = "Venkataramanan"

joined_name = join_name(first_name=first_name, last_name=last_name)

print (joined_name)
# Create the function

class Pets(object):

  

    # Initialize the class

    def __init__(self, species, color, name):

        self.species = species

        self.color = color

        self.name = name



    # For printing  

    def __str__(self):

        return "{0} {1} named {2}.".format(self.color, self.species, self.name)



    # Example function

    def change_name(self, new_name):

        self.name = new_name
# Making an instance of a class

my_dog = Pets(species="dog", color="orange", name="Guiness",)

print (my_dog)

print (my_dog.name)
# Using a class's function

my_dog.change_name(new_name="Johny Kutty")

print (my_dog)

print (my_dog.name)
import numpy as np
# Set seed for reproducability

np.random.seed(seed=1234)
# Scalars

x = np.array(6) # scalar

print ("x: ", x)

print("x ndim: ", x.ndim)

print("x shape:", x.shape)

print("x size: ", x.size)

print ("x dtype: ", x.dtype)
# 1-D Array

x = np.array([1.3 , 2.2 , 1.7])

print ("x: ", x)

print("x ndim: ", x.ndim)

print("x shape:", x.shape)

print("x size: ", x.size)

print ("x dtype: ", x.dtype) # notice the float datatype
# 3-D array (matrix)

x = np.array([[1,2,3], [4,5,6], [7,8,9]])

print ("x:\n", x)

print("x ndim: ", x.ndim)

print("x shape:", x.shape)

print("x size: ", x.size)

print ("x dtype: ", x.dtype)
# Functions

print ("np.zeros((2,2)):\n", np.zeros((2,2)))

print ("np.ones((2,2)):\n", np.ones((2,2)))

print ("np.eye((2)):\n", np.eye((2)))

print ("np.random.random((2,2)):\n", np.random.random((2,2)))
# Indexing

x = np.array([1, 2, 3])

print ("x[0]: ", x[0])

x[0] = 0

print ("x: ", x)
# Slicing

x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

print (x)

print ("x column 1: ", x[:, 1]) 

print ("x row 0: ", x[0, :]) 

print ("x rows 0,1,2 & cols 1,2: \n", x[:3, 1:3]) 
# Integer array indexing

print (x)

rows_to_get = np.arange(len(x))

print ("rows_to_get: ", rows_to_get)

cols_to_get = np.array([0, 2, 1])

print ("cols_to_get: ", cols_to_get)

print ("indexed values: ", x[rows_to_get, cols_to_get])
# Boolean array indexing

x = np.array([[1,2], [3, 4], [5, 6]])

print ("x:\n", x)

print ("x > 2:\n", x > 2)

print ("x[x > 2]:\n", x[x > 2])
# Basic math

x = np.array([[1,2], [3,4]], dtype=np.float64)

y = np.array([[1,2], [3,4]], dtype=np.float64)

print ("x + y:\n", np.add(x, y)) # or x + y

print ("x - y:\n", np.subtract(x, y)) # or x - y

print ("x * y:\n", np.multiply(x, y)) # or x * y
# Dot product

a = np.array([[1,2,3], [4,5,6]], dtype=np.float64) # we can specify dtype

b = np.array([[7,8], [9,10], [11, 12]], dtype=np.float64)

print (a.dot(b))
# Sum across a dimension

x = np.array([[1,2],[3,4]])

print (x)

print ("sum all: ", np.sum(x)) # adds all elements

print ("sum by col: ", np.sum(x, axis=0)) # add numbers in each column

print ("sum by row: ", np.sum(x, axis=1)) # add numbers in each row
# Transposing

print ("x:\n", x)

print ("x.T:\n", x.T)
# Tile

x = np.array([[1,2], [3,4]])

y = np.array([5, 6])

addent = np.tile(y, (len(x), 1))

print ("addent: \n", addent)

z = x + addent

print ("z:\n", z)
# Broadcasting

x = np.array([[1,2], [3,4]])

y = np.array([5, 6])

z = x + y

print ("z:\n", z)
# Reshaping

x = np.array([[1,2], [3,4], [5,6]])

print (x)

print ("x.shape: ", x.shape)

y = np.reshape(x, (2, 3))

print ("y.shape: ", y.shape)

print ("y: \n", y)
# Removing dimensions

x = np.array([[[1,2,1]],[[2,2,3]]])

print ("x.shape: ", x.shape)

y = np.squeeze(x, 1) # squeeze dim 1

print ("y.shape: ", y.shape) 

print ("y: \n", y)
# Adding dimensions

x = np.array([[1,2,1],[2,2,3]])

print ("x.shape: ", x.shape)

y = np.expand_dims(x, 1) # expand dim 1

print ("y.shape: ", y.shape) 

print ("y: \n", y)
import pandas as pd
# Read from CSV to Pandas DataFrame

df = pd.read_csv("../input/train.csv", header=0)
# First five items

df.head()
# Describe features

df.describe()
# Histograms

df["Age"].hist()
# Unique values

df["Embarked"].unique()
# Selecting data by feature

df["Name"].head()
# Filtering

df[df["Sex"]=="female"].head() # only the female data appear
# Sorting

df.sort_values("Age", ascending=False).head()
# Grouping

sex_group = df.groupby("Survived")

sex_group.mean()
# Selecting row

df.iloc[0, :] # iloc gets rows (or columns) at particular positions in the index (so it only takes integers)
# Selecting specific value

df.iloc[0, 1]
# Selecting by index

df.loc[0] # loc gets rows (or columns) with particular labels from the index
# Rows with at least one NaN value

df[pd.isnull(df).any(axis=1)].head()
# Drop rows with Nan values

df = df.dropna() # removes rows with any NaN values

df = df.reset_index() # reset's row indexes in case any rows were dropped

df.head()
# Dropping multiple rows

df = df.drop(["Name", "Cabin", "Ticket"], axis=1) # we won't use text features for our initial basic models

df.head()
# Map feature values

df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df["Embarked"] = df['Embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)

df.head()
# Lambda expressions to create new features

def get_family_size(sibsp, parch):

    family_size = sibsp + parch

    return family_size



df["Family_Size"] = df[["SibSp", "Parch"]].apply(lambda x: get_family_size(x["SibSp"], x["Parch"]), axis=1)

df.head()
# Reorganize headers

df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Family_Size', 'Fare', 'Embarked', 'Survived']]

df.head()
# Saving dataframe to CSV

df.to_csv("processed_titanic.csv", index=False)
# See your saved file

!ls -l
from argparse import Namespace

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
# Arguments

args = Namespace(

    seed=1234,

    data_file="sample_data.csv",

    num_samples=100,

    train_size=0.75,

    test_size=0.25,

    num_epochs=100,

)



# Set seed for reproducability

np.random.seed(args.seed)
# Generate synthetic data

def generate_data(num_samples):

    X = np.array(range(num_samples))

    y = 3.65*X + 10

    return X, y
# Generate random (linear) data

X, y = generate_data(args.num_samples)

data = np.vstack([X, y]).T

df = pd.DataFrame(data, columns=['X', 'y'])

df.head()
# Scatter plot

plt.title("Generated data")

plt.scatter(x=df["X"], y=df["y"])

plt.show()
# Import packages

from sklearn.linear_model.stochastic_gradient import SGDRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
# Create data splits

X_train, X_test, y_train, y_test = train_test_split(

    df["X"].values.reshape(-1, 1), df["y"], test_size=args.test_size, 

    random_state=args.seed)

print ("X_train:", X_train.shape)

print ("y_train:", y_train.shape)

print ("X_test:", X_test.shape)

print ("y_test:", y_test.shape)
# Standardize the data (mean=0, std=1) using training data

X_scaler = StandardScaler().fit(X_train)

y_scaler = StandardScaler().fit(y_train.values.reshape(-1,1))



# Apply scaler on training and test data

standardized_X_train = X_scaler.transform(X_train)

standardized_y_train = y_scaler.transform(y_train.values.reshape(-1,1)).ravel()

standardized_X_test = X_scaler.transform(X_test)

standardized_y_test = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()





# Check

print ("mean:", np.mean(standardized_X_train, axis=0), 

       np.mean(standardized_y_train, axis=0)) # mean should be ~0

print ("std:", np.std(standardized_X_train, axis=0), 

       np.std(standardized_y_train, axis=0))   # std should be 1
# Initialize the model

lm = SGDRegressor(loss="squared_loss", penalty="none", max_iter=args.num_epochs)
# Train

lm.fit(X=standardized_X_train, y=standardized_y_train)
# Predictions (unstandardize them)

pred_train = (lm.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_

pred_test = (lm.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
import matplotlib.pyplot as plt
# Train and test MSE

train_mse = np.mean((y_train - pred_train) ** 2)

test_mse = np.mean((y_test - pred_test) ** 2)

print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(train_mse, test_mse))
# Figure size

plt.figure(figsize=(15,5))



# Plot train data

plt.subplot(1, 2, 1)

plt.title("Train")

plt.scatter(X_train, y_train, label="y_train")

plt.plot(X_train, pred_train, color="red", linewidth=1, linestyle="-", label="lm")

plt.legend(loc='lower right')



# Plot test data

plt.subplot(1, 2, 2)

plt.title("Test")

plt.scatter(X_test, y_test, label="y_test")

plt.plot(X_test, pred_test, color="red", linewidth=1, linestyle="-", label="lm")

plt.legend(loc='lower right')



# Show plots

plt.show()
# Feed in your own inputs

X_infer = np.array((0, 1, 2), dtype=np.float32)

standardized_X_infer = X_scaler.transform(X_infer.reshape(-1, 1))

pred_infer = (lm.predict(standardized_X_infer) * np.sqrt(y_scaler.var_)) + y_scaler.mean_

print (pred_infer)

df.head(3)
# Unstandardize coefficients 

coef = lm.coef_ * (y_scaler.scale_/X_scaler.scale_)

intercept = lm.intercept_ * y_scaler.scale_ + y_scaler.mean_ - np.sum(coef*X_scaler.mean_)

print (coef) # ~3.65

print (intercept) # ~10
# Initialize the model with L2 regularization

lm = SGDRegressor(loss="squared_loss", penalty='l2', alpha=1e-2, 

                  max_iter=args.num_epochs)
# Train

lm.fit(X=standardized_X_train, y=standardized_y_train)
# Predictions (unstandardize them)

pred_train = (lm.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_

pred_test = (lm.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
# Train and test MSE

train_mse = np.mean((y_train - pred_train) ** 2)

test_mse = np.mean((y_test - pred_test) ** 2)

print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(

    train_mse, test_mse))
# Unstandardize coefficients 

coef = lm.coef_ * (y_scaler.scale_/X_scaler.scale_)

intercept = lm.intercept_ * y_scaler.scale_ + y_scaler.mean_ - (coef*X_scaler.mean_)

print (coef) # ~3.65

print (intercept) # ~10
# Create data with categorical features

cat_data = pd.DataFrame(['a', 'b', 'c', 'a'], columns=['favorite_letter'])

cat_data.head()
dummy_cat_data = pd.get_dummies(cat_data)

dummy_cat_data.head()
from argparse import Namespace

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import urllib
# Arguments

args = Namespace(

    seed=1234,

    data_file="titanic.csv",

    train_size=0.75,

    test_size=0.25,

    num_epochs=100,

)



# Set seed for reproducability

np.random.seed(args.seed)
# Upload data from GitHub to notebook's local drive

url = "https://raw.githubusercontent.com/ArunkumarRamanan/practicalAI/master/data/titanic.csv"

response = urllib.request.urlopen(url)

html = response.read()

with open(args.data_file, 'wb') as f:

    f.write(html)
# Read from CSV to Pandas DataFrame

df = pd.read_csv(args.data_file, header=0)

df.head()
# Import packages

from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
# Preprocessing

def preprocess(df):

  

    # Drop rows with NaN values

    df = df.dropna()



    # Drop text based features (we'll learn how to use them in later lessons)

    features_to_drop = ["name", "cabin", "ticket"]

    df = df.drop(features_to_drop, axis=1)



    # pclass, sex, and embarked are categorical features

    categorical_features = ["pclass","embarked","sex"]

    df = pd.get_dummies(df, columns=categorical_features)



    return df
# Preprocess the dataset

df = preprocess(df)

df.head()
# Split the data

mask = np.random.rand(len(df)) < args.train_size

train_df = df[mask]

test_df = df[~mask]

print ("Train size: {0}, test size: {1}".format(len(train_df), len(test_df)))
# Separate X and y

X_train = train_df.drop(["survived"], axis=1)

y_train = train_df["survived"]

X_test = test_df.drop(["survived"], axis=1)

y_test = test_df["survived"]
# Standardize the data (mean=0, std=1) using training data

X_scaler = StandardScaler().fit(X_train)



# Apply scaler on training and test data (don't standardize outputs for classification)

standardized_X_train = X_scaler.transform(X_train)

standardized_X_test = X_scaler.transform(X_test)



# Check

print ("mean:", np.mean(standardized_X_train, axis=0)) # mean should be ~0

print ("std:", np.std(standardized_X_train, axis=0))   # std should be 1
# Initialize the model

log_reg = SGDClassifier(loss="log", penalty="none", max_iter=args.num_epochs, 

                        random_state=args.seed)
# Train

log_reg.fit(X=standardized_X_train, y=y_train)
# Probabilities

pred_test = log_reg.predict_proba(standardized_X_test)

print (pred_test[:5])
# Predictions (unstandardize them)

pred_train = log_reg.predict(standardized_X_train) 

pred_test = log_reg.predict(standardized_X_test)

print (pred_test)