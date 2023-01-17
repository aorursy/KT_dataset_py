# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv", index_col = "customerID")
print(df.columns)
df.describe(include = "all")
df.head()
randome_data = df.sample(500)
sns.scatterplot(x=randome_data['MonthlyCharges'], y=randome_data['TotalCharges'], hue=randome_data['Churn'])
sns.swarmplot(x=randome_data['Churn'],
              y=randome_data['MonthlyCharges'])
sns.swarmplot(x=randome_data['Churn'],
              y=randome_data['tenure'])
## Generate seperate columns for categoriacal values
dummy_fields=['gender', 'Partner', 'Dependents', 'PhoneService', "MultipleLines", "InternetService", 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
for each in dummy_fields:
    dummies= pd.get_dummies(df[each], prefix= each, drop_first=False)
    df = pd.concat([df, dummies], axis=1)   

## Drop unwanted columns
fields_to_drop=['gender', 'Partner', 'Dependents', 'PhoneService', "MultipleLines", "InternetService", 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
df=df.drop(fields_to_drop,axis=1)

## Show new columns
print(df.columns )
print(df.isnull().sum().sum())
## Check document type
print(df[['SeniorCitizen', 'tenure', 'MonthlyCharges',
       'TotalCharges', 'Churn']].dtypes)

df[['SeniorCitizen', 'tenure', 'MonthlyCharges',
       'TotalCharges', 'Churn']].head()
#Convert total charges to type float 
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], downcast="float", errors='coerce')

#Check type
print(df[['SeniorCitizen', 'tenure', 'MonthlyCharges',
       'TotalCharges', 'Churn']].dtypes)
df.isnull().sum().sum()
#drop 11 rows with Nan valuse
df = df.dropna()
df.isnull().sum().sum()
max_tenure = df['tenure'].max()
max_monthly_charges = df['MonthlyCharges'].max()
max_total_charges = df['TotalCharges'].max()

# Scaling the columns
df["tenure"] = df["tenure"]/max_tenure
df["MonthlyCharges"] = df["MonthlyCharges"]/max_monthly_charges
df['TotalCharges'] = df['TotalCharges']/max_total_charges

df.head()
## Convert Churn values from Yes, No to 1 and 0
df.Churn = pd.Series(map(lambda x: dict(Yes=1, No=0)[x],
              df.Churn.values.tolist()), df.index)
df.head()
## Divide dataframe into test and train data
msk = np.random.rand(len(df)) < 0.8
train_data = df[msk]
test_data = df[~msk]

features = train_data.drop('Churn', axis=1)
targets = train_data['Churn']
features_test = test_data.drop('Churn', axis=1)
targets_test = test_data['Churn']

print(targets.head())
features.head()
# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

def error_term_formula(x, y, output):
    return (y - output)*sigmoid_prime(x)
# Neural Network hyperparameters
epochs = 1000
learnrate = 0.1

# Training function
def train_nn(features, targets, epochs, learnrate):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable 
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            error = error_formula(y, output)

            # The error term
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term * x

        # Update the weights here. The learning rate times the 
        # change in weights, divided by the number of records to average
        weights += learnrate * del_w / n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
    
weights = train_nn(features, targets, epochs, learnrate)
# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))