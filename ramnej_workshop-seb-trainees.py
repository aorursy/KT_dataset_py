print("Hello, World!")
import math

math.sqrt(27)
# importing the libraries we need

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



#########################################################

## JUST COPY-PASTE THE FOLLOWING

# Set some viewing options for convience

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)

# set a reasonable size for the figures we create

plt.rcParams['figure.figsize'] = [15, 8]

# setting the seed so that we always have the same sample

np.random.seed(0)

##########################################################
path = "../input/folkhalsomyndigheten/Folkhalsomyndigheten_Covid19.xlsx" # change to where you have put the file

covid = pd.read_excel(path, sheet_name='Antal per dag region')

covid.head(5)
covid['Örebro']
interesting_columns = ['Halland', 'Stockholm']

covid[interesting_columns]

# we could also write covid[['Halland', 'Stockholm']] straight away
ing_columns = ['Halland', 'Stockholm']

covid[interesting_columns].sum()
# we use the covid = covid... notation here to "save" our change

covid = covid.set_index('Statistikdatum')
sns.lineplot(data=covid['Stockholm'])
# this will not run in the current state

import requests

headers = {'Ocp-Apim-Subscription-Key': 'MY_API_TOKEN'}

r = requests.get('https://api-extern.systembolaget.se/product/v1/product', headers=headers)

pd.read_json(r.text).to_excel('viner.xlsx')
# Load the data

filepath = "../input/creditcardfraud/creditcard.csv"

df = pd.read_csv(filepath) # Change to the right filepath for you



# View the first 5 entries

df.head(5)
# We access a particular column in the dataframe through df['COLUMN_NAME']

df['Class']
# We can call the function value_counts to count the number of observations with each given value

v_count = df['Class'].value_counts()

print(v_count)
v_count[0] # 0 -> non-fraudulent, 1 -> fraudulent

# indices in Python ALWAYS start at 0
proportion_nofraud = v_count[0] / (v_count[0] + v_count[1])

proportion_fraud = v_count[1] / (v_count[0] + v_count[1])



# We can discuss this print statement another day

print(f"Frauds: {proportion_fraud*100:.2f}%\nNo Frauds: {proportion_nofraud*100:.2f}%")
sns.countplot(data=df, x='Class')

plt.title('Class Distributions')

plt.xticks(ticks=[0, 1], labels=["No Fraud", "Fraud"])
# Amount

amount = df['Amount']

sns.distplot(amount, color='red')

plt.title("Distribution of Transaction Amount")
# investigate the mean, median and standard deviation

amount_mean = amount.mean()

amount_median = amount.median()

amount_std = amount.std()



print(f"The mean amount is {amount_mean:.2f} USD, the median {amount_median:.2f} USD, and the standard deviation {amount_std:.2f} USD")
# Time

time = df['Time']

sns.distplot(time, color='blue')

plt.title("Distribution of Time")
# investigate the mean, median and standard deviation

time_mean = time.mean()

time_median = time.median()

time_std = time.std()



print(f"The mean amount is {time_mean:.2f} seconds, the median is{time_median:.2f} seconds, and the standard deviation is {time_std:.2f} seconds")
from sklearn.preprocessing import RobustScaler



# The following notation is a little disconcerting if you don't have any experience 

# with objects in a programming language before. Think of this as creating the function

# That we'll use.

robust_scaler = RobustScaler()



# the RobustScaler function only takes data in column-form so we reshape it

scaled_amount = robust_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

scaled_time = robust_scaler.fit_transform(df['Time'].values.reshape(-1,1))



# in pandas we can create a new column in the dataframe be simply assigning a data 

# (in the right shape) to a new name. In this case "scaled_amount" and "scaled_time" do 

# not exist as columns in the dataframe before we create it in the code below.

df['scaled_amount'] = scaled_amount

df['scaled_time'] = scaled_time
df.head(10)
# Creating a new dataframe, scaled_df, with only scaled features

scaled_df = df.drop(['Time', 'Amount'], axis=1)

scaled_df.head()
# Create a shuffled dataframe

shuffled_scaled_df = scaled_df.sample(frac=1, random_state=42)



# Use the first 80% of the observations for training

idx_80 = int(len(shuffled_scaled_df)*0.8) # index of the observation 80% down the dataframe



# .iloc is a function for getting a subsample of a dataframe based on index

# the :idx_80 means "up to, but not including, index `idx_80`"

scaled_df_train = shuffled_scaled_df.iloc[:idx_80]



# Use the last 20% for testing

# .iloc[idx_80:] gives us all the observations from `idx_80` to the end

scaled_df_test = shuffled_scaled_df.iloc[idx_80:]



# Double-check the proportions of of fraud to non-fraud are the same in train and test

counts_train = scaled_df_train['Class'].value_counts()

counts_test = scaled_df_test['Class'].value_counts()



prop_fraud_train = counts_train[1]/ (counts_train[0] + counts_train[1])

prop_fraud_test = counts_test[1]/ (counts_test[0] + counts_test[1])



print(f"Fraud in the training set: {prop_fraud_train*100:.4f}% ({counts_train[1]} in total)\nFraud in the test set: {prop_fraud_test*100:.4f}% ({counts_test[1]} in total)")
# Extract all the observations of fraud.

# .loc is a function that gives you a subset of the dataframe, in this case where 

# the 'Class' column is equal to 1

fraud_df = scaled_df_train.loc[scaled_df_train['Class'] == 1]



num_frauds = len(fraud_df)



# Extract an equal non-fraud observations in our shuffled dataframe

non_fraud_df = scaled_df_train.loc[scaled_df_train['Class'] == 0][:num_frauds]



# Glue (concatenate) these two together into one dataframe

ru_df_train = pd.concat([fraud_df, non_fraud_df])



# shuffle the dataframe that we glued together

ru_df_train = ru_df_train.sample(frac=1)



ru_df_train
sns.countplot('Class', data=ru_df_train)

plt.title("Equally Distributed Classes", fontsize=14)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier



# Create the dependent and independent variables

X_train = ru_df_train.drop("Class", axis=1)

y_train = ru_df_train['Class']
logistic_regression = LogisticRegression(penalty="l2", C=1000, solver="liblinear")

logistic_regression.fit(X_train, y_train)
knear_neighbors = KNeighborsClassifier(n_neighbors=4, weights="uniform")

knear_neighbors.fit(X_train, y_train)
# Separate out the dependent and independent variables in the testing data

X_test = scaled_df_test.drop("Class", axis=1)

y_test = scaled_df_test['Class']



lr_predictions = logistic_regression.predict(X_test)

knn_predictions = knear_neighbors.predict(X_test)
# How well did the models perform



# Number of correct predictions

lr_correct_preds = sum(y_test == lr_predictions)

lr_success_rate = lr_correct_preds / len(y_test)



knn_correct_preds = sum(y_test == knn_predictions)

knn_success_rate = knn_correct_preds / len(y_test)



print(f"Logistic Regression success rate: {lr_success_rate*100:.4f}%")

print(f"k-Nearest Neighbors success rate: {knn_success_rate*100:.4f}%")
only_nofraud = np.zeros(len(y_test))

nofraud_correct_preds = sum(y_test == only_nofraud)

nofraud_success_rate = nofraud_correct_preds / len(y_test)



print(f"No Fraud model success rate: {nofraud_success_rate*100:.4f}%")
from sklearn.metrics import confusion_matrix



nofraud_confusion = confusion_matrix(y_test, only_nofraud)

lr_confusion = confusion_matrix(y_test, lr_predictions)

knn_confusion = confusion_matrix(y_test, knn_predictions)



explanation = np.array([["True negatives", "False positives"], ["False negatives", "True positives"]])



print(f"A confusion matrix consists of:\n{explanation}")

print('-'*50)



print(f"If we only predict No Fraud:\n{nofraud_confusion}")

print('-'*50)



print(f"The confusion matrix for our logistic regression:\n{lr_confusion}")

print('-'*50)



print(f"The confusion matrix for our k-Nearest Neighbors:\n{knn_confusion}")

print('-'*50)
# Balanced accuracy score for the logistic regression model

lr_recall_fraud = lr_confusion[1, 1] / (lr_confusion[0, 1] + lr_confusion[1, 1]) # true positives / (true positives + false positives)

lr_recall_nofraud = lr_confusion[0, 0] / (lr_confusion[0, 0] + lr_confusion[1, 0]) # true negative / (true negatives + false negatives)



lr_balanced_score = (lr_recall_fraud + lr_recall_nofraud) / 2



# Balanced accuracy score for our k-Nearest Neighbors model

knn_recall_fraud = knn_confusion[1, 1] / (knn_confusion[0, 1] + knn_confusion[1, 1]) # true positives / (true positives + false positives)

knn_recall_nofraud = knn_confusion[0, 0] / (knn_confusion[0, 0] + knn_confusion[1, 0]) # true negative / (true negatives + false negatives)



knn_balanced_score = (knn_recall_fraud + knn_recall_nofraud) / 2



# Balanced accuracy score for the nofraud modell

nofraud_recall_fraud = 0 # we have no false or true positives, so this is technically undefined



nofraud_recall_nofraud = nofraud_confusion[0, 0] / (nofraud_confusion[0, 0] + nofraud_confusion[1, 0]) # true negative / (true negatives + false negatives)



nofraud_balanced_score = (nofraud_recall_fraud + nofraud_recall_nofraud) / 2



print("Balanced Accuracy Scores:")

print(f"Logistic Regression: {lr_balanced_score*100:.4f}%")

print(f"k-Nearest Neighbors: {knn_balanced_score*100:.4f}%")

print(f"Nofraud model: {nofraud_balanced_score*100:.4f}%")