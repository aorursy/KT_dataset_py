# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        creditcard_dataset_path = os.path.join(dirname, filename)

        creditcard_data = pd.read_csv(creditcard_dataset_path)

        

# Any results you write to the current directory are saved as output.

print("Number of transactions: ", len(creditcard_data))

creditcard_data.tail()
fraudulent_amount = 0

number_of_fraudulent_transactions = 0;

for record in creditcard_data.itertuples():

    if record.Class == True:

        fraudulent_amount += record.Amount

        number_of_fraudulent_transactions += 1

print('The total amount of fraudulent transactions: ', fraudulent_amount)    

print('The total number of fraudulent transactions: ', number_of_fraudulent_transactions)  

fraudulent_mean = fraudulent_amount / number_of_fraudulent_transactions

        

nonfraudulent_amount = 0

number_of_nonfraudulent_transactions = 0

for record in creditcard_data.itertuples():

    if record.Class == False:

        nonfraudulent_amount += record.Amount

        number_of_nonfraudulent_transactions += 1

print('The total amount of non-fraudulent transactions: ', nonfraudulent_amount)    

print('The total number of fraudulent transactions: ', number_of_nonfraudulent_transactions)  

nonfraudulent_mean = nonfraudulent_amount / number_of_nonfraudulent_transactions
creditcard_data.isnull().any()
# Feature Extraction with PCA,without any normalization on data

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

# load data



all_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

X = creditcard_data[all_features]

y = creditcard_data['Class']



# feature extraction

pca = PCA()

fit = pca.fit(X.T)

# summarize components

print("Explained Variance: %s" % fit.explained_variance_ratio_)

print("Sum of eigenvalues: ", sum(fit.explained_variance_ratio_))

pca_result = fit.components_

# print(pca_result.shape)

# X = pca_result.T

pca_df = pd.DataFrame({'var':fit.explained_variance_ratio_,

             'PC':all_features})

plt.figure(figsize=(18,12))

sns.barplot(x='PC',y="var", 

           data=pca_df, color="c")

plt.show()
from sklearn import preprocessing



all_features_except_time = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

y = creditcard_data['Class']



# Time column is dropped, because it might be not helpful

X = creditcard_data.drop(['Class'], axis = 1)

min_max_scaler = preprocessing.MinMaxScaler()

X = min_max_scaler.fit_transform(X, y)



# feature extraction

pca = PCA()

fit = pca.fit(X.T)

# summarize components

print("Explained Variance: %s" % fit.explained_variance_ratio_)

print("Sum of eigenvalues: ", sum(fit.explained_variance_ratio_))



pca_df = pd.DataFrame({'var':fit.explained_variance_ratio_,

             'PC':all_features_except_time})

# print(pca_df)

plt.figure(figsize=(18,12))

sns.barplot(x='PC',y="var", 

           data=pca_df, color="c")

plt.show()
selected_features = pca_df.head(17)['PC']

X = creditcard_data[selected_features]
# from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeClassifier



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

print('train data dimensions: ', train_X.shape)

print('validation data dimensions: ', val_X.shape)



# fd_model = RandomForestRegressor(random_state=1)

tree_model = DecisionTreeClassifier(random_state=0)



# fd_model.fit(train_X, train_y)

tree_model.fit(train_X, train_y)



# fd_predicted = fd_model.predict(val_X)

tree_predicted = tree_model.predict(val_X)



# Tree visualization of our fitted model

from sklearn import tree

import graphviz 



dot_data = tree.export_graphviz(tree_model)

graph = graphviz.Source(dot_data)

graph
val_y = val_y.reset_index(drop = False) # Convert the selected validation data from series to dataframe as well as reseting its index.

print(val_y)
predicted_fraud_as_nonfraud_falsely = 0

predicted_fraud_as_fraud_truely = 0

predicted_nonfraud_as_fraud_falsely = 0

predicted_nonfraud_as_nonfraud_truely = 0



for index, record in val_y.Class.iteritems():

    if(record == 1): #a real fraud record

        if(tree_predicted[index] == 0): #predicted as nonfraud

            predicted_fraud_as_nonfraud_falsely = predicted_fraud_as_nonfraud_falsely + 1

        else: #predicted as fraud

            predicted_fraud_as_fraud_truely = predicted_fraud_as_fraud_truely + 1

    else: #a real nonfraud record

        if(tree_predicted[index] == 0): #predicted as nonfraud

            predicted_nonfraud_as_nonfraud_truely = predicted_nonfraud_as_nonfraud_truely + 1

        else: #predicted as fraud

            predicted_nonfraud_as_fraud_falsely = predicted_nonfraud_as_fraud_falsely + 1

 

print('predicted_fraud_as_nonfraud_falsely: ', predicted_fraud_as_nonfraud_falsely,

'\npredicted_fraud_as_fraud_truely: ', predicted_fraud_as_fraud_truely,

'\npredicted_nonfraud_as_fraud_falsely: ', predicted_nonfraud_as_fraud_falsely,

'\npredicted_nonfraud_as_nonfraud_truely: ', predicted_nonfraud_as_nonfraud_truely)



print('Rate of true prediction of real fraud transactions: ', predicted_fraud_as_fraud_truely / (predicted_fraud_as_fraud_truely + predicted_fraud_as_nonfraud_falsely))



print('Rate of false prediction of real nonfraud transactions: ', predicted_nonfraud_as_fraud_falsely / (predicted_nonfraud_as_fraud_falsely + predicted_nonfraud_as_nonfraud_truely))

from sklearn.metrics import mean_absolute_error

# print('forest model MAE: ', mean_absolute_error(val_y, fd_predicted))

print('tree model MAE: ', mean_absolute_error(val_y.Class, tree_predicted))
from sklearn.metrics import confusion_matrix

predicted_y = tree_predicted

y_test = np.array(val_y.Class)



print("Confusion Matrix:::")

print(confusion_matrix(y_test, predicted_y))



tn, fp, fn, tp = confusion_matrix(y_test, predicted_y).ravel()

print("\nTN: ", tn,

      "\nFP: ", fp,

      "\nFN: ", fn,

      "\nTP: ", tp)