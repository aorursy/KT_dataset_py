# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings("ignore")



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor



from sklearn.tree import export_graphviz

import pydot



import lightgbm



from sklearn import metrics



from sklearn.utils import shuffle, class_weight

from sklearn import preprocessing

from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate, KFold

from sklearn.metrics import recall_score, roc_auc_score, f1_score, accuracy_score, classification_report, confusion_matrix





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read data

df_original = pd.read_csv('/kaggle/input/new-age-bank-loans-cleaned/New Age Bank Loans.csv')



#Select Significant Columns

selected_cols = [ 'num_loan_status_binary', 'annual_inc', \

                'num_addr_state', 'total_acc', 'num_home_ownership', 'open_acc',  \

                'num_purpose', 'revol_bal', 'num_char_desc', 'num_emp_length', 'num_zip_code', \

                'delinq_2yrs', 'num_verification_status', 'dti', \

                'num_pub_rec_bankruptcies','loan_amnt', 'pub_rec', \

                'inq_last_6mths', 'revol_util', 'num_term', 'num_sub_grade']

feature_cols =  selected_cols[1:]



df = df_original[selected_cols]
corr = df.corr()

plt.figure(figsize = (10, 8))

sns.heatmap(corr)

plt.show()



corr['num_loan_status_binary'].sort_values(ascending = False)
df.num_loan_status_binary.value_counts(normalize=True)
df.num_loan_status_binary.value_counts()
df_major = df[df.num_loan_status_binary == 2]

df_minor = df[df.num_loan_status_binary == 1]

df_minor_upsampled = resample(df_minor, replace = True, n_samples = 34104, random_state = 2018)

df_minor_upsampled = pd.concat([df_minor_upsampled, df_major])

df_minor_upsampled.num_loan_status_binary.value_counts()
X = df_minor_upsampled.drop('num_loan_status_binary', axis = 1)

Y = df_minor_upsampled.num_loan_status_binary

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25, random_state=0)

mms = StandardScaler()

mms.fit(xtrain)

xtrain_scaled = mms.transform(xtrain)

xtest_scaled = mms.transform(xtest)
# instantiate and fit

linearRegr = LinearRegression()

linearRegr.fit(xtrain_scaled, ytrain)



# Predict

ypred = linearRegr.predict(xtest_scaled)



# Print Results

print("Mean Squared Error:",np.sqrt(metrics.mean_squared_error(ytest, ypred)))

print("Intercept:", linearRegr.intercept_)

list(zip(feature_cols, linearRegr.coef_))



# Calculate the absolute errors

errors = abs(ytest - ypred)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')



# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / ytest)



# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
# Running the Logistic Regression

logisticRegr = LogisticRegression()

logisticRegr.fit(xtrain_scaled, ytrain)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

ypred = logisticRegr.predict(xtest_scaled)
# Calculate/Print Precision Scores

print("Mean Squared Error:", np.sqrt(metrics.mean_squared_error(ytest, ypred)))

print("Accuracy (LR):", logisticRegr.score(xtest, ytest))

print("Accuracy (ME):",metrics.accuracy_score(ytest, ypred))

print("Precision:",metrics.precision_score(ytest, ypred))

print("Recall:",metrics.recall_score(ytest, ypred))



# Calculate and Print the Confusion Matrix

cnf_matrix = metrics.confusion_matrix(ytest, ypred)

class_names=[1,2] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)



# Create Heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')



#Print Coefficients

#list(zip(feature_cols, logisticRegr.coef_))

logisticRegr.coef_

randomForest = RandomForestRegressor(n_estimators = 1000, random_state = 42)

randomForest.fit(xtrain_scaled, ytrain);

ypred = randomForest.predict(xtest_scaled)

xtrain_scaled.shape[1]

df_minor_upsampled.head()
# Calculate the absolute errors

errors = abs(ytest - ypred)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')



# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / ytest)



# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
# Get numerical feature importances

importances = list(randomForest.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_cols, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];