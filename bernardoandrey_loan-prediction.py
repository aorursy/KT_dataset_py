# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.impute import SimpleImputer
# Read train csv

dataset = pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
# identify null value

dataset.isnull().sum()
# Divide category and number

cat_data = []

num_data = []

for i,c in enumerate(dataset.dtypes):

    if c == object:

        cat_data.append(dataset.iloc[:, i])

    else :

        num_data.append(dataset.iloc[:, i])  

cat_data = pd.DataFrame(cat_data).transpose()

num_data = pd.DataFrame(num_data).transpose()
# fill missing value

cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
# checking whether there is missing value

cat_data.isnull().sum().any()
# Encoding cat_data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_cat_data_2 = LabelEncoder()

cat_data.values[:, 2] = labelencoder_cat_data_2.fit_transform(cat_data.values[:, 2])

labelencoder_cat_data_5 = LabelEncoder()

cat_data.values[:, 5] = labelencoder_cat_data_5.fit_transform(cat_data.values[:, 5])

labelencoder_cat_data_6 = LabelEncoder()

cat_data.values[:, 6] = labelencoder_cat_data_6.fit_transform(cat_data.values[:, 6])

cat_data.Education = cat_data.Education.replace({"Graduate": "1"})

cat_data.Education = cat_data.Education.replace({"Not Graduate": "0"})
# Replace "3+" to "3"

cat_data.Dependents = cat_data.Dependents.replace({"3+": "3"})
# Change type object to int

cat_data['Married'] = cat_data.Married.astype(str).astype(int)

cat_data['Dependents'] = cat_data.Dependents.astype(str).astype(int)

cat_data['Education'] = cat_data.Education.astype(str).astype(int)

cat_data['Self_Employed'] = cat_data.Self_Employed.astype(str).astype(int)

cat_data['Property_Area'] = cat_data.Property_Area.astype(str).astype(int)
# drop Loan Status from cat_data and encode Loan_status

target_values = {'Y': 1 , 'N' : 0}

target = cat_data['Loan_Status']

cat_data.drop('Loan_Status', axis=1, inplace=True)

target = target.map(target_values)


imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer.fit(num_data.values[:, (2,3)])

num_data.values[:, (2,3)] = imputer.transform(num_data.values[:, (2,3)])

imputer1 = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

imputer1.fit(num_data.values[:,(3,4)])

num_data.values[:, (3,4)] = imputer1.transform(num_data.values[:, (3,4)])
num_data.isnull().sum()
# Combine cat_data, num_data, target

data = pd.concat([cat_data, num_data, target], axis=1)

data = data.iloc[:, 2:13]
# Show correlation of variable

corr = data.corr()

sns.heatmap(corr)

columns = np.full((corr.shape[0],), True, dtype=bool)
# Select Column Selected from Matrix Correlation

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False

selected_columns = data.columns[columns]

data = data[selected_columns]

selected_columns = selected_columns[0:10].values
# Backward Elimination for select column that effect to result

import statsmodels.api as sm

def backwardElimination(x, Y, sl, columns):

    numVars = len(x[0])

    for i in range(0, numVars):

        regressor_OLS = sm.OLS(Y, x).fit()

        maxVar = max(regressor_OLS.pvalues).astype(float)

        if maxVar > sl:

            for j in range(0, numVars - i):

                if (regressor_OLS.pvalues[j].astype(float) == maxVar):

                    x = np.delete(x, j, 1)

                    columns = np.delete(columns, j)

    

    regressor_OLS.summary()

    return x, columns

SL = 0.05

data_modeled, selected_columns = backwardElimination(data.iloc[:,0:10].values, data.iloc[:,10].values, SL, selected_columns)

y = pd.DataFrame()

y['Loan_Status'] = data.iloc[:,10]

X = pd.DataFrame(data = data_modeled, columns = selected_columns)
X
y
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scalling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Classifier



# Logistic Regression

from sklearn.linear_model import LogisticRegression

classifierLR = LogisticRegression(random_state = 0)

classifierLR.fit(X_train, y_train.values.ravel())

# KNearest Neighbor

from sklearn.neighbors import KNeighborsClassifier

classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifierKNN.fit(X_train, y_train.values.ravel())

# SVM GAUSSIAN

from sklearn.svm import SVC

classifierSVM = SVC(kernel = 'rbf', random_state = 0)

classifierSVM.fit(X_train, y_train.values.ravel())

# NAIVE BAYES

from sklearn.naive_bayes import GaussianNB

classifierNB = GaussianNB()

classifierNB.fit(X_train, y_train.values.ravel())

# Decision Tree Classification

from sklearn.tree import DecisionTreeClassifier

classifierDT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifierDT.fit(X_train, y_train.values.ravel())

# RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier

classifierRM = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)

classifierRM.fit(X_train, y_train.values.ravel())

# Prediction y_pred

y_pred_LR = classifierLR.predict(X_test)

y_pred_KNN = classifierKNN.predict(X_test)

y_pred_SVM = classifierSVM.predict(X_test)

y_pred_NB = classifierLR.predict(X_test)

y_pred_DT = classifierDT.predict(X_test)

y_pred_RM = classifierRM.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
# Calculate Confusion Matrix and 

cm_logreg = confusion_matrix(y_test, y_pred_LR)

as_logreg=accuracy_score(y_test, y_pred_LR)



cm_knn = confusion_matrix(y_test, y_pred_KNN)

as_knn=accuracy_score(y_test, y_pred_KNN)



cm_svm_gaussian = confusion_matrix(y_test, y_pred_SVM)

as_svm_gaussian = accuracy_score(y_test, y_pred_SVM)



cm_nb = confusion_matrix(y_test, y_pred_NB)

as_nb = accuracy_score(y_test, y_pred_NB)



cm_dtc = confusion_matrix(y_test, y_pred_DT)

as_dtc = accuracy_score(y_test, y_pred_DT)



cm_rfc = confusion_matrix(y_test, y_pred_RM)

as_rfc = accuracy_score(y_test, y_pred_RM)
# Find best Classifier

score={'as_logreg':as_logreg, 'as_knn':as_knn, 'as_svm_gaussian':as_svm_gaussian, 'as_nb':as_nb, 'as_dtc':as_dtc, 'as_rfc':as_rfc}

score_list=[]

for i in score:

    score_list.append(score[i])

    u=max(score_list)

    if score[i]==u:

        v=i  

    print(f"{i}={score[i]}");   

print(f"The best method to use in this case is {v} with accuracy score {u}")
dataset_test = pd.read_csv("../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv")
# identify null value

dataset_test.isnull().sum()
# Divide category and number

cat_data_test = []

num_data_test = []

for i,c in enumerate(dataset_test.dtypes):

    if c == object:

        cat_data_test.append(dataset_test.iloc[:, i])

    else :

        num_data_test.append(dataset_test.iloc[:, i])  

cat_data_test = pd.DataFrame(cat_data_test).transpose()

num_data_test = pd.DataFrame(num_data_test).transpose()
# cat_data

cat_data_test = cat_data_test.apply(lambda x:x.fillna(x.value_counts().index[0]))

labelencoder_cat_data_test = LabelEncoder()

cat_data_test.values[:, 2] = labelencoder_cat_data_test.fit_transform(cat_data_test.values[:, 2])

cat_data_test.values[:, 5] = labelencoder_cat_data_test.fit_transform(cat_data_test.values[:, 5])

cat_data_test.values[:, 6] = labelencoder_cat_data_test.fit_transform(cat_data_test.values[:, 6])

cat_data_test.Dependents = cat_data_test.Dependents.replace({"3+": "3"})

cat_data_test.Education = cat_data_test.Education.replace({"Graduate": "1"})

cat_data_test.Education = cat_data_test.Education.replace({"Not Graduate": "0"})
cat_data_test['Married'] = cat_data_test.Married.astype(str).astype(int)

cat_data_test['Dependents'] = cat_data_test.Dependents.astype(str).astype(int)

cat_data_test['Education'] = cat_data_test.Education.astype(str).astype(int)

cat_data_test['Self_Employed'] = cat_data_test.Self_Employed.astype(str).astype(int)

cat_data_test['Property_Area'] = cat_data_test.Property_Area.astype(str).astype(int)
imputer_test = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer_test.fit(num_data_test.values[:, (2,3)])

num_data_test.values[:, (2,3)] = imputer_test.transform(num_data_test.values[:, (2,3)])

imputer1_test = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

imputer1_test.fit(num_data_test.values[:,(3,4)])

num_data_test.values[:, (3,4)] = imputer1_test.transform(num_data_test.values[:, (3,4)])
# combine cat_data and num_data

data_test = pd.concat([cat_data_test, num_data_test], axis=1)

X_test_real = data_test.iloc[:, 2:12]
X_real_test = X_test_real.drop(['Dependents','Education', 'Self_Employed', 'Property_Area', 'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
X_test_real_scale = sc.fit_transform(X_real_test)
# Choose Classifier Naive Bayes because high accuracy score

y_pred_real = classifierNB.predict(X_test_real_scale)
y_pred_real